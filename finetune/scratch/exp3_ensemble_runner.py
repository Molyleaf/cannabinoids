import argparse
import copy
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Dynamically ensure project root is in Python module search path
scratch_dir = Path(__file__).resolve().parent
finetune_dir = scratch_dir.parent
project_root = finetune_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.augmentation import SpectrumAugmentation
from finetune.lib import (
    BinaryClassifier,
    evaluate_and_record_predictions,
    evaluate_positive_per_smiles,
    export_results_to_excel,
    load_pretrained_encoder,
    plot_comprehensive_results,
    plot_confusion_matrices,
    prepare_finetune_dataset,
    save_model_checkpoint,
)
from finetune.lib.dataset import split_positive_by_smiles_negative_random
from finetune.lib.trainer import safe_auc


class FocalLoss(nn.Module):
    """
    Focal Loss (with pos_weight adjustment)
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, pos_weight=math.sqrt(1.8)):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: (B,), targets: (B,)
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.pos_weight is not None and self.pos_weight != 1.0:
            alpha_weight = targets * self.pos_weight + (1 - targets) * 1.0
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * bce_loss
        return loss.mean()


class AugmentedSpectrumDataset(Dataset):
    """
    PyTorch Dataset supporting dynamic mass spectrum data augmentation
    Applies SpectrumAugmentation on-the-fly during training iteration in __getitem__
    """
    def __init__(self, X, y, augmentor=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augmentor = augmentor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_vec = self.X[idx]
        if self.augmentor is not None:
            x_vec = self.augmentor(x_vec)
        return x_vec, self.y[idx]


def train_single_fold_model(
    model,
    train_loader,
    val_loader,
    device='cuda',
    warmup_epochs=15,
    unfreeze_epochs=75,
    classifier_lr=3e-4,
    encoder_lr=3e-5,
    patience=15,
    pos_weight=math.sqrt(1.8),
    use_focal_loss=True,
    max_grad_norm=1.0,
):
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    model = model.to(dev)

    if use_focal_loss:
        criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    else:
        pw_tensor = torch.tensor([pos_weight], device=dev, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    best_val_loss = float('inf')
    best_model_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Stage 1: Warmup Classifier
    model.freeze_encoder = True
    for param in model.encoder.parameters():
        param.requires_grad = False

    warmup_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_spec, batch_labels in train_loader:
            batch_spec, batch_labels = batch_spec.to(dev), batch_labels.to(dev)
            warmup_opt.zero_grad(set_to_none=True)
            logits = model(batch_spec)
            loss = criterion(logits, batch_labels)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            warmup_opt.step()

            train_loss += loss.item() * batch_spec.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)

        # Validation evaluation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec, batch_labels = batch_spec.to(dev), batch_labels.to(dev)
                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * batch_spec.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_spec.size(0)

        avg_val_loss = val_loss / val_total
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # Stage 2: Unfreeze Joint Fine-Tuning
    model.freeze_encoder = False
    for param in model.encoder.parameters():
        param.requires_grad = True

    joint_opt = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': classifier_lr, 'weight_decay': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        joint_opt, mode='min', factor=0.5, patience=4, min_lr=1e-7
    )

    patience_counter = 0
    total_epochs = warmup_epochs + unfreeze_epochs

    for epoch in range(warmup_epochs + 1, total_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_spec, batch_labels in train_loader:
            batch_spec, batch_labels = batch_spec.to(dev), batch_labels.to(dev)
            joint_opt.zero_grad(set_to_none=True)
            logits = model(batch_spec)
            loss = criterion(logits, batch_labels)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            joint_opt.step()

            train_loss += loss.item() * batch_spec.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)

        # Validation evaluation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec, batch_labels = batch_spec.to(dev), batch_labels.to(dev)
                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * batch_spec.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_spec.size(0)

        avg_val_loss = val_loss / val_total
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def evaluate_ensemble_predictions(models, data_loader, device='cuda', threshold=0.5):
    """
    Compute Soft-Voting ensemble predictions and evaluation metrics from 5 Fold models
    """
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    for m in models:
        m.eval()
        m.to(dev)

    all_ensemble_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_spec, batch_labels in data_loader:
            batch_spec = batch_spec.to(dev)
            batch_labels = batch_labels.to(dev)

            batch_probs_list = []
            for m in models:
                logits = m(batch_spec)
                probs = torch.sigmoid(logits)
                batch_probs_list.append(probs.cpu().numpy().ravel())

            # Soft Voting ensemble: average prediction probabilities across 5 models
            mean_probs = np.mean(batch_probs_list, axis=0)
            all_ensemble_probs.extend(mean_probs)
            all_labels.extend(batch_labels.cpu().numpy().ravel())

    all_probs = np.array(all_ensemble_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(float)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = safe_auc(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm,
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
        'n_samples': int(len(all_labels)),
        'n_neg': int((all_labels == 0).sum()),
        'n_pos': int((all_labels == 1).sum()),
    }


def find_optimal_threshold(val_res, metric='f1'):
    probs = val_res['probs']
    labels = val_res['labels']
    thresholds = np.linspace(0.10, 0.90, 81)
    best_th = 0.50
    best_score = -1.0
    records = []

    for th in thresholds:
        preds = (probs >= th).astype(float)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_j = rec + spec - 1.0

        score = f1 if metric == 'f1' else youden_j
        records.append({
            'threshold': float(th), 'f1': float(f1), 'accuracy': float(acc),
            'precision': float(prec), 'recall': float(rec), 'youden_j': float(youden_j)
        })

        if score > best_score:
            best_score = score
            best_th = float(th)

    return best_th, best_score, records


def plot_threshold_search_curve(records, best_th, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ths = [r['threshold'] for r in records]
    f1s = [r['f1'] for r in records]
    accs = [r['accuracy'] for r in records]
    precs = [r['precision'] for r in records]
    recs = [r['recall'] for r in records]
    j_stats = [r['youden_j'] for r in records]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ths, f1s, label='F1-Score', color='#1f77b4', lw=2.5)
    ax.plot(ths, accs, label='Accuracy', color='#2ca02c', lw=2.0, linestyle='--')
    ax.plot(ths, precs, label='Precision', color='#ff7f0e', lw=1.8, linestyle=':')
    ax.plot(ths, recs, label='Recall', color='#d62728', lw=1.8, linestyle='-.')
    ax.plot(ths, j_stats, label="Youden's J Index", color='#9467bd', lw=1.5, linestyle='-')

    ax.axvline(best_th, color='black', linestyle=':', lw=1.5, label=f'Optimal Threshold ({best_th:.2f})')
    ax.axvline(0.50, color='gray', linestyle='--', lw=1.0, label='Default Threshold (0.50)')

    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.set_title("5-Fold Ensemble Threshold Optimization", fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = output_dir / 'threshold_optimization_curve.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="High-performance 5-Fold Unfreeze Fine-Tuning + Dynamic Augmentation + Soft Voting Ensemble")
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=str(project_root / "embedding_pretrain" / "results_20260725_095428" / "best_model.pt"),
        help="Base pre-trained encoder weights (best_model.pt)"
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--unfreeze_epochs", type=int, default=75)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--classifier_lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pos_weight", type=float, default=math.sqrt(1.8))
    args = parser.parse_args()

    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Pre-trained weights file not found: {encoder_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 75, flush=True)
    print("[Ultimate Experiment] 5-Fold SMILES Cross-Validation + Dynamic Augmentation + Focal Loss + Soft Voting Ensemble", flush=True)
    print("=" * 75, flush=True)
    print(f"Base pre-trained model: {encoder_path}", flush=True)
    print(f"Device: {device.upper()}", flush=True)
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 1. Prepare full feature matrix and index splits (reserve fixed 15% independent test set)
    _, _, _, test_loader, meta = prepare_finetune_dataset(
        batch_size=args.batch_size,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        use_cuda=(device == 'cuda')
    )

    X, y = meta['X'], meta['y']
    train_val_idx = np.concatenate([meta['train_idx'], meta['val_idx']])
    test_idx = meta['test_idx']

    print(f"  Development Pool (Train+Val): {len(train_val_idx)} samples", flush=True)
    print(f"  Independent Test Set:         {len(test_idx)} samples", flush=True)

    # Build 5-Fold K-Fold indices on the development pool
    np.random.seed(42)
    pool_indices = train_val_idx.copy()
    np.random.shuffle(pool_indices)

    fold_size = len(pool_indices) // args.n_folds
    folds_idx = []
    for k in range(args.n_folds):
        start_k = k * fold_size
        end_k = (k + 1) * fold_size if k < args.n_folds - 1 else len(pool_indices)
        folds_idx.append(pool_indices[start_k:end_k])

    # Data augmentor (applied only to training set)
    finetune_augmentor = SpectrumAugmentation(mode='finetune_train')

    models = []
    fold_metrics = []

    print("\n" + "=" * 75, flush=True)
    print(f"Step 2: Sequentially training {args.n_folds}-Fold models (Two-Stage Unfreeze + Dynamic Augmentation + Focal Loss)", flush=True)
    print("=" * 75, flush=True)

    for k in range(args.n_folds):
        val_idx_k = folds_idx[k]
        train_idx_k = np.concatenate([folds_idx[i] for i in range(args.n_folds) if i != k])

        train_ds_k = AugmentedSpectrumDataset(X[train_idx_k], y[train_idx_k], augmentor=finetune_augmentor)
        val_ds_k = AugmentedSpectrumDataset(X[val_idx_k], y[val_idx_k], augmentor=None)

        train_loader_k = DataLoader(train_ds_k, batch_size=args.batch_size, shuffle=True)
        val_loader_k = DataLoader(val_ds_k, batch_size=args.batch_size, shuffle=False)

        # Initialize and load independent encoder weights for this Fold
        encoder_k = load_pretrained_encoder(encoder_path, input_dim=561, hidden_dim=256, device=device)
        model_k = BinaryClassifier(encoder_k, input_dim=256, freeze_encoder=True)

        print(f"\n>>> Training Fold {k+1}/{args.n_folds} (Train: {len(train_ds_k)} samples, Val: {len(val_ds_k)} samples)...", flush=True)
        model_k, best_val_loss_k = train_single_fold_model(
            model=model_k,
            train_loader=train_loader_k,
            val_loader=val_loader_k,
            device=device,
            warmup_epochs=args.warmup_epochs,
            unfreeze_epochs=args.unfreeze_epochs,
            classifier_lr=args.classifier_lr,
            encoder_lr=args.encoder_lr,
            patience=args.patience,
            pos_weight=args.pos_weight,
            use_focal_loss=True
        )

        models.append(model_k)
        print(f"    [Fold {k+1} OK] Best Val Loss: {best_val_loss_k:.4f}", flush=True)

    # 3. Threshold optimization on development pool validation set using 5-Fold ensemble
    print("\n" + "=" * 75, flush=True)
    print("Step 3: Decision threshold optimization on development pool using 5-Fold ensemble", flush=True)
    print("=" * 75, flush=True)

    val_pool_ds = AugmentedSpectrumDataset(X[train_val_idx], y[train_val_idx], augmentor=None)
    val_pool_loader = DataLoader(val_pool_ds, batch_size=args.batch_size, shuffle=False)

    val_ensemble_res = evaluate_ensemble_predictions(models, val_pool_loader, device=device, threshold=0.50)
    best_th, best_f1, th_records = find_optimal_threshold(val_ensemble_res, metric='f1')

    print(f"  [5-Fold Ensemble Optimization Results]:", flush=True)
    print(f"     Default threshold (T=0.50): Acc: {val_ensemble_res['accuracy']:.2%} | F1: {val_ensemble_res['f1']:.4f} | AUC: {val_ensemble_res['auc']:.4f}", flush=True)
    print(f"     Optimal threshold (T_opt={best_th:.2f}): F1: {best_f1:.4f}", flush=True)

    # 4. Ultimate evaluation on independent test set using Soft-Voting ensemble
    print("\n" + "=" * 75, flush=True)
    print("Step 4: Ultimate evaluation on independent test set (Soft Voting ensemble prediction)", flush=True)
    print("=" * 75, flush=True)

    test_ensemble_05 = evaluate_ensemble_predictions(models, test_loader, device=device, threshold=0.50)
    test_ensemble_opt = evaluate_ensemble_predictions(models, test_loader, device=device, threshold=best_th)

    print("\n" + "=" * 75, flush=True)
    print(f"[Ultimate Test Set Soft Voting Ensemble Results (T=0.50)]", flush=True)
    print(f"  Accuracy:  {test_ensemble_05['accuracy']:.2%}  (Target: 95%+)")
    print(f"  Precision: {test_ensemble_05['precision']:.2%}")
    print(f"  Recall:    {test_ensemble_05['recall']:.2%}")
    print(f"  F1-Score:  {test_ensemble_05['f1']:.4f}")
    print(f"  AUC:       {test_ensemble_05['auc']:.4f}")
    print("=" * 75 + "\n", flush=True)

    print(f"[Ultimate Test Set Soft Voting Ensemble Results (T_opt={best_th:.2f})]", flush=True)
    print(f"  Accuracy:  {test_ensemble_opt['accuracy']:.2%}")
    print(f"  Precision: {test_ensemble_opt['precision']:.2%}")
    print(f"  Recall:    {test_ensemble_opt['recall']:.2%}")
    print(f"  F1-Score:  {test_ensemble_opt['f1']:.4f}")
    print(f"  AUC:       {test_ensemble_opt['auc']:.4f}")

    smiles_res_05 = evaluate_positive_per_smiles(
        test_indices=meta['test_idx'],
        smiles_all=meta['smiles_all'],
        labels_all=meta['y'],
        probs=test_ensemble_05['probs'],
        preds=test_ensemble_05['preds'],
        threshold=0.50
    )

    smiles_res_opt = evaluate_positive_per_smiles(
        test_indices=meta['test_idx'],
        smiles_all=meta['smiles_all'],
        labels_all=meta['y'],
        probs=test_ensemble_opt['probs'],
        preds=test_ensemble_opt['preds'],
        threshold=best_th
    )

    # 5. Save models and export results
    print("\n" + "=" * 75, flush=True)
    print("Step 5: Saving 5-Fold ensemble model weights and all comparison charts", flush=True)
    print("=" * 75, flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = scratch_dir / f"results_exp3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save all 5-Fold models
    for k, m in enumerate(models):
        torch.save({
            'encoder_state_dict': m.encoder.state_dict(),
            'classifier_state_dict': m.classifier.state_dict(),
            'fold': k + 1
        }, str(run_dir / f"ensemble_fold_{k+1}_classifier.pt"))

    print(f"  [OK] All 5 Fold classifier model weights saved to: {run_dir}", flush=True)

    # Export ultimate test comparison report CSV
    with open(run_dir / 'ultimate_test_comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Model_Configuration', 'Threshold', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC', 'SMILES_Recognized'])
        writer.writerow([
            'Baseline (Frozen Encoder)', 0.50, 0.9137, 0.9013, 0.8739, 0.8874, 0.9697, '81/98 (82.65%)'
        ])
        writer.writerow([
            'Exp1 (Single Unfrozen Model)', 0.50, 0.9255, 0.9189, 0.8870, 0.9027, 0.9730, '81/98 (82.65%)'
        ])
        writer.writerow([
            'Exp3 (5-Fold Ensemble + Dynamic Aug + Focal Loss)', 0.50,
            test_ensemble_05['accuracy'], test_ensemble_05['precision'], test_ensemble_05['recall'],
            test_ensemble_05['f1'], test_ensemble_05['auc'],
            f"{smiles_res_05['n_correct']}/{smiles_res_05['n_smiles']} ({smiles_res_05['accuracy']:.2%})" if smiles_res_05 else "N/A"
        ])
        writer.writerow([
            f'Exp3 (5-Fold Ensemble + Optimal T={best_th:.2f})', best_th,
            test_ensemble_opt['accuracy'], test_ensemble_opt['precision'], test_ensemble_opt['recall'],
            test_ensemble_opt['f1'], test_ensemble_opt['auc'],
            f"{smiles_res_opt['n_correct']}/{smiles_res_opt['n_smiles']} ({smiles_res_opt['accuracy']:.2%})" if smiles_res_opt else "N/A"
        ])

    plot_threshold_search_curve(th_records, best_th, run_dir)
    print(f"\n  [OK] All results from Ultimate Experiment Exp3 exported to:\n       --> {run_dir}", flush=True)
    print("=" * 75, flush=True)


if __name__ == '__main__':
    main()