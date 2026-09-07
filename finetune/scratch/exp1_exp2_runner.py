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

# Dynamically ensure project root is in Python module search path
scratch_dir = Path(__file__).resolve().parent
finetune_dir = scratch_dir.parent
project_root = finetune_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
from finetune.lib.trainer import safe_auc


def train_two_stage_binary_classifier(
    model,
    train_loader,
    val_loader,
    device='cuda',
    warmup_epochs=15,
    unfreeze_epochs=85,
    classifier_lr=3e-4,
    encoder_lr=3e-5,
    patience=15,
    pos_weight=math.sqrt(1.8),
    max_grad_norm=1.0,
):
    """
    Two-stage unfreeze encoder fine-tuning training pipeline
    - Stage 1 (Warmup): Freeze Encoder, train only Classifier Head
    - Stage 2 (Joint Fine-Tuning): Unfreeze Encoder, end-to-end fine-tuning with differential learning rates
    """
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    print(f"\nTraining device: {dev}", flush=True)
    model = model.to(dev)

    if pos_weight is not None and pos_weight != 1.0:
        pw_tensor = torch.tensor([pos_weight], device=dev, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"  [INFO] Positive class weighting enabled: pos_weight = {pos_weight:.4f}", flush=True)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []

    # ==================== Stage 1: Freeze Encoder, Warmup Classifier ====================
    print("\n" + "=" * 60, flush=True)
    print(f"[Stage 1] Classifier Warmup ({warmup_epochs} Epochs, Encoder Frozen)", flush=True)
    print("=" * 60, flush=True)

    model.freeze_encoder = True
    for param in model.encoder.parameters():
        param.requires_grad = False

    warmup_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_probs_all, train_labels_all = [], []

        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(dev)
            batch_labels = batch_labels.to(dev)

            warmup_optimizer.zero_grad(set_to_none=True)
            logits = model(batch_spec)
            loss = criterion(logits, batch_labels)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            warmup_optimizer.step()

            train_loss += loss.item() * batch_spec.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)

            train_probs_all.extend(probs.detach().cpu().numpy().ravel())
            train_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = safe_auc(train_labels_all, train_probs_all)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs_all, val_labels_all = [], []

        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec = batch_spec.to(dev)
                batch_labels = batch_labels.to(dev)

                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_spec.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_spec.size(0)

                val_probs_all.extend(probs.cpu().numpy().ravel())
                val_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = safe_auc(val_labels_all, val_probs_all)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Warmup Epoch {epoch:2d}/{warmup_epochs:2d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | Val AUC: {val_auc:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"  [OK] Stage 1 complete. Best Val Loss: {best_val_loss:.4f}", flush=True)

    # ==================== Stage 2: Unfreeze Encoder, Joint Fine-Tuning ====================
    print("\n" + "=" * 60, flush=True)
    print(f"[Stage 2] Unfreeze Encoder End-to-End Fine-Tuning (Encoder LR: {encoder_lr:.1e}, Classifier LR: {classifier_lr:.1e})", flush=True)
    print("=" * 60, flush=True)

    model.freeze_encoder = False
    for param in model.encoder.parameters():
        param.requires_grad = True

    joint_optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': classifier_lr, 'weight_decay': 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        joint_optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7
    )

    patience_counter = 0
    total_epochs = warmup_epochs + unfreeze_epochs

    for epoch in range(warmup_epochs + 1, total_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_probs_all, train_labels_all = [], []

        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(dev)
            batch_labels = batch_labels.to(dev)

            joint_optimizer.zero_grad(set_to_none=True)
            logits = model(batch_spec)
            loss = criterion(logits, batch_labels)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            joint_optimizer.step()

            train_loss += loss.item() * batch_spec.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)

            train_probs_all.extend(probs.detach().cpu().numpy().ravel())
            train_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = safe_auc(train_labels_all, train_probs_all)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs_all, val_labels_all = [], []

        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec = batch_spec.to(dev)
                batch_labels = batch_labels.to(dev)

                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_spec.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_spec.size(0)

                val_probs_all.extend(probs.cpu().numpy().ravel())
                val_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = safe_auc(val_labels_all, val_probs_all)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        scheduler.step(avg_val_loss)
        enc_lr_curr = joint_optimizer.param_groups[0]['lr']
        cls_lr_curr = joint_optimizer.param_groups[1]['lr']

        if (epoch - warmup_epochs) % 5 == 0 or epoch == warmup_epochs + 1:
            print(f"Epoch {epoch:3d}/{total_epochs:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | Val AUC: {val_auc:.4f} | Enc LR: {enc_lr_curr:.1e}", flush=True)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (Patience={patience}) at Epoch {epoch}", flush=True)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"  [OK] Restored best global model weights (Best Val Loss: {best_val_loss:.4f})", flush=True)

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'train_auc': train_aucs,
        'val_auc': val_aucs,
    }

    return model, history


def find_optimal_threshold(val_res, metric='accuracy'):
    """
    Automatically search for optimal decision threshold T_opt on validation set
    Maximizes the specified metric (default: accuracy)
    """
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

        # Use accuracy as the optimization metric
        score = acc

        records.append({
            'threshold': float(th),
            'accuracy': float(acc),
            'f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'youden_j': float(youden_j)
        })

        if score > best_score:
            best_score = score
            best_th = float(th)

    return best_th, best_score, records


def plot_threshold_search_curve(records, best_th, output_dir):
    """Plot validation set threshold search curve"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ths = [r['threshold'] for r in records]
    accs = [r['accuracy'] for r in records]
    f1s = [r['f1'] for r in records]
    precs = [r['precision'] for r in records]
    recs = [r['recall'] for r in records]
    j_stats = [r['youden_j'] for r in records]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ths, accs, label='Accuracy', color='#1f77b4', lw=2.5)
    ax.plot(ths, f1s, label='F1-Score', color='#2ca02c', lw=2.0, linestyle='--')
    ax.plot(ths, precs, label='Precision', color='#ff7f0e', lw=1.8, linestyle=':')
    ax.plot(ths, recs, label='Recall', color='#d62728', lw=1.8, linestyle='-.')
    ax.plot(ths, j_stats, label="Youden's J Index", color='#9467bd', lw=1.5, linestyle='-')

    ax.axvline(best_th, color='black', linestyle=':', lw=1.5, label=f'Optimal Threshold ({best_th:.2f})')
    ax.axvline(0.50, color='gray', linestyle='--', lw=1.0, label='Default Threshold (0.50)')

    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.set_title("Validation Set Decision Threshold Optimization (Maximizing Accuracy)", fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = output_dir / 'threshold_optimization_curve.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  [OK] Threshold optimization curve saved: {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 (Unfreeze Fine-Tuning) + Experiment 2 (Threshold Optimization)")
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=str(project_root / "embedding_pretrain" / "results_20260725_095428" / "best_model.pt"),
        help="Base pre-trained encoder weights (best_model.pt)"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--unfreeze_epochs", type=int, default=85)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--classifier_lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pos_weight", type=float, default=math.sqrt(1.8))
    args = parser.parse_args()

    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Pre-trained weights file not found: {encoder_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70, flush=True)
    print("Experiment 1 (Two-Stage Unfreeze Fine-Tuning) & Experiment 2 (Threshold Optimization)", flush=True)
    print("=" * 70, flush=True)
    print(f"Base pre-trained model: {encoder_path}", flush=True)
    print(f"Device: {device.upper()}", flush=True)
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Encoder fine-tuning LR: {args.encoder_lr}", flush=True)
    print(f"Classifier fine-tuning LR: {args.classifier_lr}", flush=True)

    # 1. Data preparation
    print("\n" + "=" * 70, flush=True)
    print("Step 1: Loading and splitting dataset", flush=True)
    print("=" * 70, flush=True)
    train_loader, train_eval_loader, val_loader, test_loader, meta = prepare_finetune_dataset(
        batch_size=args.batch_size,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        use_cuda=(device == 'cuda')
    )

    # 2. Build model and load pre-trained weights
    print("\n" + "=" * 70, flush=True)
    print("Step 2: Instantiating model and loading pre-trained Encoder (best_model.pt)", flush=True)
    print("=" * 70, flush=True)
    encoder = load_pretrained_encoder(
        encoder_path=encoder_path,
        input_dim=561,
        hidden_dim=256,
        device=device
    )
    model = BinaryClassifier(encoder, input_dim=256, freeze_encoder=True)

    # 3. Experiment 1: Two-stage unfreeze fine-tuning
    print("\n" + "=" * 70, flush=True)
    print("Step 3: Running [Experiment 1] Two-Stage Unfreeze End-to-End Fine-Tuning", flush=True)
    print("=" * 70, flush=True)
    start_time = datetime.now()
    model, history = train_two_stage_binary_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        warmup_epochs=args.warmup_epochs,
        unfreeze_epochs=args.unfreeze_epochs,
        classifier_lr=args.classifier_lr,
        encoder_lr=args.encoder_lr,
        patience=args.patience,
        pos_weight=args.pos_weight
    )
    elapsed = datetime.now() - start_time
    print(f"\n  [OK] Unfreeze fine-tuning complete. Total time: {str(elapsed).split('.')[0]}", flush=True)

    # 4. Experiment 2: Threshold optimization on validation set
    print("\n" + "=" * 70, flush=True)
    print("Step 4: Running [Experiment 2] Validation Set Threshold Optimization", flush=True)
    print("=" * 70, flush=True)

    val_res_default = evaluate_and_record_predictions(model, val_loader, sample_names=meta['val_sample_names'], device=device, threshold=0.50)
    best_th, best_acc, th_records = find_optimal_threshold(val_res_default, metric='accuracy')

    print(f"  [Search Result] Validation set threshold scan complete (0.10 ~ 0.90):", flush=True)
    print(f"     Default threshold: T = 0.50 | Val Acc: {val_res_default['accuracy']:.2%} | Val F1: {val_res_default['f1']:.4f} | Val AUC: {val_res_default['auc']:.4f}", flush=True)
    print(f"     Optimal threshold: T_opt = {best_th:.2f} | Val Accuracy: {best_acc:.2%}", flush=True)

    # 5. Test set evaluation with both thresholds
    print("\n" + "=" * 70, flush=True)
    print("Step 5: Test Set Performance Comparison (Default T=0.50 vs Optimal T_opt)", flush=True)
    print("=" * 70, flush=True)

    train_res_05 = evaluate_and_record_predictions(model, train_eval_loader, sample_names=meta['train_sample_names'], device=device, threshold=0.50)
    val_res_05 = evaluate_and_record_predictions(model, val_loader, sample_names=meta['val_sample_names'], device=device, threshold=0.50)
    test_res_05 = evaluate_and_record_predictions(model, test_loader, sample_names=meta['test_sample_names'], device=device, threshold=0.50)

    test_res_opt = evaluate_and_record_predictions(model, test_loader, sample_names=meta['test_sample_names'], device=device, threshold=best_th)

    print("\n--- Default Threshold (T=0.50) Test Set Results ---", flush=True)
    print(f"  Test Accuracy: {test_res_05['accuracy']:.2%} | Precision: {test_res_05['precision']:.2%} | Recall: {test_res_05['recall']:.2%} | F1: {test_res_05['f1']:.4f} | AUC: {test_res_05['auc']:.4f}", flush=True)

    print(f"\n--- Optimal Threshold (T_opt={best_th:.2f}) Test Set Results ---", flush=True)
    print(f"  Test Accuracy: {test_res_opt['accuracy']:.2%} | Precision: {test_res_opt['precision']:.2%} | Recall: {test_res_opt['recall']:.2%} | F1: {test_res_opt['f1']:.4f} | AUC: {test_res_opt['auc']:.4f}", flush=True)

    smiles_res_05 = evaluate_positive_per_smiles(
        test_indices=meta['test_idx'],
        smiles_all=meta['smiles_all'],
        labels_all=meta['y'],
        probs=test_res_05['probs'],
        preds=test_res_05['preds'],
        threshold=0.50
    )

    smiles_res_opt = evaluate_positive_per_smiles(
        test_indices=meta['test_idx'],
        smiles_all=meta['smiles_all'],
        labels_all=meta['y'],
        probs=test_res_opt['probs'],
        preds=test_res_opt['preds'],
        threshold=best_th
    )

    # 6. Save and export results
    print("\n" + "=" * 70, flush=True)
    print("Step 6: Saving model and exporting results", flush=True)
    print("=" * 70, flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = scratch_dir / f"results_exp1_exp2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / f"unfrozen_binary_classifier_{timestamp}.pt"
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'classifier_state_dict': model.classifier.state_dict(),
        'optimal_threshold': best_th,
        'history': history,
        'test_metrics_default': test_res_05,
        'test_metrics_opt': test_res_opt,
        'base_encoder_path': str(encoder_path)
    }, str(checkpoint_path))
    print(f"  [OK] Unfrozen model weights saved to: {checkpoint_path}", flush=True)

    export_results_to_excel(
        history=history,
        model=model,
        train_results=train_res_05,
        val_results=val_res_05,
        test_results=test_res_opt,
        smiles_results=smiles_res_opt,
        meta=meta,
        output_dir=run_dir
    )

    with open(run_dir / 'threshold_search_records.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'accuracy', 'f1', 'precision', 'recall', 'youden_j'])
        writer.writeheader()
        writer.writerows(th_records)

    with open(run_dir / 'experiment_comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Threshold', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC', 'SMILES_Recognized'])
        writer.writerow([
            'Exp1 (Unfrozen Encoder, Default T=0.50)',
            0.50,
            test_res_05['accuracy'],
            test_res_05['precision'],
            test_res_05['recall'],
            test_res_05['f1'],
            test_res_05['auc'],
            f"{smiles_res_05['n_correct']}/{smiles_res_05['n_smiles']} ({smiles_res_05['accuracy']:.2%})" if smiles_res_05 else "N/A"
        ])
        writer.writerow([
            'Exp1+Exp2 (Unfrozen Encoder + Optimal T_opt)',
            best_th,
            test_res_opt['accuracy'],
            test_res_opt['precision'],
            test_res_opt['recall'],
            test_res_opt['f1'],
            test_res_opt['auc'],
            f"{smiles_res_opt['n_correct']}/{smiles_res_opt['n_smiles']} ({smiles_res_opt['accuracy']:.2%})" if smiles_res_opt else "N/A"
        ])

    plot_comprehensive_results(history, train_res_05, val_res_05, test_res_opt, smiles_res_opt, run_dir)
    plot_confusion_matrices(train_res_05, val_res_05, test_res_opt, run_dir)
    plot_threshold_search_curve(th_records, best_th, run_dir)

    print(f"\n  [OK] All results from Experiment 1 and Experiment 2 exported to:\n       --> {run_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()