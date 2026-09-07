import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
import pandas as pd
from datetime import datetime
import re

warnings.filterwarnings('ignore')


# ==================== Encoder Architecture ====================

class SpectrumEncoder(nn.Module):
    """Encoder architecture identical to the pre-trained model"""
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        return F.normalize(embed, p=2, dim=1)


# ==================== Classifier ====================

class UltimateClassifier(nn.Module):
    """Classifier with pre-trained encoder + classification head"""
    def __init__(self, encoder, input_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        
        self.classifier = nn.ModuleDict({
            'deep_branch': nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.7),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            ),
            'shallow_branch': nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 64),
            ),
            'fusion': nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.3),
                nn.Linear(64, 1),
            ),
        })
    
    def forward(self, x):
        embed = self.encoder(x)
        deep_out = self.classifier['deep_branch'](embed)
        shallow_out = self.classifier['shallow_branch'](embed)
        combined = torch.cat([deep_out, shallow_out], dim=1)
        logit = self.classifier['fusion'](combined)
        return logit.squeeze(-1)


# ==================== Loss Functions ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# ==================== Mixup ====================

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== Training Functions ====================

def train_two_stage(model, train_loader, val_loader, device, config):
    """Two-stage progressive unfreezing: freeze encoder then fully unfreeze"""
    
    # Stage 1: Train classification head only (encoder frozen)
    print("\n  === Stage 1: Training classification head (encoder frozen) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    stage1_config = {
        'epochs': 20,
        'lr': 1e-3,
        'patience': 10,
        'use_focal': config.get('use_focal', True),
        'use_mixup': config.get('use_mixup', False),
        'mixup_alpha': 0.2,
        'warmup_epochs': 5,
        'weight_decay': 1e-4,
        'encoder_lr_ratio': 0.0,
    }
    model, _ = train_single_stage(model, train_loader, val_loader, device, stage1_config)
    
    # Stage 2: Fully unfreeze and fine-tune
    print("\n  === Stage 2: Fully unfrozen fine-tuning ===")
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    stage2_config = {
        'epochs': 80,
        'lr': 5e-5,
        'patience': 25,
        'use_focal': config.get('use_focal', True),
        'use_mixup': config.get('use_mixup', False),
        'mixup_alpha': 0.2,
        'warmup_epochs': 5,
        'weight_decay': 1e-4,
        'encoder_lr_ratio': 0.1,
    }
    model, _ = train_single_stage(model, train_loader, val_loader, device, stage2_config)
    
    return model


def train_single_stage(model, train_loader, val_loader, device, config):
    """Single-stage training with optional layer-wise learning rates"""
    
    # Split parameters
    encoder_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                classifier_params.append(param)
    
    lr = config.get('lr', 1e-4)
    encoder_lr_ratio = config.get('encoder_lr_ratio', 1.0)
    
    param_groups = [
        {'params': classifier_params, 'lr': lr},
    ]
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': lr * encoder_lr_ratio})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.get('weight_decay', 1e-4))
    
    # Loss function
    use_focal = config.get('use_focal', True)
    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print(f"  Using Focal Loss (gamma=2.0, alpha=0.25)")
    else:
        # Compute class weights for BCE
        all_labels = []
        for _, batch_labels in train_loader:
            all_labels.extend(batch_labels.numpy())
        all_labels = np.array(all_labels)
        n_pos = (all_labels == 1).sum()
        n_neg = (all_labels == 0).sum()
        pos_weight = torch.tensor([n_neg / n_pos]).to(device) if n_pos > 0 else torch.tensor([1.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  Using BCE Loss (pos_weight={pos_weight.item():.3f})")
    
    epochs = config.get('epochs', 100)
    warmup_epochs = config.get('warmup_epochs', 5)
    patience = config.get('patience', 15)
    use_mixup = config.get('use_mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        
        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(device)
            batch_labels = batch_labels.to(device)
            
            if use_mixup:
                batch_spec, labels_a, labels_b, lam = mixup_data(batch_spec, batch_labels, mixup_alpha)
                optimizer.zero_grad()
                logits = model(batch_spec)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_spec.size(0)
            train_total += batch_spec.size(0)
        
        avg_train_loss = train_loss / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        
        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec = batch_spec.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * batch_spec.size(0)
                val_total += batch_spec.size(0)
        
        avg_val_loss = val_loss / val_total
        
        # Learning rate scheduling
        if epoch <= warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        if (epoch - 1) % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, None


# ==================== Data Processing ====================

def parse_msp_with_smiles(msp_file, min_peaks=5):
    """Parse MSP file and extract spectra with SMILES"""
    with open(msp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    compounds = []
    current_comp = None
    in_peaks = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('Name:'):
            if current_comp is not None and 'peaks' in current_comp:
                if len(current_comp['peaks']) >= min_peaks:
                    compounds.append(current_comp)
            current_comp = {'name': line.split(':', 1)[1].strip(), 'smiles': '', 'peaks': []}
            in_peaks = False
        elif line.startswith('SMILES:'):
            if current_comp is not None:
                current_comp['smiles'] = line.split(':', 1)[1].strip()
        elif line.startswith('Num peaks:'):
            in_peaks = True
        elif in_peaks and line:
            fragments = re.split(r'[;|,]', line)
            for frag in fragments:
                frag = frag.strip()
                if not frag:
                    continue
                tokens = frag.split()
                if len(tokens) >= 2:
                    try:
                        mz = float(tokens[0])
                        intensity = float(tokens[1])
                        if mz > 0 and intensity > 0:
                            current_comp['peaks'].append((mz, intensity))
                    except ValueError:
                        pass
    
    if current_comp is not None and 'peaks' in current_comp:
        if len(current_comp['peaks']) >= min_peaks:
            compounds.append(current_comp)
    return compounds


def peaks_to_vector(peaks, mz_min=40, mz_max=600):
    """Convert peak list to 561-dim vector"""
    dim = mz_max - mz_min + 1
    vec = np.zeros(dim)
    for mz, intensity in peaks:
        if mz_min <= mz <= mz_max:
            idx = int(round(mz - mz_min))
            if 0 <= idx < dim:
                vec[idx] += intensity
    return vec


def preprocess_spectra_train(spectra):
    """TIC normalization + sqrt transform (fit on training set only)"""
    tic = spectra.sum(axis=1, keepdims=True)
    tic = np.maximum(tic, 1e-8)
    spectra_norm = spectra / tic
    spectra_sqrt = np.sqrt(spectra_norm)
    return spectra_sqrt, {'tic': tic}


def preprocess_spectra_test(spectra, fit_params):
    """Apply training-set preprocessing to test/validation data"""
    tic = spectra.sum(axis=1, keepdims=True)
    tic = np.maximum(tic, 1e-8)
    spectra_norm = spectra / tic
    spectra_sqrt = np.sqrt(spectra_norm)
    return spectra_sqrt


def split_by_smiles(pos_indices, neg_indices, pos_smiles_list, test_size=0.15, val_size=0.15, random_state=42):
    """Split by SMILES for positive samples; random split for negatives"""
    np.random.seed(random_state)
    
    pos_unique_smiles = np.unique(pos_smiles_list)
    n_pos_smiles = len(pos_unique_smiles)
    shuffled_smiles = pos_unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)
    
    n_pos_test = max(1, int(n_pos_smiles * test_size))
    n_pos_val = max(1, int(n_pos_smiles * val_size))
    
    pos_test_smiles = set(shuffled_smiles[:n_pos_test])
    pos_val_smiles = set(shuffled_smiles[n_pos_test:n_pos_test + n_pos_val])
    pos_train_smiles = set(shuffled_smiles[n_pos_test + n_pos_val:])
    
    pos_train_idx = [i for i in pos_indices if pos_smiles_list[i] in pos_train_smiles]
    pos_val_idx = [i for i in pos_indices if pos_smiles_list[i] in pos_val_smiles]
    pos_test_idx = [i for i in pos_indices if pos_smiles_list[i] in pos_test_smiles]
    
    n_neg = len(neg_indices)
    neg_shuffled = neg_indices.copy()
    np.random.shuffle(neg_shuffled)
    
    n_neg_test = int(n_neg * test_size)
    n_neg_val = int(n_neg * val_size)
    
    neg_test_idx = neg_shuffled[:n_neg_test].tolist()
    neg_val_idx = neg_shuffled[n_neg_test:n_neg_test + n_neg_val].tolist()
    neg_train_idx = neg_shuffled[n_neg_test + n_neg_val:].tolist()
    
    train_idx = np.array(pos_train_idx + neg_train_idx)
    val_idx = np.array(pos_val_idx + neg_val_idx)
    test_idx = np.array(pos_test_idx + neg_test_idx)
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    return train_idx, val_idx, test_idx


# ==================== Evaluation ====================

def evaluate(model, data_loader, device):
    """Evaluate model on a given data loader"""
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch_spec, batch_labels in data_loader:
            batch_spec = batch_spec.to(device)
            logits = model(batch_spec)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
    }


def evaluate_per_smiles(test_indices, smiles_all, y, probs, preds):
    """Evaluate positive samples at SMILES (compound) level"""
    test_labels = y[test_indices]
    test_smiles = smiles_all[test_indices]
    pos_mask = test_labels == 1
    
    if pos_mask.sum() == 0:
        return None
    
    pos_smiles = test_smiles[pos_mask]
    pos_probs = probs[pos_mask]
    pos_preds = preds[pos_mask]
    pos_labels = test_labels[pos_mask]
    
    smiles_to_indices = defaultdict(list)
    for i, smi in enumerate(pos_smiles):
        smiles_to_indices[smi].append(i)
    
    correct = 0
    total = len(smiles_to_indices)
    
    for smi, idx_list in smiles_to_indices.items():
        mean_prob = np.mean(pos_probs[idx_list])
        pred = 1.0 if mean_prob >= 0.5 else 0.0
        if pred == pos_labels[idx_list[0]]:
            correct += 1
    
    acc = correct / total if total > 0 else 0
    return {'accuracy': acc, 'n_smiles': total, 'n_correct': correct}


# ==================== Main ====================

if __name__ == "__main__":
    base_dir = Path(r"D:\DL\cann\建模")
    positive_msp = base_dir / "阳性-含CanonicalSMILES-5类骨架.msp"
    negative_msp = base_dir / "阴性.msp"
    encoder_path = base_dir / "best_model_0725.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ===== 1. Load and parse data =====
    print("\n" + "="*60)
    print("1. Loading data")
    print("="*60)
    
    pos_compounds = parse_msp_with_smiles(str(positive_msp))
    neg_compounds = parse_msp_with_smiles(str(negative_msp))
    
    # Move '.alpha.-pbp' from positive to negative
    alpha_pbp_indices = [i for i, comp in enumerate(pos_compounds) if comp['name'] == '.alpha.-pbp']
    if alpha_pbp_indices:
        removed_comps = [pos_compounds[i] for i in alpha_pbp_indices]
        pos_compounds = [comp for i, comp in enumerate(pos_compounds) if i not in alpha_pbp_indices]
        neg_compounds.extend(removed_comps)
    
    pos_spectra_raw = np.array([peaks_to_vector(c['peaks']) for c in pos_compounds])
    pos_smiles = np.array([c['smiles'] if c['smiles'] else c['name'] for c in pos_compounds])
    neg_spectra_raw = np.array([peaks_to_vector(c['peaks']) for c in neg_compounds])
    
    print(f"Positive: {len(pos_spectra_raw)} spectra, {len(np.unique(pos_smiles))} unique SMILES")
    print(f"Negative: {len(neg_spectra_raw)} spectra")
    
    # ===== 2. Split data (before preprocessing) =====
    print("\n" + "="*60)
    print("2. Splitting data (no leakage)")
    print("="*60)
    
    pos_indices = np.arange(len(pos_spectra_raw))
    neg_indices = np.arange(len(pos_spectra_raw), len(pos_spectra_raw) + len(neg_spectra_raw))
    
    train_idx, val_idx, test_idx = split_by_smiles(
        pos_indices, neg_indices, pos_smiles,
        test_size=0.15, val_size=0.15, random_state=42
    )
    
    X_raw = np.concatenate([pos_spectra_raw, neg_spectra_raw], axis=0)
    y = np.concatenate([np.ones(len(pos_spectra_raw)), np.zeros(len(neg_spectra_raw))])
    smiles_all = np.concatenate([pos_smiles, np.array([f'neg_{i}' for i in range(len(neg_spectra_raw))])])
    
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"Train pos: {(y[train_idx]==1).sum():.0f}, neg: {(y[train_idx]==0).sum():.0f}")
    
    # ===== 3. Preprocess each split separately =====
    print("\n" + "="*60)
    print("3. Preprocessing (fit on train only)")
    print("="*60)
    
    X_train_raw = X_raw[train_idx]
    X_val_raw = X_raw[val_idx]
    X_test_raw = X_raw[test_idx]
    
    X_train, fit_params = preprocess_spectra_train(X_train_raw)
    X_val = preprocess_spectra_test(X_val_raw, fit_params)
    X_test = preprocess_spectra_test(X_test_raw, fit_params)
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    # ===== 4. Load pre-trained encoder =====
    print("\n" + "="*60)
    print("4. Loading pre-trained encoder")
    print("="*60)
    
    checkpoint = torch.load(str(encoder_path), map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('encoder_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # Clean up state dict keys
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if any(k.startswith('encoder.') for k in state_dict.keys()):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
    
    # ===== 5. Prepare data loaders =====
    print("\n" + "="*60)
    print("5. Preparing data loaders")
    print("="*60)
    
    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )
    
    print(f"Batch size: {batch_size}")
    
    # ===== 6. Run four fine-tuning configurations =====
    print("\n" + "="*80)
    print("6. Fine-tuning experiments")
    print("="*80)
    
    # Four configurations:
    # (1) Two-stage progressive unfreezing + Focal Loss
    # (2) Two-stage progressive unfreezing + Focal Loss + Mixup
    # (3) Single-stage fully unfrozen + Focal Loss
    # (4) Single-stage fully unfrozen + BCE Loss
    
    experiment_configs = [
        {'name': 'TwoStage_FocalLoss', 'train_type': 'two_stage', 'use_focal': True, 'use_mixup': False},
        {'name': 'TwoStage_Focal_Mixup', 'train_type': 'two_stage', 'use_focal': True, 'use_mixup': True},
        {'name': 'SingleStage_Focal', 'train_type': 'single_stage', 'use_focal': True, 'use_mixup': False},
        {'name': 'SingleStage_BCE', 'train_type': 'single_stage', 'use_focal': False, 'use_mixup': False},
    ]
    
    results = []
    models = []
    
    for cfg in experiment_configs:
        print(f"\n{'='*60}")
        print(f"Experiment: {cfg['name']}")
        print(f"{'='*60}")
        
        # Initialize encoder with pre-trained weights
        encoder = SpectrumEncoder(input_dim=561, hidden_dim=256).to(device)
        encoder.load_state_dict(state_dict, strict=False)
        
        model = UltimateClassifier(encoder, input_dim=256, dropout=0.3).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params:,}")
        
        # Train
        if cfg['train_type'] == 'two_stage':
            model = train_two_stage(model, train_loader, val_loader, device, cfg)
        else:
            single_cfg = {
                'epochs': 100,
                'lr': 1e-4,
                'encoder_lr_ratio': 1.0,
                'patience': 25,
                'use_focal': cfg['use_focal'],
                'use_mixup': cfg['use_mixup'],
                'warmup_epochs': 5,
                'weight_decay': 1e-4,
            }
            model, _ = train_single_stage(model, train_loader, val_loader, device, single_cfg)
        
        models.append(model)
        
        # Evaluate
        test_results = evaluate(model, test_loader, device)
        smiles_results = evaluate_per_smiles(test_idx, smiles_all, y, test_results['probs'], test_results['preds'])
        
        print(f"\n  Test: Acc={test_results['accuracy']:.2%}, AUC={test_results['auc']:.4f}")
        print(f"        Prec={test_results['precision']:.2%}, Rec={test_results['recall']:.2%}, F1={test_results['f1']:.2%}")
        if smiles_results:
            print(f"  Compound-level: {smiles_results['n_correct']}/{smiles_results['n_smiles']} ({smiles_results['accuracy']:.2%})")
        
        results.append({
            'name': cfg['name'],
            'test': test_results,
            'smiles': smiles_results,
        })
    
    # ===== 7. Ensemble =====
    print("\n" + "="*80)
    print("7. Ensemble Evaluation")
    print("="*80)
    
    # Simple average ensemble
    ensemble_probs = np.mean([r['test']['probs'] for r in results], axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_prec = precision_score(y_test, ensemble_preds, zero_division=0)
    ensemble_rec = recall_score(y_test, ensemble_preds, zero_division=0)
    ensemble_f1 = f1_score(y_test, ensemble_preds, zero_division=0)
    
    ensemble_smiles = evaluate_per_smiles(test_idx, smiles_all, y, ensemble_probs, ensemble_preds)
    
    print(f"  Ensemble: Acc={ensemble_acc:.2%}, AUC={ensemble_auc:.4f}")
    print(f"            Prec={ensemble_prec:.2%}, Rec={ensemble_rec:.2%}, F1={ensemble_f1:.2%}")
    if ensemble_smiles:
        print(f"  Compound-level: {ensemble_smiles['n_correct']}/{ensemble_smiles['n_smiles']} ({ensemble_smiles['accuracy']:.2%})")
    
    # ===== 8. Summary =====
    print("\n" + "="*80)
    print("8. Summary")
    print("="*80)
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Method': r['name'],
            'Acc': f"{r['test']['accuracy']:.2%}",
            'AUC': f"{r['test']['auc']:.4f}",
            'Prec': f"{r['test']['precision']:.2%}",
            'Rec': f"{r['test']['recall']:.2%}",
            'F1': f"{r['test']['f1']:.2%}",
            'SMILES_Acc': f"{r['smiles']['accuracy']:.2%}" if r['smiles'] else 'N/A',
        })
    
    summary_data.append({
        'Method': 'Ensemble',
        'Acc': f"{ensemble_acc:.2%}",
        'AUC': f"{ensemble_auc:.4f}",
        'Prec': f"{ensemble_prec:.2%}",
        'Rec': f"{ensemble_rec:.2%}",
        'F1': f"{ensemble_f1:.2%}",
        'SMILES_Acc': f"{ensemble_smiles['accuracy']:.2%}" if ensemble_smiles else 'N/A',
    })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(r"D:\DL\cann\建模\binary_classification_ultimate") / f'summary_{timestamp}.xlsx'
    summary_path.parent.mkdir(exist_ok=True)
    df_summary.to_excel(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")