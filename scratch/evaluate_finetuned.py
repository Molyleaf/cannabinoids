import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Pure NumPy Metric Implementations
def calc_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.float32)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # Calculate AUC using Mann-Whitney U / trapezoidal method
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        auc = 0.5
    else:
        # vectorized rank calculation
        # AUC = P(pos_score > neg_score) + 0.5 * P(pos_score == neg_score)
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        diff = pos_scores[:, None] - neg_scores[None, :]
        auc = (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (n_pos * n_neg)
        
    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'spec': spec,
        'f1': f1,
        'auc': auc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'y_pred': y_pred
    }

# Define Model Architecture exactly as used in simclr_finetune
class SpectrumEncoder(nn.Module):
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


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.encoder.parameters())):
            embed = self.encoder(x)
        logit = self.classifier(embed)
        return logit.squeeze(-1)

# MSP Parser
def parse_msp(msp_file, min_peaks=5):
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
            current_comp = {
                'name': line.split(':', 1)[1].strip(),
                'smiles': '',
                'peaks': []
            }
            in_peaks = False
        elif line.startswith('SMILES:'):
            if current_comp is not None:
                current_comp['smiles'] = line.split(':', 1)[1].strip()
        elif line.startswith('Num peaks:'):
            in_peaks = True
        elif in_peaks and line:
            parts = line.replace(';', '').replace('\t', ' ').split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    if mz > 0 and intensity > 0:
                        current_comp['peaks'].append((mz, intensity))
                except ValueError:
                    in_peaks = False
    
    if current_comp is not None and 'peaks' in current_comp:
        if len(current_comp['peaks']) >= min_peaks:
            compounds.append(current_comp)
    
    return compounds

def peaks_to_vector(peaks, mz_min=40, mz_max=600):
    dim = mz_max - mz_min + 1
    vec = np.zeros(dim, dtype=np.float32)
    for mz, intensity in peaks:
        if mz_min <= mz <= mz_max:
            idx = int(round(mz - mz_min))
            if 0 <= idx < dim:
                vec[idx] += intensity
    return vec

def preprocess_spectra(vecs):
    tic = vecs.sum(axis=1, keepdims=True)
    vecs = vecs / (tic + 1e-8)
    vecs = np.sqrt(vecs)
    return vecs

def main():
    project_root = r"d:\UserFiles\Documents\PyCharm\cannabinoids"
    pos_msp = os.path.join(project_root, "known_compound_recognize", "positive.msp")
    neg_msp = os.path.join(project_root, "scratch", "阴性.msp")
    
    pos_comps = parse_msp(pos_msp)
    neg_comps = parse_msp(neg_msp)
    
    print(f"Parsed {len(pos_comps)} positive compounds and {len(neg_comps)} negative compounds.")
    
    pos_vecs = np.array([peaks_to_vector(c['peaks']) for c in pos_comps], dtype=np.float32)
    neg_vecs = np.array([peaks_to_vector(c['peaks']) for c in neg_comps], dtype=np.float32)
    
    pos_norm = preprocess_spectra(pos_vecs)
    neg_norm = preprocess_spectra(neg_vecs)
    
    X = np.vstack([pos_norm, neg_norm])
    y = np.array([1]*len(pos_norm) + [0]*len(neg_norm), dtype=np.float32)
    
    ckpt_paths = {
        "binary_classifier_latest.pt": os.path.join(project_root, "simclr_finetune", "binary_classifier_latest.pt"),
        "binary_classifier_weights_20260716_164027.pt": os.path.join(project_root, "simclr_finetune", "binary_classifier_weights_20260716_164027.pt")
    }
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    for name, path in ckpt_paths.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        print(f"\n==================== Evaluating Model: {name} ====================")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        
        model = BinaryClassifier(encoder=encoder, input_dim=256, freeze_encoder=True)
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])
        model.eval()
        
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).numpy()
        
        m = calc_metrics(y, probs, threshold=0.5)
        
        print(f"Overall Dataset ({len(y)} spectra, Pos={len(pos_comps)}, Neg={len(neg_comps)}):")
        print(f"  Accuracy   : {m['acc']:.4f} ({m['acc']*100:.2f}%)")
        print(f"  Precision  : {m['prec']:.4f} ({m['prec']*100:.2f}%)")
        print(f"  Recall/Sens: {m['rec']:.4f} ({m['rec']*100:.2f}%)")
        print(f"  Specificity: {m['spec']:.4f} ({m['spec']*100:.2f}%)")
        print(f"  F1 Score   : {m['f1']:.4f}")
        print(f"  AUC        : {m['auc']:.4f}")
        print(f"  Confusion Matrix: TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")
        
        # Positive compounds (SMILES aggregation if SMILES exists)
        pos_probs = probs[:len(pos_comps)]
        pos_smiles_map = defaultdict(list)
        for i, c in enumerate(pos_comps):
            smi = c['smiles'] if c['smiles'] else c['name']
            pos_smiles_map[smi].append(pos_probs[i])
        
        compound_probs = [np.mean(p_list) for p_list in pos_smiles_map.values()]
        compound_preds = [(p >= 0.5) for p in compound_probs]
        comp_acc = np.mean(compound_preds)
        print(f"  Positive Compound-Level Accuracy ({len(pos_smiles_map)} unique compounds): {comp_acc*100:.2f}% ({sum(compound_preds)}/{len(pos_smiles_map)})")
        
        # Score distribution summary
        print(f"  Positive Scores: Mean={pos_probs.mean():.4f}, Min={pos_probs.min():.4f}, Max={pos_probs.max():.4f}, Median={np.median(pos_probs):.4f}")
        neg_probs = probs[len(pos_comps):]
        print(f"  Negative Scores: Mean={neg_probs.mean():.4f}, Min={neg_probs.min():.4f}, Max={neg_probs.max():.4f}, Median={np.median(neg_probs):.4f}")
        
        # Print misclassified samples if any
        if m['fn'] > 0 or m['fp'] > 0:
            print(f"\n  --- Misclassified Samples Details ---")
            if m['fn'] > 0:
                print(f"  False Negatives (Positive samples predicted as Negative, n={m['fn']}):")
                for i in range(len(pos_comps)):
                    if pos_probs[i] < 0.5:
                        print(f"    - [{pos_comps[i]['name']}] prob={pos_probs[i]:.4f}")
            if m['fp'] > 0:
                print(f"  False Positives (Negative samples predicted as Positive, n={m['fp']}):")
                for i in range(len(neg_comps)):
                    if neg_probs[i] >= 0.5:
                        print(f"    - [{neg_comps[i]['name']}] prob={neg_probs[i]:.4f}")

if __name__ == "__main__":
    main()
