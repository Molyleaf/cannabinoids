import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, silhouette_score
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ==================== Model Definition ====================

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
    def __init__(self, encoder, input_dim=256):
        super().__init__()
        self.encoder = encoder
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
    
    def forward(self, x, return_embed=False):
        embed = self.encoder(x)
        logit = self.classifier(embed)
        if return_embed:
            return logit.squeeze(-1), embed
        return logit.squeeze(-1)


# ==================== Data Processing ====================

def parse_msp_with_smiles(msp_file, min_peaks=5):
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
            parts = line.replace(';', '').replace('\t', ' ').split()
            if len(parts) >= 2:
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
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
    vec = np.zeros(dim)
    for mz, intensity in peaks:
        if mz_min <= mz <= mz_max:
            idx = int(round(mz - mz_min))
            if 0 <= idx < dim:
                vec[idx] += intensity
    return vec


def preprocess_spectra(spectra):
    tic = spectra.sum(axis=1, keepdims=True)
    spectra = spectra / (tic + 1e-8)
    return np.sqrt(spectra)


# ==================== Evaluation & Visualization ====================

def evaluate_model(model, test_loader, device, temperature=1.0):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch_spec, batch_labels in test_loader:
            batch_spec = batch_spec.to(device)
            logits = model(batch_spec)
            probs = torch.sigmoid(logits / temperature)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n{'='*60}")
    print("Test Set Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  Confusion Matrix:\n    TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}\n    FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 
            'confusion_matrix': cm, 'probs': all_probs, 'labels': all_labels, 'preds': all_preds}


def analyze_embeddings(model, loader, device, output_dir):
    """Feature space analysis: t-SNE + PCA"""
    model.eval()
    embeddings, labels = [], []
    
    with torch.no_grad():
        for batch_spec, batch_labels in loader:
            batch_spec = batch_spec.to(device)
            _, embed = model(batch_spec, return_embed=True)
            embeddings.append(embed.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"\n  Embedding matrix: {embeddings.shape}, Positive: {np.sum(labels==1)}, Negative: {np.sum(labels==0)}")
    
    # High-dimensional evaluation
    sil_score = silhouette_score(embeddings, labels)
    print(f"  Silhouette Score: {sil_score:.4f}")
    
    # Dimensionality reduction
    n_samples = min(1500, len(embeddings))
    np.random.seed(42)
    idx = np.random.choice(len(embeddings), n_samples, replace=False)
    emb_sample, labels_sample = embeddings[idx], labels[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    emb_tsne = tsne.fit_transform(emb_sample)
    tsne_acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), emb_tsne, labels_sample, cv=5).mean()
    
    ax = axes[0]
    scatter = ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels_sample, cmap='coolwarm', alpha=0.6, s=10)
    ax.set_title(f't-SNE (KNN Acc={tsne_acc:.3f}, Sil={silhouette_score(emb_tsne, labels_sample):.3f})')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    # PCA
    print("  Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(emb_sample)
    pca_acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), emb_pca, labels_sample, cv=5).mean()
    
    ax = axes[1]
    scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels_sample, cmap='coolwarm', alpha=0.6, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'PCA (KNN Acc={pca_acc:.3f}, Sil={silhouette_score(emb_pca, labels_sample):.3f})')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'embedding_analysis.png'), dpi=150)
    plt.show()
    
    return sil_score


def grad_cam_visualization(model, spectrum, output_dir):
    """Grad-CAM visualization"""
    model.eval()
    device = next(model.parameters()).device
    
    if torch.is_tensor(spectrum):
        spectrum_np = spectrum.cpu().numpy() if spectrum.is_cuda else spectrum.numpy()
    else:
        spectrum_np = np.array(spectrum)
    spectrum_np = spectrum_np.squeeze()
    
    spectrum_tensor = torch.tensor(spectrum_np, dtype=torch.float32).to(device)
    spectrum_tensor.requires_grad_()
    
    logits = model(spectrum_tensor.unsqueeze(0))
    model.zero_grad()
    logits[0].backward()
    
    gradients = spectrum_tensor.grad
    if gradients is not None:
        weights = gradients.mean(dim=0, keepdim=True)
        cam = (weights * spectrum_tensor).sum(dim=0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam.detach().cpu().numpy()
    else:
        cam = np.zeros_like(spectrum_np)
    
    mz_axis = np.linspace(40, 600, len(spectrum_np))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(mz_axis, spectrum_np, 'b-', alpha=0.7, linewidth=1)
    axes[0].set_xlabel('m/z'); axes[0].set_ylabel('Intensity')
    axes[0].set_title('Original Spectrum'); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mz_axis, spectrum_np, 'b-', alpha=0.3, linewidth=0.8)
    axes[1].fill_between(mz_axis, 0, spectrum_np * cam, color='red', alpha=0.5, label='Important Regions')
    axes[1].set_xlabel('m/z'); axes[1].set_ylabel('Intensity')
    axes[1].set_title('Grad-CAM - Red regions show important m/z')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'grad_cam.png'), dpi=150)
    plt.show()


def plot_summary(eval_results, output_dir):
    """Summary results plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC
    fpr, tpr, _ = roc_curve(eval_results['labels'], eval_results['probs'])
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC={eval_results["auc"]:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('FPR'); axes[0, 0].set_ylabel('TPR')
    axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = eval_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Cannabinoid', 'Cannabinoid'],
                yticklabels=['Non-Cannabinoid', 'Cannabinoid'], ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix (Acc={eval_results["accuracy"]:.2%})')
    
    # Probability distribution
    pos_probs = eval_results['probs'][eval_results['labels'] == 1]
    neg_probs = eval_results['probs'][eval_results['labels'] == 0]
    axes[1, 0].hist(neg_probs, bins=30, alpha=0.6, label='Negative', color='blue', edgecolor='black')
    axes[1, 0].hist(pos_probs, bins=30, alpha=0.6, label='Positive', color='red', edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Probability'); axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Probability Distribution'); axes[1, 0].legend()
    
    # Performance metrics
    metrics = {'Acc': eval_results['accuracy'], 'Prec': eval_results['precision'],
               'Rec': eval_results['recall'], 'F1': eval_results['f1'], 'AUC': eval_results['auc']}
    colors = ['#2ecc71' if v > 0.9 else '#f39c12' if v > 0.8 else '#e74c3c' for v in metrics.values()]
    axes[1, 1].bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black')
    axes[1, 1].set_ylim(0, 1.1); axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Summary')
    for bar, val in zip(axes[1, 1].patches, metrics.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'evaluation_summary.png'), dpi=150)
    plt.show()


# ==================== Main Program ====================

if __name__ == "__main__":
    base_dir = Path(r"D:\DL\cann\建模")
    
    # Find all possible model locations
    possible_paths = [
        base_dir / "binary_classifier_20260716_155622.pt",
        base_dir / "binary_classifier_enhanced.pt",
        base_dir / "binary_classification_enhanced" / "binary_classifier_enhanced.pt",
        base_dir / "visualization" / "binary_classifier_*.pt",
    ]
    
    # Find all pt files starting with binary_classifier
    all_models = list(base_dir.glob("binary_classifier_*.pt"))
    all_models.extend(list((base_dir / "binary_classification_enhanced").glob("binary_classifier_*.pt"))) if (base_dir / "binary_classification_enhanced").exists() else None
    all_models.extend(list((base_dir / "visualization").glob("binary_classifier_*.pt"))) if (base_dir / "visualization").exists() else None
    
    # Remove duplicates and sort by modification time
    all_models = sorted(set(all_models), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if all_models:
        model_path = all_models[0]
        print(f"Found model: {model_path.name} (Modified: {datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        # Try direct specified file
        specified_path = base_dir / "binary_classifier_20260716_155622.pt"
        if specified_path.exists():
            model_path = specified_path
            print(f"Using specified model: {model_path.name}")
        else:
            print(f"Error: Binary classification model file not found")
            print(f"Search path: {base_dir}")
            print(f"Please ensure the model file exists, or modify the path in the code")
            exit(1)
    
    # Create output directory (with timestamp)
    output_dir = base_dir / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\nOutput: {output_dir}")
    
    # ===== 1. Load Data =====
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    pos_compounds = parse_msp_with_smiles(str(base_dir / "positive_cannabinoids.msp"))
    neg_compounds = parse_msp_with_smiles(str(base_dir / "negative.msp"))
    
    X_pos = preprocess_spectra(np.array([peaks_to_vector(c['peaks']) for c in pos_compounds]))
    X_neg = preprocess_spectra(np.array([peaks_to_vector(c['peaks']) for c in neg_compounds]))
    pos_smiles = np.array([c['smiles'] if c['smiles'] else c['name'] for c in pos_compounds])
    
    print(f"  Positive: {len(X_pos)} spectra, {len(np.unique(pos_smiles))} compounds")
    print(f"  Negative: {len(X_neg)} spectra")
    
    # ===== 2. Split Test Set (consistent with training) =====
    print("\n" + "="*60)
    print("Splitting Test Set")
    print("="*60)
    
    np.random.seed(42)
    pos_indices = np.arange(len(X_pos))
    neg_indices = np.arange(len(X_pos), len(X_pos) + len(X_neg))
    
    # Split positive by SMILES
    pos_unique_smiles = np.unique(pos_smiles)
    shuffled_smiles = pos_unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)
    n_pos_test = max(1, int(len(pos_unique_smiles) * 0.15))
    pos_test_smiles = set(shuffled_smiles[:n_pos_test])
    pos_test_idx = [i for i in pos_indices if pos_smiles[i] in pos_test_smiles]
    
    # Random split for negative
    neg_shuffled = neg_indices.copy()
    np.random.shuffle(neg_shuffled)
    n_neg_test = int(len(neg_indices) * 0.15)
    neg_test_idx = neg_shuffled[:n_neg_test].tolist()
    
    test_idx = np.array(pos_test_idx + neg_test_idx)
    np.random.shuffle(test_idx)
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    print(f"  Test set: {len(test_idx)} spectra (Positive: {(y[test_idx]==1).sum()})")
    
    # ===== 3. Create DataLoader =====
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X[test_idx], dtype=torch.float32),
                      torch.tensor(y[test_idx], dtype=torch.float32)),
        batch_size=128, shuffle=False
    )
    
    # ===== 4. Load Model =====
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    
    checkpoint = torch.load(str(model_path), map_location=device)
    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256).to(device)
    model = BinaryClassifier(encoder, input_dim=256).to(device)
    
    # Load weights
    if 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        print(f"  Loaded encoder weights")
    else:
        print(f"  Warning: Encoder weights not found, using random initialization")
    
    if 'classifier_state_dict' in checkpoint:
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
        print(f"  Loaded classifier weights")
    else:
        print(f"  Warning: Classifier weights not found, using random initialization")
    
    temperature = checkpoint.get('temperature', 1.0)
    model.eval()
    print(f"  Temperature parameter: {temperature:.4f}")
    
    # ===== 5. Evaluation =====
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    eval_results = evaluate_model(model, test_loader, device, temperature)
    
    # ===== 6. Feature Space Analysis =====
    print("\n" + "="*60)
    print("Feature Space Analysis")
    print("="*60)
    
    analyze_embeddings(model, test_loader, device, output_dir)
    
    # ===== 7. Grad-CAM Visualization =====
    print("\n" + "="*60)
    print("Grad-CAM Visualization")
    print("="*60)
    
    # Select positive sample
    pos_test = [i for i in test_idx if y[i] == 1]
    if pos_test:
        sample_idx = pos_test[0]
        sample_label = 'Cannabinoid'
    else:
        sample_idx = test_idx[0]
        sample_label = 'Non-Cannabinoid'
    
    sample_spectrum = torch.tensor(X[sample_idx], dtype=torch.float32)
    print(f"  Sample label: {sample_label}")
    grad_cam_visualization(model, sample_spectrum, output_dir)
    
    # ===== 8. Results Summary =====
    print("\n" + "="*60)
    print("Generating Results Summary")
    print("="*60)
    
    plot_summary(eval_results, output_dir)
    
    # ===== 9. Export Data =====
    print("\n" + "="*60)
    print("Exporting Data")
    print("="*60)
    
    # Export evaluation results
    pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
        'Value': [eval_results['accuracy'], eval_results['precision'], 
                  eval_results['recall'], eval_results['f1'], eval_results['auc']]
    }).to_excel(output_dir / 'metrics.xlsx', index=False)
    
    # Export predictions
    pd.DataFrame({
        'True_Label': eval_results['labels'],
        'Pred_Prob': eval_results['probs'],
        'Pred_Label': eval_results['preds']
    }).to_excel(output_dir / 'predictions.xlsx', index=False)
    
    # Export confusion matrix
    cm_df = pd.DataFrame(eval_results['confusion_matrix'], 
                         index=['True:Non-Cannabinoid', 'True:Cannabinoid'],
                         columns=['Pred:Non-Cannabinoid', 'Pred:Cannabinoid'])
    cm_df.to_excel(output_dir / 'confusion_matrix.xlsx')
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Summary plot: {output_dir / 'evaluation_summary.png'}")
    print(f"  Embedding plot: {output_dir / 'embedding_analysis.png'}")
    print(f"  Grad-CAM: {output_dir / 'grad_cam.png'}")
    print(f"  Excel data:")
    print(f"    - {output_dir / 'metrics.xlsx'}")
    print(f"    - {output_dir / 'predictions.xlsx'}")
    print(f"    - {output_dir / 'confusion_matrix.xlsx'}")