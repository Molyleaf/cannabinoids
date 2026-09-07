import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ==================== Grad-CAM ====================

def compute_grad_cam(model, spectrum_tensor):
    """
    Compute Grad-CAM weights for a single spectrum
    Returns: (cam_weights, gradients)
    """
    model.eval()
    device = next(model.parameters()).device
    
    if spectrum_tensor.device != device:
        spectrum_tensor = spectrum_tensor.to(device)
    
    # Ensure 2D input [features]
    if spectrum_tensor.dim() == 1:
        spectrum_tensor = spectrum_tensor.unsqueeze(0)
    
    spectrum_tensor.requires_grad_()
    
    # Forward pass
    logits = model(spectrum_tensor)
    
    # Backward pass
    model.zero_grad()
    logits[0].backward()
    
    # Get gradients
    gradients = spectrum_tensor.grad
    if gradients is None:
        return np.zeros(spectrum_tensor.shape[1]), None
    
    # Compute CAM weights: average gradients as weights
    weights = gradients.mean(dim=0, keepdim=True)
    
    # Weighted sum to obtain CAM
    cam = (weights * spectrum_tensor).sum(dim=0)
    cam = F.relu(cam)
    
    # Normalize
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    
    return cam.detach().cpu().numpy(), gradients.detach().cpu().numpy()


def get_top_ions(cam_weights, mz_min=40, mz_max=600, top_k=20, threshold=0.5):
    """Extract key ions"""
    mz_axis = np.linspace(mz_min, mz_max, len(cam_weights))
    
    # Find ions with weights above threshold
    threshold_value = threshold * cam_weights.max()
    important_indices = np.where(cam_weights > threshold_value)[0]
    
    if len(important_indices) == 0:
        # If none above threshold, take top_k
        important_indices = np.argsort(cam_weights)[-top_k:][::-1]
    
    # Sort by weight
    sorted_idx = np.argsort(cam_weights[important_indices])[::-1]
    important_indices = important_indices[sorted_idx]
    
    # Take top_k
    top_indices = important_indices[:top_k]
    top_mz = mz_axis[top_indices]
    top_weights = cam_weights[top_indices]
    
    return top_mz, top_weights


# ==================== Main Program ====================

if __name__ == "__main__":
    base_dir = Path(r"D:\DL\cann\建模")
    
    # Find model
    model_files = list(base_dir.glob("binary_classifier_*.pt"))
    model_files.extend(list((base_dir / "binary_classification_enhanced").glob("binary_classifier_*.pt"))) if (base_dir / "binary_classification_enhanced").exists() else None
    
    model_files = sorted(set(model_files), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if model_files:
        model_path = model_files[0]
        print(f"Using model: {model_path.name}")
    else:
        print("Model file not found, please verify path")
        exit(1)
    
    # Output directory
    output_dir = base_dir / f"gradcam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    print(f"  Positive: {len(X_pos)} spectra")
    print(f"  Negative: {len(X_neg)} spectra")
    
    # ===== 2. Split Test Set =====
    np.random.seed(42)
    pos_indices = np.arange(len(X_pos))
    neg_indices = np.arange(len(X_pos), len(X_pos) + len(X_neg))
    
    pos_unique_smiles = np.unique(pos_smiles)
    shuffled_smiles = pos_unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)
    n_pos_test = max(1, int(len(pos_unique_smiles) * 0.15))
    pos_test_smiles = set(shuffled_smiles[:n_pos_test])
    pos_test_idx = [i for i in pos_indices if pos_smiles[i] in pos_test_smiles]
    
    neg_shuffled = neg_indices.copy()
    np.random.shuffle(neg_shuffled)
    n_neg_test = int(len(neg_indices) * 0.15)
    neg_test_idx = neg_shuffled[:n_neg_test].tolist()
    
    test_idx = np.array(pos_test_idx + neg_test_idx)
    np.random.shuffle(test_idx)
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    print(f"  Test set: {len(test_idx)} spectra (Positive: {(y[test_idx]==1).sum()})")
    
    # ===== 3. Load Model =====
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    
    checkpoint = torch.load(str(model_path), map_location=device)
    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256).to(device)
    model = BinaryClassifier(encoder, input_dim=256).to(device)
    
    if 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    if 'classifier_state_dict' in checkpoint:
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
    
    model.eval()
    print(f"  Model loaded successfully")
    
    # ===== 4. Grad-CAM Analysis =====
    print("\n" + "="*60)
    print("Grad-CAM Attention Analysis")
    print("="*60)
    
    mz_min, mz_max = 40, 600
    mz_axis = np.linspace(mz_min, mz_max, 561)
    
    # Store all results
    all_cam_weights = []  # CAM weights for all samples
    all_labels = []       # Corresponding labels
    
    pos_cam_weights = []  # CAM weights for positive samples
    neg_cam_weights = []  # CAM weights for negative samples
    
    sample_top_ions = []  # Top ions for each sample
    
    print("\n  Computing Grad-CAM weights...")
    
    for i, idx in enumerate(test_idx):
        spectrum = torch.tensor(X[idx], dtype=torch.float32)
        label = y[idx]
        label_text = 'Cannabinoid' if label == 1 else 'Non-Cannabinoid'
        
        # Compute Grad-CAM
        cam_weights, gradients = compute_grad_cam(model, spectrum)
        
        # Ensure cam_weights is 1D
        if cam_weights.ndim > 1:
            cam_weights = cam_weights.squeeze()
        
        all_cam_weights.append(cam_weights)
        all_labels.append(label)
        
        # Store by class
        if label == 1:
            pos_cam_weights.append(cam_weights)
        else:
            neg_cam_weights.append(cam_weights)
        
        # Extract top ions
        top_mz, top_weights = get_top_ions(cam_weights, top_k=15)
        for j, (mz, w) in enumerate(zip(top_mz, top_weights)):
            sample_top_ions.append({
                'Sample_ID': i,
                'Label': label_text,
                'Rank': j + 1,
                'm/z': round(mz, 2),
                'CAM_Weight': round(w, 4)
            })
        
        # Progress display
        if (i + 1) % 50 == 0:
            print(f"    Processed: {i+1}/{len(test_idx)} samples")
    
    print(f"    Complete! Processed {len(test_idx)} samples")
    
    # ===== 5. Summary Analysis =====
    print("\n  Performing summary analysis...")
    
    # Compute average weights for all samples
    all_avg = np.mean(all_cam_weights, axis=0)
    all_std = np.std(all_cam_weights, axis=0)
    
    # Compute average weights for positive samples
    if pos_cam_weights:
        pos_avg = np.mean(pos_cam_weights, axis=0)
        pos_std = np.std(pos_cam_weights, axis=0)
    else:
        pos_avg = np.zeros_like(all_avg)
        pos_std = np.zeros_like(all_avg)
    
    # Compute average weights for negative samples
    if neg_cam_weights:
        neg_avg = np.mean(neg_cam_weights, axis=0)
        neg_std = np.std(neg_cam_weights, axis=0)
    else:
        neg_avg = np.zeros_like(all_avg)
        neg_std = np.zeros_like(all_avg)
    
    # Build summary DataFrame
    summary_data = []
    for i, mz in enumerate(mz_axis):
        summary_data.append({
            'm/z': round(mz, 2),
            'Mean_Weight_All': round(all_avg[i], 6),
            'Std_Weight_All': round(all_std[i], 6),
            'Mean_Weight_Pos': round(pos_avg[i], 6),
            'Mean_Weight_Neg': round(neg_avg[i], 6),
            'Weight_Diff_Pos_Neg': round(pos_avg[i] - neg_avg[i], 6)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # ===== 6. Identify Key Ions =====
    print("\n  Identifying key ions...")
    
    # Cannabinoid-focused ions (highest positive weights)
    top_pos = summary_df.nlargest(30, 'Mean_Weight_Pos')[['m/z', 'Mean_Weight_Pos', 'Mean_Weight_Neg', 'Weight_Diff_Pos_Neg']]
    top_pos['Type'] = 'Cannabinoid-Focused'
    
    # Non-cannabinoid-focused ions (highest negative weights)
    top_neg = summary_df.nlargest(30, 'Mean_Weight_Neg')[['m/z', 'Mean_Weight_Neg', 'Mean_Weight_Pos', 'Weight_Diff_Pos_Neg']]
    top_neg['Type'] = 'Non-Cannabinoid-Focused'
    
    # Cannabinoid-specific ions (positive >> negative)
    top_diff_pos = summary_df.nlargest(20, 'Weight_Diff_Pos_Neg')[['m/z', 'Mean_Weight_Pos', 'Mean_Weight_Neg', 'Weight_Diff_Pos_Neg']]
    top_diff_pos['Type'] = 'Cannabinoid-Specific'
    
    # Non-cannabinoid-specific ions (negative >> positive)
    top_diff_neg = summary_df.nsmallest(20, 'Weight_Diff_Pos_Neg')[['m/z', 'Mean_Weight_Pos', 'Mean_Weight_Neg', 'Weight_Diff_Pos_Neg']]
    top_diff_neg['Type'] = 'Non-Cannabinoid-Specific'
    
    # ===== 7. Export to Excel =====
    print("\n  Exporting to Excel...")
    
    with pd.ExcelWriter(output_dir / 'gradcam_analysis.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Summary statistics for all m/z
        summary_df.to_excel(writer, sheet_name='All_Ion_Weights', index=False)
        
        # Sheet 2: Cannabinoid-focused ions
        top_pos.to_excel(writer, sheet_name='Cannabinoid_Focused', index=False)
        
        # Sheet 3: Non-cannabinoid-focused ions
        top_neg.to_excel(writer, sheet_name='NonCannabinoid_Focused', index=False)
        
        # Sheet 4: Cannabinoid-specific ions
        top_diff_pos.to_excel(writer, sheet_name='Cannabinoid_Specific', index=False)
        
        # Sheet 5: Non-cannabinoid-specific ions
        top_diff_neg.to_excel(writer, sheet_name='NonCannabinoid_Specific', index=False)
        
        # Sheet 6: Top 15 ions per sample
        pd.DataFrame(sample_top_ions).to_excel(writer, sheet_name='PerSample_Top15_Ions', index=False)
    
    print(f"  Excel exported: {output_dir / 'gradcam_analysis.xlsx'}")
    
    # ===== 8. Print Key Ions Summary =====
    print("\n" + "="*60)
    print("Key Ion Summary")
    print("="*60)
    
    print("\n[Cannabinoid-Specific Ions] (Positive weight significantly higher than Negative)")
    print("-" * 70)
    for _, row in top_diff_pos.iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  Pos={row['Mean_Weight_Pos']:.5f}, Neg={row['Mean_Weight_Neg']:.5f}, Diff={row['Weight_Diff_Pos_Neg']:.5f}")
    
    print("\n[Non-Cannabinoid-Specific Ions] (Negative weight significantly higher than Positive)")
    print("-" * 70)
    for _, row in top_diff_neg.iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  Neg={row['Mean_Weight_Neg']:.5f}, Pos={row['Mean_Weight_Pos']:.5f}, Diff={abs(row['Weight_Diff_Pos_Neg']):.5f}")
    
    print("\n[Top 15 Ions Most Focused by Cannabinoid Samples]")
    print("-" * 70)
    for _, row in top_pos.head(15).iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  Weight={row['Mean_Weight_Pos']:.5f}")
    
    print("\n[Top 15 Ions Most Focused by Non-Cannabinoid Samples]")
    print("-" * 70)
    for _, row in top_neg.head(15).iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  Weight={row['Mean_Weight_Neg']:.5f}")
    
    # ===== 9. Statistics by m/z Range =====
    print("\n" + "="*60)
    print("Statistics by m/z Range")
    print("="*60)
    
    ranges = [(40, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
    range_labels = ['40-100', '100-200', '200-300', '300-400', '400-500', '500-600']
    
    print("\n  Cannabinoid-focused ion distribution:")
    for (low, high), label in zip(ranges, range_labels):
        count = len(top_pos[(top_pos['m/z'] >= low) & (top_pos['m/z'] < high)])
        print(f"    {label}: {count}")
    
    print("\n  Non-cannabinoid-focused ion distribution:")
    for (low, high), label in zip(ranges, range_labels):
        count = len(top_neg[(top_neg['m/z'] >= low) & (top_neg['m/z'] < high)])
        print(f"    {label}: {count}")
    
    print("\n  Cannabinoid-specific ion distribution:")
    for (low, high), label in zip(ranges, range_labels):
        count = len(top_diff_pos[(top_diff_pos['m/z'] >= low) & (top_diff_pos['m/z'] < high)])
        print(f"    {label}: {count}")
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"  Output file: {output_dir / 'gradcam_analysis.xlsx'}")
    print(f"  Contains the following sheets:")
    print(f"    1. All_Ion_Weights - Mean CAM weights for all m/z")
    print(f"    2. Cannabinoid_Focused - Ions most focused by cannabinoid samples")
    print(f"    3. NonCannabinoid_Focused - Ions most focused by non-cannabinoid samples")
    print(f"    4. Cannabinoid_Specific - Ions unique to cannabinoids")
    print(f"    5. NonCannabinoid_Specific - Ions unique to non-cannabinoids")
    print(f"    6. PerSample_Top15_Ions - Top 15 ions per sample")