# ============================================================================
# Open Set Recognition – Export Excel Plot Data
# Training set contains only known classes (positive). Test set contains both
# known (positive) and unknown (negative / novel) classes.
# ============================================================================
print("\n" + "="*60)
print("[Open Set Recognition – Export Excel Plot Data]")
print("="*60)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Assumes X, y, smiles_all are pre-defined externally
# X: (N, 561) spectral data
# y: (N,) labels, 1=active (known), 0=inactive (unknown / novel)
# smiles_all: (N,) SMILES strings

# ============================================================================
# 0. Data Split – SMILES-Stratified to Prevent Leakage
# ============================================================================
print("\n" + "="*60)
print("[0. Data Split – Only Positives as Known Class]")
print("="*60)

# Extract positive sample indices
pos_indices = np.where(y == 1)[0]
neg_indices = np.where(y == 0)[0]

print(f"Total positive samples: {len(pos_indices)}")
print(f"Total negative samples: {len(neg_indices)}")

# Get unique SMILES for positive samples
pos_smiles = smiles_all[pos_indices]
unique_pos_smiles = list(set(pos_smiles))
print(f"Unique positive SMILES: {len(unique_pos_smiles)}")

# Split positive SMILES into train/val/test (60% / 20% / 20%)
train_pos_smiles, temp_pos_smiles = train_test_split(
    unique_pos_smiles, 
    test_size=0.4,
    random_state=42
)
val_pos_smiles, test_pos_smiles = train_test_split(
    temp_pos_smiles,
    test_size=0.5,
    random_state=42
)

print(f"Training positive SMILES: {len(train_pos_smiles)}")
print(f"Validation positive SMILES: {len(val_pos_smiles)}")
print(f"Test positive SMILES: {len(test_pos_smiles)}")

# Assign positive indices by SMILES
train_pos_idx = []
val_pos_idx = []
test_pos_idx = []

for i, smi in enumerate(pos_smiles):
    if smi in train_pos_smiles:
        train_pos_idx.append(pos_indices[i])
    elif smi in val_pos_smiles:
        val_pos_idx.append(pos_indices[i])
    elif smi in test_pos_smiles:
        test_pos_idx.append(pos_indices[i])

# Negative samples: all treated as unknown, split 50/50 into val/test
val_neg_idx, test_neg_idx = train_test_split(
    neg_indices,
    test_size=0.5,
    random_state=42
)

print(f"\nTraining positive samples: {len(train_pos_idx)}")
print(f"Validation: known={len(val_pos_idx)}, novel={len(val_neg_idx)}")
print(f"Test: known={len(test_pos_idx)}, novel={len(test_neg_idx)}")

# Assert no overlap
train_set = set(train_pos_idx)
val_set = set(val_pos_idx) | set(val_neg_idx)
test_set = set(test_pos_idx) | set(test_neg_idx)
assert len(train_set & val_set) == 0, "Training and validation sets overlap!"
assert len(train_set & test_set) == 0, "Training and test sets overlap!"
assert len(val_set & test_set) == 0, "Validation and test sets overlap!"
print("All splits are mutually exclusive – no leakage!")

# Create output directory
output_dir = Path(r"D:\DL\cann\建模\open_set_no_leakage")
output_dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 1. Encoder Architecture
# ============================================================================
print("\n" + "="*60)
print("[1. Encoder Definition]")
print("="*60)

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

class TripletModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

print("Model definition complete")

# ============================================================================
# 2. Train Triplet Encoder – Positive-Only Training Set
# ============================================================================
print("\n" + "="*60)
print("[2. Training Triplet Encoder – Positive-Only Training Set]")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Extract training positive data
X_train_pos = X[train_pos_idx]
smiles_train_pos = smiles_all[train_pos_idx]

# Initialize encoder
encoder = SpectrumEncoder(input_dim=561, hidden_dim=256).to(device)
triplet_model = TripletModel(encoder).to(device)

# Group by SMILES to build positive pairs
smiles_to_indices = defaultdict(list)
for i, smi in enumerate(smiles_train_pos):
    smiles_to_indices[smi].append(i)

anchor_pairs = []
for smi, indices in smiles_to_indices.items():
    if len(indices) >= 2:
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                anchor_pairs.append((indices[i], indices[j]))

print(f"Positive pairs: {len(anchor_pairs)}")

if len(anchor_pairs) == 0:
    raise ValueError("No positive pairs can be constructed. Check data or use alternative method.")

optimizer = torch.optim.Adam(triplet_model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=0.3)

X_tensor = torch.tensor(X_train_pos, dtype=torch.float32)

triplet_model.train()
for epoch in range(50):
    epoch_loss = 0
    np.random.shuffle(anchor_pairs)
    for i in range(0, len(anchor_pairs), 64):
        batch_pairs = anchor_pairs[i:i+64]
        anchors = []
        positives = []
        negatives = []
        for idx1, idx2 in batch_pairs:
            anchors.append(X_tensor[idx1])
            positives.append(X_tensor[idx2])
            # Negative: randomly sample a different SMILES
            neg_smi = None
            while neg_smi is None or neg_smi == smiles_train_pos[idx1]:
                neg_idx = np.random.randint(len(X_train_pos))
                neg_smi = smiles_train_pos[neg_idx]
            negatives.append(X_tensor[neg_idx])
        
        anchors = torch.stack(anchors).to(device)
        positives = torch.stack(positives).to(device)
        negatives = torch.stack(negatives).to(device)
        
        optimizer.zero_grad()
        anchor_emb = triplet_model(anchors)
        positive_emb = triplet_model(positives)
        negative_emb = triplet_model(negatives)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {epoch_loss/len(anchor_pairs):.4f}")

print("Triplet encoder training complete")
triplet_model.eval()

# ============================================================================
# 3. Build Known-Class Library (Training Positive Embeddings)
# ============================================================================
print("\n" + "="*60)
print("[3. Building Known-Class Library]")
print("="*60)

library_embeddings = []
with torch.no_grad():
    for idx in train_pos_idx:
        spec = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)
        embed = triplet_model(spec)
        library_embeddings.append(embed.cpu().numpy().flatten())
library_embeddings = np.array(library_embeddings)
print(f"Library size: {len(library_embeddings)}")

# ============================================================================
# 4. Helper Function: Maximum Cosine Similarity
# ============================================================================
def get_scores(indices):
    """Compute maximum cosine similarity between given samples and the known library."""
    scores = []
    with torch.no_grad():
        for idx in indices:
            spec = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)
            embed = triplet_model(spec).cpu().numpy().flatten()
            embed = embed / (np.linalg.norm(embed) + 1e-8)
            lib_norm = library_embeddings / (np.linalg.norm(library_embeddings, axis=1, keepdims=True) + 1e-8)
            sims = np.dot(lib_norm, embed)
            scores.append(np.max(sims))
    return np.array(scores)

print("Helper function defined")

# ============================================================================
# 5. Validation Set – Threshold Selection
# ============================================================================
print("\n" + "="*60)
print("[5. Optimal Threshold Selection on Validation Set]")
print("="*60)

val_known_scores = get_scores(val_pos_idx)
val_novel_scores = get_scores(val_neg_idx)

val_scores = np.concatenate([val_known_scores, val_novel_scores])
val_labels = np.concatenate([np.ones(len(val_known_scores)), np.zeros(len(val_novel_scores))])

best_threshold = 0.5
best_f1 = 0
thresholds = np.linspace(0.0, 1.0, 201)
threshold_f1_scores = []

for thresh in thresholds:
    preds = (val_scores >= thresh).astype(int)
    f1 = f1_score(val_labels, preds, zero_division=0)
    threshold_f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"Optimal threshold: {best_threshold:.4f} (Val F1 = {best_f1:.4f})")

# ============================================================================
# 6. Test Set Evaluation
# ============================================================================
print("\n" + "="*60)
print("[6. Test Set Performance Evaluation]")
print("="*60)

test_known_scores = get_scores(test_pos_idx)
test_novel_scores = get_scores(test_neg_idx)

test_scores = np.concatenate([test_known_scores, test_novel_scores])
test_labels = np.concatenate([np.ones(len(test_known_scores)), np.zeros(len(test_novel_scores))])
preds = (test_scores >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, preds)
acc = (preds == test_labels).mean()
f1 = f1_score(test_labels, preds, zero_division=0)
auc = roc_auc_score(test_labels, test_scores)

print("\nConfusion Matrix:")
print(f"              Pred Known    Pred Novel")
print(f"True Known     {cm[1,1]:5d}  {cm[1,0]:5d}")
print(f"True Novel     {cm[0,1]:5d}  {cm[0,0]:5d}")
print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC:      {auc:.4f}")

known_mis = cm[1,0]
novel_mis = cm[0,1]
print(f"\nKnown-class recall: {cm[1,1]/(cm[1,1]+cm[1,0]+1e-8):.4f}")
print(f"Known misclassified as novel: {known_mis} ({known_mis/len(test_pos_idx)*100:.2f}%)")
print(f"Novel-class recall: {cm[0,0]/(cm[0,0]+cm[0,1]+1e-8):.4f}")
print(f"Novel misclassified as known: {novel_mis} ({novel_mis/len(test_neg_idx)*100:.2f}%)")

# ============================================================================
# 7. Export Excel Plot Data
# ============================================================================
print("\n" + "="*60)
print("[7. Exporting Excel Plot Data]")
print("="*60)

excel_path = output_dir / 'openset_plot_data.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    
    # Sheet 1: Validation scores
    val_df = pd.DataFrame({
        'Score': np.concatenate([val_known_scores, val_novel_scores]),
        'Class': ['Known'] * len(val_known_scores) + ['Novel'] * len(val_novel_scores)
    })
    val_df.to_excel(writer, sheet_name='Validation_Scores', index=False)
    
    # Sheet 2: Test scores
    test_df = pd.DataFrame({
        'Score': np.concatenate([test_known_scores, test_novel_scores]),
        'Class': ['Known'] * len(test_known_scores) + ['Novel'] * len(test_novel_scores)
    })
    test_df.to_excel(writer, sheet_name='Test_Scores', index=False)
    
    # Sheet 3: Validation histogram statistics
    bins = np.linspace(0, 1, 31)
    val_known_hist, _ = np.histogram(val_known_scores, bins=bins)
    val_novel_hist, _ = np.histogram(val_novel_scores, bins=bins)
    val_hist_df = pd.DataFrame({
        'Bin_Center': (bins[:-1] + bins[1:]) / 2,
        'Bin_Lower': bins[:-1],
        'Bin_Upper': bins[1:],
        'Known_Count': val_known_hist,
        'Novel_Count': val_novel_hist
    })
    val_hist_df.to_excel(writer, sheet_name='Val_Histogram', index=False)
    
    # Sheet 4: Test histogram statistics
    test_known_hist, _ = np.histogram(test_known_scores, bins=bins)
    test_novel_hist, _ = np.histogram(test_novel_scores, bins=bins)
    test_hist_df = pd.DataFrame({
        'Bin_Center': (bins[:-1] + bins[1:]) / 2,
        'Bin_Lower': bins[:-1],
        'Bin_Upper': bins[1:],
        'Known_Count': test_known_hist,
        'Novel_Count': test_novel_hist
    })
    test_hist_df.to_excel(writer, sheet_name='Test_Histogram', index=False)
    
    # Sheet 5: Threshold-F1 curve
    threshold_df = pd.DataFrame({
        'Threshold': thresholds,
        'F1_Score': threshold_f1_scores
    })
    threshold_df.to_excel(writer, sheet_name='Threshold_F1_Curve', index=False)
    
    # Sheet 6: Performance summary
    summary_df = pd.DataFrame({
        'Metric': ['Optimal Threshold', 'Val F1', 'Test Accuracy', 'Test F1', 'Test AUC',
                   'Known Recall', 'Known Misclass.', 'Novel Recall', 'Novel Misclass.'],
        'Value': [best_threshold, best_f1, acc, f1, auc,
                  cm[1,1]/(cm[1,1]+cm[1,0]+1e-8), known_mis,
                  cm[0,0]/(cm[0,0]+cm[0,1]+1e-8), novel_mis],
        'Note': ['', '', '', '', '',
                 f'({cm[1,1]}/{cm[1,1]+cm[1,0]})', f'/{len(test_pos_idx)}',
                 f'({cm[0,0]}/{cm[0,0]+cm[0,1]})', f'/{len(test_neg_idx)}']
    })
    summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
    
    # Sheet 7: Confusion matrix
    cm_df = pd.DataFrame({
        '': ['True Known', 'True Novel'],
        'Pred Known': [cm[1,1], cm[0,1]],
        'Pred Novel': [cm[1,0], cm[0,0]]
    })
    cm_df.to_excel(writer, sheet_name='Confusion_Matrix', index=False)

print(f"Excel data saved: {excel_path}")

# ============================================================================
# 8. Generate Visualization (PDF/PNG)
# ============================================================================
print("\n" + "="*60)
print("[8. Generating Visualization]")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Validation set
ax1 = axes[0]
ax1.hist(val_known_scores, bins=30, alpha=0.6, label='Known (Val)', color='green', edgecolor='black')
ax1.hist(val_novel_scores, bins=30, alpha=0.6, label='Novel (Val)', color='red', edgecolor='black')
ax1.axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.3f}')
ax1.set_xlabel('Max Cosine Similarity')
ax1.set_ylabel('Frequency')
ax1.set_title('Validation Set Score Distribution')
ax1.legend()
ax1.grid(alpha=0.3)
ks_val, _ = ks_2samp(val_known_scores, val_novel_scores)
ax1.text(0.02, 0.95, f'KS = {ks_val:.4f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')

# Test set
ax2 = axes[1]
ax2.hist(test_known_scores, bins=30, alpha=0.6, label='Known (Test)', color='green', edgecolor='black')
ax2.hist(test_novel_scores, bins=30, alpha=0.6, label='Novel (Test)', color='red', edgecolor='black')
ax2.axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.3f}')
ax2.set_xlabel('Max Cosine Similarity')
ax2.set_ylabel('Frequency')
ax2.set_title('Test Set Score Distribution')
ax2.legend()
ax2.grid(alpha=0.3)
ks_test, _ = ks_2samp(test_known_scores, test_novel_scores)
ax2.text(0.02, 0.95, f'KS = {ks_test:.4f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top')

plt.tight_layout()
save_path = output_dir / 'openset_score_distribution.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {save_path}")
plt.show()

print("\n" + "="*60)
print("[Done!]")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"Excel plot data: {excel_path}")
print(f"Figure: {save_path}")