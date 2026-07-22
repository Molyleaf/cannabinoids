import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

project_root = r"d:\UserFiles\Documents\PyCharm\cannabinoids"
sys.path.insert(0, project_root)

# GN/LN Encoder
class CorrectSpectrumEncoder(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        norm = torch.norm(embed, p=2, dim=1, keepdim=True)
        return embed / torch.clamp(norm, min=1e-3)

# Correct Classifier Head with LN
class CorrectBinaryClassifier(nn.Module):
    def __init__(self, encoder, input_dim=256):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        with torch.no_grad():
            embed = self.encoder(x)
        logit = self.classifier(embed)
        return logit.squeeze(-1)

# Helper functions
def parse_large_msp_generator(file_path):
    current_block = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("Name:"):
                if current_block:
                    yield "\n".join(current_block) + "\n\n"
                current_block = [line.strip()]
            else:
                if current_block:
                    current_block.append(line.strip())
        if current_block:
            yield "\n".join(current_block) + "\n\n"

def parse_block(block):
    peaks = []
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" in line:
            continue
        parts = line.replace(';', '').replace('\t', ' ').split()
        if len(parts) >= 2:
            try:
                peaks.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return peaks

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
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]
    tic = vecs.sum(axis=1, keepdims=True)
    vecs = vecs / (tic + 1e-8)
    vecs = np.sqrt(vecs)
    return vecs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    encoder = CorrectSpectrumEncoder().to(device)
    model = CorrectBinaryClassifier(encoder).to(device)
    
    # Load original weights from binary_classifier_weights_20260716_164027.pt
    ckpt_path = os.path.join(project_root, "simclr_finetune", "binary_classifier_weights_20260716_164027.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load weights with strict=True to verify exact architectural alignment
    encoder.load_state_dict(ckpt['encoder_state_dict'], strict=True)
    model.classifier.load_state_dict(ckpt['classifier_state_dict'], strict=True)
    print("Loaded weights successfully with strict=True!")
    
    model.eval()
    
    # Evaluate on MoNA
    mona_path = os.path.join(project_root, "scratch", "MONA_GCMS-18914.MSP")
    if not os.path.exists(mona_path):
        print(f"MoNA not found at {mona_path}")
        return
        
    print("Evaluating on MoNA...")
    mona_probs = []
    processed = 0
    with torch.no_grad():
        for block in parse_large_msp_generator(mona_path):
            processed += 1
            peaks = parse_block(block)
            if not peaks:
                mona_probs.append(0.0)
                continue
            vec = peaks_to_vector(peaks)
            vec_norm = preprocess_spectra(vec)
            x_tensor = torch.tensor(vec_norm, dtype=torch.float32).to(device)
            logit = model(x_tensor)
            prob = torch.sigmoid(logit).item()
            mona_probs.append(prob)
            
            if processed % 5000 == 0:
                print(f"Processed {processed}...")
                
    mona_probs = np.array(mona_probs)
    high_risk_count = np.sum(mona_probs >= 0.5)
    print(f"MoNA High Risk count: {high_risk_count} ({high_risk_count/len(mona_probs)*100:.2f}%)")
    print(f"MoNA Average prob: {mona_probs.mean():.4f}")

if __name__ == "__main__":
    main()
