import os, sys, torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.pipeline import peaks_to_vector, preprocess_spectra, get_classifier_model

def parse_msp_correct(file_bytes):
    text = file_bytes.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    compounds = []
    current_comp = None
    in_peaks = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith('name:'):
            if current_comp is not None and len(current_comp['peaks']) > 0:
                compounds.append(current_comp)
            current_comp = {'name': stripped.split(':', 1)[1].strip(), 'peaks': []}
            in_peaks = False
        elif current_comp is not None:
            if lower.startswith('num peaks:') or lower.startswith('num_peaks:'):
                in_peaks = True
            elif in_peaks:
                sub_items = stripped.split(';')
                for sub in sub_items:
                    sub = sub.strip()
                    if not sub:
                        continue
                    parts = sub.replace('\t', ' ').split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            if mz > 0 and intensity > 0:
                                current_comp['peaks'].append([mz, intensity])
                        except ValueError:
                            pass
    if current_comp is not None and len(current_comp['peaks']) > 0:
        compounds.append(current_comp)
    return compounds

models = {
    "162614_pt": r"simclr_finetune/results_20260723_162614/binary_classifier_20260723_162614.pt",
    "195942_pt": r"simclr_finetune/results_20260723_195942/binary_classifier_20260723_195942.pt",
    "211225_safetensors": r"app/models/binary_classifier_weights_20260722_211225.safetensors"
}

with open(r"scratch/data_source/20260709.MSP", "rb") as f:
    file_bytes = f.read()

comps = parse_msp_correct(file_bytes)

print("=== CORRECT PARSED PEAKS & MODEL PREDICTIONS ===")
for m_name, m_path in models.items():
    if not os.path.exists(m_path): continue
    model = get_classifier_model(model_path=m_path)
    print(f"\n--- Model: {m_name} ---")
    high_count = 0
    for c in comps:
        peaks = c['peaks']
        vec = peaks_to_vector(peaks)
        vec_norm = preprocess_spectra(vec)
        x_tensor = torch.tensor(vec_norm, dtype=torch.float32)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(x_tensor)).item())
        if prob >= 0.5: high_count += 1
        print(f"  {c['name']} (Peaks: {len(peaks)}) -> Risk Prob: {prob:.4f} | High Risk: {prob >= 0.5}")
    print(f"High Risk Count: {high_count} / {len(comps)}")
