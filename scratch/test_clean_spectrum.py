import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ms_entropy import read_one_spectrum, clean_spectrum

def peaks_to_vector_original(peaks, mz_min=40, mz_max=600):
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
    
    raw_vecs = []
    clean_vecs = []
    
    for spec in read_one_spectrum(pos_msp):
        raw_peaks = spec.get("peaks")
        if not raw_peaks:
            continue
        peaks_list = []
        for p in raw_peaks:
            try:
                mz = float(p[0])
                intensity = float(p[1].replace(';', '') if isinstance(p[1], str) else p[1])
                peaks_list.append([mz, intensity])
            except Exception:
                continue
        if not peaks_list:
            continue
            
        peaks_arr = np.array(peaks_list, dtype=np.float32)
        cleaned_arr = clean_spectrum(peaks_arr)
        
        v_raw = peaks_to_vector_original(peaks_list)
        v_clean = peaks_to_vector_original(cleaned_arr)
        
        raw_vecs.append(v_raw)
        clean_vecs.append(v_clean)
        
    raw_norm = preprocess_spectra(np.array(raw_vecs, dtype=np.float32))
    clean_norm = preprocess_spectra(np.array(clean_vecs, dtype=np.float32))
    
    diff = np.abs(raw_norm - clean_norm).max()
    mean_diff = np.abs(raw_norm - clean_norm).mean()
    print(f"Comparison between raw binned vector vs clean_spectrum binned vector:")
    print(f"  Max absolute difference : {diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

if __name__ == "__main__":
    main()
