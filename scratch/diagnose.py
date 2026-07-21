# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ms_entropy import read_one_spectrum
from simclr_pretrain.lib.models import SpectrumEncoder

def peaks_to_vector(raw_peaks, input_dim=561, start_mz=40):
    vec = np.zeros(input_dim, dtype=np.float32)
    for p in raw_peaks:
        try:
            mz = float(p[0])
            intensity = float(p[1]) if not isinstance(p[1], str) else float(p[1].replace(';', ''))
            mz_idx = int(round(mz)) - start_mz
            if 0 <= mz_idx < input_dim:
                vec[mz_idx] += intensity
        except Exception:
            continue
    tic = np.sum(vec)
    if tic > 1e-8:
        vec = vec / tic
    # 开根号
    vec = np.sqrt(vec)
    return vec

def load_msp_vectors(msp_path):
    print(f"Loading {os.path.basename(msp_path)}...")
    vectors = []
    names = []
    for spec in read_one_spectrum(msp_path):
        raw_peaks = spec.get("peaks")
        if not raw_peaks:
            continue
        name = spec.get("name") or spec.get("Name") or "Unknown"
        vec = peaks_to_vector(raw_peaks)
        vectors.append(vec)
        names.append(name)
    return np.array(vectors, dtype=np.float32), names

def analyze_similarities(name, S_matrix, is_symmetric=False):
    # 如果是对称矩阵，排除对角线
    if is_symmetric:
        M = S_matrix.shape[0]
        mask = ~np.eye(M, dtype=bool)
        sims = S_matrix[mask]
    else:
        sims = S_matrix.flatten()
        
    mean = np.mean(sims)
    std = np.std(sims)
    minimum = np.min(sims)
    maximum = np.max(sims)
    p25 = np.percentile(sims, 25)
    p50 = np.percentile(sims, 50)
    p75 = np.percentile(sims, 75)
    p90 = np.percentile(sims, 90)
    
    print(f"--- {name} Similarity Distribution ---")
    print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
    print(f"  Min / Max : {minimum:.4f} / {maximum:.4f}")
    print(f"  Quantiles : 25%={p25:.4f}, 50%={p50:.4f}, 75%={p75:.4f}, 90%={p90:.4f}")
    print()

def main():
    positive_msp = os.path.join(project_root, "known_compound_recognize", "positive.msp")
    negative_msp = os.path.join(current_dir, "阴性.msp")
    
    # 1. 加载数据向量
    pos_vecs, pos_names = load_msp_vectors(positive_msp)
    neg_vecs, neg_names = load_msp_vectors(negative_msp)
    print(f"Loaded {len(pos_vecs)} positive spectra and {len(neg_vecs)} negative spectra.")
    print()
    
    # 2. 分析原始 561 维特征向量的相似度 (即输入余弦相似度)
    print("=" * 60)
    print("ANALYSIS 1: Raw Vector Cosine Similarities (Input Space)")
    print("=" * 60)
    
    # 计算原始相似度矩阵
    # 因为已经归一化和sqrt，L2范数应该接近 1.0 (其实是1.0)
    pos_norms = np.linalg.norm(pos_vecs, axis=1, keepdims=True)
    neg_norms = np.linalg.norm(neg_vecs, axis=1, keepdims=True)
    
    pos_vecs_norm = pos_vecs / np.clip(pos_norms, 1e-8, None)
    neg_vecs_norm = neg_vecs / np.clip(neg_norms, 1e-8, None)
    
    S_pos_pos_raw = np.dot(pos_vecs_norm, pos_vecs_norm.T)
    S_neg_neg_raw = np.dot(neg_vecs_norm, neg_vecs_norm.T)
    S_neg_pos_raw = np.dot(neg_vecs_norm, pos_vecs_norm.T)
    
    analyze_similarities("Positive-Positive (Raw)", S_pos_pos_raw, is_symmetric=True)
    analyze_similarities("Negative-Negative (Raw)", S_neg_neg_raw, is_symmetric=True)
    analyze_similarities("Negative-Positive (Raw)", S_neg_pos_raw, is_symmetric=False)
    
    # 3. 加载 SimCLR 模型，提取 Embedding
    print("=" * 60)
    print("ANALYSIS 2: SimCLR Embedding Cosine Similarities (Latent Space)")
    print("=" * 60)
    
    model_path = os.path.join(project_root, "model_archive", "best_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu")
    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    
    with torch.no_grad():
        pos_tensor = torch.tensor(pos_vecs, dtype=torch.float32)
        neg_tensor = torch.tensor(neg_vecs, dtype=torch.float32)
        
        pos_embeddings = []
        batch_size = 512
        for i in range(0, len(pos_tensor), batch_size):
            pos_embeddings.append(encoder(pos_tensor[i:i+batch_size]))
        pos_embeddings = torch.cat(pos_embeddings, dim=0).numpy()
        
        neg_embeddings = []
        for i in range(0, len(neg_tensor), batch_size):
            neg_embeddings.append(encoder(neg_tensor[i:i+batch_size]))
        neg_embeddings = torch.cat(neg_embeddings, dim=0).numpy()
        
    S_pos_pos_sim = np.dot(pos_embeddings, pos_embeddings.T)
    S_neg_neg_sim = np.dot(neg_embeddings, neg_embeddings.T)
    S_neg_pos_sim = np.dot(neg_embeddings, pos_embeddings.T)
    
    analyze_similarities("Positive-Positive (SimCLR)", S_pos_pos_sim, is_symmetric=True)
    analyze_similarities("Negative-Negative (SimCLR)", S_neg_neg_sim, is_symmetric=True)
    analyze_similarities("Negative-Positive (SimCLR)", S_neg_pos_sim, is_symmetric=False)
    
    # 4. 分析高丰度 m/z 通道
    print("=" * 60)
    print("ANALYSIS 3: High Abundance m/z Channels Analysis")
    print("=" * 60)
    
    # 计算在 [40, 600] 范围内的平均相对强度
    # 还原到线性强度空间 (平方)
    pos_linear = pos_vecs ** 2
    neg_linear = neg_vecs ** 2
    
    pos_mean_intensity = np.mean(pos_linear, axis=0)
    neg_mean_intensity = np.mean(neg_linear, axis=0)
    
    # 获取平均强度最高的前 10 个 m/z
    pos_top_idx = np.argsort(pos_mean_intensity)[::-1][:10]
    neg_top_idx = np.argsort(neg_mean_intensity)[::-1][:10]
    
    print("Top 10 highest abundance m/z in Positive samples:")
    for idx in pos_top_idx:
        print(f"  m/z {idx + 40}: average intensity = {pos_mean_intensity[idx]:.4f}")
    print()
    
    print("Top 10 highest abundance m/z in Negative samples:")
    for idx in neg_top_idx:
        print(f"  m/z {idx + 40}: average intensity = {neg_mean_intensity[idx]:.4f}")
    print()
    
    # 看看两者共有的高丰度峰
    common_high_mz = set(pos_top_idx + 40).intersection(set(neg_top_idx + 40))
    print(f"Common high-abundance m/z channels between positive and negative samples: {list(common_high_mz)}")
    print()

if __name__ == "__main__":
    main()
