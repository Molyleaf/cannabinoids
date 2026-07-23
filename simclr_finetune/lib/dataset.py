import torch
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

def parse_msp_with_smiles(msp_file, min_peaks=5):
    """解析 MSP 文件，返回包含 SMILES 的化合物字典列表"""
    msp_file = Path(msp_file)
    print(f"  正在解析 MSP 文件: {msp_file.name}...")
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
    
    print(f"  [OK] 成功解析出 {len(compounds)} 个化合物谱图")
    return compounds


def peaks_to_vector(peaks, mz_min=40, mz_max=600):
    """将 (mz, intensity) 峰列表映射为 561 维数值向量"""
    dim = mz_max - mz_min + 1
    vec = np.zeros(dim, dtype=np.float32)
    for mz, intensity in peaks:
        if mz_min <= mz <= mz_max:
            idx = int(round(mz - mz_min))
            if 0 <= idx < dim:
                vec[idx] += intensity
    return vec


def preprocess_spectra(spectra):
    """对质谱向量进行 TIC 归一化 + 平方根缩放"""
    tic = spectra.sum(axis=1, keepdims=True)
    spectra = spectra / (tic + 1e-8)
    spectra = np.sqrt(spectra)
    return spectra.astype(np.float32)


def split_positive_by_smiles_negative_random(
    pos_indices, neg_indices, 
    pos_smiles_list, 
    test_size=0.15, val_size=0.15, 
    random_state=42
):
    """
    划分数据集逻辑：
    阳性样本：按 SMILES 分组，同一 SMILES 的所有谱图放在同一集合
    阴性样本：随机划分
    """
    np.random.seed(random_state)
    
    # 阳性分组划分
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
    
    # 阴性随机划分
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


def prepare_finetune_dataset(
    pos_msp_path, neg_msp_path, 
    batch_size=128, test_size=0.15, val_size=0.15, 
    random_state=42, use_cuda=True
):
    """完整的质谱加载、清洗、转换与 DataLoader 划分流水线"""
    pos_compounds = parse_msp_with_smiles(pos_msp_path)
    neg_compounds = parse_msp_with_smiles(neg_msp_path)
    
    # 自动把 '.alpha.-pbp' 从阳性改为阴性
    alpha_pbp_indices = [i for i, c in enumerate(pos_compounds) if c['name'] == '.alpha.-pbp']
    if alpha_pbp_indices:
        removed_comps = [pos_compounds[i] for i in alpha_pbp_indices]
        pos_compounds = [comp for i, comp in enumerate(pos_compounds) if i not in alpha_pbp_indices]
        neg_compounds.extend(removed_comps)
        print(f"  [OK] 已将 {len(removed_comps)} 个 '.alpha.-pbp' 样本从阳性移至阴性集")
        
    pos_spectra = np.array([peaks_to_vector(c['peaks']) for c in pos_compounds])
    pos_smiles = np.array([c['smiles'] if c['smiles'] else c['name'] for c in pos_compounds])
    neg_spectra = np.array([peaks_to_vector(c['peaks']) for c in neg_compounds])
    
    X_pos = preprocess_spectra(pos_spectra)
    X_neg = preprocess_spectra(neg_spectra)
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos), dtype=np.float32), np.zeros(len(X_neg), dtype=np.float32)])
    smiles_all = np.concatenate([pos_smiles, np.array([f'neg_{i}' for i in range(len(X_neg))])])
    
    pos_indices = np.arange(len(pos_spectra))
    neg_indices = np.arange(len(pos_spectra), len(pos_spectra) + len(neg_spectra))
    
    train_idx, val_idx, test_idx = split_positive_by_smiles_negative_random(
        pos_indices, neg_indices, pos_smiles,
        test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 生成所有样本的名称数组
    sample_names_all = np.array([c['name'] for c in pos_compounds] + [c['name'] for c in neg_compounds])
    
    train_sample_names = sample_names_all[train_idx]
    val_sample_names = sample_names_all[val_idx]
    test_sample_names = sample_names_all[test_idx]
    
    pin_mem = use_cuda and torch.cuda.is_available()
    
    train_dataset = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_dataset = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))
    test_dataset = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    
    meta = {
        'X': X,
        'y': y,
        'smiles_all': smiles_all,
        'sample_names_all': sample_names_all,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'train_sample_names': train_sample_names,
        'val_sample_names': val_sample_names,
        'test_sample_names': test_sample_names,
        'pos_compounds': pos_compounds,
        'neg_compounds': neg_compounds
    }
    
    return train_loader, train_eval_loader, val_loader, test_loader, meta
