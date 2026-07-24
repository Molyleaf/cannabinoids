from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

# 动态确保项目根目录在 python 模块路径中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.augmentation import SpectrumAugmentation


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
    
    pos_idx_to_smiles = dict(zip(pos_indices, pos_smiles_list))
    
    pos_train_idx = [i for i in pos_indices if pos_idx_to_smiles[i] in pos_train_smiles]
    pos_val_idx = [i for i in pos_indices if pos_idx_to_smiles[i] in pos_val_smiles]
    pos_test_idx = [i for i in pos_indices if pos_idx_to_smiles[i] in pos_test_smiles]
    
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
    parquet_path=None,
    pos_msp_path=None,
    neg_msp_path=None,
    batch_size=128, 
    test_size=0.15, 
    val_size=0.15, 
    random_state=42, 
    use_cuda=True
):
    """
    后训练 (Finetune) 数据加载与划分流水线。
    读取预处理完成的合并单文件 Parquet (preprocessed_finetune.parquet)。
    无任何 MSP 解析与兜底/回退逻辑。
    """
    if parquet_path is None:
        parquet_path = Path(__file__).resolve().parent.parent / "data_source" / "preprocessed_finetune.parquet"
        
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"错误: 未找到后训练预处理 Parquet 文件 {parquet_file}")

    print(f"[Dataset] 正在从 Parquet 直接加载后训练数据集: {parquet_file.name}...")
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    
    # 直接提取 561 维归一化向量与 0/1 标签
    X = np.array(df['spectrum_vector'].tolist(), dtype=np.float32)
    y = df['label'].values.astype(np.float32)
    
    pos_indices = np.where(y == 1.0)[0]
    neg_indices = np.where(y == 0.0)[0]
    
    print(f"  [OK] 数据集加载成功: 总计 {len(df)} 条 (阳性 {len(pos_indices)}, 阴性 {len(neg_indices)})")

    pos_smiles = df.iloc[pos_indices]['smiles'].values
    pos_names = df.iloc[pos_indices]['name'].values
    pos_group_keys = np.array([s if (isinstance(s, str) and len(s.strip()) > 0) else n for s, n in zip(pos_smiles, pos_names)])
    
    train_idx, val_idx, test_idx = split_positive_by_smiles_negative_random(
        pos_indices, neg_indices, pos_group_keys,
        test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    sample_names_all = df['name'].values
    smiles_all = np.array([s if (isinstance(s, str) and len(s.strip()) > 0) else f"neg_{i}" for i, s in enumerate(df['smiles'].values)])
    
    train_sample_names = sample_names_all[train_idx]
    val_sample_names = sample_names_all[val_idx]
    test_sample_names = sample_names_all[test_idx]
    
    pin_mem = False
    
    train_dataset = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_dataset = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))
    test_dataset = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    
    pos_compounds = df[df['label'] == 1].to_dict('records')
    neg_compounds = df[df['label'] == 0].to_dict('records')
    
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

