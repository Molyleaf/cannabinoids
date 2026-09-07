from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

# Dynamically ensure project root is in Python module path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.augmentation import SpectrumAugmentation


def split_by_smiles(
    pos_indices,
    neg_indices,
    pos_smiles_list,
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    """
    Split dataset by SMILES for all samples (both positive and negative).
    All spectra sharing the same SMILES are kept within the same split.
    """
    np.random.seed(random_state)

    # Build complete SMILES mapping for both positive and negative samples
    all_indices = np.concatenate([pos_indices, neg_indices])
    all_smiles = np.concatenate([
        pos_smiles_list,
        np.array([f'neg_{i}' for i in range(len(neg_indices))])
    ])

    # Get unique SMILES
    unique_smiles = np.unique(all_smiles)
    n_smiles = len(unique_smiles)

    # Shuffle and split by SMILES
    shuffled_smiles = unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)

    n_test = max(1, int(n_smiles * test_size))
    n_val = max(1, int(n_smiles * val_size))

    test_smiles = set(shuffled_smiles[:n_test])
    val_smiles = set(shuffled_smiles[n_test:n_test + n_val])
    train_smiles = set(shuffled_smiles[n_test + n_val:])

    # Assign indices based on SMILES
    train_idx = []
    val_idx = []
    test_idx = []

    for idx, smi in zip(all_indices, all_smiles):
        if smi in train_smiles:
            train_idx.append(idx)
        elif smi in val_smiles:
            val_idx.append(idx)
        elif smi in test_smiles:
            test_idx.append(idx)

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # Shuffle each split
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def prepare_finetune_dataset(
    parquet_path=None,
    batch_size=128,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    use_cuda=True
):
    """
    Finetune data loading pipeline with SMILES-based splitting.
    Reads preprocessed Parquet file and splits all samples by SMILES.
    """
    if parquet_path is None:
        parquet_path = Path(__file__).resolve().parent.parent / "data_source" / "preprocessed_finetune.parquet"

    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Error: Finetune Parquet file not found: {parquet_file}")

    print(f"[Dataset] Loading finetune dataset from Parquet: {parquet_file.name}...")
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    # Extract 561-dim normalized vectors and labels
    X = np.array(df['spectrum_vector'].tolist(), dtype=np.float32)
    y = df['label'].values.astype(np.float32)

    pos_indices = np.where(y == 1.0)[0]
    neg_indices = np.where(y == 0.0)[0]

    print(f"  [OK] Dataset loaded: {len(df)} total samples (positive: {len(pos_indices)}, negative: {len(neg_indices)})")

    # Build SMILES keys for positive samples (fallback to name if SMILES is empty)
    pos_smiles = df.iloc[pos_indices]['smiles'].values
    pos_names = df.iloc[pos_indices]['name'].values
    pos_group_keys = np.array([
        s if (isinstance(s, str) and len(s.strip()) > 0) else n
        for s, n in zip(pos_smiles, pos_names)
    ])

    # Split all samples by SMILES (both positive and negative)
    train_idx, val_idx, test_idx = split_by_smiles(
        pos_indices,
        neg_indices,
        pos_group_keys,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Build SMILES array for all samples
    sample_names_all = df['name'].values
    smiles_all = np.array([
        s if (isinstance(s, str) and len(s.strip()) > 0) else f"neg_{i}"
        for i, s in enumerate(df['smiles'].values)
    ])

    train_sample_names = sample_names_all[train_idx]
    val_sample_names = sample_names_all[val_idx]
    test_sample_names = sample_names_all[test_idx]

    pin_mem = False

    # Create datasets and loaders
    train_dataset = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_dataset = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))
    test_dataset = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    # Metadata for reference
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