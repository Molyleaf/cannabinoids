"""
finetune/data_preprocess.py

Fine-tuning mass spectrum data preprocessing script (aligned with pre-training/inference unified standard)

Features:
1. Uses common.data_processor to parse negative and positive MSP mass spectrum files;
2. Negative samples are labeled as label=0, positive samples as label=1;
3. Uses ms-entropy (clean_spectrum) for mass spectrum outlier peak denoising and deduplication;
4. Maps cleaned mass spectrum peaks to [40, 600] range with 1 Da resolution, producing 561-dimensional feature vectors;
5. Performs total ion current (TIC) normalization and square root (sqrt) nonlinear scaling transform;
6. Exports a merged Parquet file with labels and full metadata to finetune/data_source/preprocessed_finetune.parquet.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Dynamically ensure project root is in Python module path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.data_processor import clean_spectrum, parse_msp, peaks_to_vector, preprocess_spectra


def process_single_compound(comp, label, mz_min=40, mz_max=600):
    """
    Process a single compound: denoising -> vectorization -> TIC normalization + sqrt transform
    """
    peaks = comp.get('peaks', [])
    cleaned_peaks = clean_spectrum(peaks)
    num_peaks = len(cleaned_peaks)
    
    vec = peaks_to_vector(cleaned_peaks, mz_min=mz_min, mz_max=mz_max)
    vec_norm = preprocess_spectra(vec).squeeze(0)  # shape (561,)
    
    return {
        'name': comp.get('name', ''),
        'smiles': comp.get('smiles', ''),
        'precursor_mz': float(comp['precursor_mz']) if comp.get('precursor_mz') is not None else None,
        'num_peaks': int(num_peaks),
        'label': int(label),
        'spectrum_vector': vec_norm.tolist()
    }


def _process_chunk_with_label(chunk_compounds, label, mz_min=40, mz_max=600):
    """Multi-process chunk processing function"""
    results = []
    for comp in chunk_compounds:
        results.append(process_single_compound(comp, label=label, mz_min=mz_min, mz_max=mz_max))
    return results


def preprocess_finetune_msp_to_parquet(
    pos_msp, neg_msp, output_parquet, 
    min_peaks=5, mz_min=40, mz_max=600, num_workers=None
):
    pos_path = Path(pos_msp)
    neg_path = Path(neg_msp)
    output_path = Path(output_parquet)
    
    if not pos_path.exists():
        raise FileNotFoundError(f"Positive MSP file not found: {pos_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"Negative MSP file not found: {neg_path}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"[Preprocess] Step 1/3: Parsing MSP data...")
    pos_compounds = parse_msp(pos_path, min_peaks=min_peaks)
    neg_compounds = parse_msp(neg_path, min_peaks=min_peaks)
    
    print(f"  - Positive compounds (Label 1): {len(pos_compounds)}")
    print(f"  - Negative compounds (Label 0): {len(neg_compounds)}")
    
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
        
    print(f"[Preprocess] Step 2/3: Multi-process acceleration for ms-entropy cleaning and [{mz_min}, {mz_max}] 561-dim mapping (Workers: {num_workers})...")
    
    processed_records = []
    
    # Process positive and negative samples separately with parallel execution
    for comps, label, desc in [(pos_compounds, 1, "Positive (Label 1)"), (neg_compounds, 0, "Negative (Label 0)")]:
        if not comps:
            continue
        chunk_size = max(50, len(comps) // (num_workers * 4))
        chunks = [comps[i:i + chunk_size] for i in range(0, len(comps), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_chunk_with_label, chunk, label, mz_min, mz_max) for chunk in chunks]
            for future in tqdm(futures, desc=f"[Processing {desc}]", unit="chunk"):
                processed_records.extend(future.result())
                
    print(f"[Preprocess] Step 3/3: Exporting to unified Parquet format...")
    
    df = pd.DataFrame(processed_records)
    
    schema = pa.schema([
        ('name', pa.string()),
        ('smiles', pa.string()),
        ('precursor_mz', pa.float64()),
        ('num_peaks', pa.int32()),
        ('label', pa.int32()),
        ('spectrum_vector', pa.list_(pa.float32()))
    ])
    
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, output_path, compression='snappy')
    
    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    
    print(f"[Preprocess] [OK] Export complete!")
    print(f"  - Output path: {output_path}")
    print(f"  - Total samples: {len(df)} (Positive Label 1: {pos_count}, Negative Label 0: {neg_count})")
    print(f"  - Feature dimension: {len(df['spectrum_vector'].iloc[0])} dims (List[float32])")
    print(f"  - File size: {file_size_mb:.2f} MB")
    print(f"  - Total time: {elapsed:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning mass spectrum data preprocessing (MSP -> Parquet)")
    base_dir = Path(__file__).resolve().parent
    default_pos = base_dir / "data_source" / "positive_cannabinoids.msp"
    default_neg = base_dir / "data_source" / "negative.msp"
    default_output = base_dir / "data_source" / "preprocessed_finetune.parquet"
    
    parser.add_argument("--pos_msp", type=str, default=str(default_pos), help="Positive MSP file path")
    parser.add_argument("--neg_msp", type=str, default=str(default_neg), help="Negative MSP file path")
    parser.add_argument("-o", "--output_parquet", type=str, default=str(default_output), help="Output Parquet file path")
    parser.add_argument("--min_peaks", type=int, default=5, help="Minimum number of valid peaks threshold (default: 5)")
    parser.add_argument("--mz_min", type=float, default=40.0, help="Minimum m/z value (default: 40)")
    parser.add_argument("--mz_max", type=float, default=600.0, help="Maximum m/z value (default: 600)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel worker processes")
    
    args = parser.parse_args()
    
    preprocess_finetune_msp_to_parquet(
        pos_msp=args.pos_msp,
        neg_msp=args.neg_msp,
        output_parquet=args.output_parquet,
        min_peaks=args.min_peaks,
        mz_min=int(args.mz_min),
        mz_max=int(args.mz_max),
        num_workers=args.num_workers
    )