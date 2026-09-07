"""
embedding_pretrain/data_preprocess.py

Mass spectrum data preprocessing module (unified standard for pre-training, fine-tuning, and inference)

Features:
1. Uses common.data_processor for robust parsing of MSP mass spectrum data files;
2. Uses ms-entropy (clean_spectrum) for mass spectrum outlier peak denoising and deduplication;
3. Maps cleaned mass spectrum peaks to [40, 600] range with 1 Da resolution, producing 561-dimensional feature vectors;
4. Performs total ion current (TIC) normalization and square root (sqrt) nonlinear scaling transform;
5. Supports multi-process parallel acceleration;
6. Exports Parquet files with full metadata (name, smiles, precursor_mz, num_peaks, spectrum_vector).
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


def process_single_compound(comp, mz_min=40, mz_max=600):
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
        'spectrum_vector': vec_norm.tolist()
    }


def _process_chunk(chunk_compounds, mz_min=40, mz_max=600):
    """Multi-process chunk processing function"""
    results = []
    for comp in chunk_compounds:
        results.append(process_single_compound(comp, mz_min=mz_min, mz_max=mz_max))
    return results


def preprocess_msp_to_parquet(input_msp, output_parquet, min_peaks=5, mz_min=40, mz_max=600, num_workers=None):
    """
    Parse MSP file, preprocess using ms-entropy, and store as Parquet.
    """
    input_path = Path(input_msp)
    output_path = Path(output_parquet)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input MSP file not found: {input_path}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[Preprocess] Step 1/3: Parsing MSP nodes from {input_path.name}...")
    start_time = time.time()
    compounds = parse_msp(input_path, min_peaks=min_peaks)
    print(f"[Preprocess] Parsing complete, retrieved {len(compounds)} compounds with peaks >= {min_peaks} (Time: {time.time() - start_time:.2f}s)")
    
    if not compounds:
        print("[Warning] No valid compound spectra found!")
        return
        
    print(f"[Preprocess] Step 2/3: Performing ms-entropy cleaning denoising, [{mz_min}, {mz_max}] 561-dim mapping, and TIC+sqrt normalization...")
    
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
        
    print(f"[Preprocess] Launching multi-process acceleration (Workers: {num_workers})...")
    
    # Split into chunks for batch parallel processing
    chunk_size = max(100, len(compounds) // (num_workers * 4))
    chunks = [compounds[i:i + chunk_size] for i in range(0, len(compounds), chunk_size)]
    
    processed_records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk, mz_min, mz_max) for chunk in chunks]
        for future in tqdm(futures, desc="[Processing Chunks]", unit="chunk"):
            processed_records.extend(future.result())
            
    print(f"[Preprocess] Step 3/3: Exporting to unified standard Parquet data file...")
    
    # Build DataFrame and PyArrow Schema
    df = pd.DataFrame(processed_records)
    
    schema = pa.schema([
        ('name', pa.string()),
        ('smiles', pa.string()),
        ('precursor_mz', pa.float64()),
        ('num_peaks', pa.int32()),
        ('spectrum_vector', pa.list_(pa.float32()))
    ])
    
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, output_path, compression='snappy')
    
    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Preprocess] [OK] Export complete!")
    print(f"  - Output path: {output_path}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Feature dimension: {len(df['spectrum_vector'].iloc[0])} dims (List[float32])")
    print(f"  - File size: {file_size_mb:.2f} MB")
    print(f"  - Total time: {elapsed:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretraining mass spectrum data preprocessing (MSP -> Parquet)")
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "data_source" / "combined_spectra_normalized.msp"
    default_output = base_dir / "data_source" / "preprocessed_spectra.parquet"
    
    parser.add_argument("-i", "--input_msp", type=str, default=str(default_input), help="Input MSP file path")
    parser.add_argument("-o", "--output_parquet", type=str, default=str(default_output), help="Output Parquet file path")
    parser.add_argument("--min_peaks", type=int, default=5, help="Minimum number of valid peaks threshold (default: 5)")
    parser.add_argument("--mz_min", type=float, default=40.0, help="Minimum m/z value (default: 40)")
    parser.add_argument("--mz_max", type=float, default=600.0, help="Maximum m/z value (default: 600)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel worker processes (default: CPU cores - 1)")
    
    args = parser.parse_args()
    
    preprocess_msp_to_parquet(
        input_msp=args.input_msp,
        output_parquet=args.output_parquet,
        min_peaks=args.min_peaks,
        mz_min=int(args.mz_min),
        mz_max=int(args.mz_max),
        num_workers=args.num_workers
    )