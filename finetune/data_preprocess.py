"""
finetune/data_preprocess.py

微调质谱数据预处理脚本 (对齐预训练/推理统一标准)

功能：
1. 引用 common.data_processor 解析阴性与阳性 MSP 质谱文件；
2. 阴性样本标注为 label=0，阳性样本标注为 label=1；
3. 使用 ms-entropy (clean_spectrum) 进行质谱离群峰滤噪与去重；
4. 将清洗后的质谱峰映射为 [40, 600] 范围 1 Da 分辨率 561 维特征向量；
5. 执行总离子强度 (TIC) 归一化与平方根 (sqrt) 开方非线性变换转换；
6. 导出带 label 及全量元数据的合并 Parquet 文件 finetune/data_source/preprocessed_finetune.parquet。
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

# 动态确保项目根目录在 python 模块路径中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.data_processor import clean_spectrum, parse_msp, peaks_to_vector, preprocess_spectra


def process_single_compound(comp, label, mz_min=40, mz_max=600):
    """
    处理单个化合物：滤噪 -> 向量化 -> TIC 归一化 + sqrt 变换
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
    """多进程 chunk 处理函数"""
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
        raise FileNotFoundError(f"未找到阳性 MSP 文件: {pos_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"未找到阴性 MSP 文件: {neg_path}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"[Preprocess] 步骤 1/3: 正在解析 MSP 数据...")
    pos_compounds = parse_msp(pos_path, min_peaks=min_peaks)
    neg_compounds = parse_msp(neg_path, min_peaks=min_peaks)
    
    print(f"  - 阳性化合物数量 (Label 1): {len(pos_compounds)}")
    print(f"  - 阴性化合物数量 (Label 0): {len(neg_compounds)}")
    
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
        
    print(f"[Preprocess] 步骤 2/3: 多进程加速执行 ms-entropy 清洗与 [{mz_min}, {mz_max}] 561维映射 (Worker 数量: {num_workers})...")
    
    processed_records = []
    
    # 分别并发处理阳性和阴性样本
    for comps, label, desc in [(pos_compounds, 1, "阳性样本 (Label 1)"), (neg_compounds, 0, "阴性样本 (Label 0)")]:
        if not comps:
            continue
        chunk_size = max(50, len(comps) // (num_workers * 4))
        chunks = [comps[i:i + chunk_size] for i in range(0, len(comps), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_chunk_with_label, chunk, label, mz_min, mz_max) for chunk in chunks]
            for future in tqdm(futures, desc=f"[Processing {desc}]", unit="chunk"):
                processed_records.extend(future.result())
                
    print(f"[Preprocess] 步骤 3/3: 导出为统一标准 Parquet 格式...")
    
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
    
    print(f"[Preprocess] [OK] 导出完成!")
    print(f"  - 输出路径: {output_path}")
    print(f"  - 总样本数: {len(df)} (阳性 Label 1: {pos_count}, 阴性 Label 0: {neg_count})")
    print(f"  - 特征维度: {len(df['spectrum_vector'].iloc[0])} 维 (List[float32])")
    print(f"  - 文件体积: {file_size_mb:.2f} MB")
    print(f"  - 总耗时: {elapsed:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="微调质谱数据预处理 (MSP -> Parquet)")
    base_dir = Path(__file__).resolve().parent
    default_pos = base_dir / "data_source" / "阳性-含CanonicalSMILES-5类骨架(4).msp"
    default_neg = base_dir / "data_source" / "阴性(4).msp"
    default_output = base_dir / "data_source" / "preprocessed_finetune.parquet"
    
    parser.add_argument("--pos_msp", type=str, default=str(default_pos), help="阳性 MSP 文件路径")
    parser.add_argument("--neg_msp", type=str, default=str(default_neg), help="阴性 MSP 文件路径")
    parser.add_argument("-o", "--output_parquet", type=str, default=str(default_output), help="输出 Parquet 文件路径")
    parser.add_argument("--min_peaks", type=int, default=5, help="质谱最小有效峰数阈值 (默认: 5)")
    parser.add_argument("--mz_min", type=float, default=40.0, help="m/z 最小取值 (默认: 40)")
    parser.add_argument("--mz_max", type=float, default=600.0, help="m/z 最大取值 (默认: 600)")
    parser.add_argument("--num_workers", type=int, default=None, help="多进程并行进程数")
    
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
