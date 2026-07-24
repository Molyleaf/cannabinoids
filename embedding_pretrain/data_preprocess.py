"""
embedding_pretrain/data_preprocess.py

质谱数据预处理模块 (预训练 - 后训练 - 推理 统一标准)

功能：
1. 引用 common.data_processor 鲁棒解析 MSP 质谱数据文件；
2. 使用 ms-entropy (clean_spectrum) 进行质谱离群峰滤噪与去重；
3. 将清洗后的质谱峰映射为 [40, 600] 范围 1 Da 分辨率 561 维特征向量；
4. 执行总离子强度 (TIC) 归一化与平方根 (sqrt) 开方非线性变换转换；
5. 支持多进程并行加速处理；
6. 导出带全量元数据 (name, smiles, precursor_mz, num_peaks, spectrum_vector) 的 Parquet 文件。
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


def process_single_compound(comp, mz_min=40, mz_max=600):
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
        'spectrum_vector': vec_norm.tolist()
    }


def _process_chunk(chunk_compounds, mz_min=40, mz_max=600):
    """多进程 chunk 处理函数"""
    results = []
    for comp in chunk_compounds:
        results.append(process_single_compound(comp, mz_min=mz_min, mz_max=mz_max))
    return results


def preprocess_msp_to_parquet(input_msp, output_parquet, min_peaks=5, mz_min=40, mz_max=600, num_workers=None):
    """
    解析 MSP 文件，使用 ms-entropy 预处理并存储为 Parquet。
    """
    input_path = Path(input_msp)
    output_path = Path(output_parquet)
    
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入的 MSP 文件: {input_path}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[Preprocess] 步骤 1/3: 正在从 {input_path.name} 解析 MSP 节点数据...")
    start_time = time.time()
    compounds = parse_msp(input_path, min_peaks=min_peaks)
    print(f"[Preprocess] 解析完成，获取 {len(compounds)} 个满足峰数 >= {min_peaks} 的化合物质谱 (耗时: {time.time() - start_time:.2f}s)")
    
    if not compounds:
        print("[Warning] 未找到有效化合物谱图！")
        return
        
    print(f"[Preprocess] 步骤 2/3: 正在执行 ms-entropy 清洗滤噪、[{mz_min}, {mz_max}] 561维映射与 TIC+sqrt 归一化...")
    
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
        
    print(f"[Preprocess] 启动多进程加速 (Worker 数量: {num_workers})...")
    
    # 拆分 chunk 批量并行
    chunk_size = max(100, len(compounds) // (num_workers * 4))
    chunks = [compounds[i:i + chunk_size] for i in range(0, len(compounds), chunk_size)]
    
    processed_records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk, mz_min, mz_max) for chunk in chunks]
        for future in tqdm(futures, desc="[Processing Chunks]", unit="chunk"):
            processed_records.extend(future.result())
            
    print(f"[Preprocess] 步骤 3/3: 导出为统一标准 Parquet 数据文件...")
    
    # 构建 DataFrame 与 PyArrow Schema
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
    print(f"[Preprocess] [OK] 导出完成!")
    print(f"  - 输出路径: {output_path}")
    print(f"  - 样本总数: {len(df)}")
    print(f"  - 特征维度: {len(df['spectrum_vector'].iloc[0])} 维 (List[float32])")
    print(f"  - 文件体积: {file_size_mb:.2f} MB")
    print(f"  - 总耗时: {elapsed:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="预训练质谱数据预处理 (MSP -> Parquet)")
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "data_source" / "combined_spectra_normalized.msp"
    default_output = base_dir / "data_source" / "preprocessed_spectra.parquet"
    
    parser.add_argument("-i", "--input_msp", type=str, default=str(default_input), help="输入 MSP 文件路径")
    parser.add_argument("-o", "--output_parquet", type=str, default=str(default_output), help="输出 Parquet 文件路径")
    parser.add_argument("--min_peaks", type=int, default=5, help="质谱最小有效峰数阈值 (默认: 5)")
    parser.add_argument("--mz_min", type=float, default=40.0, help="m/z 最小取值 (默认: 40)")
    parser.add_argument("--mz_max", type=float, default=600.0, help="m/z 最大取值 (默认: 600)")
    parser.add_argument("--num_workers", type=int, default=None, help="多进程并行进程数 (默认: CPU核心数-1)")
    
    args = parser.parse_args()
    
    preprocess_msp_to_parquet(
        input_msp=args.input_msp,
        output_parquet=args.output_parquet,
        min_peaks=args.min_peaks,
        mz_min=int(args.mz_min),
        mz_max=int(args.mz_max),
        num_workers=args.num_workers
    )
