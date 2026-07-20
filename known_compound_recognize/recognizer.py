import os
import numpy as np
from ms_entropy import clean_spectrum, FlashEntropySearch

# Global singleton for the search engine
_entropy_search = None

def get_entropy_search():
    """
    Lazily initialize and load the pre-built FlashEntropySearch index.
    """
    global _entropy_search
    if _entropy_search is None:
        # mz_index_step must match the step used during build_index (0.01)
        _entropy_search = FlashEntropySearch(mz_index_step=0.01)
        
        # Load index files from the same directory as this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_dir = os.path.join(current_dir, "positive_idx")
        if not os.path.exists(index_dir):
            raise FileNotFoundError(
                f"Index directory not found at {index_dir}. "
                "Please run build_index.py to pre-build the index first."
            )
        _entropy_search.read(index_dir)
    return _entropy_search

def check_spectrum_similarity(
    query_spectrum,
    ms1_tolerance_in_da=0.2,
    ms2_tolerance_in_da=0.02,
    min_similarity=0.75
) -> bool:
    """
    检查输入质谱数据与 positive.msp 内所有数据的相似性。
    如果有高于 min_similarity (默认 0.75) 的相似度匹配结果则返回 True，没有则返回 False。

    参数:
    - query_spectrum: 可以是字典, 包含:
        - "precursor_mz" (可选): float, 前体离子 m/z
        - "peaks": np.ndarray 或 list, 形如 [[mz1, int1], [mz2, int2], ...] 的质谱峰列表
      或者是单纯的 peaks 列表/数组。
    - ms1_tolerance_in_da: MS1 质量允差 (Da)。
      如果为 None，或者 query_spectrum 中未提供 precursor_mz，则在此检索中不限制前体离子质量（对比库中所有数据，即 Open Search 模式）。
    - ms2_tolerance_in_da: MS2 质量允差 (Da)。
    - min_similarity: 匹配相似度的阈值 (默认 0.75)。

    返回:
    - bool: 如果存在高于 min_similarity 的相似度匹配结果则返回 True，否则返回 False。
    """
    # 1. 解析输入质谱
    precursor_mz = None
    peaks_raw = None
    
    if isinstance(query_spectrum, dict):
        precursor_mz = query_spectrum.get("precursor_mz")
        peaks_raw = query_spectrum.get("peaks")
    else:
        # 认为输入直接是 peaks 列表或数组
        peaks_raw = query_spectrum

    if peaks_raw is None or len(peaks_raw) == 0:
        raise ValueError("Invalid query spectrum: 'peaks' list is empty or missing.")

    # 2. 清洗输入质谱峰 (将原始峰信息转换为 float numpy 数组并做 clean)
    cleaned_peaks = []
    for p in peaks_raw:
        try:
            mz = float(p[0])
            # 兼容带有分号的强度字符串
            intensity = float(p[1]) if not isinstance(p[1], str) else float(p[1].replace(';', ''))
            cleaned_peaks.append([mz, intensity])
        except Exception:
            continue
            
    if not cleaned_peaks:
        raise ValueError("No valid peaks could be parsed from the query spectrum.")
        
    cleaned_peaks = np.array(cleaned_peaks, dtype=np.float32)
    cleaned_peaks = clean_spectrum(cleaned_peaks)

    # 3. 获取检索器
    entropy_search = get_entropy_search()

    # 4. 执行检索 (根据是否有 precursor_mz 决定检索模式)
    # 发布包和运行时强制 target="cpu"，绝对不使用 GPU
    if precursor_mz is not None and ms1_tolerance_in_da is not None:
        # Identity Search 模式 (前体 m/z 在指定范围内的条目)
        results = entropy_search.search(
            precursor_mz=float(precursor_mz),
            peaks=cleaned_peaks,
            ms1_tolerance_in_da=float(ms1_tolerance_in_da),
            ms2_tolerance_in_da=float(ms2_tolerance_in_da),
            method="identity",
            target="cpu"
        )
        scores = results.get("identity_search", [])
    else:
        # Open Search 模式 (不限制前体 m/z，对比库中全部条目)
        results = entropy_search.search(
            precursor_mz=0.0,  # 仅作为占位符，open_search 会忽略它
            peaks=cleaned_peaks,
            ms2_tolerance_in_da=float(ms2_tolerance_in_da),
            method="open",
            target="cpu"
        )
        scores = results.get("open_search", [])

    # 5. 判断是否存在相似度高于阈值的匹配并返回对应 SMILES
    if len(scores) > 0:
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]
        if max_score > min_similarity:
            # 获取相似度最高的元数据
            matched_metadata = entropy_search[int(max_idx)]
            return matched_metadata.get("smiles", "")
            
    return False
