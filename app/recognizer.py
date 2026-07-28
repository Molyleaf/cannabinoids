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
        index_dir = os.path.join(current_dir, "cache/positive_idx")
        if not os.path.exists(index_dir):
            raise FileNotFoundError(
                f"Index directory not found at {index_dir}. "
                "Please run build_index.py to pre-build the index first."
            )
        _entropy_search.read(index_dir)
    return _entropy_search

def search_entropy_detail(
    query_spectrum,
    ms1_tolerance_in_da=0.2,
    ms2_tolerance_in_da=0.02,
    min_similarity=0.90
) -> dict:
    """
    检索输入质谱与 positive.msp 已知阳性库数据的熵相似度。

    返回:
    - dict: {
        "is_matched": bool,           # max_score > min_similarity (默认 0.90)
        "similarity_score": float,     # 最高相似度得分 (0~1)
        "matched_smiles": str,        # 当 > min_similarity 时返回 SMILES，否则为空字符串
        "matched_name": str,          # 命中化合物名称
        "min_similarity_threshold": float
      }
    """
    # 1. 解析输入质谱
    precursor_mz = None
    peaks_raw = None
    
    if isinstance(query_spectrum, dict):
        precursor_mz = query_spectrum.get("precursor_mz")
        peaks_raw = query_spectrum.get("peaks")
    else:
        peaks_raw = query_spectrum

    if peaks_raw is None or len(peaks_raw) == 0:
        raise ValueError("Invalid query spectrum: 'peaks' list is empty or missing.")

    # 2. 清洗输入质谱峰
    cleaned_peaks = []
    for p in peaks_raw:
        try:
            mz = float(p[0])
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

    # 4. 执行检索
    if precursor_mz is not None and ms1_tolerance_in_da is not None:
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
        results = entropy_search.search(
            precursor_mz=0.0,
            peaks=cleaned_peaks,
            ms2_tolerance_in_da=float(ms2_tolerance_in_da),
            method="open",
            target="cpu"
        )
        scores = results.get("open_search", [])

    if len(scores) > 0:
        max_idx = int(np.argmax(scores))
        max_score = float(scores[max_idx])
        matched_metadata = entropy_search[max_idx]
        matched_smiles = matched_metadata.get("smiles", "")
        matched_name = matched_metadata.get("name", "")

        is_matched = max_score > min_similarity
        return {
            "is_matched": is_matched,
            "similarity_score": round(max_score, 4),
            "matched_smiles": matched_smiles if is_matched else "",
            "raw_matched_smiles": matched_smiles, # 即使未超过阈值也提供参考
            "matched_name": matched_name,
            "min_similarity_threshold": min_similarity
        }
            
    return {
        "is_matched": False,
        "similarity_score": 0.0,
        "matched_smiles": "",
        "raw_matched_smiles": "",
        "matched_name": "",
        "min_similarity_threshold": min_similarity
    }

def check_spectrum_similarity(
    query_spectrum,
    ms1_tolerance_in_da=0.2,
    ms2_tolerance_in_da=0.02,
    min_similarity=0.90
):
    """保持向下兼容接口"""
    res = search_entropy_detail(
        query_spectrum=query_spectrum,
        ms1_tolerance_in_da=ms1_tolerance_in_da,
        ms2_tolerance_in_da=ms2_tolerance_in_da,
        min_similarity=min_similarity
    )
    if res["is_matched"]:
        return res["matched_smiles"]
    return False

