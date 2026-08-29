# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Union

import numpy as np
from ms_entropy import clean_spectrum, FlashEntropySearch

# Global singleton for the search engine
_entropy_search = None


def get_entropy_search() -> FlashEntropySearch:
    """
    Lazily initialize and load the pre-built FlashEntropySearch index.
    """
    global _entropy_search
    if _entropy_search is None:
        # mz_index_step must match the step used during build_index (0.01)
        _entropy_search = FlashEntropySearch(mz_index_step=0.01)

        # Load index files from the cache directory
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
    query_spectrum: Union[Dict[str, Any], list, np.ndarray],
    ms1_tolerance_in_da: float = 0.2,
    ms2_tolerance_in_da: float = 0.02,
    min_similarity: float = 0.80
) -> Dict[str, Any]:
    """
    Search entropy similarity between query mass spectrum and positive library.

    Returns:
    - dict: {
        "is_matched": bool,           # max_score > min_similarity (default 0.80)
        "similarity_score": float,     # highest similarity score (0~1)
        "matched_smiles": str,        # returns SMILES if > min_similarity, else empty string
        "matched_name": str,          # matched compound name
        "min_similarity_threshold": float
      }
    """
    # 1. Parse query spectrum
    precursor_mz = None
    peaks_raw = None

    if isinstance(query_spectrum, dict):
        precursor_mz = query_spectrum.get("precursor_mz")
        peaks_raw = query_spectrum.get("peaks")
    else:
        peaks_raw = query_spectrum

    if peaks_raw is None or len(peaks_raw) == 0:
        raise ValueError("Invalid query spectrum: 'peaks' list is empty or missing.")

    # 2. Extract and clean query spectrum peaks
    if isinstance(peaks_raw, np.ndarray) and peaks_raw.ndim == 2 and peaks_raw.shape[1] == 2:
        cleaned_peaks = clean_spectrum(np.ascontiguousarray(peaks_raw, dtype=np.float32))
    else:
        cleaned_peaks_list = []
        for p in peaks_raw:
            try:
                mz = float(p[0])
                intensity = float(p[1]) if not isinstance(p[1], str) else float(p[1].replace(';', ''))
                if mz > 0 and intensity > 0:
                    cleaned_peaks_list.append([mz, intensity])
            except Exception:
                continue

        if not cleaned_peaks_list:
            raise ValueError("No valid peaks could be parsed from the query spectrum.")

        cleaned_peaks_arr = np.array(cleaned_peaks_list, dtype=np.float32)
        cleaned_peaks = clean_spectrum(cleaned_peaks_arr)

    if len(cleaned_peaks) == 0:
        raise ValueError("No valid peaks remaining after spectrum cleaning.")

    # 3. Get entropy search instance
    entropy_search = get_entropy_search()

    # 4. Perform search
    has_valid_precursor = (
        precursor_mz is not None
        and isinstance(precursor_mz, (int, float, str))
        and float(precursor_mz) > 0
        and ms1_tolerance_in_da is not None
    )

    if has_valid_precursor:
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
        if max_score > 0:
            matched_metadata = entropy_search[max_idx]
            matched_smiles = matched_metadata.get("smiles", "")
            matched_name = matched_metadata.get("name", "")

            is_matched = max_score > min_similarity
            return {
                "is_matched": is_matched,
                "similarity_score": round(max_score, 4),
                "matched_smiles": matched_smiles if is_matched else "",
                "raw_matched_smiles": matched_smiles,
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
    query_spectrum: Union[Dict[str, Any], list, np.ndarray],
    ms1_tolerance_in_da: float = 0.2,
    ms2_tolerance_in_da: float = 0.02,
    min_similarity: float = 0.80
) -> Union[str, bool]:
    """Backward-compatible similarity check interface."""
    res = search_entropy_detail(
        query_spectrum=query_spectrum,
        ms1_tolerance_in_da=ms1_tolerance_in_da,
        ms2_tolerance_in_da=ms2_tolerance_in_da,
        min_similarity=min_similarity
    )
    if res["is_matched"]:
        return res["matched_smiles"]
    return False
