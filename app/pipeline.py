# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Dynamically ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.data_processor import (
    peaks_to_vector,
    preprocess_spectra,
    clean_spectrum,
    parse_msp_bytes,
)
from app.recognizer import check_spectrum_similarity, search_entropy_detail
from app.models import (
    SpectrumEncoder,
    BinaryClassifier,
    MultiClassSpectrumEncoder,
    MultiClassClassifier,
    get_ensemble_models,
    get_classifier_model,
    predict_risk_ensemble,
    get_multi_class_model,
    predict_multi_class,
    load_single_safetensors_or_pt,
)


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    model_type: str = "binary",
    min_similarity: float = 0.80,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main analysis pipeline:
    1. Parse and clean mass spectrum data (MSP / MGF)
    2. Perform inference based on model selection (Binary Model or Multi-class Model)
    3. If determined as Positive, trigger FlashEntropySearch against positive library (threshold min_similarity)
    4. When similarity > min_similarity, return matched known molecular SMILES
    """
    compounds = parse_msp_bytes(file_bytes, min_peaks=1)
    if not compounds:
        raise ValueError(f"Failed to parse valid mass spectrum data from {filename}.")

    comp = compounds[0]
    spec_name = comp['name']
    precursor_mz = comp['precursor_mz']
    peaks_raw = comp['peaks']

    cleaned_peaks_arr = clean_spectrum(peaks_raw)

    query_dict = {
        "precursor_mz": precursor_mz,
        "peaks": cleaned_peaks_arr
    }

    # Vectorization & Preprocessing
    vec = peaks_to_vector(cleaned_peaks_arr)
    vec_norm = preprocess_spectra(vec)

    is_positive = False
    is_ood = False
    model_inference_data: Dict[str, Any] = {}

    if model_type == "multi":
        multi_res = predict_multi_class(vec_norm, model_path=model_path)
        possibility = float(multi_res["confidence"])
        if possibility < 0.98:
            is_ood = True
            is_positive = False
            model_inference_data = {
                "model_type": "multi",
                "model_name": "Multi-class Model (nine NPS categories)",
                "pred_class": "OOD",
                "confidence": possibility,
                "confidence_percentage": multi_res["confidence_percentage"],
                "probabilities": multi_res["probabilities"],
                "is_positive": False,
                "is_ood": True,
                "status_text": "OOD"
            }
        else:
            is_ood = False
            is_positive = True
            model_inference_data = {
                "model_type": "multi",
                "model_name": "Multi-class Model (nine NPS categories)",
                "pred_class": multi_res["pred_class"],
                "confidence": possibility,
                "confidence_percentage": multi_res["confidence_percentage"],
                "probabilities": multi_res["probabilities"],
                "is_positive": True,
                "is_ood": False,
                "status_text": f"Positive (Category: {multi_res['pred_class']})"
            }
    else:
        is_ood = False
        models = get_ensemble_models(model_path=model_path)
        probs = predict_risk_ensemble(vec_norm, models)
        risk_probability = float(probs[0])
        is_positive = risk_probability >= 0.50

        model_inference_data = {
            "model_type": "binary",
            "model_name": "Binary Model (SC versus NSC)",
            "risk_probability": round(risk_probability, 4),
            "risk_percentage": f"{risk_probability * 100:.2f}%",
            "risk_level": "High Risk" if is_positive else "Low Risk",
            "is_positive": is_positive,
            "is_ood": False,
            "status_text": "Positive (High Risk)" if is_positive else "Negative (Low Risk)"
        }

    # If evaluated as positive and not OOD, trigger entropy search
    entropy_match_result = {
        "is_triggered": is_positive and not is_ood,
        "is_matched": False,
        "is_ood": is_ood,
        "similarity_score": 0.0,
        "matched_smiles": "",
        "raw_matched_smiles": "",
        "matched_name": "",
        "min_similarity_threshold": min_similarity,
        "message": (
            "OOD sample, library search not triggered."
            if is_ood
            else ("Negative sample, library search not triggered." if not is_positive else "")
        )
    }

    if is_positive and not is_ood:
        search_res = search_entropy_detail(
            query_dict,
            min_similarity=min_similarity
        )
        entropy_match_result.update({
            "is_matched": search_res["is_matched"],
            "similarity_score": search_res["similarity_score"],
            "matched_smiles": search_res["matched_smiles"],
            "raw_matched_smiles": search_res["raw_matched_smiles"],
            "matched_name": search_res["matched_name"],
            "message": (
                f"Known molecule matched with similarity {search_res['similarity_score']:.4f} > {min_similarity:.2f}."
                if search_res["is_matched"]
                else f"Max similarity is {search_res['similarity_score']:.4f} (<= {min_similarity:.2f}), below threshold."
            )
        })

    return {
        "filename": filename,
        "name": spec_name,
        "num_cleaned_peaks": len(cleaned_peaks_arr),
        "precursor_mz": precursor_mz,
        "model_type_selected": model_type,
        "is_positive": is_positive,
        "is_ood": is_ood,
        "model_inference": model_inference_data,
        "entropy_match": entropy_match_result,
        "peaks": cleaned_peaks_arr.tolist()
    }


def run_pipeline_batch(
    file_bytes: bytes,
    filename: str,
    min_similarity: float = 0.80,
    batch_size: int = 1024,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Batch processing pipeline:
    Performs batch inference with 5-Fold soft voting ensemble and library similarity matching.
    """
    compounds = parse_msp_bytes(file_bytes, min_peaks=1)
    if not compounds:
        raise ValueError(f"Failed to parse any valid mass spectra from {filename}.")

    total_count = len(compounds)
    results_detail = []
    vectors = []

    for idx, comp in enumerate(compounds):
        spec_name = comp['name']
        precursor_mz = comp['precursor_mz']
        peaks_raw = comp['peaks']

        cleaned_peaks_arr = clean_spectrum(peaks_raw)

        query_dict = {
            "precursor_mz": precursor_mz,
            "peaks": cleaned_peaks_arr
        }

        # Known library match
        matched_smiles_or_bool = check_spectrum_similarity(
            query_dict,
            min_similarity=min_similarity
        )
        is_known_compound = bool(matched_smiles_or_bool)
        matched_smiles = matched_smiles_or_bool if isinstance(matched_smiles_or_bool, str) else ""

        vec = peaks_to_vector(cleaned_peaks_arr)
        vectors.append(vec)

        results_detail.append({
            "index": idx,
            "name": spec_name,
            "num_cleaned_peaks": len(cleaned_peaks_arr),
            "precursor_mz": precursor_mz,
            "is_matched": is_known_compound,
            "matched_smiles": matched_smiles,
            "risk_probability": 0.0,
            "is_high_risk": False
        })

    # Batch 5-Fold Soft-Voting inference
    if vectors:
        vecs_arr = np.array(vectors, dtype=np.float32)
        vecs_norm = preprocess_spectra(vecs_arr)

        models = get_ensemble_models(model_path=model_path)
        all_probs = predict_risk_ensemble(vecs_norm, models, batch_size=batch_size)

        for detail_item, prob in zip(results_detail, all_probs):
            prob_val = float(prob)
            detail_item["risk_probability"] = round(prob_val, 4)
            detail_item["is_high_risk"] = prob_val >= 0.5

    # Statistical summary
    matched_count = sum(1 for item in results_detail if item["is_matched"])
    high_risk_count = sum(1 for item in results_detail if item["is_high_risk"])
    low_risk_count = len(results_detail) - high_risk_count

    probs_arr = np.array([item["risk_probability"] for item in results_detail]) if results_detail else np.array([0.0])

    stats = {
        "filename": filename,
        "total_parsed_spectra": len(results_detail),
        "total_file_spectra": total_count,
        "known_library_matches": {
            "matched_count": matched_count,
            "unmatched_count": len(results_detail) - matched_count,
            "match_rate_percentage": f"{(matched_count / len(results_detail) * 100):.2f}%" if results_detail else "0.00%",
            "min_similarity_threshold": min_similarity
        },
        "model_risk_inference": {
            "high_risk_count": high_risk_count,
            "low_risk_count": low_risk_count,
            "high_risk_percentage": f"{(high_risk_count / len(results_detail) * 100):.2f}%" if results_detail else "0.00%",
            "low_risk_percentage": f"{(low_risk_count / len(results_detail) * 100):.2f}%" if results_detail else "0.00%",
            "mean_risk_probability": round(float(np.mean(probs_arr)), 4),
            "median_risk_probability": round(float(np.median(probs_arr)), 4),
            "std_risk_probability": round(float(np.std(probs_arr)), 4),
            "percentiles": {
                "p25": round(float(np.percentile(probs_arr, 25)), 4),
                "p50": round(float(np.percentile(probs_arr, 50)), 4),
                "p75": round(float(np.percentile(probs_arr, 75)), 4),
                "p90": round(float(np.percentile(probs_arr, 90)), 4),
                "p95": round(float(np.percentile(probs_arr, 95)), 4)
            }
        },
        "details": results_detail
    }

    return stats
