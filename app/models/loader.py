# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from safetensors.torch import load_file

from app.models.networks import (
    SpectrumEncoder,
    BinaryClassifier,
    MultiClassSpectrumEncoder,
    MultiClassClassifier,
)

# Singleton caches
_ensemble_models: Optional[List[BinaryClassifier]] = None
_multi_class_model: Optional[MultiClassClassifier] = None
_multi_class_names: List[str] = [
    'Fentanyls', 'Cathinones', 'Synthetic Cannabinoids', 'Arylcyclohexylamines',
    'Benzodiazepines', 'Nitazenes', 'Opiates', 'Phenethylamines', 'Tryptamines'
]

CLASS_NAME_NORMALIZATION: Dict[str, str] = {
    '芬太尼': 'Fentanyls',
    'Fentanils': 'Fentanyls',
    'Fentanyls': 'Fentanyls',
    '卡西酮': 'Cathinones',
    'Synthetic Cathinones': 'Cathinones',
    'Cathinones': 'Cathinones',
    '大麻素': 'Synthetic Cannabinoids',
    'Synthetic Cannabinoids': 'Synthetic Cannabinoids',
    'Cannabinoids': 'Synthetic Cannabinoids',
    'Arylcyclohexylamines': 'Arylcyclohexylamines',
    'Benzodiazepines': 'Benzodiazepines',
    'Nitazenes': 'Nitazenes',
    'Opiates': 'Opiates',
    'Phenethylamines': 'Phenethylamines',
    'Tryptamines': 'Tryptamines',
}


def get_multi_class_model(models_dir: Optional[str] = None, model_path: Optional[str] = None) -> Tuple[MultiClassClassifier, List[str]]:
    """Load multi-class model weights from models directory or specific path."""
    global _multi_class_model, _multi_class_names

    if _multi_class_model is not None and model_path is None:
        return _multi_class_model, _multi_class_names

    if model_path is None:
        if models_dir is None:
            models_dir_path = Path(__file__).resolve().parent
        else:
            models_dir_path = Path(models_dir)

        patterns = [
            "best_multi_class_model_*.pt",
            "best_multi_class_model_*.safetensors",
            "*multi_class*.pt",
            "*multi_class*.safetensors",
        ]
        candidates = []
        for pat in patterns:
            matched = sorted(list(models_dir_path.glob(pat)), key=lambda p: p.stat().st_mtime, reverse=True)
            if matched:
                candidates = matched
                break

        if not candidates:
            raise FileNotFoundError(f"Multi-class model weights not found in directory: {models_dir_path}")
        model_path = str(candidates[0])

    print(f"[ModelLoader] Loading multi-class model weights: {model_path}")
    if model_path.endswith(".safetensors"):
        checkpoint = load_file(model_path)
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    class_names = _multi_class_names
    if isinstance(checkpoint, dict) and 'config' in checkpoint and isinstance(checkpoint['config'], dict) and 'class_names' in checkpoint['config']:
        raw_names = checkpoint['config']['class_names']
        class_names = [CLASS_NAME_NORMALIZATION.get(c, c) for c in raw_names]

    num_classes = len(class_names)
    encoder = MultiClassSpectrumEncoder(input_dim=561, hidden_dim=256)
    model = MultiClassClassifier(encoder=encoder, num_classes=num_classes, freeze_encoder=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        m_state = checkpoint['model_state_dict']
        if 'encoder_state_dict' in m_state and 'classifier_state_dict' in m_state:
            model.encoder.load_state_dict(m_state['encoder_state_dict'])
            model.classifier.load_state_dict(m_state['classifier_state_dict'])
        else:
            model.load_state_dict(m_state)
    elif isinstance(checkpoint, dict) and 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    _multi_class_model = model
    _multi_class_names = class_names
    return model, class_names


def predict_multi_class(vec_norm: np.ndarray, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Inference for multi-class model."""
    model, class_names = get_multi_class_model(model_path=model_path)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)

    if vec_norm.ndim == 1:
        vec_norm = vec_norm[np.newaxis, :]

    with torch.inference_mode():
        batch_tensor = torch.tensor(vec_norm, dtype=torch.float32).to(dev)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_prob = float(probs[top_idx])

    prob_dict = {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}

    return {
        "pred_class": top_class,
        "confidence": round(top_prob, 4),
        "confidence_percentage": f"{top_prob * 100:.2f}%",
        "probabilities": prob_dict,
        "is_positive": True
    }


def load_single_safetensors_or_pt(target_path: str) -> BinaryClassifier:
    """Load weights from single .safetensors or .pt file into BinaryClassifier."""
    target_path = str(target_path)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Model weights file not found: {target_path}")

    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
    model = BinaryClassifier(encoder=encoder, input_dim=256, freeze_encoder=False)

    if target_path.endswith(".pt"):
        ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt and "classifier_state_dict" in ckpt:
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            model.classifier.load_state_dict(ckpt["classifier_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model = ckpt
    else:
        state_dict = load_file(target_path)
        model.load_state_dict(state_dict)

    model.eval()
    return model


def get_ensemble_models(models_dir: Optional[str] = None, model_path: Optional[str] = None) -> List[BinaryClassifier]:
    """
    Lazily load 5-Fold .safetensors ensemble models.
    If a specific model_path is provided, loads and returns a single model wrapped in a list.
    """
    global _ensemble_models

    if model_path is not None:
        return [load_single_safetensors_or_pt(model_path)]

    if _ensemble_models is not None:
        return _ensemble_models

    if models_dir is None:
        models_dir_path = Path(__file__).resolve().parent
    else:
        models_dir_path = Path(models_dir)

    # Priority: search official_ensemble_fold_*.safetensors
    st_files = sorted(list(models_dir_path.glob("official_ensemble_fold_*.safetensors")))

    if len(st_files) == 5:
        print(f"[ModelLoader] Detected 5-Fold official .safetensors models, loading soft-voting ensemble...")
        models = []
        for st_p in st_files:
            m = load_single_safetensors_or_pt(str(st_p))
            models.append(m)
        _ensemble_models = models
        return _ensemble_models

    # Fallback mechanism: look for any .safetensors or .pt
    candidates = sorted(list(models_dir_path.glob("*.safetensors")), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        single_path = str(candidates[0])
    else:
        candidates_pt = sorted(list(models_dir_path.glob("*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates_pt:
            single_path = str(candidates_pt[0])
        else:
            raise FileNotFoundError(f"No valid .safetensors or .pt model weights found in directory: {models_dir_path}")

    print(f"[ModelLoader] 5-Fold ensemble not completely found, falling back to single model: {single_path}")
    _ensemble_models = [load_single_safetensors_or_pt(single_path)]
    return _ensemble_models


def get_classifier_model(safetensors_path: Optional[str] = None, model_path: Optional[str] = None) -> BinaryClassifier:
    """Legacy interface: returns first model in ensemble."""
    models = get_ensemble_models(model_path=model_path or safetensors_path)
    return models[0]


def predict_risk_ensemble(vecs_norm: np.ndarray, models: List[BinaryClassifier], batch_size: int = 1024) -> np.ndarray:
    """
    Perform vectorized Soft-Voting batch inference using 5-Fold ensemble models.
    """
    if len(models) == 0:
        return np.array([], dtype=np.float32)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for m in models:
        m.eval()
        m.to(dev)

    if vecs_norm.ndim == 1:
        vecs_norm = vecs_norm[np.newaxis, :]

    all_probs = []

    with torch.inference_mode():
        for i in range(0, len(vecs_norm), batch_size):
            batch_tensor = torch.tensor(vecs_norm[i:i+batch_size], dtype=torch.float32, device=dev)
            # Vectorized multi-model forward pass
            batch_logits = torch.stack([m(batch_tensor) for m in models], dim=0)  # Shape: (num_models, B)
            batch_probs = torch.sigmoid(batch_logits).mean(dim=0).cpu().numpy()
            if batch_probs.ndim == 0:
                batch_probs = np.array([batch_probs])
            all_probs.append(batch_probs)

    return np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
