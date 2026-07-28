import os
import sys
from pathlib import Path

import ms_entropy
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

# 动态确保项目根目录在 python 模块路径中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.data_processor import peaks_to_vector, preprocess_spectra, clean_spectrum
from app.recognizer import check_spectrum_similarity, search_entropy_detail

# 单例缓存
_ensemble_models = None
_classifier_model = None
_multi_class_model = None
_multi_class_names = [
    '芬太尼', '卡西酮', '大麻素', 'Arylcyclohexylamines',
    'Benzodiazepines', 'Nitazenes', 'Opiates', 'Phenethylamines', 'Tryptamines'
]

class SpectrumEncoder(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        norm = torch.norm(embed, p=2, dim=1, keepdim=True)
        return embed / torch.clamp(norm, min=1e-3)


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.encoder.parameters())):
            embed = self.encoder(x)
        logit = self.classifier(embed)
        return logit.squeeze(-1)


# 多分类模型网络结构定义
class MultiClassSpectrumEncoder(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        return torch.nn.functional.normalize(embed, p=2, dim=1)


class MultiClassClassifier(nn.Module):
    def __init__(self, encoder, num_classes=9, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embed = self.encoder(x)
        logits = self.classifier(embed)
        return logits


def get_multi_class_model(model_path: str = None):
    """加载多分类模型 PyTorch 权重"""
    global _multi_class_model, _multi_class_names

    if _multi_class_model is not None and model_path is None:
        return _multi_class_model, _multi_class_names

    if model_path is None:
        multi_dir = project_root / "multi_classifier"
        candidates = sorted(list(multi_dir.glob("best_multi_class_model_*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            candidates = sorted(list(multi_dir.glob("*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"在 {multi_dir} 下未找到多分类模型权重 .pt 文件")
        model_path = str(candidates[0])

    print(f"[Pipeline] 正在加载多分类模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    class_names = _multi_class_names
    if isinstance(checkpoint, dict) and 'config' in checkpoint and 'class_names' in checkpoint['config']:
        class_names = checkpoint['config']['class_names']

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


def predict_multi_class(vec_norm: np.ndarray, model_path: str = None) -> dict:
    """多分类模型概率推断"""
    model, class_names = get_multi_class_model(model_path=model_path)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)

    if vec_norm.ndim == 1:
        vec_norm = vec_norm[np.newaxis, :]

    with torch.no_grad():
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
        "is_positive": True  # 属于阳性毒品多分类模型
    }


def load_single_safetensors_or_pt(target_path: str) -> BinaryClassifier:
    """从单个 .safetensors 或 .pt 加载权重构建 BinaryClassifier 模型"""
    target_path = str(target_path)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"未找到模型权重文件: {target_path}")

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


def get_ensemble_models(models_dir: str = None, model_path: str = None) -> list:
    """
    懒加载获取 5-Fold .safetensors 正式集成模型列表。
    如果指定了单个 model_path 则加载单模型并包装为列表返回。
    """
    global _ensemble_models

    if model_path is not None:
        return [load_single_safetensors_or_pt(model_path)]

    if _ensemble_models is not None:
        return _ensemble_models

    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    # 优先查找 official_ensemble_fold_*.safetensors
    st_files = sorted(list(Path(models_dir).glob("official_ensemble_fold_*.safetensors")))

    if len(st_files) == 5:
        print(f"[Pipeline] 成功检测到 5-Fold 正式 .safetensors 模型，正在加载 Soft Voting 集成模型组...")
        models = []
        for st_p in st_files:
            m = load_single_safetensors_or_pt(str(st_p))
            models.append(m)
        _ensemble_models = models
        return _ensemble_models

    # 回退机制：查找任意候选 .safetensors 或 .pt
    candidates = sorted(list(Path(models_dir).glob("*.safetensors")), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        single_path = str(candidates[0])
    else:
        candidates_pt = sorted(list(Path(models_dir).glob("*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates_pt:
            single_path = str(candidates_pt[0])
        else:
            raise FileNotFoundError(f"在目录 {models_dir} 下未检测到有效的 .safetensors 或 .pt 模型权重。")

    print(f"[Pipeline] 未完全找到 5-Fold 集成文件，回退加载单模型: {single_path}")
    _ensemble_models = [load_single_safetensors_or_pt(single_path)]
    return _ensemble_models


def get_classifier_model(safetensors_path: str = None, model_path: str = None) -> BinaryClassifier:
    """兼容旧接口：返回集成模型中的第一个模型"""
    models = get_ensemble_models(model_path=model_path or safetensors_path)
    return models[0]


def predict_risk_ensemble(vecs_norm: np.ndarray, models: list, batch_size: int = 1024) -> np.ndarray:
    """
    使用 5-Fold .safetensors 集成模型进行 Soft-Voting 批量预测
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for m in models:
        m.eval()
        m.to(dev)

    if vecs_norm.ndim == 1:
        vecs_norm = vecs_norm[np.newaxis, :]

    all_fold_probs = []

    with torch.no_grad():
        for m in models:
            m_probs = []
            for i in range(0, len(vecs_norm), batch_size):
                batch_tensor = torch.tensor(vecs_norm[i:i+batch_size], dtype=torch.float32).to(dev)
                logits = m(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 0:
                    probs = np.array([probs])
                m_probs.extend(probs)
            all_fold_probs.append(m_probs)

    # Soft Voting: 计算 5 个模型预测概率的算术平均值
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    return ensemble_probs


def peaks_to_vector(peaks, mz_min=40, mz_max=600) -> np.ndarray:
    """将质谱峰 [mz, intensity] 映射为固定维度的 1D 特征向量 (561 维)"""
    dim = mz_max - mz_min + 1
    vec = np.zeros(dim, dtype=np.float32)
    for p in peaks:
        try:
            mz = float(p[0])
            intensity = float(p[1]) if not isinstance(p[1], str) else float(p[1].replace(';', ''))
            if mz_min <= mz <= mz_max:
                idx = int(round(mz - mz_min))
                if 0 <= idx < dim:
                    vec[idx] += intensity
        except (ValueError, TypeError):
            continue
    return vec


def preprocess_spectra(vecs: np.ndarray) -> np.ndarray:
    """总离子强度(TIC)归一化与平方根变换"""
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]
    tic = vecs.sum(axis=1, keepdims=True)
    vecs = vecs / (tic + 1e-8)
    vecs = np.sqrt(vecs)
    return vecs


def parse_msp_bytes(file_bytes: bytes, min_peaks: int = 1) -> list:
    """
    状态机模式鲁棒解析 MSP 文本数据。
    严格防范 Comments 或 Metadata 行泄漏至质谱峰矩阵中。
    """
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        text = str(file_bytes)

    lines = text.splitlines()
    compounds = []
    current_comp = None
    in_peaks = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith('name:'):
            if current_comp is not None and len(current_comp['peaks']) >= min_peaks:
                compounds.append(current_comp)
            current_comp = {
                'name': stripped.split(':', 1)[1].strip() if ':' in stripped else stripped,
                'smiles': '',
                'precursor_mz': None,
                'peaks': []
            }
            in_peaks = False
        elif current_comp is not None:
            if lower.startswith('smiles:'):
                current_comp['smiles'] = stripped.split(':', 1)[1].strip()
            elif lower.startswith('precursormz:') or lower.startswith('precursor_mz:'):
                try:
                    current_comp['precursor_mz'] = float(stripped.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif lower.startswith('num peaks:') or lower.startswith('num_peaks:'):
                in_peaks = True
            elif in_peaks:
                sub_items = stripped.split(';')
                for sub in sub_items:
                    sub = sub.strip()
                    if not sub:
                        continue
                    parts = sub.replace('\t', ' ').split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            if mz > 0 and intensity > 0:
                                current_comp['peaks'].append([mz, intensity])
                        except ValueError:
                            pass

    if current_comp is not None and len(current_comp['peaks']) >= min_peaks:
        compounds.append(current_comp)

    return compounds


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    model_type: str = "binary",
    min_similarity: float = 0.90,
    model_path: str = None
) -> dict:
    """
    测样管线核心执行函数：
    1. 导入质谱 (MSP / MGF) 格式数据并清洗
    2. 根据选择模型类型进行推理 (二分类 finetune 或 多分类 multi_classifier)
    3. 若模型判定为阳性 (Positive)，自动触发 @entropy 库相似度检索 (阈值 0.90)
    4. 相似度 > 0.90 时返回已知分子的 SMILES 结构式
    """
    compounds = parse_msp_bytes(file_bytes, min_peaks=1)
    if not compounds:
        raise ValueError(f"无法从 {filename} 解析出有效的质谱数据。")

    comp = compounds[0]
    spec_name = comp['name']
    precursor_mz = comp['precursor_mz']
    peaks_raw = comp['peaks']

    cleaned_peaks_arr = np.array(peaks_raw, dtype=np.float32)
    cleaned_peaks_arr = ms_entropy.clean_spectrum(cleaned_peaks_arr)

    query_dict = {
        "precursor_mz": precursor_mz,
        "peaks": cleaned_peaks_arr
    }

    # 特征向量化
    vec = peaks_to_vector(cleaned_peaks_arr)
    vec_norm = preprocess_spectra(vec)

    is_positive = False
    model_inference_data = {}

    if model_type == "multi":
        multi_res = predict_multi_class(vec_norm, model_path=model_path)
        is_positive = True
        model_inference_data = {
            "model_type": "multi",
            "model_name": "Multi-class Model (@multi_classifier)",
            "pred_class": multi_res["pred_class"],
            "confidence": multi_res["confidence"],
            "confidence_percentage": multi_res["confidence_percentage"],
            "probabilities": multi_res["probabilities"],
            "is_positive": True,
            "status_text": f"Positive 🎯 (Category: {multi_res['pred_class']})"
        }
    else:
        models = get_ensemble_models(model_path=model_path)
        probs = predict_risk_ensemble(vec_norm, models)
        risk_probability = float(probs[0])
        is_positive = risk_probability >= 0.50

        model_inference_data = {
            "model_type": "binary",
            "model_name": "Binary Model (@finetune 5-Fold Ensemble)",
            "risk_probability": round(risk_probability, 4),
            "risk_percentage": f"{risk_probability * 100:.2f}%",
            "risk_level": "High Risk" if is_positive else "Low Risk",
            "is_positive": is_positive,
            "status_text": "Positive 🎯 (High Risk)" if is_positive else "Negative 🛡️ (Low Risk)"
        }

    # 如果模型表征为阳性，进入 @entropy 信息熵检索
    entropy_match_result = {
        "is_triggered": is_positive,
        "is_matched": False,
        "similarity_score": 0.0,
        "matched_smiles": "",
        "raw_matched_smiles": "",
        "matched_name": "",
        "min_similarity_threshold": min_similarity,
        "message": "阴性样本未触发 @entropy 熵检索" if not is_positive else ""
    }

    if is_positive:
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
            "message": f"已知分子相似度 {search_res['similarity_score']:.4f} > {min_similarity:.2f}，成功命中并返回 SMILES 结构式！" if search_res["is_matched"] else f"与已知库最高相似度为 {search_res['similarity_score']:.4f} (≤ {min_similarity:.2f})，未命中已知分子 SMILES"
        })

    return {
        "filename": filename,
        "name": spec_name,
        "num_cleaned_peaks": len(cleaned_peaks_arr),
        "precursor_mz": precursor_mz,
        "model_type_selected": model_type,
        "is_positive": is_positive,
        "model_inference": model_inference_data,
        "entropy_match": entropy_match_result,
        "known_library_match": {
            "is_matched": entropy_match_result["is_matched"],
            "matched_smiles": entropy_match_result["matched_smiles"],
            "min_similarity_threshold": min_similarity
        },
        "peaks": cleaned_peaks_arr.tolist()
    }



def run_pipeline_batch(file_bytes: bytes, filename: str, min_similarity: float = 0.75, batch_size: int = 1024, model_path: str = None) -> dict:
    """
    测样管线批量处理函数：
    高效处理成千上万条质谱，基于 5-Fold .safetensors 集成模型进行 Soft-Voting 风险概率推断
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

        cleaned_peaks_arr = np.array(peaks_raw, dtype=np.float32)
        cleaned_peaks_arr = ms_entropy.clean_spectrum(cleaned_peaks_arr)

        query_dict = {
            "precursor_mz": precursor_mz,
            "peaks": cleaned_peaks_arr
        }

        # 已知库匹配
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

    # 批量 5-Fold .safetensors Soft-Voting 推断
    if vectors:
        vecs_arr = np.array(vectors, dtype=np.float32)
        vecs_norm = preprocess_spectra(vecs_arr)

        models = get_ensemble_models(model_path=model_path)
        all_probs = predict_risk_ensemble(vecs_norm, models, batch_size=batch_size)

        for detail_item, prob in zip(results_detail, all_probs):
            prob_val = float(prob)
            detail_item["risk_probability"] = round(prob_val, 4)
            detail_item["is_high_risk"] = prob_val >= 0.5

    # 统计汇总分析
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
