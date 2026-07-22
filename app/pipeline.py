import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
import ms_entropy
from app.recognizer import check_spectrum_similarity

# Global singleton for the model
_classifier_model = None

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


def get_classifier_model(safetensors_path: str = None) -> BinaryClassifier:
    """
    懒加载模式初始化并获取二进制风险分类模型。
    支持从 .safetensors 文件读取权重。
    """
    global _classifier_model
    if _classifier_model is None:
        if safetensors_path is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            candidates = sorted(list(Path(models_dir).glob("*.safetensors")), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                safetensors_path = str(candidates[0])
            else:
                safetensors_path = os.path.join(models_dir, "binary_classifier_weights_20260722_211225.safetensors")
            
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"Safetensors weight file not found at {safetensors_path}. "
                "Please run offline conversion script scratch/convert_weights_to_safetensors.py first."
            )
            
        encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
        model = BinaryClassifier(encoder=encoder, input_dim=256, freeze_encoder=True)
        
        # 从 safetensors 文件中载入权重
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        model.eval()
        _classifier_model = model

    return _classifier_model


def peaks_to_vector(peaks, mz_min=40, mz_max=600) -> np.ndarray:
    """
    将质谱峰 [mz, intensity] 映射为固定维度的 1D 特征向量 (561 维)。
    """
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
    """
    总离子强度(TIC)归一化与平方根变换。
    """
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
                parts = stripped.replace(';', ' ').replace('\t', ' ').split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        if mz > 0 and intensity > 0:
                            current_comp['peaks'].append([mz, intensity])
                    except ValueError:
                        in_peaks = False
                        
    if current_comp is not None and len(current_comp['peaks']) >= min_peaks:
        compounds.append(current_comp)
        
    return compounds


def run_pipeline(file_bytes: bytes, filename: str, min_similarity: float = 0.75) -> dict:
    """
    测样管线核心执行函数：
    1. 导入单个 msp 或 mgf 质谱，使用 parse_msp_bytes 鲁棒解析
    2. 使用 app/recognizer.py 的 check_spectrum_similarity 获取已知库匹配
    3. 使用 safetensors 模型推断该质谱数据的风险概率
    """
    compounds = parse_msp_bytes(file_bytes, min_peaks=1)
    if not compounds:
        raise ValueError(f"Failed to parse any valid mass spectrum from {filename}.")

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

    # 已知库熵检索匹配
    matched_smiles_or_bool = check_spectrum_similarity(
        query_dict,
        min_similarity=min_similarity
    )
    is_known_compound = bool(matched_smiles_or_bool)
    matched_smiles = matched_smiles_or_bool if isinstance(matched_smiles_or_bool, str) else ""

    # safetensors 模型深度学习风险推断 (使用原始离子强度的 peaks_raw 进行 1D 特征向量转换)
    vec = peaks_to_vector(peaks_raw)
    vec_norm = preprocess_spectra(vec)
    x_tensor = torch.tensor(vec_norm, dtype=torch.float32)

    model = get_classifier_model()
    dev = next(model.parameters()).device
    x_tensor = x_tensor.to(dev)

    with torch.no_grad():
        logit = model(x_tensor)
        risk_probability = float(torch.sigmoid(logit).item())

    risk_level = "High Risk (高风险)" if risk_probability >= 0.5 else "Low Risk (低风险)"

    return {
        "filename": filename,
        "name": spec_name,
        "num_cleaned_peaks": len(cleaned_peaks_arr),
        "precursor_mz": precursor_mz,
        "known_library_match": {
            "is_matched": is_known_compound,
            "matched_smiles": matched_smiles,
            "min_similarity_threshold": min_similarity
        },
        "model_inference": {
            "risk_probability": round(risk_probability, 4),
            "risk_percentage": f"{risk_probability * 100:.2f}%",
            "risk_level": risk_level,
            "is_high_risk": risk_probability >= 0.5
        },
        "peaks": cleaned_peaks_arr.tolist()
    }


def run_pipeline_batch(file_bytes: bytes, filename: str, min_similarity: float = 0.75, batch_size: int = 1024) -> dict:
    """
    测样管线批量处理函数：
    能够高效处理包含成千上万条质谱的 MSP 或 MGF 文件。
    1. 使用 parse_msp_bytes 鲁棒状态机解析全部质谱
    2. 使用 FlashEntropySearch 批量检索已知库匹配
    3. 批量使用 PyTorch safetensors 神经分类器进行风险推理
    4. 返回总体统计数据与逐条测试详情
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

        # 使用原始相对强度的 peaks_raw 进行 1D 特征向量转换
        vec = peaks_to_vector(peaks_raw)
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

    # 批量 PyTorch 风险概率推断
    if vectors:
        vecs_arr = np.array(vectors, dtype=np.float32)
        vecs_norm = preprocess_spectra(vecs_arr)
        model = get_classifier_model()
        dev = next(model.parameters()).device
        
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(vecs_norm), batch_size):
                batch_tensor = torch.tensor(vecs_norm[i:i+batch_size], dtype=torch.float32).to(dev)
                logits = model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(probs)

        for detail_item, prob in zip(results_detail, all_probs):
            detail_item["risk_probability"] = round(prob, 4)
            detail_item["is_high_risk"] = prob >= 0.5

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
