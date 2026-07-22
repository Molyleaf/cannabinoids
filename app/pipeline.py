import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return F.normalize(embed, p=2, dim=1)


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
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
            current_dir = os.path.dirname(os.path.abspath(__file__))
            safetensors_path = os.path.join(
                current_dir, "models", "binary_classifier_weights_20260716_164027.safetensors"
            )
            
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


def run_pipeline(file_bytes: bytes, filename: str, min_similarity: float = 0.75) -> dict:
    """
    测样管线核心执行函数：
    1. 导入单个 msp 或 mgf 格式质谱，使用 ms_entropy.read_one_spectrum 与 ms_entropy.clean_spectrum 清洗数据
    2. 使用 app/recognizer.py 的 get_entropy_search() 获取输入的数据是否在已知库中
    3. 使用 safetensors 模型推断该质谱数据的风险概率
    """
    ext = os.path.splitext(filename)[1].lower().replace('.', '')
    if ext not in ['msp', 'mgf']:
        # 默认尝试解析为 msp
        ext = 'msp'

    # 步骤 1: 导入并清洗质谱数据
    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        tmp_file.write(file_bytes)
        tmp_file.close()
        tmp_path = tmp_file.name

        raw_spectrum = ms_entropy.read_one_spectrum(tmp_path, file_type=ext)
        if hasattr(raw_spectrum, "__iter__") and not isinstance(raw_spectrum, dict):
            spectra_list = list(raw_spectrum)
            raw_spectrum = spectra_list[0] if spectra_list else None
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if raw_spectrum is None:
        raise ValueError(f"Failed to parse mass spectrum from {filename}.")

    # 提取 precursor_mz 与 peaks
    precursor_mz = None
    peaks_raw = None

    if isinstance(raw_spectrum, dict):
        precursor_mz = raw_spectrum.get("precursor_mz")
        peaks_raw = raw_spectrum.get("peaks")
    elif hasattr(raw_spectrum, "peaks"):
        peaks_raw = raw_spectrum.peaks
        precursor_mz = getattr(raw_spectrum, "precursor_mz", None)
    else:
        peaks_raw = raw_spectrum

    if peaks_raw is not None:
        peaks_raw = list(peaks_raw)

    if peaks_raw is None or len(peaks_raw) == 0:
        raise ValueError("No peak data found in the input spectrum.")

    # 格式转换与 ms_entropy.clean_spectrum 清洗
    cleaned_peaks = []
    for p in peaks_raw:
        try:
            mz = float(p[0])
            intensity = float(p[1]) if not isinstance(p[1], str) else float(p[1].replace(';', ''))
            cleaned_peaks.append([mz, intensity])
        except Exception:
            continue

    if not cleaned_peaks:
        raise ValueError("No valid peaks could be parsed from the spectrum.")

    cleaned_peaks_arr = np.array(cleaned_peaks, dtype=np.float32)
    cleaned_peaks_arr = ms_entropy.clean_spectrum(cleaned_peaks_arr)

    query_dict = {
        "precursor_mz": precursor_mz,
        "peaks": cleaned_peaks_arr
    }

    # 步骤 2: 已知库熵检索匹配
    matched_smiles_or_bool = check_spectrum_similarity(
        query_dict,
        min_similarity=min_similarity
    )

    is_known_compound = bool(matched_smiles_or_bool)
    matched_smiles = matched_smiles_or_bool if isinstance(matched_smiles_or_bool, str) else ""

    # 步骤 3: safetensors 模型深度学习风险推断
    vec = peaks_to_vector(cleaned_peaks_arr)
    vec_norm = preprocess_spectra(vec)
    x_tensor = torch.tensor(vec_norm, dtype=torch.float32)

    model = get_classifier_model()
    with torch.no_grad():
        logit = model(x_tensor)
        risk_probability = float(torch.sigmoid(logit).item())

    risk_level = "High Risk (高风险)" if risk_probability >= 0.5 else "Low Risk (低风险)"

    return {
        "filename": filename,
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
