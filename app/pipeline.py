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
            current_dir = os.path.dirname(os.path.abspath(__file__))
            safetensors_path = os.path.join(
                current_dir, "models", "binary_classifier_weights_20260722_172808.safetensors"
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


def format_msp_content(content_str: str) -> str:
    """
    处理可能带有分号的单行多质谱峰格式（例如：37 4; 38 6;），
    将其格式化为标准的单行单峰格式（例如：37 4\n38 6\n）。
    """
    lines = content_str.splitlines()
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue
            
        # 判断是否为元数据行
        is_metadata = False
        if ":" in stripped:
            first_char = stripped[0]
            if not first_char.isdigit():
                is_metadata = True
                
        if is_metadata:
            formatted_lines.append(stripped)
        else:
            # 解析并提取质谱峰，按行写入
            parts = stripped.split(";")
            for part in parts:
                p_str = part.strip()
                if p_str:
                    tokens = p_str.split()
                    if len(tokens) >= 2:
                        formatted_lines.append(f"{tokens[0]} {tokens[1]}")
                        
    return "\n".join(formatted_lines)


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

    # 如果是 msp 文件，先进行格式标准化清洗以支持单行多 peaks 结构
    if ext == 'msp':
        try:
            # 兼容多种编码尝试解码
            content_str = file_bytes.decode('utf-8', errors='ignore')
            formatted_str = format_msp_content(content_str)
            file_bytes = formatted_str.encode('utf-8')
        except Exception as e:
            # 容错：如果解码失败，使用原数据
            print(f"Warning: msp formatting failed, using raw bytes. Error: {str(e)}")

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


def run_pipeline_batch(file_bytes: bytes, filename: str, min_similarity: float = 0.75, batch_size: int = 1024) -> dict:
    """
    测样管线批量处理函数：
    能够高效处理包含成千上万条质谱的 MSP 或 MGF 文件。
    1. 使用 format_msp_content 预处理标准化文本
    2. 使用 ms_entropy 提取全部质谱数据并清洗峰
    3. 批量使用 FlashEntropySearch 进行已知库匹配
    4. 批量使用 PyTorch safetensors 神经分类器进行风险推理
    5. 返回总体统计数据与逐条测试详情
    """
    ext = os.path.splitext(filename)[1].lower().replace('.', '')
    if ext not in ['msp', 'mgf']:
        ext = 'msp'

    if ext == 'msp':
        try:
            content_str = file_bytes.decode('utf-8', errors='ignore')
            formatted_str = format_msp_content(content_str)
            file_bytes = formatted_str.encode('utf-8')
        except Exception as e:
            print(f"Warning: msp formatting failed, using raw bytes. Error: {str(e)}")

    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        tmp_file.write(file_bytes)
        tmp_file.close()
        tmp_path = tmp_file.name

        raw_spectra_gen = ms_entropy.read_one_spectrum(tmp_path, file_type=ext)
        if hasattr(raw_spectra_gen, "__iter__") and not isinstance(raw_spectra_gen, dict):
            spectra_list = list(raw_spectra_gen)
        else:
            spectra_list = [raw_spectra_gen] if raw_spectra_gen else []
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not spectra_list:
        raise ValueError(f"Failed to parse any mass spectra from {filename}.")

    total_count = len(spectra_list)
    results_detail = []
    vectors = []
    valid_indices = []

    # 1. 遍历质谱并进行清洗与准备
    for idx, spec in enumerate(spectra_list):
        spec_name = "Unknown"
        precursor_mz = None
        peaks_raw = None

        if isinstance(spec, dict):
            spec_name = spec.get("name", f"Spectrum_{idx+1}")
            precursor_mz = spec.get("precursor_mz")
            peaks_raw = spec.get("peaks")
        elif hasattr(spec, "peaks"):
            peaks_raw = spec.peaks
            spec_name = getattr(spec, "name", f"Spectrum_{idx+1}")
            precursor_mz = getattr(spec, "precursor_mz", None)

        if peaks_raw is None:
            continue

        cleaned_peaks = []
        for p in peaks_raw:
            try:
                mz = float(p[0])
                intensity = float(p[1]) if not isinstance(p[1], str) else float(p[1].replace(';', ''))
                cleaned_peaks.append([mz, intensity])
            except Exception:
                continue

        if not cleaned_peaks:
            continue

        cleaned_peaks_arr = np.array(cleaned_peaks, dtype=np.float32)
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
        valid_indices.append(idx)

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

    # 2. 批量 PyTorch 风险概率推断
    if vectors:
        vecs_arr = np.array(vectors, dtype=np.float32)
        vecs_norm = preprocess_spectra(vecs_arr)
        model = get_classifier_model()
        
        # 分批推断避免 GPU/CPU 内存峰值
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(vecs_norm), batch_size):
                batch_tensor = torch.tensor(vecs_norm[i:i+batch_size], dtype=torch.float32)
                logits = model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(probs)

        for detail_item, prob in zip(results_detail, all_probs):
            detail_item["risk_probability"] = round(prob, 4)
            detail_item["is_high_risk"] = prob >= 0.5

    # 3. 统计汇总分析
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

