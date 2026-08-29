import argparse
from pathlib import Path
import ms_entropy
import numpy as np


def parse_msp_text(text: str, min_peaks: int = 1) -> list:
    """
    使用鲁棒状态机解析 MSP/MGF 质谱文本数据。
    严格防范 Comments 或 Metadata 行泄漏至质谱峰矩阵中。
    返回包含 name, smiles, precursor_mz, peaks 的化合物列表。
    """
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
            if current_comp is not None and 'peaks' in current_comp and len(current_comp['peaks']) >= min_peaks:
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
            elif lower.startswith('num peaks:') or lower.startswith('num_peaks:') or lower.startswith('numpeaks:'):
                in_peaks = True
            elif in_peaks or (stripped and (stripped[0].isdigit() or stripped[0] == '.')):
                sub_items = stripped.split(';')
                has_valid_peak = False
                for sub in sub_items:
                    sub = sub.strip()
                    if not sub:
                        continue
                    parts = sub.replace('\t', ' ').split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity_str = parts[1].replace(';', '')
                            intensity = float(intensity_str)
                            if mz > 0 and intensity > 0:
                                current_comp['peaks'].append([mz, intensity])
                                has_valid_peak = True
                        except (ValueError, TypeError):
                            pass
                if has_valid_peak:
                    in_peaks = True

    if current_comp is not None and 'peaks' in current_comp and len(current_comp['peaks']) >= min_peaks:
        compounds.append(current_comp)

    return compounds


def parse_msp_bytes(file_bytes: bytes, min_peaks: int = 1) -> list:
    """从二进制字节流解析 MSP/MGF 质谱数据。"""
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        text = str(file_bytes)
    return parse_msp_text(text, min_peaks=min_peaks)


def parse_msp(msp_file, min_peaks: int = 5) -> list:
    """
    使用鲁棒状态机解析本地 MSP/MGF 质谱文件。
    返回包含 name, smiles, precursor_mz, peaks 的化合物列表。
    """
    msp_path = Path(msp_file)
    print(f"[DataProcessor] 正在解析 MSP 文件: {msp_path.name}...")

    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    compounds = parse_msp_text(text, min_peaks=min_peaks)
    print(f"[DataProcessor] 成功解析出 {len(compounds)} 个化合物谱图")
    return compounds



def clean_spectrum(peaks):
    """
    使用 ms_entropy.clean_spectrum 进行质谱离群峰滤噪与质谱清洗
    """
    arr = np.array(peaks, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return ms_entropy.clean_spectrum(arr)


def peaks_to_vector(peaks, mz_min=40, mz_max=600):
    """
    将 (mz, intensity) 峰列表按 1 Da 分辨率映射为固定维度的 1D 特征向量 (561 维)
    """
    dim = mz_max - mz_min + 1
    vec = np.zeros(dim, dtype=np.float32)
    for mz, intensity in peaks:
        if mz_min <= mz <= mz_max:
            idx = int(round(mz - mz_min))
            if 0 <= idx < dim:
                vec[idx] += intensity
    return vec


def preprocess_spectra(spectra):
    """
    对 1D 质谱向量矩阵进行总离子强度 (TIC) 归一化 + 平方根缩放转换
    """
    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]
    tic = spectra.sum(axis=1, keepdims=True)
    spectra = spectra / (tic + 1e-8)
    spectra = np.sqrt(spectra)
    return spectra.astype(np.float32)


def process_msp_file_to_vectors(msp_file, mz_min=40, mz_max=600):
    """
    从 MSP 文件直接端到端提取已清洗并归一化的 561 维特征矩阵
    """
    compounds = parse_msp(msp_file)
    vectors = []
    for comp in compounds:
        cleaned_peaks = clean_spectrum(comp['peaks'])
        vec = peaks_to_vector(cleaned_peaks, mz_min=mz_min, mz_max=mz_max)
        vectors.append(vec)
    
    if not vectors:
        dim = mz_max - mz_min + 1
        return np.zeros((0, dim), dtype=np.float32)
    
    raw_vecs = np.array(vectors, dtype=np.float32)
    norm_vecs = preprocess_spectra(raw_vecs)
    return norm_vecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cannabinoids 数据清洗与向量提取工具")
    parser.add_argument("input_msp", type=str, help="输入的 MSP 文件路径")
    parser.add_argument("-o", "--output_npy", type=str, default=None, help="输出的 .npy 文件路径")
    args = parser.parse_args()
    
    output_path = args.output_npy
    if not output_path:
        output_path = Path(args.input_msp).with_suffix('.npy')
        
    vecs = process_msp_file_to_vectors(args.input_msp)
    np.save(output_path, vecs)
    print(f"[OK] 已生成归一化向量数据: {output_path} (Shape: {vecs.shape})")
