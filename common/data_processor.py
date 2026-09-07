import argparse
from pathlib import Path
import ms_entropy
import numpy as np


def parse_msp_text(text: str, min_peaks: int = 1) -> list:
    """
    Parse MSP/MGF mass spectrometry text data using a robust state machine.
    Strictly prevents Comments or Metadata lines from leaking into the mass spectrum peak matrix.
    Returns a list of compounds containing name, smiles, precursor_mz, and peaks.
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
    """Parse MSP/MGF mass spectrometry data from binary byte stream."""
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        text = str(file_bytes)
    return parse_msp_text(text, min_peaks=min_peaks)


def parse_msp(msp_file, min_peaks: int = 5) -> list:
    """
    Parse local MSP/MGF mass spectrometry file using a robust state machine.
    Returns a list of compounds containing name, smiles, precursor_mz, and peaks.
    """
    msp_path = Path(msp_file)
    print(f"[DataProcessor] Parsing MSP file: {msp_path.name}...")

    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    compounds = parse_msp_text(text, min_peaks=min_peaks)
    print(f"[DataProcessor] Successfully parsed {len(compounds)} compound spectra")
    return compounds


def clean_spectrum(peaks):
    """
    Use ms_entropy.clean_spectrum for mass spectrum outlier peak denoising and spectrum cleaning
    """
    arr = np.array(peaks, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return ms_entropy.clean_spectrum(arr)


def peaks_to_vector(peaks, mz_min=40, mz_max=600):
    """
    Map (mz, intensity) peak list to a fixed-dimension 1D feature vector (561 dimensions) at 1 Da resolution
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
    Apply total ion current (TIC) normalization + square root scaling transform to 1D mass spectrum vector matrix
    """
    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]
    tic = spectra.sum(axis=1, keepdims=True)
    spectra = spectra / (tic + 1e-8)
    spectra = np.sqrt(spectra)
    return spectra.astype(np.float32)


def process_msp_file_to_vectors(msp_file, mz_min=40, mz_max=600):
    """
    End-to-end extraction of cleaned and normalized 561-dimensional feature matrix directly from MSP file
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
    parser = argparse.ArgumentParser(description="Cannabinoids data cleaning and vector extraction tool")
    parser.add_argument("input_msp", type=str, help="Input MSP file path")
    parser.add_argument("-o", "--output_npy", type=str, default=None, help="Output .npy file path")
    args = parser.parse_args()
    
    output_path = args.output_npy
    if not output_path:
        output_path = Path(args.input_msp).with_suffix('.npy')
        
    vecs = process_msp_file_to_vectors(args.input_msp)
    np.save(output_path, vecs)
    print(f"[OK] Normalized vector data generated: {output_path} (Shape: {vecs.shape})")