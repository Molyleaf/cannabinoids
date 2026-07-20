import os
import numpy as np
from ms_entropy import read_one_spectrum, clean_spectrum, FlashEntropySearch

def build_positive_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    msp_file = os.path.join(current_dir, "positive.msp")
    index_dir = os.path.join(current_dir, "positive_idx")

    if not os.path.exists(msp_file):
        raise FileNotFoundError(f"Could not find positive.msp at {msp_file}")

    print(f"Reading spectra from {msp_file}...")
    spectra = []
    
    # ms-entropy's read_one_spectrum returns a generator yielding dicts representing each spectrum
    for spec in read_one_spectrum(msp_file):
        raw_peaks = spec.get("peaks")
        if not raw_peaks:
            continue
            
        peaks = []
        for p in raw_peaks:
            try:
                # Remove semicolons and convert to float
                mz = float(p[0])
                intensity = float(p[1].replace(';', ''))
                peaks.append([mz, intensity])
            except Exception:
                continue
                
        if not peaks:
            continue
            
        peaks = np.array(peaks, dtype=np.float32)
        # Clean spectrum (normalizes intensity, centroids and sorts by m/z)
        peaks = clean_spectrum(peaks)
        
        # Resolve precursor_mz (estimate protonated ion [M+H]+ in positive mode)
        exactmass = spec.get("exactmass")
        mw = spec.get("mw")
        if exactmass is not None:
            precursor_mz = float(exactmass) + 1.007276
        elif mw is not None:
            precursor_mz = float(mw) + 1.007276
        else:
            precursor_mz = float(np.max(peaks[:, 0]))
            
        # Keep name and smiles in metadata
        name = spec.get("name") or spec.get("Name") or "Unknown"
        smiles = spec.get("smiles") or spec.get("SMILES") or ""
        
        spectra.append({
            "precursor_mz": precursor_mz,
            "peaks": peaks,
            "name": name,
            "smiles": smiles
        })
        
    print(f"Loaded {len(spectra)} spectra. Building FlashEntropySearch index...")
    
    # Use mz_index_step=0.01 because peaks are integers, saving substantial disk space
    entropy_search = FlashEntropySearch(mz_index_step=0.01)
    
    # Build index with clean_spectra=False since they are pre-cleaned
    entropy_search.build_index(spectra, clean_spectra=False)
    
    # Write to directory
    print(f"Writing index to {index_dir}...")
    entropy_search.write(index_dir)
    print("Index build completed successfully!")

if __name__ == "__main__":
    build_positive_index()
