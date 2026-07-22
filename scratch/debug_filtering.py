import os
import sys
import numpy as np
import ms_entropy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def debug_filtering():
    msp_path = os.path.join(project_root, "scratch", "20260709.MSP")
    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    blocks = content.strip().split("\n\n")
    sample_block = blocks[0] + "\n\n"
    
    # Standardize format
    from app.pipeline import format_msp_content
    formatted_block = format_msp_content(sample_block)
    
    # Save to temp
    tmp_path = os.path.join(project_root, "scratch", "tmp_filt_debug.msp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(formatted_block)
        
    try:
        raw_spectrum = ms_entropy.read_one_spectrum(tmp_path, file_type='msp')
        raw_spectrum = list(raw_spectrum)[0]
        peaks = np.array(list(raw_spectrum.get("peaks")), dtype=np.float32)
        
        print(f"Original peaks count: {len(peaks)}")
        orig_max = np.max(peaks[:, 1])
        print(f"Original max intensity: {orig_max}")
        print("Original peaks below 1% of max:", [p for p in peaks if p[1] < 0.01 * orig_max])
        
        cleaned = ms_entropy.clean_spectrum(peaks)
        print(f"Cleaned peaks count: {len(cleaned)}")
        cleaned_max = np.max(cleaned[:, 1])
        print(f"Cleaned max intensity: {cleaned_max}")
        print("Cleaned peaks below 1% of cleaned max:", [p for p in cleaned if p[1] < 0.01 * cleaned_max])
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    debug_filtering()
