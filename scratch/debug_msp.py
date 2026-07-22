import os
import sys
import ms_entropy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def format_msp_content(content_str: str) -> str:
    lines = content_str.splitlines()
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue
            
        # Check if it is metadata
        is_metadata = False
        if ":" in stripped:
            first_char = stripped[0]
            if not first_char.isdigit():
                is_metadata = True
                
        if is_metadata:
            formatted_lines.append(stripped)
        else:
            # Parse peaks from the line
            parts = stripped.split(";")
            for part in parts:
                p_str = part.strip()
                if p_str:
                    tokens = p_str.split()
                    if len(tokens) >= 2:
                        # Write peak: mz intensity
                        formatted_lines.append(f"{tokens[0]} {tokens[1]}")
                        
    return "\n".join(formatted_lines)

def debug_msp():
    msp_path = os.path.join(project_root, "scratch", "20260709.MSP")
    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    blocks = content.strip().split("\n\n")
    sample_block = blocks[0] + "\n\n"
    
    formatted_block = format_msp_content(sample_block)
    print("--- Formatted Block (First 15 lines) ---")
    print("\n".join(formatted_block.splitlines()[:15]))
    print("---------------------------------------")
    
    # Save formatted block to temp file
    tmp_path = os.path.join(project_root, "scratch", "tmp_debug_formatted.msp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(formatted_block)
        
    try:
        raw_spectrum = ms_entropy.read_one_spectrum(tmp_path, file_type='msp')
        raw_spectrum = list(raw_spectrum)[0]
        print("Raw spectrum keys:", raw_spectrum.keys())
        print("Precursor m/z:", raw_spectrum.get("precursor_mz"))
        peaks = list(raw_spectrum.get("peaks"))
        print(f"Number of peaks read by ms_entropy: {len(peaks)}")
        print("First 10 peaks:", peaks[:10])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    debug_msp()
