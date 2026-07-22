import os
import sys
import json
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.pipeline import run_pipeline

def evaluate_test_set():
    msp_path = os.path.join(project_root, "scratch", "20260709.MSP")
    if not os.path.exists(msp_path):
        print(f"Error: Test MSP file not found at {msp_path}")
        return
        
    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Split the file by Name: to separate different spectra
    # But to be precise, we split by empty lines or Name: blocks
    blocks = []
    current_block = []
    for line in content.splitlines():
        if line.strip().startswith("Name:"):
            if current_block:
                blocks.append("\n".join(current_block) + "\n\n")
            current_block = [line]
        else:
            if current_block:
                current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block) + "\n\n")
        
    print(f"Total blocks split from MSP file: {len(blocks)}")
    
    results = []
    for i, block in enumerate(blocks):
        # Extract name
        name = "Unknown"
        for line in block.splitlines():
            if line.strip().startswith("Name:"):
                name = line.split(":", 1)[1].strip()
                break
                
        block_bytes = block.encode('utf-8')
        try:
            res = run_pipeline(block_bytes, f"sample_{i}.msp", min_similarity=0.75)
            known = res["known_library_match"]
            model = res["model_inference"]
            results.append({
                "Index": i + 1,
                "Name": name,
                "Cleaned Peaks": res["num_cleaned_peaks"],
                "Precursor mz": res["precursor_mz"],
                "Known Lib Matched": known["is_matched"],
                "Matched SMILES": known["matched_smiles"] if known["is_matched"] else "N/A",
                "Model Risk Prob": model["risk_probability"],
                "Model Risk Level": model["risk_level"]
            })
        except Exception as e:
            print(f"Error processing block {i+1} ({name}): {str(e)}")
            
    # Output to console
    df = pd.DataFrame(results)
    print("\n--- Test Set Evaluation Results ---")
    print(df.to_string(index=False))
    
    # Save to JSON and CSV
    output_json = os.path.join(project_root, "scratch", "evaluation_results_20260709.json")
    output_csv = os.path.join(project_root, "scratch", "evaluation_results_20260709.csv")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nSaved evaluation results to:\n- {output_json}\n- {output_csv}")

if __name__ == "__main__":
    evaluate_test_set()
