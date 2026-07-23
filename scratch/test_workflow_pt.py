import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.pipeline import run_pipeline_batch

MODEL_PATH = r"simclr_finetune/results_20260723_162614/binary_classifier_20260723_162614.pt"
FILES = [
    r"scratch/data_source/20260709.MSP",
    r"scratch/data_source/MONA_GCMS-18914.MSP"
]

def main():
    print(f"=== Starting Workflow Test with Model: {MODEL_PATH} ===")
    
    for file_path in FILES:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"\nProcessing {file_path} ...")
        t0 = time.time()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        filename = Path(file_path).name
        result = run_pipeline_batch(
            file_bytes=file_bytes,
            filename=filename,
            min_similarity=0.75,
            model_path=MODEL_PATH
        )
        elapsed = time.time() - t0
        
        print(f"Done in {elapsed:.2f}s!")
        print(f"Total Parsed Spectra: {result['total_parsed_spectra']}")
        print("Known Library Matches:", json.dumps(result['known_library_matches'], ensure_ascii=False))
        print("Model Risk Inference:", json.dumps(result['model_risk_inference'], ensure_ascii=False, indent=2))
        
        # Display top 5 predictions for inspection if count is small
        print("\nSample Details (Top 5):")
        for item in result['details'][:5]:
            print(f"  [{item['index']}] {item['name']} | m/z: {item['precursor_mz']} | Matched: {item['is_matched']} | Risk Prob: {item['risk_probability']} | High Risk: {item['is_high_risk']}")

if __name__ == "__main__":
    main()
