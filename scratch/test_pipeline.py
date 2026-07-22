import sys
import os
import json

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.pipeline import run_pipeline

def test_single_msp_sample():
    # Construct a sample single MSP content from positive.msp
    pos_msp_path = os.path.join(project_root, "known_compound_recognize", "positive.msp")
    
    with open(pos_msp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Take first compound block from positive.msp
    blocks = content.strip().split("\n\n")
    sample_block = blocks[0] + "\n\n"
    sample_bytes = sample_block.encode('utf-8')
    
    print("Testing pipeline with a positive compound sample...")
    result = run_pipeline(sample_bytes, "positive_sample.msp", min_similarity=0.75)
    
    print("\n--- Pipeline Analysis Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    assert result["num_cleaned_peaks"] > 0, "Cleaned peaks should be > 0"
    assert "risk_probability" in result["model_inference"], "Model inference should return risk_probability"
    assert result["known_library_match"]["is_matched"] is True, "Positive library sample should match in library"
    
    print("\n[SUCCESS] Single MSP pipeline test passed!")

if __name__ == "__main__":
    test_single_msp_sample()
