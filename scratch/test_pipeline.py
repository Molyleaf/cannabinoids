import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from app.pipeline import run_pipeline

pos_msp_path = root / "entropy" / "positive.msp"
if pos_msp_path.exists():
    with open(pos_msp_path, "rb") as f:
        file_bytes = f.read()

    print("Testing pipeline on positive.msp spectrum (expecting high similarity match > 0.90)...")
    res_pos = run_pipeline(file_bytes, "positive_sample.msp", model_type="multi", min_similarity=0.90)
    print("Multi Class Pred:", res_pos["model_inference"]["pred_class"])
    print("Entropy Match (>0.90):", res_pos["entropy_match"]["is_matched"], "Score:", res_pos["entropy_match"]["similarity_score"])
    print("Returned SMILES:", res_pos["entropy_match"]["matched_smiles"])
    print("Matched Compound Name:", res_pos["entropy_match"]["matched_name"])

