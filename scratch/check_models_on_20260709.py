import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.pipeline import run_pipeline_batch

models = [
    r"app/models/binary_classifier_weights_20260722_211225.safetensors",
    r"simclr_finetune/results_20260723_162614/binary_classifier_20260723_162614.pt",
    r"simclr_finetune/results_20260723_195942/binary_classifier_20260723_195942.pt"
]

with open(r"scratch/data_source/20260709.MSP", "rb") as f:
    content = f.read()

for m in models:
    if os.path.exists(m):
        res = run_pipeline_batch(content, "20260709.MSP", model_path=m)
        print(f"=== MODEL: {m} ===")
        print(f"High risk count: {res['model_risk_inference']['high_risk_count']} / {res['total_parsed_spectra']}")
        for d in res["details"]:
            print(f"  {d['name']} | Risk Prob: {d['risk_probability']} | High Risk: {d['is_high_risk']}")
        print()
