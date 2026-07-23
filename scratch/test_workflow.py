import os
import sys
import json
import numpy as np

# 将项目根目录添加进 python 路径
project_root = r"d:\UserFiles\Documents\PyCharm\cannabinoids"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.pipeline import run_pipeline_batch

model_path = os.path.join(project_root, r"simclr_finetune\results_20260723_204612\binary_classifier_20260723_204612.pt")
msp_files = [
    os.path.join(project_root, r"scratch\data_source\MONA_GCMS-18914.MSP"),
    os.path.join(project_root, r"scratch\data_source\20260709.MSP")
]

print(f"Testing model: {model_path}")

for msp_path in msp_files:
    filename = os.path.basename(msp_path)
    print(f"\n================ Running Workflow for: {filename} ================")
    if not os.path.exists(msp_path):
        print(f"File not found: {msp_path}")
        continue
    
    with open(msp_path, 'rb') as f:
        file_bytes = f.read()
    
    stats = run_pipeline_batch(
        file_bytes=file_bytes,
        filename=filename,
        min_similarity=0.75,
        batch_size=1024,
        model_path=model_path
    )
    
    # 打印概览
    print(f"Total parsed spectra: {stats['total_parsed_spectra']}")
    print("Library Matches:", json.dumps(stats['known_library_matches'], indent=2, ensure_ascii=False))
    print("Model Risk Inference:", json.dumps(stats['model_risk_inference'], indent=2, ensure_ascii=False))

    # 显示风险预测分值的采样/分布
    details = stats['details']
    high_risk_sample = [item for item in details if item['is_high_risk']][:5]
    low_risk_sample = [item for item in details if not item['is_high_risk']][:5]
    
    print(f"\n--- High Risk Samples (Total: {len([item for item in details if item['is_high_risk']])}) ---")
    for s in high_risk_sample:
        print(f"  Name: {s['name']}, Prob: {s['risk_probability']}, Matched: {s['is_matched']}")
        
    print(f"\n--- Low Risk Samples (Total: {len([item for item in details if not item['is_high_risk']])}) ---")
    for s in low_risk_sample:
        print(f"  Name: {s['name']}, Prob: {s['risk_probability']}, Matched: {s['is_matched']}")
