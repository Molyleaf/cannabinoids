import time
import json
import os
import sys

# 确保 app 目录加入 Python 搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import run_pipeline_batch

def main():
    msp_path = os.path.join("scratch", "MONA_GCMS-18914.MSP")
    print(f"Loading {msp_path} for testing via app.pipeline...")

    with open(msp_path, "rb") as f:
        file_bytes = f.read()

    file_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    print("\n[Pipeline Execution Starting]")
    t0 = time.time()
    
    # 严格调用系统的 app.pipeline 核心流程
    results = run_pipeline_batch(
        file_bytes=file_bytes,
        filename="MONA_GCMS-18914.MSP",
        min_similarity=0.75,
        batch_size=2048
    )
    
    t1 = time.time()
    elapsed = t1 - t0
    throughput = results["total_parsed_spectra"] / elapsed if elapsed > 0 else 0

    results["execution_metrics"] = {
        "elapsed_seconds": round(elapsed, 2),
        "throughput_spectra_per_sec": round(throughput, 2),
        "file_size_mb": round(file_size_mb, 2)
    }

    print("\n[Pipeline Execution Finished]")
    print(f"Total Parsed Spectra: {results['total_parsed_spectra']}")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Throughput: {throughput:.2f} spectra/sec")

    print("\n--- Known Library Match (Entropy Search) ---")
    lib_match = results["known_library_matches"]
    print(f"Matched Count: {lib_match['matched_count']}")
    print(f"Unmatched Count: {lib_match['unmatched_count']}")
    print(f"Match Rate: {lib_match['match_rate_percentage']}")

    print("\n--- Model Risk Inference (Safetensors Classifier) ---")
    model_inf = results["model_risk_inference"]
    print(f"High Risk Count (>= 0.5): {model_inf['high_risk_count']} ({model_inf['high_risk_percentage']})")
    print(f"Low Risk Count (< 0.5): {model_inf['low_risk_count']} ({model_inf['low_risk_percentage']})")
    print(f"Mean Risk Probability: {model_inf['mean_risk_probability']}")
    print(f"Median Risk Probability: {model_inf['median_risk_probability']}")
    print(f"Std Risk Probability: {model_inf['std_risk_probability']}")
    print(f"Percentiles: {model_inf['percentiles']}")

    # 保存测试输出 json (不包含逐条明细以保持轻量，明细另存)
    output_summary_path = os.path.join("scratch", "pipeline_mona_summary.json")
    summary_data = {
        "filename": results["filename"],
        "total_parsed_spectra": results["total_parsed_spectra"],
        "total_file_spectra": results["total_file_spectra"],
        "known_library_matches": results["known_library_matches"],
        "model_risk_inference": results["model_risk_inference"],
        "execution_metrics": results["execution_metrics"]
    }
    
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved test summary report to {output_summary_path}")

if __name__ == "__main__":
    main()
