import os
import sys
import time
import json
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.pipeline import run_pipeline

def parse_large_msp_generator(file_path):
    """
    流式读取大 MSP 文件，以生成器形式返回每一个谱图块的 string
    避免一次性加载大文件到内存中造成内存溢出
    """
    current_block = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("Name:"):
                if current_block:
                    yield "\n".join(current_block) + "\n\n"
                current_block = [line.strip()]
            else:
                if current_block:
                    current_block.append(line.strip())
        if current_block:
            yield "\n".join(current_block) + "\n\n"

def evaluate_large_test_set():
    msp_path = os.path.join(project_root, "scratch", "MONA_GCMS-18914.MSP")
    if not os.path.exists(msp_path):
        print(f"Error: Large MSP file not found at {msp_path}")
        return
        
    print("Starting evaluation of MONA_GCMS-18914.MSP with updated pipeline...")
    start_time = time.time()
    
    results = []
    processed_count = 0
    error_count = 0
    
    for block in parse_large_msp_generator(msp_path):
        processed_count += 1
        
        # 提取 Name
        name = "Unknown"
        for line in block.splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
                break
        
        block_bytes = block.encode('utf-8')
        try:
            res = run_pipeline(block_bytes, f"large_sample_{processed_count}.msp", min_similarity=0.75)
            known = res["known_library_match"]
            model = res["model_inference"]
            results.append({
                "Index": processed_count,
                "Name": name,
                "Cleaned Peaks": res["num_cleaned_peaks"],
                "Known Lib Matched": known["is_matched"],
                "Matched SMILES": known["matched_smiles"] if known["is_matched"] else "N/A",
                "Model Risk Prob": model["risk_probability"],
                "Model Risk Level": model["risk_level"],
                "Is High Risk": model["is_high_risk"]
            })
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"Error processing block {processed_count} ({name}): {str(e)}")
        
        if processed_count % 3000 == 0:
            print(f"Processed {processed_count} spectra...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nFinished processing in {total_time:.2f} seconds.")
    print(f"Total processed: {processed_count}, Errors: {error_count}")
    
    if not results:
        print("No valid results computed.")
        return
        
    df = pd.DataFrame(results)
    
    # 计算统计指标
    total_valid = len(df)
    matched_count = df["Known Lib Matched"].sum()
    high_risk_count = df["Is High Risk"].sum()
    
    matched_pct = (matched_count / total_valid) * 100 if total_valid > 0 else 0
    high_risk_pct = (high_risk_count / total_valid) * 100 if total_valid > 0 else 0
    
    print("\n==================== 统计汇总结果 ====================")
    print(f"有效样本总数: {total_valid}")
    print(f"已知库命中数量: {matched_count} ({matched_pct:.2f}%)")
    print(f"深度学习判定高风险数量: {high_risk_count} ({high_risk_pct:.2f}%)")
    print(f"平均高风险概率值: {df['Model Risk Prob'].mean():.4f}")
    
    # 筛选风险概率最高的前 15 个样本展示
    print("\n--- 风险概率最高的前 15 个样本 ---")
    top_high_risk = df.sort_values(by="Model Risk Prob", ascending=False).head(15)
    print(top_high_risk[["Index", "Name", "Cleaned Peaks", "Known Lib Matched", "Model Risk Prob"]].to_string(index=False))
    
    # 保存结果
    output_csv = os.path.join(project_root, "scratch", "evaluation_results_MONA.csv")
    output_json = os.path.join(project_root, "scratch", "evaluation_results_MONA.json")
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    summary_stats = {
        "total_processed": processed_count,
        "total_valid": total_valid,
        "errors": error_count,
        "matched_count": int(matched_count),
        "matched_percentage": f"{matched_pct:.2f}%",
        "high_risk_count": int(high_risk_count),
        "high_risk_percentage": f"{high_risk_pct:.2f}%",
        "avg_risk_probability": float(df["Model Risk Prob"].mean()),
        "top_high_risk_samples": top_high_risk.to_dict(orient="records")
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved detailed evaluation to:\n- {output_csv}\n- {output_json}")

if __name__ == "__main__":
    evaluate_large_test_set()
