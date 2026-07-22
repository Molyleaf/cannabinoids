import os
import sys
import time
import json
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.pipeline import run_pipeline

def parse_large_msp_generator(file_path):
    """
    流式读取大 MSP 文件，以生成器形式返回每一个谱图块的 string
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

def evaluate_msp_file(msp_path, dataset_name):
    print(f"\n==================== 开始评估数据集: {dataset_name} ====================")
    print(f"文件路径: {msp_path}")
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
            res = run_pipeline(block_bytes, f"sample_{processed_count}.msp", min_similarity=0.75)
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
            if error_count <= 5:
                print(f"Error processing block {processed_count} ({name}): {str(e)}")
        
        if processed_count % 3000 == 0:
            print(f"  已处理 {processed_count} 张谱图...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"处理完成，耗时: {total_time:.2f} 秒。有效处理: {len(results)}, 错误: {error_count}")
    
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    total_valid = len(df)
    matched_count = df["Known Lib Matched"].sum()
    high_risk_count = df["Is High Risk"].sum()
    low_risk_count = total_valid - high_risk_count
    
    matched_pct = (matched_count / total_valid) * 100 if total_valid > 0 else 0
    high_risk_pct = (high_risk_count / total_valid) * 100 if total_valid > 0 else 0
    low_risk_pct = (low_risk_count / total_valid) * 100 if total_valid > 0 else 0
    
    probs = df["Model Risk Prob"].values
    
    stats = {
        "dataset_name": dataset_name,
        "total_processed": processed_count,
        "total_valid": total_valid,
        "matched_count": int(matched_count),
        "matched_pct": f"{matched_pct:.2f}%",
        "high_risk_count": int(high_risk_count),
        "high_risk_pct": f"{high_risk_pct:.2f}%",
        "low_risk_count": int(low_risk_count),
        "low_risk_pct": f"{low_risk_pct:.2f}%",
        "mean_prob": float(np.mean(probs)),
        "std_prob": float(np.std(probs)),
        "median_prob": float(np.median(probs)),
        "min_prob": float(np.min(probs)),
        "max_prob": float(np.max(probs)),
        "p25": float(np.percentile(probs, 25)),
        "p75": float(np.percentile(probs, 75)),
        "p90": float(np.percentile(probs, 90)),
    }
    
    print(f"\n--- {dataset_name} 汇总统计指标 ---")
    print(f"  有效样本总数: {total_valid}")
    print(f"  已知库命中数: {matched_count} ({matched_pct:.2f}%)")
    print(f"  高风险判定数 (>= 0.5): {high_risk_count} ({high_risk_pct:.2f}%)")
    print(f"  低风险判定数 (< 0.5) : {low_risk_count} ({low_risk_pct:.2f}%)")
    print(f"  风险概率均值 ± 标准差: {stats['mean_prob']:.4f} ± {stats['std_prob']:.4f}")
    print(f"  风险概率中位数       : {stats['median_prob']:.4f}")
    print(f"  风险概率分位数       : 25%={stats['p25']:.4f}, 75%={stats['p75']:.4f}, 90%={stats['p90']:.4f}")
    print(f"  最小/最大风险概率   : {stats['min_prob']:.4f} / {stats['max_prob']:.4f}")
    
    print(f"\n--- {dataset_name} 风险概率最低的前 10 个样本 ---")
    top_low = df.sort_values(by="Model Risk Prob", ascending=True).head(10)
    print(top_low[["Index", "Name", "Cleaned Peaks", "Model Risk Prob"]].to_string(index=False))
    
    print(f"\n--- {dataset_name} 风险概率最高的前 10 个样本 ---")
    top_high = df.sort_values(by="Model Risk Prob", ascending=False).head(10)
    print(top_high[["Index", "Name", "Cleaned Peaks", "Known Lib Matched", "Model Risk Prob"]].to_string(index=False))
    
    return stats, df

def main():
    neg_msp = os.path.join(project_root, "scratch", "阴性.msp")
    mona_msp = os.path.join(project_root, "scratch", "MONA_GCMS-18914.MSP")
    
    all_stats = {}
    
    if os.path.exists(neg_msp):
        stats_neg, df_neg = evaluate_msp_file(neg_msp, "阴性.msp (本地阴性测试集)")
        all_stats["neg"] = stats_neg
        df_neg.to_csv(os.path.join(project_root, "scratch", "evaluation_results_neg.csv"), index=False, encoding="utf-8-sig")
    else:
        print(f"未找到文件: {neg_msp}")
        
    if os.path.exists(mona_msp):
        stats_mona, df_mona = evaluate_msp_file(mona_msp, "MONA_GCMS-18914.MSP (MoNA通用测试集)")
        all_stats["mona"] = stats_mona
        df_mona.to_csv(os.path.join(project_root, "scratch", "evaluation_results_MONA.csv"), index=False, encoding="utf-8-sig")
    else:
        print(f"未找到文件: {mona_msp}")
        
    with open(os.path.join(project_root, "scratch", "evaluation_summary_both.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
    print("\n评估完成，结果已保存至 scratch/ 目录下的 CSV 与 JSON 文件。")

if __name__ == "__main__":
    main()
