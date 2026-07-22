import pandas as pd
import numpy as np

# 读取 evaluation_results_neg.csv
df = pd.read_csv("scratch/evaluation_results_neg.csv")
print("Total rows:", len(df))
print("Matched in known library:", df["Known Lib Matched"].sum())
print("High risk predicted (>= 0.5):", df["Is High Risk"].sum())

# 找出 Known Lib Matched == True 的样本
matched_df = df[df["Known Lib Matched"] == True]
print("\n--- Known Library Matched Samples (Top 15) ---")
print(matched_df[["Index", "Name", "Matched SMILES", "Model Risk Prob"]].head(15).to_string())

# 找出 High Risk 预测中，Model Risk Prob 最高的前 20 个样本
high_risk_df = df.sort_values(by="Model Risk Prob", ascending=False)
print("\n--- Model Highest Risk Samples in Negative MSP (Top 20) ---")
print(high_risk_df[["Index", "Name", "Known Lib Matched", "Matched SMILES", "Model Risk Prob"]].head(20).to_string())

# 分析概率分布
print("\n--- Risk Prob Percentiles ---")
percentiles = [50, 80, 90, 95, 98, 99, 99.5, 100]
for p in percentiles:
    print(f"P{p}: {np.percentile(df['Model Risk Prob'], p):.4f}")
