import pandas as pd
import numpy as np

df = pd.read_csv("scratch/evaluation_results_neg.csv")
high_risk = df[df["Is High Risk"] == True].copy()

print(f"Total High Risk Count in 阴性.msp: {len(high_risk)} / {len(df)} ({len(high_risk)/len(df)*100:.2f}%)")

# 1. 已知库直接匹配上阳性 SMILES 的样本（绝对的数据污染/阳性混入）
matched_in_high_risk = high_risk[high_risk["Known Lib Matched"] == True]
print(f"1. Verified Positive Compounds matched with Positive SMILES Library: {len(matched_in_high_risk)}")

# 2. 未在已知库匹配上，但名称包含大麻素/NPS/违禁药物特征词的样本
cannabinoid_nps_keywords = [
    r'AM\d+', r'JWH', r'HU-\d+', r'CP-47', r'XLR-11', r'UR-144', r'AKB48', r'APICA', r'APINACA', 
    r'PB-22', r'5F-', r'AB-FUBINACA', r'ADB-FUBINACA', r'5F-ADB', r'MDMB', r'NNEI', r'MAM', r'F2201',
    r'MDA 19', r'PX 1', r'PX 2', r'CH-P', r'PCN', r'PeGACLONE', r'FUB', r'FICA', r'PINACA', r'CUMYL',
    r'Fentanyl', r'Isotonitazene', r'Nitazene', r'Morphine', r'Heroin', r'Cocaine', r'Methamphetamine'
]
pattern = '|'.join(cannabinoid_nps_keywords)

unmatched_high_risk = high_risk[high_risk["Known Lib Matched"] == False]
nps_unmatched = unmatched_high_risk[unmatched_high_risk["Name"].str.contains(pattern, case=False, na=False)]
print(f"2. Unmatched in Positive Lib, but Name clearly indicates Synthetic Cannabinoid / NPS: {len(nps_unmatched)}")

# 3. 剩下的真正“疑似误报”（阴性常规化合物被模型判高风险）
true_false_positives = unmatched_high_risk[~unmatched_high_risk["Name"].str.contains(pattern, case=False, na=False)]
print(f"3. Pure Negative / Generic Compounds wrongly predicted as High Risk (Real False Positives): {len(true_false_positives)}")

print("\n--- Top 20 Real False Positives (Pure Negative/Generic Compounds with High Risk Prob) ---")
print(true_false_positives.sort_values(by="Model Risk Prob", ascending=False)[["Index", "Name", "Model Risk Prob"]].head(20).to_string())
