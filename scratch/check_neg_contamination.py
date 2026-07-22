import pandas as pd
import re

# 读取 negative 预测结果
df = pd.read_csv("scratch/evaluation_results_neg.csv")

# 正则匹配常见的合成大麻素关键词
cannabinoid_keywords = [
    r'AM\d+', r'JWH', r'HU-\d+', r'CP-47', r'XLR-11', r'UR-144', r'AKB48', r'APICA', r'APINACA', 
    r'PB-22', r'5F-', r'AB-FUBINACA', r'ADB-FUBINACA', r'5F-ADB', r'MDMB', r'NNEI', r'MAM', r'F2201',
    r'MDA 19', r'PX 1', r'PX 2', r'CH-P', r'PCN', r'PeGACLONE', r'FUB', r'FICA', r'PINACA', r'CUMYL'
]

pattern = '|'.join(cannabinoid_keywords)

cannabinoid_in_neg = df[df['Name'].str.contains(pattern, case=False, na=False)]
print(f"Total entries in 阴性.msp: {len(df)}")
print(f"Entries matching known synthetic cannabinoid naming patterns: {len(cannabinoid_in_neg)}")

print("\n--- High Risk Preds in 阴性.msp with Synthetic Cannabinoid Keywords (Top 20) ---")
print(cannabinoid_in_neg.sort_values(by="Model Risk Prob", ascending=False)[["Index", "Name", "Known Lib Matched", "Model Risk Prob"]].head(20).to_string())

# 检查这些包含大麻素关键词的样本中，有多少被预测为 High Risk (>=0.5)
high_risk_cannabinoids = cannabinoid_in_neg[cannabinoid_in_neg['Is High Risk'] == True]
print(f"\nIn 阴性.msp, {len(high_risk_cannabinoids)} out of {len(cannabinoid_in_neg)} cannabinoid-named compounds were predicted as HIGH RISK (>= 0.5) by the model!")

# 反过来看：被预测为 High Risk (202 个) 中，有多少名字带有大麻素关键词？
high_risk_all = df[df['Is High Risk'] == True]
high_risk_matching_keyword = high_risk_all[high_risk_all['Name'].str.contains(pattern, case=False, na=False)]
print(f"Out of ALL 202 High Risk false positives, {len(high_risk_matching_keyword)} ({len(high_risk_matching_keyword)/len(high_risk_all)*100:.1f}%) actually carry synthetic cannabinoid names!")
