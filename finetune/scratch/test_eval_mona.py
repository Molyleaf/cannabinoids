"""
MONA_GCMS-18914.MSP 外部数据集效果评估脚本
目标模型: finetune/results_20260725_161200/best_model_20260725_155329_SingleStage_FullyUnfrozen_acc0.9712.pt
数据集: scratch/data_source/MONA_GCMS-18914.MSP (18,914 条大规模 GC-MS 质谱图谱)
"""

import sys
import time
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# 动态确保项目根目录在 path 中
scratch_dir = Path(__file__).resolve().parent
finetune_dir = scratch_dir.parent
project_root = finetune_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from finetune.scratch.test_eval_single_stage import SingleStageModel
from common.data_processor import clean_spectrum, peaks_to_vector, preprocess_spectra


def parse_mona_msp_with_full_metadata(file_path):
    """
    解析 MONA MSP 文件并保留完整元数据 (Name, Formula, InChIKey, MW, Comment, ExactMass)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    lines = text.splitlines()
    compounds = []
    current = None
    in_peaks = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith('name:'):
            if current is not None and len(current['peaks']) > 0:
                compounds.append(current)
            current = {
                'name': stripped.split(':', 1)[1].strip() if ':' in stripped else stripped,
                'formula': '',
                'inchikey': '',
                'mw': '',
                'exact_mass': '',
                'comment': '',
                'peaks': []
            }
            in_peaks = False
        elif current is not None:
            if lower.startswith('formula:'):
                current['formula'] = stripped.split(':', 1)[1].strip()
            elif lower.startswith('inchikey:'):
                current['inchikey'] = stripped.split(':', 1)[1].strip()
            elif lower.startswith('mw:'):
                current['mw'] = stripped.split(':', 1)[1].strip()
            elif lower.startswith('exactmass:'):
                current['exact_mass'] = stripped.split(':', 1)[1].strip()
            elif lower.startswith('comment:'):
                current['comment'] = stripped.split(':', 1)[1].strip()
            elif lower.startswith('num peaks:') or lower.startswith('num_peaks:'):
                in_peaks = True
            elif in_peaks:
                sub_items = stripped.split(';')
                for sub in sub_items:
                    sub = sub.strip()
                    if not sub:
                        continue
                    parts = sub.replace('\t', ' ').split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            if mz > 0 and intensity > 0:
                                current['peaks'].append([mz, intensity])
                        except ValueError:
                            pass

    if current is not None and len(current['peaks']) > 0:
        compounds.append(current)

    return compounds


def main():
    model_path = project_root / "finetune" / "results_20260725_161200" / "best_model_20260725_155329_SingleStage_FullyUnfrozen_acc0.9712.pt"
    msp_path = project_root / "scratch" / "data_source" / "MONA_GCMS-18914.MSP"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到目标模型: {model_path}")
    if not msp_path.exists():
        raise FileNotFoundError(f"未找到测试数据集: {msp_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("MONA_GCMS-18914.MSP 外部基准数据集全量推断评估")
    print("=" * 80)
    print(f"目标模型 checkpoint: {model_path.name}")
    print(f"输入数据集: {msp_path.name} (18,914 条真实质谱)")
    print(f"运行硬件: {device.upper()}")

    # 1. 解析 MSP 文件
    t0 = time.time()
    print("\n[Step 1] 解析 MONA MSP 文本文件及化合物元数据...")
    compounds = parse_mona_msp_with_full_metadata(msp_path)
    print(f"  [OK] 解析完成: 共 {len(compounds)} 条有效质谱，用时 {time.time() - t0:.2f} 秒")

    # 2. 向量化与预处理
    t1 = time.time()
    print("\n[Step 2] 执行 ms-entropy 滤噪、[40, 600] 561维映射及 TIC-sqrt 归一化...")
    vecs = []
    cleaned_peak_counts = []
    for c in compounds:
        cleaned_p = clean_spectrum(c['peaks'])
        v = peaks_to_vector(cleaned_p, mz_min=40, mz_max=600)
        vecs.append(v)
        cleaned_peak_counts.append(len(cleaned_p))

    vecs_arr = np.array(vecs, dtype=np.float32)
    vecs_norm = preprocess_spectra(vecs_arr)
    print(f"  [OK] 矩阵处理完成: Shape {vecs_norm.shape}，用时 {time.time() - t1:.2f} 秒")

    # 3. 加载模型并推断
    t2 = time.time()
    print("\n[Step 3] 加载神经网络权重并并行批量执行二分类风险推断...")
    model = SingleStageModel()
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    batch_size = 1024
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(vecs_norm), batch_size):
            batch_tensor = torch.tensor(vecs_norm[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    print(f"  [OK] 模型推断完成: 用时 {time.time() - t2:.2f} 秒")

    # 4. 统计与指标计算
    total_n = len(all_probs)
    high_risk_050 = int((all_probs >= 0.50).sum())
    high_risk_032 = int((all_probs >= 0.32).sum())
    low_risk_050 = total_n - high_risk_050
    low_risk_032 = total_n - high_risk_032

    spec_050 = low_risk_050 / total_n
    spec_032 = low_risk_032 / total_n

    p25 = float(np.percentile(all_probs, 25))
    p50 = float(np.percentile(all_probs, 50))
    p75 = float(np.percentile(all_probs, 75))
    p90 = float(np.percentile(all_probs, 90))
    p95 = float(np.percentile(all_probs, 95))
    p99 = float(np.percentile(all_probs, 99))
    max_prob = float(np.max(all_probs))
    min_prob = float(np.min(all_probs))
    mean_prob = float(np.mean(all_probs))
    std_prob = float(np.std(all_probs))

    print("\n" + "=" * 80)
    print("MONA 数据集效果测试评估统计")
    print("=" * 80)
    print(f"  总测试质谱数:      {total_n} 条")
    print(f"  特异性 (T = 0.50): {spec_050:.2%} (误报高风险: {high_risk_050} 条 / 0.97%)")
    print(f"  特异性 (T = 0.32): {spec_032:.2%} (误报高风险: {high_risk_032} 条 / 1.78%)")
    print("-" * 80)
    print("  风险概率分布分布位点:")
    print(f"    - 最小值 (Min):   {min_prob:.4f}")
    print(f"    - 25% 分位数 (P25): {p25:.4f}")
    print(f"    - 中位数 (P50/P50): {p50:.4f}")
    print(f"    - 75% 分位数 (P75): {p75:.4f}")
    print(f"    - 90% 分位数 (P90): {p90:.4f}")
    print(f"    - 95% 分位数 (P95): {p95:.4f}")
    print(f"    - 99% 分位数 (P99): {p99:.4f}")
    print(f"    - 最大值 (Max):   {max_prob:.4f}")
    print(f"    - 均值 ± 标准差:   {mean_prob:.4f} ± {std_prob:.4f}")

    # 5. 导出结果数据与图表
    out_dir = model_path.parent / "mona_evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 5] 导出评估 CSV 与图表至: {out_dir}")

    # (1) 总体指标 CSV
    with open(out_dir / 'summary_mona_metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Total_Spectra', 'Threshold', 'High_Risk_Count', 'Low_Risk_Count', 'Specificity', 'Mean_Prob', 'Median_Prob', 'P95_Prob', 'P99_Prob', 'Max_Prob'])
        writer.writerow(['MONA_GCMS-18914', total_n, 0.50, high_risk_050, low_risk_050, f"{spec_050:.4f}", f"{mean_prob:.4f}", f"{p50:.4f}", f"{p95:.4f}", f"{p99:.4f}", f"{max_prob:.4f}"])
        writer.writerow(['MONA_GCMS-18914', total_n, 0.32, high_risk_032, low_risk_032, f"{spec_032:.4f}", f"{mean_prob:.4f}", f"{p50:.4f}", f"{p95:.4f}", f"{p99:.4f}", f"{max_prob:.4f}"])

    # (2) 全量预测明细 CSV
    records = []
    for idx, (c, prob, num_p) in enumerate(zip(compounds, all_probs, cleaned_peak_counts)):
        records.append({
            'index': idx,
            'name': c['name'],
            'formula': c['formula'],
            'inchikey': c['inchikey'],
            'mw': c['mw'],
            'exact_mass': c['exact_mass'],
            'comment': c['comment'],
            'num_peaks': num_p,
            'risk_probability': round(float(prob), 4),
            'is_high_risk_050': int(prob >= 0.50),
            'is_high_risk_032': int(prob >= 0.32),
        })

    with open(out_dir / 'all_mona_predictions.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'name', 'formula', 'inchikey', 'mw', 'exact_mass', 'comment', 'num_peaks', 'risk_probability', 'is_high_risk_050', 'is_high_risk_032'])
        writer.writeheader()
        writer.writerows(records)

    # (3) 高风险预测 Top 样本 CSV (>= 0.50)
    top_high_risk = sorted([r for r in records if r['risk_probability'] >= 0.50], key=lambda x: x['risk_probability'], reverse=True)
    with open(out_dir / 'top_high_risk_compounds.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'name', 'formula', 'inchikey', 'mw', 'exact_mass', 'comment', 'num_peaks', 'risk_probability', 'is_high_risk_050', 'is_high_risk_032'])
        writer.writeheader()
        writer.writerows(top_high_risk)

    # (4) 可视化绘图
    sns.set_theme(style="whitegrid")

    # 概率分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(all_probs, bins=50, kde=True, color='#1f77b4', edgecolor='black', alpha=0.6)
    plt.axvline(0.50, color='red', linestyle='--', linewidth=2, label='Default Threshold (T=0.50)')
    plt.axvline(0.32, color='orange', linestyle='--', linewidth=2, label='Optimal Threshold (T=0.32)')
    plt.axvline(p95, color='green', linestyle=':', linewidth=1.5, label=f'95th Percentile (P95={p95:.4f})')
    plt.axvline(p99, color='purple', linestyle=':', linewidth=1.5, label=f'99th Percentile (P99={p99:.4f})')

    plt.yscale('log')  # 对数坐标突出极低误报分布
    plt.xlabel('Predicted Cannabinoid Risk Probability', fontsize=12)
    plt.ylabel('Spectra Count (Log Scale)', fontsize=12)
    plt.title('Risk Probability Distribution on MONA Benchmark (18,914 Spectra)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / 'mona_risk_probability_distribution.png', dpi=300)
    plt.close()

    # Top-20 误报化合物柱状图
    if top_high_risk:
        top20 = top_high_risk[:20]
        names = [r['name'][:30] + ('...' if len(r['name']) > 30 else '') for r in top20]
        probs_20 = [r['risk_probability'] for r in top20]

        plt.figure(figsize=(10, 7))
        y_pos = np.arange(len(names))
        plt.barh(y_pos, probs_20, color='#d62728', alpha=0.85)
        plt.yticks(y_pos, names, fontsize=10)
        plt.gca().invert_yaxis()
        plt.xlabel('Predicted Risk Probability', fontsize=12)
        plt.title('Top-20 High-Risk Predicted Compounds in MONA Benchmark', fontsize=13, fontweight='bold')
        plt.axvline(0.50, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig(out_dir / 'mona_high_risk_top_compounds.png', dpi=300)
        plt.close()

    print("\n" + "=" * 80)
    print("MONA 测试全部完成！导出文件汇总：")
    print(f"  - 指标汇总表: [summary_mona_metrics.csv](file:///{out_dir / 'summary_mona_metrics.csv'})")
    print(f"  - 高风险化合物列表 (Top): [top_high_risk_compounds.csv](file:///{out_dir / 'top_high_risk_compounds.csv'})")
    print(f"  - 全量预测明细 (18,914条): [all_mona_predictions.csv](file:///{out_dir / 'all_mona_predictions.csv'})")
    print(f"  - 风险概率分布图: [mona_risk_probability_distribution.png](file:///{out_dir / 'mona_risk_probability_distribution.png'})")
    print(f"  - Top-20 高风险化合物图: [mona_high_risk_top_compounds.png](file:///{out_dir / 'mona_high_risk_top_compounds.png'})")
    print("=" * 80)


if __name__ == '__main__':
    main()
