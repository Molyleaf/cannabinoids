import sys
import json
from pathlib import Path

# 将项目根目录添加到 python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.pipeline import run_pipeline_batch

def main():
    msp_path = project_root / "known_compound_recognize" / "positive.msp"
    print(f"Reading file: {msp_path}")
    
    if not msp_path.exists():
        print(f"Error: {msp_path} does not exist.")
        return

    with open(msp_path, "rb") as f:
        file_bytes = f.read()

    print(f"Loaded {len(file_bytes)} bytes. Running full pipeline batch processing...")
    results = run_pipeline_batch(file_bytes, filename="positive.msp", min_similarity=0.75)

    stats = results["model_risk_inference"]
    known_stats = results["known_library_matches"]

    print("\n" + "=" * 60)
    print("      全工作流测试结果汇总 (positive.msp - 阳性样本集)")
    print("=" * 60)
    print(f"解析质谱总数: {results['total_parsed_spectra']} / 文件总数: {results['total_file_spectra']}")
    print("-" * 60)
    print("1. 已知库熵匹配结果:")
    print(f"   - 匹配成功数: {known_stats['matched_count']}")
    print(f"   - 未匹配数:   {known_stats['unmatched_count']}")
    print(f"   - 匹配率:     {known_stats['match_rate_percentage']}")
    print("-" * 60)
    print("2. 二分类模型风险推断结果:")
    print(f"   - 高风险样本数 (Positive): {stats['high_risk_count']} ({stats['high_risk_percentage']})")
    print(f"   - 低风险样本数 (Negative): {stats['low_risk_count']} ({stats['low_risk_percentage']})")
    print(f"   - 平均风险概率:            {stats['mean_risk_probability']:.4f}")
    print(f"   - 中位数风险概率:          {stats['median_risk_probability']:.4f}")
    print(f"   - 标准差:                  {stats['std_risk_probability']:.4f}")
    print(f"   - 概率分位数 [P25, P50, P75, P90, P95]:")
    print(f"     {stats['percentiles']}")
    print("=" * 60)

    # 导出 summary json
    summary_out = project_root / "scratch" / "positive_workflow_results.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump({
            "total_parsed_spectra": results["total_parsed_spectra"],
            "known_library_matches": known_stats,
            "model_risk_inference": stats,
            "sample_details_head": results["details"][:10]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至 {summary_out}")

if __name__ == "__main__":
    main()
