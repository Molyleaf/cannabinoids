import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import ms_entropy
from ms_entropy import clean_spectrum
from rdkit import Chem
from scipy.sparse import csr_matrix
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_canonical_identifier(spec, spec_idx):
    """
    根据统一标识符规范化化合物身份。
    优先解析 SMILES -> InChIKey / Canonical SMILES；
    无 SMILES 时退而求其次使用规范化 Name。
    """
    smiles = spec.get("smiles") or spec.get("SMILES") or ""
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchikey = Chem.MolToInchiKey(mol)
                can_smiles = Chem.MolToSmiles(mol, canonical=True)
                return inchikey, can_smiles
        except Exception:
            pass
        return f"SMILES_{smiles}", smiles
    name = spec.get("name") or spec.get("Name") or ""
    if name:
        return f"NAME_{name.strip().upper()}", ""
    return f"SPEC_{spec_idx}", ""

def parse_peaks(raw_peaks):
    """
    解析质谱峰，处理强度中的特殊字符，并使用 ms_entropy.clean_spectrum 清理。
    """
    peaks = []
    for p in raw_peaks:
        try:
            mz = float(p[0])
            intensity = float(p[1].replace(';', '')) if isinstance(p[1], str) else float(p[1])
            peaks.append([mz, intensity])
        except Exception:
            continue
    if not peaks:
        return np.empty((0, 2), dtype=np.float32)
    return clean_spectrum(np.array(peaks, dtype=np.float32))

def load_dataset(msp_path):
    print(f"正在加载 MSP 数据集: {msp_path}")
    spectra_list = []
    compound_map = {} # compound_id -> list of spectrum indices
    
    for idx, spec in enumerate(ms_entropy.read_one_spectrum(msp_path)):
        raw_peaks = spec.get("peaks")
        if not raw_peaks:
            continue
            
        peaks_arr = parse_peaks(raw_peaks)
        if len(peaks_arr) == 0:
            continue
            
        cid, can_smiles = get_canonical_identifier(spec, idx)
        name = spec.get("name") or spec.get("Name") or f"Spec_{idx}"
        
        item = {
            "spec_id": idx,
            "compound_id": cid,
            "name": name,
            "smiles": can_smiles,
            "peaks": peaks_arr
        }
        spectra_list.append(item)
        if cid not in compound_map:
            compound_map[cid] = []
        compound_map[cid].append(idx)
        
    print(f"成功加载 {len(spectra_list)} 条质谱谱图，涵盖 {len(compound_map)} 个独立化合物。")
    return spectra_list, compound_map

class FastCosineSearcher:
    """
    高效向量化质谱余弦相似度检索器 (Cosine Similarity Searcher)。
    采用质谱标准的 0.02 Da m/z 离散 Bin 和平方根强度加权 (Square Root Intensity Weighting)。
    使用 SciPy 稀疏矩阵加速全库批量矩阵点积计算。
    """
    def __init__(self, bin_size=0.02, mz_max=2000.0):
        self.bin_size = bin_size
        self.num_bins = int(np.ceil(mz_max / bin_size))
        self.db_mat = None
        self.spectra = []
        
    def build_index(self, spectra_list):
        self.spectra = spectra_list
        rows, cols, data = [], [], []
        
        for idx, spec in enumerate(spectra_list):
            peaks = spec['peaks']
            if len(peaks) == 0:
                continue
            mzs = peaks[:, 0]
            w = np.sqrt(peaks[:, 1])
            norm = np.linalg.norm(w)
            if norm == 0:
                continue
            w = (w / norm).astype(np.float32)
            
            bin_indices = np.floor(mzs / self.bin_size).astype(int)
            valid = (bin_indices >= 0) & (bin_indices < self.num_bins)
            bin_indices = bin_indices[valid]
            w = w[valid]
            
            bin_dict = {}
            for b, v in zip(bin_indices, w):
                if b not in bin_dict or v > bin_dict[b]:
                    bin_dict[b] = v
                    
            for b, v in bin_dict.items():
                rows.append(idx)
                cols.append(b)
                data.append(v)
                
        self.db_mat = csr_matrix((data, (rows, cols)), shape=(len(spectra_list), self.num_bins), dtype=np.float32)
        
    def search_batch(self, query_peaks_list):
        if not query_peaks_list:
            return np.zeros((0, self.db_mat.shape[0]), dtype=np.float32)
            
        rows, cols, data = [], [], []
        for idx, peaks in enumerate(query_peaks_list):
            if len(peaks) == 0:
                continue
            mzs = peaks[:, 0]
            w = np.sqrt(peaks[:, 1])
            norm = np.linalg.norm(w)
            if norm == 0:
                continue
            w = (w / norm).astype(np.float32)
            
            bin_indices = np.floor(mzs / self.bin_size).astype(int)
            valid = (bin_indices >= 0) & (bin_indices < self.num_bins)
            bin_indices = bin_indices[valid]
            w = w[valid]
            
            bin_dict = {}
            for b, v in zip(bin_indices, w):
                if b not in bin_dict or v > bin_dict[b]:
                    bin_dict[b] = v
                    
            for b, v in bin_dict.items():
                rows.append(idx)
                cols.append(b)
                data.append(v)
                
        q_mat = csr_matrix((data, (rows, cols)), shape=(len(query_peaks_list), self.num_bins), dtype=np.float32)
        scores_mat = q_mat.dot(self.db_mat.T).toarray()
        return scores_mat

def run_closed_set_matching(spectra_list, compound_map, bin_size=0.02):
    """
    闭集已知化合物匹配测试 (Leave-One-Out):
    目标：同一个化合物有多个谱图时，删掉一条，用这一条去库里搜，能否准确找到同一化合物的其他谱图。
    判断命中：返回的 Top-K 列表中只要包含同化合物标识符 (compound_id) 的任意谱图就算命中。
    """
    print(f"\n--- 运行闭集已知化合物匹配测试 (Leave-One-Out) ---")
    
    searcher = FastCosineSearcher(bin_size=bin_size)
    searcher.build_index(spectra_list)
    
    multi_spec_compounds = {cid: idxs for cid, idxs in compound_map.items() if len(idxs) >= 2}
    query_spec_indices = []
    for idxs in multi_spec_compounds.values():
        query_spec_indices.extend(idxs)
        
    print(f"共有 {len(multi_spec_compounds)} 个化合物包含 >=2 条谱图，共计 {len(query_spec_indices)} 条已知查询质谱。")
    
    query_peaks_list = [spectra_list[idx]["peaks"] for idx in query_spec_indices]
    t0 = time.time()
    all_scores_mat = searcher.search_batch(query_peaks_list)
    t1 = time.time()
    print(f"批量矩阵余弦相似度检索完成，耗时: {t1-t0:.4f} 秒。")
    
    top1_correct_count = 0
    top3_correct_count = 0
    rr_list = []
    known_top1_scores = []
    
    for q_i, q_idx in enumerate(query_spec_indices):
        q_spec = spectra_list[q_idx]
        q_cid = q_spec["compound_id"]
        
        scores = all_scores_mat[q_i]
        
        # 降序排序并排除自身 (Leave-One-Out: idx != q_idx)
        sorted_indices = np.argsort(scores)[::-1]
        filtered_candidates = [idx for idx in sorted_indices if idx != q_idx]
        
        if not filtered_candidates:
            rr_list.append(0.0)
            known_top1_scores.append(0.0)
            continue
            
        top1_idx = filtered_candidates[0]
        top1_score = float(scores[top1_idx])
        top1_cid = spectra_list[top1_idx]["compound_id"]
        
        # 判断 Top-1 是否命中正确化合物
        is_top1_hit = (top1_cid == q_cid)
        if is_top1_hit:
            top1_correct_count += 1
            
        # 查找同化合物首次出现的排名
        correct_rank = None
        for rank, cand_idx in enumerate(filtered_candidates, 1):
            if spectra_list[cand_idx]["compound_id"] == q_cid:
                correct_rank = rank
                break
                
        if correct_rank is not None:
            rr_list.append(1.0 / correct_rank)
            if correct_rank <= 3:
                top3_correct_count += 1
        else:
            rr_list.append(0.0)
            
        known_top1_scores.append(top1_score)

    n_queries = len(query_spec_indices)
    top1_acc = top1_correct_count / n_queries if n_queries > 0 else 0.0
    top3_acc = top3_correct_count / n_queries if n_queries > 0 else 0.0
    mrr = float(np.mean(rr_list)) if rr_list else 0.0
    
    print(f"闭集测试结果 (已知 Queries={n_queries}):")
    print(f"  Top-1 准确率 : {top1_acc*100:.2f}% ({top1_correct_count}/{n_queries})")
    print(f"  Top-3 准确率 : {top3_acc*100:.2f}% ({top3_correct_count}/{n_queries})")
    print(f"  MRR (平均倒数排名) : {mrr:.4f}")
    
    return {
        "n_queries": n_queries,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "mrr": mrr,
        "known_top1_scores": known_top1_scores
    }

def run_open_set_rejection(spectra_list, compound_map, bin_size=0.02, n_splits=5, seed=42):
    """
    开集全新化合物拒绝测试 (5-Fold Compound Split):
    目标：评估把化合物的所有谱图从库里删除作为全新未知物，去检索剩余库时的最高得分（Top-1 Score）。
    """
    print(f"\n--- 运行开集全新化合物拒绝测试 (5-Fold Compound Split) ---")
    
    np.random.seed(seed)
    all_cids = list(compound_map.keys())
    np.random.shuffle(all_cids)
    
    folds = np.array_split(all_cids, n_splits)
    unknown_top1_scores = []
    
    for fold_idx in range(n_splits):
        unknown_cids = set(folds[fold_idx])
        known_cids = set(all_cids) - unknown_cids
        
        db_spectra = [s for s in spectra_list if s["compound_id"] in known_cids]
        query_spectra = [s for s in spectra_list if s["compound_id"] in unknown_cids]
        
        db_searcher = FastCosineSearcher(bin_size=bin_size)
        db_searcher.build_index(db_spectra)
        
        query_peaks = [q["peaks"] for q in query_spectra]
        scores_mat = db_searcher.search_batch(query_peaks)
        
        for q_i in range(len(query_spectra)):
            scores = scores_mat[q_i]
            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
            unknown_top1_scores.append(max_score)
            
    print(f"开集测试完成 (未知化合物 Queries={len(unknown_top1_scores)})。")
    return {
        "n_unknown_queries": len(unknown_top1_scores),
        "unknown_top1_scores": unknown_top1_scores
    }

def evaluate_thresholds(known_scores, unknown_scores, thresholds):
    """
    评估不同相似度阈值 θ 下的开集拒绝与已知接受性能。
    """
    known_arr = np.array(known_scores, dtype=np.float32)
    unknown_arr = np.array(unknown_scores, dtype=np.float32)
    
    results = []
    for theta in thresholds:
        tp = np.sum(known_arr >= theta)
        fn = np.sum(known_arr < theta)
        fp = np.sum(unknown_arr >= theta)
        tn = np.sum(unknown_arr < theta)
        
        tpr = float(tp / len(known_arr)) if len(known_arr) > 0 else 0.0
        fpr = float(fp / len(unknown_arr)) if len(unknown_arr) > 0 else 0.0
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
        youden = tpr - fpr
        
        results.append({
            "threshold": round(float(theta), 2),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "youden_index": round(youden, 4),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn)
        })
    return results

def compute_distribution_stats(scores):
    arr = np.array(scores, dtype=np.float32)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4)
    }

def print_ascii_histogram(known_scores, unknown_scores, bins=10):
    counts_k, bin_edges = np.histogram(known_scores, bins=bins, range=(0.0, 1.0))
    counts_u, _ = np.histogram(unknown_scores, bins=bins, range=(0.0, 1.0))
    
    print("\n" + "=" * 70)
    print(" 余弦相似度得分分布直方图对比 (Known Queries vs Unknown Queries Top-1 Scores)")
    print("=" * 70)
    print(f"{'得分区间':<15} | {'已知物 (Known)':<24} | {'未知物 (Unknown)':<24}")
    print("-" * 70)
    
    max_k = max(counts_k) if max(counts_k) > 0 else 1
    max_u = max(counts_u) if max(counts_u) > 0 else 1
    
    for i in range(bins):
        low, high = bin_edges[i], bin_edges[i+1]
        k_val = counts_k[i]
        u_val = counts_u[i]
        k_bar = "#" * int((k_val / max_k) * 16)
        u_bar = "*" * int((u_val / max_u) * 16)
        print(f"[{low:.1f} - {high:.1f})     | {k_val:<5} {k_bar:<17} | {u_val:<5} {u_bar:<17}")
    print("=" * 70)

def plot_and_save_visualizations(known_scores, unknown_scores, thresh_eval, output_fig_path):
    plt.figure(figsize=(16, 5), dpi=300)
    
    # 子图 1: 分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(known_scores, bins=20, range=(0, 1), alpha=0.6, color='blue', label='Known Queries (Top-1)', density=True)
    plt.hist(unknown_scores, bins=20, range=(0, 1), alpha=0.6, color='orange', label='Unknown Queries (Top-1)', density=True)
    plt.title("Cosine Similarity Score Distribution")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 子图 2: ROC 曲线
    plt.subplot(1, 3, 2)
    thresholds_dense = np.linspace(0.0, 1.0, 201)
    known_arr = np.array(known_scores)
    unknown_arr = np.array(unknown_scores)
    tprs = [np.mean(known_arr >= t) for t in thresholds_dense]
    fprs = [np.mean(unknown_arr >= t) for t in thresholds_dense]
    # AUC 计算兼容不同 NumPy 版本
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc_score = -trapz_fn(tprs, fprs) if trapz_fn else 0.0
    
    plt.plot(fprs, tprs, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Open-Set ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 子图 3: TPR/FPR vs Threshold 阈值影响图
    plt.subplot(1, 3, 3)
    thresh_list = [r["threshold"] for r in thresh_eval]
    tpr_list = [r["tpr"] for r in thresh_eval]
    fpr_list = [r["fpr"] for r in thresh_eval]
    youden_list = [r["youden_index"] for r in thresh_eval]
    
    plt.plot(thresh_list, tpr_list, 'o-', color='green', label='TPR (Known Accept Rate)')
    plt.plot(thresh_list, fpr_list, 's-', color='red', label='FPR (Unknown Misaccept Rate)')
    plt.plot(thresh_list, youden_list, '^--', color='purple', label="Youden's J Index")
    
    best_idx = np.argmax(youden_list)
    best_thresh = thresh_list[best_idx]
    plt.axvline(x=best_thresh, color='black', linestyle=':', label=f'Optimal θ = {best_thresh:.2f}')
    
    plt.xlabel('Cosine Threshold (θ)')
    plt.ylabel('Rate')
    plt.title('TPR / FPR / Youden J vs Cosine Threshold')
    plt.legend(loc="center left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_fig_path)
    plt.close()
    print(f"\n可视化评估图表已成功保存至: {output_fig_path}")

def main():
    msp_file = os.path.join("entropy", "positive.msp")
    if not os.path.exists(msp_file):
        msp_file = os.path.join("scratch", "../data_source", "MONA_GCMS-18914.MSP")
        
    spectra_list, compound_map = load_dataset(msp_file)
    
    bin_size = 0.02 # 0.02 Da m/z 离散 bin 容差
    closed_res = run_closed_set_matching(spectra_list, compound_map, bin_size=bin_size)
    open_res = run_open_set_rejection(spectra_list, compound_map, bin_size=bin_size, n_splits=5)
    
    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    thresh_eval = evaluate_thresholds(closed_res["known_top1_scores"], open_res["unknown_top1_scores"], thresholds)
    
    known_stats = compute_distribution_stats(closed_res["known_top1_scores"])
    unknown_stats = compute_distribution_stats(open_res["unknown_top1_scores"])
    
    print_ascii_histogram(closed_res["known_top1_scores"], open_res["unknown_top1_scores"])
    
    print("\n" + "=" * 90)
    print(f"{'阈值 θ':<8} | {'TPR (已知接受率)':<16} | {'FPR (未知误接受率)':<18} | {'Precision':<10} | {'F1-Score':<10} | {'Youden J':<10}")
    print("=" * 90)
    for row in thresh_eval:
        print(f"{row['threshold']:<8.2f} | {row['tpr']*100:<15.2f}% | {row['fpr']*100:<17.2f}% | {row['precision']:<10.4f} | {row['f1_score']:<10.4f} | {row['youden_index']:<10.4f}")
    print("=" * 90)
    
    fig_path = os.path.join("scratch", "cosine_similarity_threshold_eval.png")
    plot_and_save_visualizations(closed_res["known_top1_scores"], open_res["unknown_top1_scores"], thresh_eval, fig_path)
    
    output_data = {
        "algorithm": "Cosine Similarity (Binned Matrix 0.02 Da, Square Root Intensity Weighting)",
        "bin_size_da": bin_size,
        "dataset": msp_file,
        "total_spectra": len(spectra_list),
        "total_compounds": len(compound_map),
        "closed_set_metrics": {
            "n_queries": closed_res["n_queries"],
            "top1_accuracy": round(closed_res["top1_acc"], 4),
            "top3_accuracy": round(closed_res["top3_acc"], 4),
            "mrr": round(closed_res["mrr"], 4)
        },
        "open_set_metrics": {
            "n_unknown_queries": open_res["n_unknown_queries"]
        },
        "known_top1_distribution": known_stats,
        "unknown_top1_distribution": unknown_stats,
        "threshold_evaluations": thresh_eval
    }
    
    out_json = os.path.join("scratch", "cosine_threshold_evaluation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n全量评估结果已成功保存到: {out_json}")

if __name__ == "__main__":
    main()
