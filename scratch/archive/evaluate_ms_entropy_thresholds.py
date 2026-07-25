import os
import sys
import json
import numpy as np
import ms_entropy
from ms_entropy import FlashEntropySearch, clean_spectrum
from rdkit import Chem
from pathlib import Path

def get_canonical_identifier(spec, spec_idx):
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

def load_dataset(msp_path):
    print(f"Loading MSP dataset from: {msp_path}")
    spectra_list = []
    compound_map = {} # id -> list of spectrum indices
    
    for idx, spec in enumerate(ms_entropy.read_one_spectrum(msp_path)):
        raw_peaks = spec.get("peaks")
        if not raw_peaks:
            continue
            
        peaks = []
        for p in raw_peaks:
            try:
                mz = float(p[0])
                intensity = float(p[1].replace(';', '')) if isinstance(p[1], str) else float(p[1])
                peaks.append([mz, intensity])
            except Exception:
                continue
                
        if not peaks:
            continue
            
        peaks_arr = clean_spectrum(np.array(peaks, dtype=np.float32))
        cid, can_smiles = get_canonical_identifier(spec, idx)
        name = spec.get("name") or spec.get("Name") or f"Spec_{idx}"
        
        item = {
            "spec_id": idx,
            "compound_id": cid,
            "name": name,
            "smiles": can_smiles or (smiles if 'smiles' in locals() else ""),
            "precursor_mz": -1.0,  # 使用 -1.0 执行纯粹 MS2 Spectral Entropy Search
            "peaks": peaks_arr
        }
        spectra_list.append(item)
        if cid not in compound_map:
            compound_map[cid] = []
        compound_map[cid].append(idx)
        
    print(f"Loaded {len(spectra_list)} spectra across {len(compound_map)} unique compounds.")
    return spectra_list, compound_map

def run_closed_set_matching(spectra_list, compound_map):
    """
    闭集已知化合物匹配测试 (Leave-One-Out):
    使用化合物有>=2条谱图的记录。
    循环取出1条作为query，排除自身后在全库查找。
    """
    print(f"\n--- 运行闭集已知化合物匹配测试 (Leave-One-Out) ---")
    
    searcher = FlashEntropySearch(mz_index_step=0.01)
    searcher.build_index(spectra_list, clean_spectra=False)
    
    multi_spec_compounds = {cid: idxs for cid, idxs in compound_map.items() if len(idxs) >= 2}
    query_spec_indices = []
    for idxs in multi_spec_compounds.values():
        query_spec_indices.extend(idxs)
        
    print(f"共有 {len(multi_spec_compounds)} 个化合物包含 >=2 条谱图，共计 {len(query_spec_indices)} 条查询质谱。")
    
    top1_correct_count = 0
    top3_correct_count = 0
    rr_list = []
    known_top1_scores = []
    
    for q_idx in query_spec_indices:
        q_spec = spectra_list[q_idx]
        q_cid = q_spec["compound_id"]
        
        res = searcher.search(
            precursor_mz=-1.0,
            peaks=q_spec["peaks"],
            ms2_tolerance_in_da=0.02,
            method="open",
            target="cpu"
        )
        scores = res.get("open_search", [])
            
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
            
        # 查找正确化合物首次出现排名
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
    
    print(f"闭集测试结果 (Queries={n_queries}):")
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

def run_open_set_rejection(spectra_list, compound_map, n_splits=5, seed=42):
    """
    开集全新化合物拒绝测试 (5-Fold Compound Split):
    随机将化合物分为 5 个 Fold。
    轮流将 1 个 Fold 中的化合物全部移出索引库作为“全新未知化合物查询集”。
    用剩余 4 个 Fold 建立索引库，搜索未知查询集，记录最高 Top-1 得分。
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
        
        db_searcher = FlashEntropySearch(mz_index_step=0.01)
        db_searcher.build_index(db_spectra, clean_spectra=False)
        
        for q_spec in query_spectra:
            res = db_searcher.search(
                precursor_mz=-1.0,
                peaks=q_spec["peaks"],
                ms2_tolerance_in_da=0.02,
                method="open",
                target="cpu"
            )
            scores = res.get("open_search", [])
            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
            unknown_top1_scores.append(max_score)
            
    print(f"开集测试完成 (未知化合物查询数={len(unknown_top1_scores)})。")
    return {
        "n_unknown_queries": len(unknown_top1_scores),
        "unknown_top1_scores": unknown_top1_scores
    }

def evaluate_thresholds(known_scores, unknown_scores, thresholds):
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
    print(" 相似度分布直方图对比 (Known vs Unknown Top-1 Scores)")
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

def main():
    msp_file = os.path.join("entropy", "positive.msp")
    if not os.path.exists(msp_file):
        msp_file = os.path.join("finetune", "../data_source", "阳性-含CanonicalSMILES-5类骨架(4).msp")
        
    spectra_list, compound_map = load_dataset(msp_file)
    
    closed_res = run_closed_set_matching(spectra_list, compound_map)
    open_res = run_open_set_rejection(spectra_list, compound_map, n_splits=5)
    
    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    thresh_eval = evaluate_thresholds(closed_res["known_top1_scores"], open_res["unknown_top1_scores"], thresholds)
    
    known_stats = compute_distribution_stats(closed_res["known_top1_scores"])
    unknown_stats = compute_distribution_stats(open_res["unknown_top1_scores"])
    
    print_ascii_histogram(closed_res["known_top1_scores"], open_res["unknown_top1_scores"])
    
    print("\n" + "=" * 80)
    print(f"{'阈值 θ':<8} | {'TPR (已知接受率)':<16} | {'FPR (未知误接受率)':<18} | {'Precision':<10} | {'F1-Score':<10} | {'Youden J':<10}")
    print("=" * 80)
    for row in thresh_eval:
        print(f"{row['threshold']:<8.2f} | {row['tpr']*100:<15.2f}% | {row['fpr']*100:<17.2f}% | {row['precision']:<10.4f} | {row['f1_score']:<10.4f} | {row['youden_index']:<10.4f}")
    print("=" * 80)
    
    output_data = {
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
    
    out_path = os.path.join("scratch", "ms_entropy_threshold_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {out_path}")

if __name__ == "__main__":
    main()
