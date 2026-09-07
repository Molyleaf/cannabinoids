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
    Normalize compound identity based on unified identifier.
    Priority: Parse SMILES -> InChIKey / Canonical SMILES;
    Fallback: Normalized Name when SMILES is unavailable.
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
    Parse mass spectrum peaks, handle special characters in intensities,
    and clean using ms_entropy.clean_spectrum.
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
    print(f"Loading MSP dataset: {msp_path}")
    spectra_list = []
    compound_map = {}  # compound_id -> list of spectrum indices

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

    print(f"Successfully loaded {len(spectra_list)} mass spectra, covering {len(compound_map)} unique compounds.")
    return spectra_list, compound_map


class FastCosineSearcher:
    """
    Efficient vectorized mass spectrum cosine similarity searcher.
    Uses standard 0.02 Da m/z discrete binning and square root intensity weighting.
    Uses SciPy sparse matrix for accelerated batch matrix dot product computation.
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
    Closed-set known compound matching test (Leave-One-Out):
    Goal: When a compound has multiple spectra, remove one and search it against the library.
           Can it correctly find other spectra of the same compound?
    Hit definition: The returned Top-K list contains any spectrum with the same compound_id.
    """
    print(f"\n--- Running Closed-Set Known Compound Matching Test (Leave-One-Out) ---")

    searcher = FastCosineSearcher(bin_size=bin_size)
    searcher.build_index(spectra_list)

    multi_spec_compounds = {cid: idxs for cid, idxs in compound_map.items() if len(idxs) >= 2}
    query_spec_indices = []
    for idxs in multi_spec_compounds.values():
        query_spec_indices.extend(idxs)

    print(f"Found {len(multi_spec_compounds)} compounds with >=2 spectra, totaling {len(query_spec_indices)} known query spectra.")

    query_peaks_list = [spectra_list[idx]["peaks"] for idx in query_spec_indices]
    t0 = time.time()
    all_scores_mat = searcher.search_batch(query_peaks_list)
    t1 = time.time()
    print(f"Batch matrix cosine similarity search completed, time: {t1-t0:.4f}s")

    top1_correct_count = 0
    top3_correct_count = 0
    rr_list = []
    known_top1_scores = []

    for q_i, q_idx in enumerate(query_spec_indices):
        q_spec = spectra_list[q_idx]
        q_cid = q_spec["compound_id"]

        scores = all_scores_mat[q_i]

        # Sort descending and exclude self (Leave-One-Out: idx != q_idx)
        sorted_indices = np.argsort(scores)[::-1]
        filtered_candidates = [idx for idx in sorted_indices if idx != q_idx]

        if not filtered_candidates:
            rr_list.append(0.0)
            known_top1_scores.append(0.0)
            continue

        top1_idx = filtered_candidates[0]
        top1_score = float(scores[top1_idx])
        top1_cid = spectra_list[top1_idx]["compound_id"]

        # Check if Top-1 hits the correct compound
        is_top1_hit = (top1_cid == q_cid)
        if is_top1_hit:
            top1_correct_count += 1

        # Find the first occurrence of the same compound
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

    print(f"Closed-set test results (Known Queries={n_queries}):")
    print(f"  Top-1 Accuracy : {top1_acc*100:.2f}% ({top1_correct_count}/{n_queries})")
    print(f"  Top-3 Accuracy : {top3_acc*100:.2f}% ({top3_correct_count}/{n_queries})")
    print(f"  MRR (Mean Reciprocal Rank) : {mrr:.4f}")

    return {
        "n_queries": n_queries,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "mrr": mrr,
        "known_top1_scores": known_top1_scores
    }


def run_open_set_rejection(spectra_list, compound_map, bin_size=0.02, n_splits=5, seed=42):
    """
    Open-set novel compound rejection test (5-Fold Compound Split):
    Goal: Remove all spectra of a compound from the library as a novel unknown,
          search it against the remaining library, and evaluate the Top-1 score.
    """
    print(f"\n--- Running Open-Set Novel Compound Rejection Test (5-Fold Compound Split) ---")

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

    print(f"Open-set test complete (Unknown compound Queries={len(unknown_top1_scores)}).")
    return {
        "n_unknown_queries": len(unknown_top1_scores),
        "unknown_top1_scores": unknown_top1_scores
    }


def evaluate_thresholds(known_scores, unknown_scores, thresholds):
    """
    Evaluate open-set rejection and known-set acceptance performance under different similarity thresholds θ.
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
        "median": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4)
    }


def print_ascii_histogram(known_scores, unknown_scores, bins=10):
    counts_k, bin_edges = np.histogram(known_scores, bins=bins, range=(0.0, 1.0))
    counts_u, _ = np.histogram(unknown_scores, bins=bins, range=(0.0, 1.0))

    print("\n" + "=" * 70)
    print(" Cosine Similarity Score Distribution Histogram (Known Queries vs Unknown Queries Top-1 Scores)")
    print("=" * 70)
    print(f"{'Score Range':<15} | {'Known Queries':<24} | {'Unknown Queries':<24}")
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

    # Subplot 1: Distribution histogram
    plt.subplot(1, 3, 1)
    plt.hist(known_scores, bins=20, range=(0, 1), alpha=0.6, color='blue', label='Known Queries (Top-1)', density=True)
    plt.hist(unknown_scores, bins=20, range=(0, 1), alpha=0.6, color='orange', label='Unknown Queries (Top-1)', density=True)
    plt.title("Cosine Similarity Score Distribution")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: ROC curve
    plt.subplot(1, 3, 2)
    thresholds_dense = np.linspace(0.0, 1.0, 201)
    known_arr = np.array(known_scores)
    unknown_arr = np.array(unknown_scores)
    tprs = [np.mean(known_arr >= t) for t in thresholds_dense]
    fprs = [np.mean(unknown_arr >= t) for t in thresholds_dense]
    # AUC computation with NumPy version compatibility
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

    # Subplot 3: TPR/FPR vs Threshold
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
    print(f"\nVisualization chart successfully saved to: {output_fig_path}")


def main():
    msp_file = os.path.join("entropy", "positive.msp")
    if not os.path.exists(msp_file):
        msp_file = os.path.join("scratch", "../data_source", "MONA_GCMS-18914.MSP")

    spectra_list, compound_map = load_dataset(msp_file)

    bin_size = 0.02  # 0.02 Da m/z discrete bin tolerance
    closed_res = run_closed_set_matching(spectra_list, compound_map, bin_size=bin_size)
    open_res = run_open_set_rejection(spectra_list, compound_map, bin_size=bin_size, n_splits=5)

    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    thresh_eval = evaluate_thresholds(closed_res["known_top1_scores"], open_res["unknown_top1_scores"], thresholds)

    known_stats = compute_distribution_stats(closed_res["known_top1_scores"])
    unknown_stats = compute_distribution_stats(open_res["unknown_top1_scores"])

    print_ascii_histogram(closed_res["known_top1_scores"], open_res["unknown_top1_scores"])

    print("\n" + "=" * 90)
    print(f"{'Threshold θ':<8} | {'TPR (Known Accept)':<16} | {'FPR (Unknown Misaccept)':<18} | {'Precision':<10} | {'F1-Score':<10} | {'Youden J':<10}")
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
    print(f"\nFull evaluation results successfully saved to: {out_json}")


if __name__ == "__main__":
    main()