import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

def safe_auc(labels, probs):
    """数值安全的 AUC 计算"""
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.5


def evaluate_and_record_predictions(model, data_loader, sample_names=None, device='cuda', threshold=0.5):
    print("  [DEBUG] Entering evaluate_and_record_predictions", flush=True)
    """
    标准的 PyTorch 评估流程：计算预测概率与评估指标
    """
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    model.eval()
    model.to(dev)
    
    all_probs, all_labels, all_preds = [], [], []
    
    print("  [DEBUG] Starting forward pass loop", flush=True)
    with torch.no_grad():
        for batch_spec, batch_labels in data_loader:
            batch_spec = batch_spec.to(dev)
            batch_labels = batch_labels.to(dev)
            logits = model(batch_spec)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy().ravel())
            all_labels.extend(batch_labels.cpu().numpy().ravel())
            all_preds.extend((probs >= threshold).float().cpu().numpy().ravel())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    print("  [DEBUG] Finished forward pass loop, calculating metrics", flush=True)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = safe_auc(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    df_pred = None
    
    print("  [DEBUG] Returning from evaluate_and_record_predictions", flush=True)
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm,
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
        'df_pred': df_pred,
        'n_samples': int(len(all_labels)),
        'n_neg': int((all_labels == 0).sum()),
        'n_pos': int((all_labels == 1).sum()),
    }


def evaluate_model(model, data_loader, device='cuda', threshold=0.5, sample_names=None):
    return evaluate_and_record_predictions(model, data_loader, sample_names=sample_names, device=device, threshold=threshold)


def evaluate_positive_per_smiles(test_indices, smiles_all, labels_all, probs, preds, threshold=0.5):
    """按 SMILES 聚合评估化合物级别的识别率"""
    test_labels = labels_all[test_indices]
    test_smiles = smiles_all[test_indices]
    
    pos_mask = (test_labels == 1)
    if pos_mask.sum() == 0:
        print("\n  [WARN] 测试集中无阳性样本，跳过 SMILES 聚合评估")
        return None
    
    pos_smiles = test_smiles[pos_mask]
    pos_probs = probs[pos_mask]
    pos_labels = test_labels[pos_mask]
    
    smiles_to_indices = defaultdict(list)
    for i, smi in enumerate(pos_smiles):
        smiles_to_indices[smi].append(i)
        
    smiles_probs, smiles_labels, smiles_preds = [], [], []
    smiles_names, smiles_counts = [], []
    
    for smi, idx_list in smiles_to_indices.items():
        mean_prob = np.mean(pos_probs[idx_list])
        smiles_probs.append(mean_prob)
        smiles_labels.append(pos_labels[idx_list[0]])
        smiles_preds.append(1.0 if mean_prob >= threshold else 0.0)
        smiles_names.append(smi)
        smiles_counts.append(len(idx_list))
        
    smiles_probs = np.array(smiles_probs)
    smiles_labels = np.array(smiles_labels)
    smiles_preds = np.array(smiles_preds)
    
    n_correct = int((smiles_preds == smiles_labels).sum())
    n_total = len(smiles_labels)
    acc = n_correct / n_total if n_total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"阳性样本-按 SMILES 聚合评估 (化合物级别, 判定阈值={threshold:.2f})")
    print(f"{'='*60}")
    print(f"  阳性唯一 SMILES 数: {len(smiles_to_indices)}")
    print(f"  正确识别: {n_correct}/{n_total} 种化合物 ({acc:.2%})")
    print(f"  漏检:     {n_total - n_correct}/{n_total} 种化合物")
    
    print("  [DEBUG] Returning from evaluate_and_record_predictions", flush=True)
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'n_smiles': int(len(smiles_to_indices)),
        'n_correct': n_correct,
        'n_missed': n_total - n_correct,
        'mean_prob': float(smiles_probs.mean()),
        'smiles_names': smiles_names,
        'smiles_probs': smiles_probs,
        'smiles_labels': smiles_labels,
        'smiles_preds': smiles_preds,
        'smiles_counts': smiles_counts,
    }


import csv

def export_results_to_excel(
    history, model, train_results, val_results, test_results,
    smiles_results, meta, output_dir
):
    """导出评估结果至 CSV 文件（纯 Python 标准库实现，带 BOM 格式供 Excel 直接无乱码打开，彻底消除 PyCharm 变量预览插件冲突）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = output_dir
    
    # 1. 保存 Training History
    with open(csv_dir / 'training_history.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Train_Acc', 'Val_Acc', 'Train_AUC', 'Val_AUC'])
        max_len = max(len(history['train_loss']), len(history['val_loss']))
        for i in range(max_len):
            writer.writerow([
                i + 1,
                history['train_loss'][i] if i < len(history['train_loss']) else '',
                history['val_loss'][i] if i < len(history['val_loss']) else '',
                history['train_acc'][i] if i < len(history['train_acc']) else '',
                history['val_acc'][i] if i < len(history['val_acc']) else '',
                history['train_auc'][i] if i < len(history['train_auc']) else '',
                history['val_auc'][i] if i < len(history['val_auc']) else '',
            ])
            
    # 2. 保存 Summary Metrics
    with open(csv_dir / 'summary_metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Samples', 'Positive', 'Negative', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'TN', 'FP', 'FN', 'TP'])
        for name, res in [('Train', train_results), ('Validation', val_results), ('Test', test_results)]:
            cm = res['confusion_matrix']
            writer.writerow([
                name, res['n_samples'], res['n_pos'], res['n_neg'],
                res['accuracy'], res['precision'], res['recall'], res['f1'], res['auc'],
                cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            ])
            
    # Helper for predictions
    def get_sample_names(fallback_idx, fallback_key):
        sample_names = meta.get(fallback_key)
        if sample_names is None:
            sample_names_all = meta.get('sample_names_all')
            if sample_names_all is None:
                compounds_pos = meta['pos_compounds']
                compounds_neg = meta['neg_compounds']
                sample_names_all = np.array([c['name'] for c in compounds_pos] + [c['name'] for c in compounds_neg])
            sample_names = sample_names_all[fallback_idx]
        return sample_names

    # 3. 保存 各集合预测明细 及 错误样本汇总
    all_errors_rows = []
    for name, res, idx, key in [
        ('Train', train_results, meta['train_idx'], 'train_sample_names'),
        ('Validation', val_results, meta['val_idx'], 'val_sample_names'),
        ('Test', test_results, meta['test_idx'], 'test_sample_names')
    ]:
        s_names = get_sample_names(idx, key)
        fname = f"{name.lower()}_predictions.csv"
        with open(csv_dir / fname, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_Name', 'True_Label', 'Pred_Prob', 'Pred_Label', 'Correct'])
            for sname, t_lbl, prob, p_lbl in zip(s_names, res['labels'], res['probs'], res['preds']):
                is_correct = (t_lbl == p_lbl)
                writer.writerow([sname, t_lbl, prob, p_lbl, is_correct])
                if not is_correct:
                    err_type = 'False Positive (FP)' if t_lbl == 0 and p_lbl == 1 else 'False Negative (FN)'
                    all_errors_rows.append([sname, t_lbl, prob, p_lbl, is_correct, name, err_type])
                    
    with open(csv_dir / 'error_predictions.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Name', 'True_Label', 'Pred_Prob', 'Pred_Label', 'Correct', 'Dataset', 'Error_Type'])
        writer.writerows(all_errors_rows)
        
    # 4. 保存 SMILES 级别结果
    with open(csv_dir / 'smiles_level_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if smiles_results is not None:
            writer.writerow(['SMILES', 'Spectra_Count', 'True_Label', 'Mean_Prob', 'Pred_Label', 'Correct'])
            for smi, cnt, t_lbl, m_prob, p_lbl in zip(
                smiles_results['smiles_names'],
                smiles_results['smiles_counts'],
                smiles_results['smiles_labels'],
                smiles_results['smiles_probs'],
                smiles_results['smiles_preds']
            ):
                writer.writerow([smi, cnt, t_lbl, m_prob, p_lbl, t_lbl == p_lbl])
        else:
            writer.writerow(['Note'])
            writer.writerow(['No positive samples in test set'])
            
    print(f"  [OK] 评估表格 CSV 已全部导出至目录: {csv_dir}", flush=True)
    return csv_dir


def plot_comprehensive_results(history, train_results, val_results, test_results, smiles_results, output_dir):
    """绘制全面性能图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', lw=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='orange', lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['train_acc'], label='Train Acc', color='blue', lw=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='orange', lw=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    t_m = [train_results['accuracy'], train_results['precision'], train_results['recall'], train_results['f1'], train_results['auc']]
    v_m = [val_results['accuracy'], val_results['precision'], val_results['recall'], val_results['f1'], val_results['auc']]
    te_m = [test_results['accuracy'], test_results['precision'], test_results['recall'], test_results['f1'], test_results['auc']]
    
    x = np.arange(len(metrics))
    width = 0.25
    ax3.bar(x - width, t_m, width, label='Train', color='blue', alpha=0.7)
    ax3.bar(x, v_m, width, label='Val', color='orange', alpha=0.7)
    ax3.bar(x + width, te_m, width, label='Test', color='green', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=30)
    ax3.set_ylim(0, 1.1)
    ax3.set_title('Performance Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(2, 3, 4)
    fpr_tr, tpr_tr, _ = roc_curve(train_results['labels'], train_results['probs'])
    ax4.plot(fpr_tr, tpr_tr, color='blue', lw=2, label=f'Train AUC={train_results["auc"]:.4f}')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax4.set_title('Train Set ROC')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    fpr_v, tpr_v, _ = roc_curve(val_results['labels'], val_results['probs'])
    ax5.plot(fpr_v, tpr_v, color='orange', lw=2, label=f'Val AUC={val_results["auc"]:.4f}')
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax5.set_title('Val Set ROC')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    fpr_te, tpr_te, _ = roc_curve(test_results['labels'], test_results['probs'])
    ax6.plot(fpr_te, tpr_te, color='green', lw=2, label=f'Test AUC={test_results["auc"]:.4f}')
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax6.set_title('Test Set ROC')
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'comprehensive_results.png'
    plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    plt.close('all')
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6.5))
    ax_roc.plot(fpr_tr, tpr_tr, color='#1f77b4', lw=2.5, label=f'Train (AUC = {train_results["auc"]:.4f})')
    ax_roc.plot(fpr_v, tpr_v, color='#ff7f0e', lw=2.5, label=f'Val (AUC = {val_results["auc"]:.4f})')
    ax_roc.plot(fpr_te, tpr_te, color='#2ca02c', lw=2.5, label=f'Test (AUC = {test_results["auc"]:.4f})')
    ax_roc.plot([0, 1], [0, 1], 'gray', linestyle='--', lw=1.2, label='Random (AUC = 0.5000)')
    ax_roc.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    ax_roc.legend(loc='lower right', frameon=True)
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    
    roc_compare_path = output_dir / 'roc_curves_comparison.png'
    plt.savefig(str(roc_compare_path), dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  [OK] 评估图表已保存至: {output_dir}", flush=True)


def plot_confusion_matrices(train_results, val_results, test_results, output_dir):
    """绘制混淆矩阵热力图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    class_names = ['Non-Cannabinoid', 'Cannabinoid']
    results_map = {'Train': train_results, 'Val': val_results, 'Test': test_results}
    
    for idx, (name, res) in enumerate(results_map.items()):
        cm = res['confusion_matrix']
        ax = axes[idx]
        cm_ratio = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        labels = []
        for i in range(2):
            for j in range(2):
                labels.append(f"{cm[i, j]}\n({cm_ratio[i, j]:.1%})")
        labels = np.array(labels).reshape(2, 2)
        
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=True, ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1, linecolor='#e2e2e2'
        )
        ax.set_title(f'{name} Set (n={res["n_samples"]})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        if idx == 0:
            ax.set_ylabel('True Label')
            
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  [OK] 混淆矩阵热力图已保存: {cm_path}", flush=True)
