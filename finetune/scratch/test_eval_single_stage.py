"""
测试评估脚本：评估 SingleStage FullyUnfrozen checkpoint 效果
模型路径: finetune/results_20260725_161200/best_model_20260725_155329_SingleStage_FullyUnfrozen_acc0.9712.pt
"""

import sys
from pathlib import Path
import csv
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)

# 确保项目根目录在 path 中
scratch_dir = Path(__file__).resolve().parent
finetune_dir = scratch_dir.parent
project_root = finetune_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from finetune.lib.dataset import prepare_finetune_dataset
from finetune.lib.evaluation import safe_auc, evaluate_positive_per_smiles


class CustomEncoder(nn.Module):
    """1D-CNN 质谱编码器 (使用 BatchNorm1d)"""
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        norm = torch.norm(embed, p=2, dim=1, keepdim=True)
        return embed / torch.clamp(norm, min=1e-3)


class MultiBranchHead(nn.Module):
    """双分支融合分类头 (Deep + Shallow Branch Fusion)"""
    def __init__(self, input_dim=256):
        super().__init__()
        self.deep_branch = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
        )
        self.shallow_branch = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, embed):
        d = self.deep_branch(embed)
        s = self.shallow_branch(embed)
        cat = torch.cat([d, s], dim=-1)
        logit = self.fusion(cat)
        return logit


class SingleStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomEncoder()
        self.classifier = MultiBranchHead()

    def forward(self, x):
        embed = self.encoder(x)
        logit = self.classifier(embed)
        return logit.squeeze(-1)


def evaluate_dataset(model, loader, device='cuda', threshold=0.5):
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    model.eval()
    model.to(dev)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_spec, batch_labels in loader:
            batch_spec = batch_spec.to(dev)
            logits = model(batch_spec)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().ravel())
            all_labels.extend(batch_labels.numpy().ravel())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(float)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = safe_auc(labels, probs)
    cm = confusion_matrix(labels, preds)

    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm,
        'probs': probs,
        'labels': labels,
        'preds': preds,
        'n_samples': int(len(labels)),
        'n_neg': int((labels == 0).sum()),
        'n_pos': int((labels == 1).sum()),
    }


def find_optimal_threshold(val_probs, val_labels, metric='f1'):
    best_th = 0.50
    best_score = -1.0
    records = []

    for th in np.arange(0.10, 0.91, 0.01):
        preds = (val_probs >= th).astype(float)
        acc = accuracy_score(val_labels, preds)
        prec = precision_score(val_labels, preds, zero_division=0)
        rec = recall_score(val_labels, preds, zero_division=0)
        f1 = f1_score(val_labels, preds, zero_division=0)
        cm = confusion_matrix(val_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        youden_j = tpr - fpr

        records.append({
            'threshold': round(float(th), 2),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'youden_j': float(youden_j)
        })

        score = f1 if metric == 'f1' else acc
        if score > best_score:
            best_score = score
            best_th = float(th)

    return best_th, best_score, records


def plot_evaluation_charts(train_res, val_res, test_res_05, test_res_opt, th_records, best_th, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. 混淆矩阵对比图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sets = [('Train (T=0.50)', train_res), ('Val (T=0.50)', val_res), (f'Test (T_opt={best_th:.2f})', test_res_opt)]
    
    for ax, (title, res) in zip(axes, sets):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
        ax.set_yticklabels(['Negative (0)', 'Positive (1)'])

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300)
    plt.close()

    # 2. ROC 曲线图
    plt.figure(figsize=(8, 6))
    for name, res, color in [('Train Set', train_res, 'blue'), ('Validation Set', val_res, 'green'), ('Test Set', test_res_05, 'red')]:
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})", color=color, lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Chance (AUC = 0.5000)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves (SingleStage FullyUnfrozen Model)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300)
    plt.close()

    # 3. 阈值寻优曲线
    ths = [r['threshold'] for r in th_records]
    accs = [r['accuracy'] for r in th_records]
    precs = [r['precision'] for r in th_records]
    recs = [r['recall'] for r in th_records]
    f1s = [r['f1'] for r in th_records]

    plt.figure(figsize=(9, 5.5))
    plt.plot(ths, accs, label='Accuracy', color='#2ca02c', lw=2)
    plt.plot(ths, precs, label='Precision', color='#ff7f0e', lw=2, linestyle='--')
    plt.plot(ths, recs, label='Recall', color='#d62728', lw=2, linestyle='--')
    plt.plot(ths, f1s, label='F1 Score', color='#1f77b4', lw=2.5)
    plt.axvline(x=best_th, color='black', linestyle=':', label=f'Optimal T = {best_th:.2f}')

    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Validation Threshold Search (0.10 ~ 0.90)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_search_curve.png', dpi=300)
    plt.close()


def main():
    model_path = project_root / "finetune" / "results_20260725_161200" / "best_model_20260725_155329_SingleStage_FullyUnfrozen_acc0.9712.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到目标模型 checkpoint: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print("SingleStage FullyUnfrozen 模型全量效果评估 (Evaluation Pipeline)")
    print("=" * 80)
    print(f"目标模型: {model_path.name}")
    print(f"完整路径: {model_path}")
    print(f"运行设备: {device.upper()}")

    # 1. 加载数据集
    print("\n[Step 1] 加载与划分数据集 (SMILES 阳性分组 + 阴性随机划分)...")
    train_loader, train_eval_loader, val_loader, test_loader, meta = prepare_finetune_dataset(
        batch_size=128, test_size=0.15, val_size=0.15, random_state=42, use_cuda=(device == 'cuda')
    )

    # 2. 实例化并加载模型
    print("\n[Step 2] 实例化神经网络模型并加载权重 checkpoint...")
    model = SingleStageModel()
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    print(f"  [OK] 权重加载成功! Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 3. 评估默认阈值 (T=0.50)
    print("\n[Step 3] 评估默认阈值 (T = 0.50)...")
    train_res = evaluate_dataset(model, train_eval_loader, device=device, threshold=0.50)
    val_res = evaluate_dataset(model, val_loader, device=device, threshold=0.50)
    test_res_05 = evaluate_dataset(model, test_loader, device=device, threshold=0.50)

    print(f"  训练集 (Train)   | Acc: {train_res['accuracy']:.2%} | Prec: {train_res['precision']:.2%} | Rec: {train_res['recall']:.2%} | F1: {train_res['f1']:.4f} | AUC: {train_res['auc']:.4f}")
    print(f"  验证集 (Val)     | Acc: {val_res['accuracy']:.2%} | Prec: {val_res['precision']:.2%} | Rec: {val_res['recall']:.2%} | F1: {val_res['f1']:.4f} | AUC: {val_res['auc']:.4f}")
    print(f"  测试集 (Test-0.5)| Acc: {test_res_05['accuracy']:.2%} | Prec: {test_res_05['precision']:.2%} | Rec: {test_res_05['recall']:.2%} | F1: {test_res_05['f1']:.4f} | AUC: {test_res_05['auc']:.4f}")

    # 4. 验证集决策阈值自动寻优
    print("\n[Step 4] 执行验证集决策阈值自动扫描寻优 (0.10 ~ 0.90)...")
    best_th, best_f1, th_records = find_optimal_threshold(val_res['probs'], val_res['labels'], metric='f1')
    print(f"  --> 最佳决策阈值: T_opt = {best_th:.2f} (验证集 F1: {best_f1:.4f})")

    # 5. 测试集最优阈值评估
    print(f"\n[Step 5] 使用最优阈值 (T_opt = {best_th:.2f}) 重新评估测试集...")
    test_res_opt = evaluate_dataset(model, test_loader, device=device, threshold=best_th)
    print(f"  测试集 (Test-Opt)| Acc: {test_res_opt['accuracy']:.2%} | Prec: {test_res_opt['precision']:.2%} | Rec: {test_res_opt['recall']:.2%} | F1: {test_res_opt['f1']:.4f} | AUC: {test_res_opt['auc']:.4f}")

    # 6. SMILES 级别的阳性识别效果
    print("\n[Step 6] 评估阳性化合物 (SMILES 级别) 识别覆盖率...")
    smiles_res_05 = evaluate_positive_per_smiles(
        meta['test_idx'], meta['smiles_all'], meta['y'], test_res_05['probs'], test_res_05['preds'], threshold=0.50
    )
    smiles_res_opt = evaluate_positive_per_smiles(
        meta['test_idx'], meta['smiles_all'], meta['y'], test_res_opt['probs'], test_res_opt['preds'], threshold=best_th
    )

    # 7. 导出结果与图表
    out_dir = model_path.parent / "test_evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 7] 导出评估数据表与对比图表至: {out_dir}")
    # 汇总 CSV
    with open(out_dir / 'summary_metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Threshold', 'Samples', 'Positive', 'Negative', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC', 'TN', 'FP', 'FN', 'TP'])
        for name, res in [('Train (T=0.50)', train_res), ('Validation (T=0.50)', val_res), ('Test (T=0.50)', test_res_05), (f'Test (T_opt={best_th:.2f})', test_res_opt)]:
            cm = res['confusion_matrix']
            writer.writerow([
                name, res['threshold'], res['n_samples'], res['n_pos'], res['n_neg'],
                f"{res['accuracy']:.4f}", f"{res['precision']:.4f}", f"{res['recall']:.4f}", f"{res['f1']:.4f}", f"{res['auc']:.4f}",
                cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            ])

    # 阈值扫频 CSV
    with open(out_dir / 'threshold_search_records.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'accuracy', 'precision', 'recall', 'f1', 'youden_j'])
        writer.writeheader()
        writer.writerows(th_records)

    # SMILES CSV
    if smiles_res_opt is not None:
        with open(out_dir / 'smiles_evaluation_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['SMILES', 'Num_Spectra', 'Mean_Predicted_Prob', 'Ground_Truth_Label', 'Predicted_Class', 'Is_Correct'])
            for smi, count, prob, label, pred in zip(
                smiles_res_opt['smiles_names'],
                smiles_res_opt['smiles_counts'],
                smiles_res_opt['smiles_probs'],
                smiles_res_opt['smiles_labels'],
                smiles_res_opt['smiles_preds']
            ):
                writer.writerow([smi, count, f"{prob:.4f}", int(label), int(pred), int(label == pred)])

    # 绘图
    plot_evaluation_charts(train_res, val_res, test_res_05, test_res_opt, th_records, best_th, out_dir)

    print("\n" + "=" * 80)
    print("评估全部完成！测试结果文件汇总：")
    print(f"  - 指标汇总表: [summary_metrics.csv](file:///{out_dir / 'summary_metrics.csv'})")
    print(f"  - 阈值扫描表: [threshold_search_records.csv](file:///{out_dir / 'threshold_search_records.csv'})")
    print(f"  - 化合物识别表: [smiles_evaluation_results.csv](file:///{out_dir / 'smiles_evaluation_results.csv'})")
    print(f"  - 混淆矩阵图: [confusion_matrices.png](file:///{out_dir / 'confusion_matrices.png'})")
    print(f"  - ROC 曲线图: [roc_curves.png](file:///{out_dir / 'roc_curves.png'})")
    print(f"  - 阈值寻优图: [threshold_search_curve.png](file:///{out_dir / 'threshold_search_curve.png'})")
    print("=" * 80)


if __name__ == '__main__':
    main()
