"""
大麻素 vs 非大麻素 二分类模型可视化
加载已训练好的模型: binary_classifier_20260716_155622.pt
"""

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, silhouette_score
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ==================== 模型定义 ====================

class SpectrumEncoder(nn.Module):
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
        return F.normalize(embed, p=2, dim=1)


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, input_dim=256):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, return_embed=False):
        embed = self.encoder(x)
        logit = self.classifier(embed)
        if return_embed:
            return logit.squeeze(-1), embed
        return logit.squeeze(-1)


# ==================== 数据处理 ====================

def parse_msp_with_smiles(msp_file, min_peaks=5):
    with open(msp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    compounds = []
    current_comp = None
    in_peaks = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('Name:'):
            if current_comp is not None and 'peaks' in current_comp:
                if len(current_comp['peaks']) >= min_peaks:
                    compounds.append(current_comp)
            current_comp = {'name': line.split(':', 1)[1].strip(), 'smiles': '', 'peaks': []}
            in_peaks = False
        elif line.startswith('SMILES:'):
            if current_comp is not None:
                current_comp['smiles'] = line.split(':', 1)[1].strip()
        elif line.startswith('Num peaks:'):
            in_peaks = True
        elif in_peaks and line:
            parts = line.replace(';', '').replace('\t', ' ').split()
            if len(parts) >= 2:
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
                    if mz > 0 and intensity > 0:
                        current_comp['peaks'].append((mz, intensity))
                except ValueError:
                    in_peaks = False
    
    if current_comp is not None and 'peaks' in current_comp:
        if len(current_comp['peaks']) >= min_peaks:
            compounds.append(current_comp)
    return compounds


def peaks_to_vector(peaks, mz_min=40, mz_max=600):
    dim = mz_max - mz_min + 1
    vec = np.zeros(dim)
    for mz, intensity in peaks:
        if mz_min <= mz <= mz_max:
            idx = int(round(mz - mz_min))
            if 0 <= idx < dim:
                vec[idx] += intensity
    return vec


def preprocess_spectra(spectra):
    tic = spectra.sum(axis=1, keepdims=True)
    spectra = spectra / (tic + 1e-8)
    return np.sqrt(spectra)


# ==================== 评估与可视化 ====================

def evaluate_model(model, test_loader, device, temperature=1.0):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch_spec, batch_labels in test_loader:
            batch_spec = batch_spec.to(device)
            logits = model(batch_spec)
            probs = torch.sigmoid(logits / temperature)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n{'='*60}")
    print("测试集评估结果")
    print(f"{'='*60}")
    print(f"  准确率: {acc:.2%} | 精确率: {prec:.2%} | 召回率: {rec:.2%} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  混淆矩阵:\n    TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}\n    FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 
            'confusion_matrix': cm, 'probs': all_probs, 'labels': all_labels, 'preds': all_preds}


def analyze_embeddings(model, loader, device, output_dir):
    """特征空间分析：t-SNE + pca"""
    model.eval()
    embeddings, labels = [], []
    
    with torch.no_grad():
        for batch_spec, batch_labels in loader:
            batch_spec = batch_spec.to(device)
            _, embed = model(batch_spec, return_embed=True)
            embeddings.append(embed.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"\n  嵌入矩阵: {embeddings.shape}, 阳性: {np.sum(labels==1)}, 阴性: {np.sum(labels==0)}")
    
    # 高维评估
    sil_score = silhouette_score(embeddings, labels)
    print(f"  Silhouette Score: {sil_score:.4f}")
    
    # 降维
    n_samples = min(1500, len(embeddings))
    np.random.seed(42)
    idx = np.random.choice(len(embeddings), n_samples, replace=False)
    emb_sample, labels_sample = embeddings[idx], labels[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # t-SNE
    print("  计算 t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    emb_tsne = tsne.fit_transform(emb_sample)
    tsne_acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), emb_tsne, labels_sample, cv=5).mean()
    
    ax = axes[0]
    scatter = ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels_sample, cmap='coolwarm', alpha=0.6, s=10)
    ax.set_title(f't-SNE (KNN Acc={tsne_acc:.3f}, Sil={silhouette_score(emb_tsne, labels_sample):.3f})')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    # pca
    print("  计算 pca...")
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(emb_sample)
    pca_acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), emb_pca, labels_sample, cv=5).mean()
    
    ax = axes[1]
    scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels_sample, cmap='coolwarm', alpha=0.6, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'pca (KNN Acc={pca_acc:.3f}, Sil={silhouette_score(emb_pca, labels_sample):.3f})')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'embedding_analysis.png'), dpi=150)
    plt.show()
    
    return sil_score


def grad_cam_visualization(model, spectrum, output_dir):
    """Grad-CAM可视化"""
    model.eval()
    device = next(model.parameters()).device
    
    if torch.is_tensor(spectrum):
        spectrum_np = spectrum.cpu().numpy() if spectrum.is_cuda else spectrum.numpy()
    else:
        spectrum_np = np.array(spectrum)
    spectrum_np = spectrum_np.squeeze()
    
    spectrum_tensor = torch.tensor(spectrum_np, dtype=torch.float32).to(device)
    spectrum_tensor.requires_grad_()
    
    logits = model(spectrum_tensor.unsqueeze(0))
    model.zero_grad()
    logits[0].backward()
    
    gradients = spectrum_tensor.grad
    if gradients is not None:
        weights = gradients.mean(dim=0, keepdim=True)
        cam = (weights * spectrum_tensor).sum(dim=0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam.detach().cpu().numpy()
    else:
        cam = np.zeros_like(spectrum_np)
    
    mz_axis = np.linspace(40, 600, len(spectrum_np))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(mz_axis, spectrum_np, 'b-', alpha=0.7, linewidth=1)
    axes[0].set_xlabel('m/z'); axes[0].set_ylabel('Intensity')
    axes[0].set_title('Original Spectrum'); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mz_axis, spectrum_np, 'b-', alpha=0.3, linewidth=0.8)
    axes[1].fill_between(mz_axis, 0, spectrum_np * cam, color='red', alpha=0.5, label='Important Regions')
    axes[1].set_xlabel('m/z'); axes[1].set_ylabel('Intensity')
    axes[1].set_title('heatmap - Red regions show important m/z')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'grad_cam.png'), dpi=150)
    plt.show()


def plot_summary(eval_results, output_dir):
    """结果汇总图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC
    fpr, tpr, _ = roc_curve(eval_results['labels'], eval_results['probs'])
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC={eval_results["auc"]:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('FPR'); axes[0, 0].set_ylabel('TPR')
    axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # 混淆矩阵
    cm = eval_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非大麻素', '大麻素'],
                yticklabels=['非大麻素', '大麻素'], ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix (Acc={eval_results["accuracy"]:.2%})')
    
    # 概率分布
    pos_probs = eval_results['probs'][eval_results['labels'] == 1]
    neg_probs = eval_results['probs'][eval_results['labels'] == 0]
    axes[1, 0].hist(neg_probs, bins=30, alpha=0.6, label='阴性', color='blue', edgecolor='black')
    axes[1, 0].hist(pos_probs, bins=30, alpha=0.6, label='阳性', color='red', edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Probability'); axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Probability Distribution'); axes[1, 0].legend()
    
    # 性能指标
    metrics = {'Acc': eval_results['accuracy'], 'Prec': eval_results['precision'],
               'Rec': eval_results['recall'], 'F1': eval_results['f1'], 'AUC': eval_results['auc']}
    colors = ['#2ecc71' if v > 0.9 else '#f39c12' if v > 0.8 else '#e74c3c' for v in metrics.values()]
    axes[1, 1].bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black')
    axes[1, 1].set_ylim(0, 1.1); axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Summary')
    for bar, val in zip(axes[1, 1].patches, metrics.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'evaluation_summary.png'), dpi=150)
    plt.show()


# ==================== 主程序 ====================

if __name__ == "__main__":
    base_dir = Path(r"D:\DL\cann\建模")
    
    # 查找所有可能的模型位置
    possible_paths = [
        base_dir / "binary_classifier_20260716_155622.pt",
        base_dir / "binary_classifier_enhanced.pt",
        base_dir / "binary_classification_enhanced" / "binary_classifier_enhanced.pt",
        base_dir / "visualization" / "binary_classifier_*.pt",
    ]
    
    # 查找所有binary_classifier开头的pt文件
    all_models = list(base_dir.glob("binary_classifier_*.pt"))
    all_models.extend(list((base_dir / "binary_classification_enhanced").glob("binary_classifier_*.pt"))) if (base_dir / "binary_classification_enhanced").exists() else None
    all_models.extend(list((base_dir / "visualization").glob("binary_classifier_*.pt"))) if (base_dir / "visualization").exists() else None
    
    # 去重并按修改时间排序
    all_models = sorted(set(all_models), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if all_models:
        model_path = all_models[0]
        print(f"找到模型: {model_path.name} (修改时间: {datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        # 尝试直接查找指定文件
        specified_path = base_dir / "binary_classifier_20260716_155622.pt"
        if specified_path.exists():
            model_path = specified_path
            print(f"使用指定模型: {model_path.name}")
        else:
            print(f"错误: 未找到二分类模型文件")
            print(f"搜索路径: {base_dir}")
            print(f"请确保模型文件存在，或修改代码中的路径")
            exit(1)
    
    # 创建输出目录（带时间戳）
    output_dir = base_dir / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n输出: {output_dir}")
    
    # ===== 1. 加载数据 =====
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)
    
    pos_compounds = parse_msp_with_smiles(str(base_dir / "阳性-含CanonicalSMILES-5类骨架.msp"))
    neg_compounds = parse_msp_with_smiles(str(base_dir / "阴性.msp"))
    
    X_pos = preprocess_spectra(np.array([peaks_to_vector(c['peaks']) for c in pos_compounds]))
    X_neg = preprocess_spectra(np.array([peaks_to_vector(c['peaks']) for c in neg_compounds]))
    pos_smiles = np.array([c['smiles'] if c['smiles'] else c['name'] for c in pos_compounds])
    
    print(f"  阳性: {len(X_pos)} 谱图, {len(np.unique(pos_smiles))} 种化合物")
    print(f"  阴性: {len(X_neg)} 谱图")
    
    # ===== 2. 划分测试集（与训练时一致） =====
    print("\n" + "="*60)
    print("划分测试集")
    print("="*60)
    
    np.random.seed(42)
    pos_indices = np.arange(len(X_pos))
    neg_indices = np.arange(len(X_pos), len(X_pos) + len(X_neg))
    
    # 按SMILES划分阳性
    pos_unique_smiles = np.unique(pos_smiles)
    shuffled_smiles = pos_unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)
    n_pos_test = max(1, int(len(pos_unique_smiles) * 0.15))
    pos_test_smiles = set(shuffled_smiles[:n_pos_test])
    pos_test_idx = [i for i in pos_indices if pos_smiles[i] in pos_test_smiles]
    
    # 随机划分阴性
    neg_shuffled = neg_indices.copy()
    np.random.shuffle(neg_shuffled)
    n_neg_test = int(len(neg_indices) * 0.15)
    neg_test_idx = neg_shuffled[:n_neg_test].tolist()
    
    test_idx = np.array(pos_test_idx + neg_test_idx)
    np.random.shuffle(test_idx)
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    print(f"  测试集: {len(test_idx)} 谱图 (阳性: {(y[test_idx]==1).sum()})")
    
    # ===== 3. 创建DataLoader =====
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X[test_idx], dtype=torch.float32),
                      torch.tensor(y[test_idx], dtype=torch.float32)),
        batch_size=128, shuffle=False
    )
    
    # ===== 4. 加载模型 =====
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    
    checkpoint = torch.load(str(model_path), map_location=device)
    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256).to(device)
    model = BinaryClassifier(encoder, input_dim=256).to(device)
    
    # 加载权重
    if 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        print(f"  ✓ 加载编码器权重")
    else:
        print(f"  ⚠ 未找到编码器权重，使用随机初始化")
    
    if 'classifier_state_dict' in checkpoint:
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
        print(f"  ✓ 加载分类器权重")
    else:
        print(f"  ⚠ 未找到分类器权重，使用随机初始化")
    
    temperature = checkpoint.get('temperature', 1.0)
    model.eval()
    print(f"  温度参数: {temperature:.4f}")
    
    # ===== 5. 评估 =====
    print("\n" + "="*60)
    print("评估模型")
    print("="*60)
    
    eval_results = evaluate_model(model, test_loader, device, temperature)
    
    # ===== 6. 特征空间分析 =====
    print("\n" + "="*60)
    print("特征空间分析")
    print("="*60)
    
    analyze_embeddings(model, test_loader, device, output_dir)
    
    # ===== 7. heatmap =====
    print("\n" + "="*60)
    print("heatmap 可视化")
    print("="*60)
    
    # 选择阳性样本
    pos_test = [i for i in test_idx if y[i] == 1]
    if pos_test:
        sample_idx = pos_test[0]
        sample_label = '大麻素'
    else:
        sample_idx = test_idx[0]
        sample_label = '非大麻素'
    
    sample_spectrum = torch.tensor(X[sample_idx], dtype=torch.float32)
    print(f"  样本标签: {sample_label}")
    grad_cam_visualization(model, sample_spectrum, output_dir)
    
    # ===== 8. 结果汇总 =====
    print("\n" + "="*60)
    print("生成结果汇总")
    print("="*60)
    
    plot_summary(eval_results, output_dir)
    
    # ===== 9. 导出数据 =====
    print("\n" + "="*60)
    print("导出数据")
    print("="*60)
    
    # 导出评估结果
    pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
        'Value': [eval_results['accuracy'], eval_results['precision'], 
                  eval_results['recall'], eval_results['f1'], eval_results['auc']]
    }).to_excel(output_dir / 'metrics.xlsx', index=False)
    
    # 导出预测结果
    pd.DataFrame({
        'True_Label': eval_results['labels'],
        'Pred_Prob': eval_results['probs'],
        'Pred_Label': eval_results['preds']
    }).to_excel(output_dir / 'predictions.xlsx', index=False)
    
    # 导出混淆矩阵
    cm_df = pd.DataFrame(eval_results['confusion_matrix'], 
                         index=['真实:非大麻素', '真实:大麻素'],
                         columns=['预测:非大麻素', '预测:大麻素'])
    cm_df.to_excel(output_dir / 'confusion_matrix.xlsx')
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")
    print(f"  输出目录: {output_dir}")
    print(f"  结果图: {output_dir / 'evaluation_summary.png'}")
    print(f"  特征图: {output_dir / 'embedding_analysis.png'}")
    print(f"  heatmap: {output_dir / 'grad_cam.png'}")
    print(f"  Excel数据:")
    print(f"    - {output_dir / 'metrics.xlsx'}")
    print(f"    - {output_dir / 'predictions.xlsx'}")
    print(f"    - {output_dir / 'confusion_matrix.xlsx'}")