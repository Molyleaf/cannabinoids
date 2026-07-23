"""
大麻素 vs 非大麻素 二分类模型 - Grad-CAM注意力分析
加载模型: binary_classifier_20260716_155622.pt
输出: Grad-CAM权重Excel + 重点关注离子列表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import warnings
from datetime import datetime
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


# ==================== Grad-CAM ====================

def compute_grad_cam(model, spectrum_tensor):
    """
    计算单个谱图的Grad-CAM权重
    返回: (cam_weights, gradients) 
    """
    model.eval()
    device = next(model.parameters()).device
    
    if spectrum_tensor.device != device:
        spectrum_tensor = spectrum_tensor.to(device)
    
    # 确保是2D输入 [features]
    if spectrum_tensor.dim() == 1:
        spectrum_tensor = spectrum_tensor.unsqueeze(0)
    
    spectrum_tensor.requires_grad_()
    
    # 前向传播
    logits = model(spectrum_tensor)
    
    # 反向传播
    model.zero_grad()
    logits[0].backward()
    
    # 获取梯度
    gradients = spectrum_tensor.grad
    if gradients is None:
        return np.zeros(spectrum_tensor.shape[1]), None
    
    # 计算CAM权重: 对梯度求平均作为权重
    weights = gradients.mean(dim=0, keepdim=True)
    
    # 加权求和得到CAM
    cam = (weights * spectrum_tensor).sum(dim=0)
    cam = F.relu(cam)
    
    # 归一化
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    
    return cam.detach().cpu().numpy(), gradients.detach().cpu().numpy()


def get_top_ions(cam_weights, mz_min=40, mz_max=600, top_k=20, threshold=0.5):
    """提取重点关注离子"""
    mz_axis = np.linspace(mz_min, mz_max, len(cam_weights))
    
    # 找到权重高于阈值的离子
    threshold_value = threshold * cam_weights.max()
    important_indices = np.where(cam_weights > threshold_value)[0]
    
    if len(important_indices) == 0:
        # 如果没有高于阈值的，取top_k
        important_indices = np.argsort(cam_weights)[-top_k:][::-1]
    
    # 按权重排序
    sorted_idx = np.argsort(cam_weights[important_indices])[::-1]
    important_indices = important_indices[sorted_idx]
    
    # 取前top_k个
    top_indices = important_indices[:top_k]
    top_mz = mz_axis[top_indices]
    top_weights = cam_weights[top_indices]
    
    return top_mz, top_weights


# ==================== 主程序 ====================

if __name__ == "__main__":
    base_dir = Path(r"D:\DL\cann\建模")
    
    # 查找模型
    model_files = list(base_dir.glob("binary_classifier_*.pt"))
    model_files.extend(list((base_dir / "binary_classification_enhanced").glob("binary_classifier_*.pt"))) if (base_dir / "binary_classification_enhanced").exists() else None
    
    model_files = sorted(set(model_files), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if model_files:
        model_path = model_files[0]
        print(f"使用模型: {model_path.name}")
    else:
        print("未找到模型文件，请确认路径")
        exit(1)
    
    # 输出目录
    output_dir = base_dir / f"gradcam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    print(f"  阳性: {len(X_pos)} 谱图")
    print(f"  阴性: {len(X_neg)} 谱图")
    
    # ===== 2. 划分测试集 =====
    np.random.seed(42)
    pos_indices = np.arange(len(X_pos))
    neg_indices = np.arange(len(X_pos), len(X_pos) + len(X_neg))
    
    pos_unique_smiles = np.unique(pos_smiles)
    shuffled_smiles = pos_unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)
    n_pos_test = max(1, int(len(pos_unique_smiles) * 0.15))
    pos_test_smiles = set(shuffled_smiles[:n_pos_test])
    pos_test_idx = [i for i in pos_indices if pos_smiles[i] in pos_test_smiles]
    
    neg_shuffled = neg_indices.copy()
    np.random.shuffle(neg_shuffled)
    n_neg_test = int(len(neg_indices) * 0.15)
    neg_test_idx = neg_shuffled[:n_neg_test].tolist()
    
    test_idx = np.array(pos_test_idx + neg_test_idx)
    np.random.shuffle(test_idx)
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    print(f"  测试集: {len(test_idx)} 谱图 (阳性: {(y[test_idx]==1).sum()})")
    
    # ===== 3. 加载模型 =====
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    
    checkpoint = torch.load(str(model_path), map_location=device)
    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256).to(device)
    model = BinaryClassifier(encoder, input_dim=256).to(device)
    
    if 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    if 'classifier_state_dict' in checkpoint:
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
    
    model.eval()
    print(f"  ✓ 模型加载完成")
    
    # ===== 4. Grad-CAM分析 =====
    print("\n" + "="*60)
    print("Grad-CAM 注意力分析")
    print("="*60)
    
    mz_min, mz_max = 40, 600
    mz_axis = np.linspace(mz_min, mz_max, 561)
    
    # 存储所有结果
    all_cam_weights = []  # 存储所有样本的CAM权重
    all_labels = []       # 存储对应的标签
    
    pos_cam_weights = []  # 阳性样本的CAM权重
    neg_cam_weights = []  # 阴性样本的CAM权重
    
    sample_top_ions = []  # 每个样本的top离子
    
    print("\n  计算Grad-CAM权重...")
    
    for i, idx in enumerate(test_idx):
        spectrum = torch.tensor(X[idx], dtype=torch.float32)
        label = y[idx]
        label_text = '大麻素' if label == 1 else '非大麻素'
        
        # 计算Grad-CAM
        cam_weights, gradients = compute_grad_cam(model, spectrum)
        
        # 确保cam_weights是一维数组
        if cam_weights.ndim > 1:
            cam_weights = cam_weights.squeeze()
        
        all_cam_weights.append(cam_weights)
        all_labels.append(label)
        
        # 按类别存储
        if label == 1:
            pos_cam_weights.append(cam_weights)
        else:
            neg_cam_weights.append(cam_weights)
        
        # 提取top离子
        top_mz, top_weights = get_top_ions(cam_weights, top_k=15)
        for j, (mz, w) in enumerate(zip(top_mz, top_weights)):
            sample_top_ions.append({
                'Sample_ID': i,
                'Label': label_text,
                'Rank': j + 1,
                'm/z': round(mz, 2),
                'CAM_Weight': round(w, 4)
            })
        
        # 进度显示
        if (i + 1) % 50 == 0:
            print(f"    已处理: {i+1}/{len(test_idx)} 个样本")
    
    print(f"    完成! 共处理 {len(test_idx)} 个样本")
    
    # ===== 5. 汇总分析 =====
    print("\n  汇总分析...")
    
    # 计算所有样本的平均权重
    all_avg = np.mean(all_cam_weights, axis=0)
    all_std = np.std(all_cam_weights, axis=0)
    
    # 计算阳性样本的平均权重
    if pos_cam_weights:
        pos_avg = np.mean(pos_cam_weights, axis=0)
        pos_std = np.std(pos_cam_weights, axis=0)
    else:
        pos_avg = np.zeros_like(all_avg)
        pos_std = np.zeros_like(all_avg)
    
    # 计算阴性样本的平均权重
    if neg_cam_weights:
        neg_avg = np.mean(neg_cam_weights, axis=0)
        neg_std = np.std(neg_cam_weights, axis=0)
    else:
        neg_avg = np.zeros_like(all_avg)
        neg_std = np.zeros_like(all_avg)
    
    # 构建汇总DataFrame
    summary_data = []
    for i, mz in enumerate(mz_axis):
        summary_data.append({
            'm/z': round(mz, 2),
            'Mean_Weight_All': round(all_avg[i], 6),
            'Std_Weight_All': round(all_std[i], 6),
            'Mean_Weight_Pos': round(pos_avg[i], 6),
            'Mean_Weight_Neg': round(neg_avg[i], 6),
            'Weight_Diff_Pos_Neg': round(pos_avg[i] - neg_avg[i], 6)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # ===== 6. 找出关键离子 =====
    print("\n  识别关键离子...")
    
    # 大麻素关注离子 (阳性权重最高的)
    top_pos = summary_df.nlargest(30, 'Mean_Weight_Pos')[['m/z', 'Mean_Weight_Pos', 'Mean_Weight_Neg', 'Weight_Diff_Pos_Neg']]
    top_pos['Type'] = '大麻素关注'
    
    # 非大麻素关注离子 (阴性权重最高的)
    top_neg = summary_df.nlargest(30, 'Mean_Weight_Neg')[['m/z', 'Mean_Weight_Neg', 'Mean_Weight_Pos', 'Weight_Diff_Pos_Neg']]
    top_neg['Type'] = '非大麻素关注'
    
    # 大麻素特异性离子 (阳性 >> 阴性)
    top_diff_pos = summary_df.nlargest(20, 'Weight_Diff_Pos_Neg')[['m/z', 'Mean_Weight_Pos', 'Mean_Weight_Neg', 'Weight_Diff_Pos_Neg']]
    top_diff_pos['Type'] = '大麻素特异性'
    
    # 非大麻素特异性离子 (阴性 >> 阳性)
    top_diff_neg = summary_df.nsmallest(20, 'Weight_Diff_Pos_Neg')[['m/z', 'Mean_Weight_Pos', 'Mean_Weight_Neg', 'Weight_Diff_Pos_Neg']]
    top_diff_neg['Type'] = '非大麻素特异性'
    
    # ===== 7. 导出Excel =====
    print("\n  导出Excel...")
    
    with pd.ExcelWriter(output_dir / 'gradcam_analysis.xlsx', engine='openpyxl') as writer:
        # Sheet 1: 所有m/z的汇总统计
        summary_df.to_excel(writer, sheet_name='全部离子权重', index=False)
        
        # Sheet 2: 大麻素关注离子
        top_pos.to_excel(writer, sheet_name='大麻素关注离子', index=False)
        
        # Sheet 3: 非大麻素关注离子
        top_neg.to_excel(writer, sheet_name='非大麻素关注离子', index=False)
        
        # Sheet 4: 大麻素特异性离子
        top_diff_pos.to_excel(writer, sheet_name='大麻素特异性离子', index=False)
        
        # Sheet 5: 非大麻素特异性离子
        top_diff_neg.to_excel(writer, sheet_name='非大麻素特异性离子', index=False)
        
        # Sheet 6: 各样本Top15离子
        pd.DataFrame(sample_top_ions).to_excel(writer, sheet_name='各样本Top15离子', index=False)
    
    print(f"  ✓ Excel已导出: {output_dir / 'gradcam_analysis.xlsx'}")
    
    # ===== 8. 打印重点关注离子 =====
    print("\n" + "="*60)
    print("重点关注离子汇总")
    print("="*60)
    
    print("\n【大麻素特异性离子】（阳性权重显著高于阴性）")
    print("-" * 70)
    for _, row in top_diff_pos.iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  阳性={row['Mean_Weight_Pos']:.5f}, 阴性={row['Mean_Weight_Neg']:.5f}, 差异={row['Weight_Diff_Pos_Neg']:.5f}")
    
    print("\n【非大麻素特异性离子】（阴性权重显著高于阳性）")
    print("-" * 70)
    for _, row in top_diff_neg.iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  阴性={row['Mean_Weight_Neg']:.5f}, 阳性={row['Mean_Weight_Pos']:.5f}, 差异={abs(row['Weight_Diff_Pos_Neg']):.5f}")
    
    print("\n【大麻素样本最关注的Top15离子】")
    print("-" * 70)
    for _, row in top_pos.head(15).iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  权重={row['Mean_Weight_Pos']:.5f}")
    
    print("\n【非大麻素样本最关注的Top15离子】")
    print("-" * 70)
    for _, row in top_neg.head(15).iterrows():
        print(f"  m/z = {row['m/z']:6.2f}:  权重={row['Mean_Weight_Neg']:.5f}")
    
    # ===== 9. 按m/z范围统计 =====
    print("\n" + "="*60)
    print("按m/z范围统计")
    print("="*60)
    
    ranges = [(40, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
    range_labels = ['40-100', '100-200', '200-300', '300-400', '400-500', '500-600']
    
    print("\n  大麻素关注离子分布:")
    for (low, high), label in zip(ranges, range_labels):
        count = len(top_pos[(top_pos['m/z'] >= low) & (top_pos['m/z'] < high)])
        print(f"    {label}: {count} 个")
    
    print("\n  非大麻素关注离子分布:")
    for (low, high), label in zip(ranges, range_labels):
        count = len(top_neg[(top_neg['m/z'] >= low) & (top_neg['m/z'] < high)])
        print(f"    {label}: {count} 个")
    
    print("\n  大麻素特异性离子分布:")
    for (low, high), label in zip(ranges, range_labels):
        count = len(top_diff_pos[(top_diff_pos['m/z'] >= low) & (top_diff_pos['m/z'] < high)])
        print(f"    {label}: {count} 个")
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")
    print(f"  输出文件: {output_dir / 'gradcam_analysis.xlsx'}")
    print(f"  包含以下Sheet:")
    print(f"    1. 全部离子权重 - 所有m/z的平均CAM权重")
    print(f"    2. 大麻素关注离子 - 大麻素样本最关注的离子")
    print(f"    3. 非大麻素关注离子 - 非大麻素样本最关注的离子")
    print(f"    4. 大麻素特异性离子 - 大麻素特有的离子")
    print(f"    5. 非大麻素特异性离子 - 非大麻素特有的离子")
    print(f"    6. 各样本Top15离子 - 每个样本的Top15关注离子")