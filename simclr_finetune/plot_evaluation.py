"""
后训练模型评估脚本
功能：
1. 加载阳性 (positive.msp) 和阴性 (阴性.msp) 质谱数据；
2. 执行与微调训练时一致的数据清洗 (.alpha.-pbp 从阳性改阴性) 与数据集划分 (random_state=42, test=0.15, val=0.15, 阳性按SMILES分组，阴性随机)；
3. 使用 app.pipeline 中新的加载方法 (get_classifier_model) 加载最新的 safetensors 分类器模型；
4. 在 Train、Val、Test 三个划分上运行前向推理；
5. 计算评估指标 (Accuracy, Precision, Recall, F1, AUC)；
6. 绘制 ROC 曲线对比图 (roc_curves_comparison.png) 与 混淆矩阵热力图 (confusion_matrix.png)。
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter, defaultdict

# 设置系统路径，确保能导入 app 模块
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入新的模型加载方法和预处理工具
from app.pipeline import get_classifier_model, peaks_to_vector, preprocess_spectra

def parse_msp_with_smiles(msp_file, min_peaks=5):
    """解析MSP文件，返回化合物列表（包含SMILES）"""
    print(f"  正在解析 MSP 文件: {Path(msp_file).name}...")
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
            current_comp = {
                'name': line.split(':', 1)[1].strip(),
                'smiles': '',
                'peaks': []
            }
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
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    if mz > 0 and intensity > 0:
                        current_comp['peaks'].append((mz, intensity))
                except ValueError:
                    in_peaks = False
    
    if current_comp is not None and 'peaks' in current_comp:
        if len(current_comp['peaks']) >= min_peaks:
            compounds.append(current_comp)
    
    print(f"  [OK] 成功解析出 {len(compounds)} 个化合物谱图")
    return compounds

def split_positive_by_smiles_negative_random(
    pos_indices, neg_indices, 
    pos_smiles_list, 
    test_size=0.15, val_size=0.15, 
    random_state=42
):
    """
    划分数据集逻辑（与微调训练时一致）：
    阳性样本：按SMILES分组，同一SMILES的所有谱图在同一集合
    阴性样本：随机划分
    """
    np.random.seed(random_state)
    
    # ===== 阳性：按SMILES分组划分 =====
    pos_unique_smiles = np.unique(pos_smiles_list)
    n_pos_smiles = len(pos_unique_smiles)
    
    shuffled_smiles = pos_unique_smiles.copy()
    np.random.shuffle(shuffled_smiles)
    
    n_pos_test = max(1, int(n_pos_smiles * test_size))
    n_pos_val = max(1, int(n_pos_smiles * val_size))
    
    pos_test_smiles = set(shuffled_smiles[:n_pos_test])
    pos_val_smiles = set(shuffled_smiles[n_pos_test:n_pos_test + n_pos_val])
    pos_train_smiles = set(shuffled_smiles[n_pos_test + n_pos_val:])
    
    pos_train_idx = [i for i in pos_indices if pos_smiles_list[i] in pos_train_smiles]
    pos_val_idx = [i for i in pos_indices if pos_smiles_list[i] in pos_val_smiles]
    pos_test_idx = [i for i in pos_indices if pos_smiles_list[i] in pos_test_smiles]
    
    print(f"  阳性分类结果:")
    print(f"    训练集: {len(pos_train_smiles)} 种化合物, {len(pos_train_idx)} 张谱图")
    print(f"    验证集: {len(pos_val_smiles)} 种化合物, {len(pos_val_idx)} 张谱图")
    print(f"    测试集: {len(pos_test_smiles)} 种化合物, {len(pos_test_idx)} 张谱图")
    
    # ===== 阴性：随机划分 =====
    n_neg = len(neg_indices)
    neg_shuffled = neg_indices.copy()
    np.random.shuffle(neg_shuffled)
    
    n_neg_test = int(n_neg * test_size)
    n_neg_val = int(n_neg * val_size)
    
    neg_test_idx = neg_shuffled[:n_neg_test].tolist()
    neg_val_idx = neg_shuffled[n_neg_test:n_neg_test + n_neg_val].tolist()
    neg_train_idx = neg_shuffled[n_neg_test + n_neg_val:].tolist()
    
    print(f"  阴性分类结果:")
    print(f"    训练集: {len(neg_train_idx)} 张谱图")
    print(f"    验证集: {len(neg_val_idx)} 张谱图")
    print(f"    测试集: {len(neg_test_idx)} 张谱图")
    
    train_idx = np.array(pos_train_idx + neg_train_idx)
    val_idx = np.array(pos_val_idx + neg_val_idx)
    test_idx = np.array(pos_test_idx + neg_test_idx)
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    return train_idx, val_idx, test_idx

def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return 0.5

def evaluate_predictions(y_true, y_prob):
    """根据真实标签和预测概率计算指标"""
    y_pred = (y_prob >= 0.5).astype(float)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = safe_auc(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(y_true)),
        'n_pos': int((y_true == 1).sum()),
        'n_neg': int((y_true == 0).sum())
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前推理执行设备: {device}")
    
    # 1. 拼装数据文件路径
    finetune_dir = project_root / "simclr_finetune"
    positive_msp = finetune_dir / "data_source" / "阳性-含CanonicalSMILES-5类骨架(3).msp"
    negative_msp = finetune_dir / "data_source" / "阴性(4).msp"
    
    if not positive_msp.exists():
        positive_msp = project_root / "known_compound_recognize" / "positive.msp"
    if not negative_msp.exists():
        negative_msp = project_root / "scratch" / "阴性.msp"
        
    if not positive_msp.exists():
        raise FileNotFoundError(f"阳性数据不存在: {positive_msp}")
    if not negative_msp.exists():
        raise FileNotFoundError(f"阴性数据不存在: {negative_msp}")
        
    # 2. 解析原始数据
    print("=" * 60)
    print("Step 1: 加载并清洗原始质谱数据...")
    print("=" * 60)
    pos_compounds = parse_msp_with_smiles(str(positive_msp))
    neg_compounds = parse_msp_with_smiles(str(negative_msp))
    
    # 3. 将 '.alpha.-pbp' 从阳性改为阴性 (原始清洗逻辑)
    print("\n  [INFO] 正在将 '.alpha.-pbp' 化合物从阳性改为阴性...")
    alpha_pbp_indices = []
    for i, comp in enumerate(pos_compounds):
        if comp['name'] == '.alpha.-pbp':
            alpha_pbp_indices.append(i)
            print(f"    找到: {comp['name']}, SMILES: {comp['smiles']}")
            
    if alpha_pbp_indices:
        removed_comps = [pos_compounds[i] for i in alpha_pbp_indices]
        pos_compounds = [comp for i, comp in enumerate(pos_compounds) if i not in alpha_pbp_indices]
        neg_compounds.extend(removed_comps)
        print(f"  [OK] 已将 {len(removed_comps)} 个 '.alpha.-pbp' 样本移至阴性集")
    else:
        print("  [WARN] 未找到 '.alpha.-pbp' 化合物")
        
    # 4. 特征向量化与预处理 (使用 app.pipeline 的实现)
    print("\n  [INFO] 正在将质谱转换为561维预处理向量...")
    pos_spectra = np.array([peaks_to_vector(c['peaks']) for c in pos_compounds])
    pos_smiles = np.array([c['smiles'] if c['smiles'] else c['name'] for c in pos_compounds])
    neg_spectra = np.array([peaks_to_vector(c['peaks']) for c in neg_compounds])
    
    X_pos = preprocess_spectra(pos_spectra)
    X_neg = preprocess_spectra(neg_spectra)
    
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    # 5. 划分数据集
    print("\n" + "=" * 60)
    print("Step 2: 数据集划分 (Train vs Val vs Test)")
    print("=" * 60)
    pos_indices = np.arange(len(pos_spectra))
    neg_indices = np.arange(len(pos_spectra), len(pos_spectra) + len(neg_spectra))
    
    train_idx, val_idx, test_idx = split_positive_by_smiles_negative_random(
        pos_indices, neg_indices,
        pos_smiles,
        test_size=0.15, val_size=0.15,
        random_state=42
    )
    
    print(f"\n  总计样本数:")
    print(f"    训练集: {len(train_idx)} (阳性: {(y[train_idx]==1).sum()}, 阴性: {(y[train_idx]==0).sum()})")
    print(f"    验证集: {len(val_idx)} (阳性: {(y[val_idx]==1).sum()}, 阴性: {(y[val_idx]==0).sum()})")
    print(f"    测试集: {len(test_idx)} (阳性: {(y[test_idx]==1).sum()}, 阴性: {(y[test_idx]==0).sum()})")
    
    # 6. 新加载方法加载最新的模型
    print("\n" + "=" * 60)
    print("Step 3: 使用新方法 get_classifier_model() 加载最新 safetensors 模型...")
    print("=" * 60)
    
    # get_classifier_model 内部会读取 app/models/ 中的最新 safetensors 并 eval
    model = get_classifier_model().to(device)
    model.eval()
    print("  [OK] 最新分类器模型加载成功并已部署至设备")
    
    # 7. 进行模型推理
    print("\n" + "=" * 60)
    print("Step 4: 模型预测与推理...")
    print("=" * 60)
    
    datasets = {
        'Train': TensorDataset(torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32)),
        'Val': TensorDataset(torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.float32)),
        'Test': TensorDataset(torch.tensor(X[test_idx], dtype=torch.float32), torch.tensor(y[test_idx], dtype=torch.float32))
    }
    
    results = {}
    probs_dict = {}
    
    for name, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        probs_dict[name] = (all_labels, all_probs)
        results[name] = evaluate_predictions(all_labels, all_probs)
        
        # 终端打印结果
        r = results[name]
        print(f"\n  [{name} 集合评估指标]")
        print(f"    样本总数: {r['n_samples']} (阳性: {r['n_pos']}, 阴性: {r['n_neg']})")
        print(f"    Accuracy:  {r['accuracy']:.4f}")
        print(f"    Precision: {r['precision']:.4f}")
        print(f"    Recall:    {r['recall']:.4f}")
        print(f"    F1-score:  {r['f1']:.4f}")
        print(f"    AUC-ROC:   {r['auc']:.4f}")
        
    # 保存结果到 JSON 文件中
    out_dir = project_root / "simclr_finetune"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "evaluation_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] 评估指标已保存至: {json_path}")
    
    # 8. 绘制 ROC 曲线对比图
    print("\n" + "=" * 60)
    print("Step 5: 绘制 ROC 曲线与混淆矩阵...")
    print("=" * 60)
    
    plt.figure(figsize=(8, 6.5))
    colors = {'Train': '#1f77b4', 'Val': '#ff7f0e', 'Test': '#2ca02c'}
    
    for name in ['Train', 'Val', 'Test']:
        labels, probs = probs_dict[name]
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = results[name]['auc']
        plt.plot(fpr, tpr, color=colors[name], lw=2.5, 
                 label=f'{name} (AUC = {auc_val:.4f})')
                 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Random (AUC = 0.5000)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11, labelpad=8)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11, labelpad=8)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, shadow=False, facecolor='white', edgecolor='#e2e2e2')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    roc_path = out_dir / "roc_curves_comparison.png"
    plt.savefig(str(roc_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] ROC对比曲线已保存: {roc_path}")
    
    # 9. 绘制混淆矩阵热力图 (1行3列合并展示)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    class_names = ['Non-Cannabinoid', 'Cannabinoid']
    
    for idx, name in enumerate(['Train', 'Val', 'Test']):
        cm = np.array(results[name]['confusion_matrix'])
        ax = axes[idx]
        
        # 计算比例
        cm_ratio = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 制作带具体数值和占比的文本标注
        labels = []
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                ratio = cm_ratio[i, j]
                labels.append(f"{count}\n({ratio:.1%})")
        labels = np.array(labels).reshape(2, 2)
        
        # 绘制热力图
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True, ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12, "weight": "bold"},
                    linewidths=1, linecolor='#e2e2e2',
                    vmin=0, vmax=cm.sum())
                    
        ax.set_title(f'{name} Set (n={results[name]["n_samples"]})', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10, labelpad=8)
        if idx == 0:
            ax.set_ylabel('True Label', fontsize=10, labelpad=8)
        else:
            ax.set_ylabel('')
            
    plt.suptitle('Confusion Matrices (Train vs Val vs Test)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 混淆矩阵热力图已保存: {cm_path}")
    print("\n评估全部顺利完成！图表及指标均已成功保存。")

if __name__ == '__main__':
    main()
