import argparse
import copy
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# 动态确保项目根目录在 python 模块搜索路径中
scratch_dir = Path(__file__).resolve().parent
finetune_dir = scratch_dir.parent
project_root = finetune_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from finetune.lib import (
    BinaryClassifier,
    evaluate_and_record_predictions,
    evaluate_positive_per_smiles,
    export_results_to_excel,
    load_pretrained_encoder,
    plot_comprehensive_results,
    plot_confusion_matrices,
    prepare_finetune_dataset,
    save_model_checkpoint,
)
from finetune.lib.trainer import safe_auc


def train_two_stage_binary_classifier(
    model,
    train_loader,
    val_loader,
    device='cuda',
    warmup_epochs=15,
    unfreeze_epochs=85,
    classifier_lr=3e-4,
    encoder_lr=3e-5,
    patience=15,
    pos_weight=math.sqrt(1.8),
    max_grad_norm=1.0,
):
    """
    两阶段解冻编码器微调训练流程 (Two-Stage Unfreeze Encoder Fine-Tuning)
    - 阶段 1 (Warmup): 冻结 Encoder，仅训练分类头 (Classifier Head)
    - 阶段 2 (Joint Fine-Tuning): 解冻 Encoder，差分学习率端到端微调 Encoder + Classifier
    """
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    print(f"\n训练执行设备: {dev}", flush=True)
    model = model.to(dev)

    if pos_weight is not None and pos_weight != 1.0:
        pw_tensor = torch.tensor([pos_weight], device=dev, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"  [INFO] 启用正样本损失加权 pos_weight = {pos_weight:.4f}", flush=True)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []

    # ==================== 阶段 1: 冻结 Encoder, 预热分类头 ====================
    print("\n" + "=" * 60, flush=True)
    print(f"【阶段 1】分类头预热训练 (Warmup {warmup_epochs} Epochs, 编码器已冻结)", flush=True)
    print("=" * 60, flush=True)

    model.freeze_encoder = True
    for param in model.encoder.parameters():
        param.requires_grad = False

    warmup_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_probs_all, train_labels_all = [], []

        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(dev)
            batch_labels = batch_labels.to(dev)

            warmup_optimizer.zero_grad(set_to_none=True)
            logits = model(batch_spec)
            loss = criterion(logits, batch_labels)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            warmup_optimizer.step()

            train_loss += loss.item() * batch_spec.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)

            train_probs_all.extend(probs.detach().cpu().numpy().ravel())
            train_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = safe_auc(train_labels_all, train_probs_all)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs_all, val_labels_all = [], []

        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec = batch_spec.to(dev)
                batch_labels = batch_labels.to(dev)

                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_spec.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_spec.size(0)

                val_probs_all.extend(probs.cpu().numpy().ravel())
                val_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = safe_auc(val_labels_all, val_probs_all)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Warmup Epoch {epoch:2d}/{warmup_epochs:2d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | Val AUC: {val_auc:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"  [OK] 阶段 1 完成，阶段 1 最佳 Val Loss: {best_val_loss:.4f}", flush=True)

    # ==================== 阶段 2: 解冻 Encoder, 联合差分微调 ====================
    print("\n" + "=" * 60, flush=True)
    print(f"【阶段 2】解冻编码器端到端微调 (Encoder LR: {encoder_lr:.1e}, Classifier LR: {classifier_lr:.1e})", flush=True)
    print("=" * 60, flush=True)

    model.freeze_encoder = False
    for param in model.encoder.parameters():
        param.requires_grad = True

    joint_optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': classifier_lr, 'weight_decay': 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        joint_optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7
    )

    patience_counter = 0
    total_epochs = warmup_epochs + unfreeze_epochs

    for epoch in range(warmup_epochs + 1, total_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_probs_all, train_labels_all = [], []

        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(dev)
            batch_labels = batch_labels.to(dev)

            joint_optimizer.zero_grad(set_to_none=True)
            logits = model(batch_spec)
            loss = criterion(logits, batch_labels)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            joint_optimizer.step()

            train_loss += loss.item() * batch_spec.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)

            train_probs_all.extend(probs.detach().cpu().numpy().ravel())
            train_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = safe_auc(train_labels_all, train_probs_all)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs_all, val_labels_all = [], []

        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec = batch_spec.to(dev)
                batch_labels = batch_labels.to(dev)

                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_spec.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_spec.size(0)

                val_probs_all.extend(probs.cpu().numpy().ravel())
                val_labels_all.extend(batch_labels.cpu().numpy().ravel())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = safe_auc(val_labels_all, val_probs_all)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        scheduler.step(avg_val_loss)
        enc_lr_curr = joint_optimizer.param_groups[0]['lr']
        cls_lr_curr = joint_optimizer.param_groups[1]['lr']

        if (epoch - warmup_epochs) % 5 == 0 or epoch == warmup_epochs + 1:
            print(f"Epoch {epoch:3d}/{total_epochs:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | Val AUC: {val_auc:.4f} | Enc LR: {enc_lr_curr:.1e}", flush=True)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n触发早停机制 (Patience={patience})，早停于 Epoch {epoch}", flush=True)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"  [OK] 恢复全局最佳权重模型 (最佳 Val Loss: {best_val_loss:.4f})", flush=True)

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'train_auc': train_aucs,
        'val_auc': val_aucs,
    }

    return model, history


def find_optimal_threshold(val_res, metric='f1'):
    """
    在验证集上自动搜索最优决策阈值 T_opt
    """
    probs = val_res['probs']
    labels = val_res['labels']

    thresholds = np.linspace(0.10, 0.90, 81)
    best_th = 0.50
    best_score = -1.0

    records = []

    for th in thresholds:
        preds = (probs >= th).astype(float)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_j = rec + spec - 1.0

        score = f1 if metric == 'f1' else youden_j

        records.append({
            'threshold': float(th),
            'f1': float(f1),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'youden_j': float(youden_j)
        })

        if score > best_score:
            best_score = score
            best_th = float(th)

    return best_th, best_score, records


def plot_threshold_search_curve(records, best_th, output_dir):
    """绘制验证集阈值搜索曲线图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ths = [r['threshold'] for r in records]
    f1s = [r['f1'] for r in records]
    accs = [r['accuracy'] for r in records]
    precs = [r['precision'] for r in records]
    recs = [r['recall'] for r in records]
    j_stats = [r['youden_j'] for r in records]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ths, f1s, label='F1-Score', color='#1f77b4', lw=2.5)
    ax.plot(ths, accs, label='Accuracy', color='#2ca02c', lw=2.0, linestyle='--')
    ax.plot(ths, precs, label='Precision', color='#ff7f0e', lw=1.8, linestyle=':')
    ax.plot(ths, recs, label='Recall', color='#d62728', lw=1.8, linestyle='-.')
    ax.plot(ths, j_stats, label="Youden's J Index", color='#9467bd', lw=1.5, linestyle='-')

    ax.axvline(best_th, color='black', linestyle=':', lw=1.5, label=f'Optimal Threshold ({best_th:.2f})')
    ax.axvline(0.50, color='gray', linestyle='--', lw=1.0, label='Default Threshold (0.50)')

    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.set_title("Validation Set Decision Threshold Optimization (Experiment 2)", fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = output_dir / 'threshold_optimization_curve.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  [OK] 阈值寻优曲线图已保存: {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="试验1 (解冻微调) 与 试验2 (验证集决策阈值寻优)")
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=str(project_root / "embedding_pretrain" / "results_20260725_095428" / "best_model.pt"),
        help="基础预训练编码器权重 (best_model.pt)"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--unfreeze_epochs", type=int, default=85)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--classifier_lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pos_weight", type=float, default=math.sqrt(1.8))
    args = parser.parse_args()

    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"未找到指定的预训练权重文件: {encoder_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70, flush=True)
    print("试验1 (两阶段解冻微调) & 试验2 (验证集决策阈值寻优)", flush=True)
    print("=" * 70, flush=True)
    print(f"基础预训练模型: {encoder_path}", flush=True)
    print(f"运行设备: {device.upper()}", flush=True)
    if device == 'cuda':
        print(f"显卡型号: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"编码器微调学习率: {args.encoder_lr}", flush=True)
    print(f"分类头微调学习率: {args.classifier_lr}", flush=True)

    # 1. 数据准备
    print("\n" + "=" * 70, flush=True)
    print("Step 1: 加载与划分数据集", flush=True)
    print("=" * 70, flush=True)
    train_loader, train_eval_loader, val_loader, test_loader, meta = prepare_finetune_dataset(
        batch_size=args.batch_size,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        use_cuda=(device == 'cuda')
    )

    # 2. 构建模型并加载预训练权重 best_model.pt
    print("\n" + "=" * 70, flush=True)
    print("Step 2: 实例化模型并加载预训练 Encoder (best_model.pt)", flush=True)
    print("=" * 70, flush=True)
    encoder = load_pretrained_encoder(
        encoder_path=encoder_path,
        input_dim=561,
        hidden_dim=256,
        device=device
    )
    # 初始化 BinaryClassifier，初始挂载 Encoder
    model = BinaryClassifier(encoder, input_dim=256, freeze_encoder=True)

    # 3. 试验1: 开始两阶段解冻微调
    print("\n" + "=" * 70, flush=True)
    print("Step 3: 执行【试验1】两阶段解冻 Encoder 端到端微调", flush=True)
    print("=" * 70, flush=True)
    start_time = datetime.now()
    model, history = train_two_stage_binary_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        warmup_epochs=args.warmup_epochs,
        unfreeze_epochs=args.unfreeze_epochs,
        classifier_lr=args.classifier_lr,
        encoder_lr=args.encoder_lr,
        patience=args.patience,
        pos_weight=args.pos_weight
    )
    elapsed = datetime.now() - start_time
    print(f"\n  [OK] 解冻微调完成，总用时: {str(elapsed).split('.')[0]}", flush=True)

    # 4. 试验2: 验证集决策阈值自动寻优
    print("\n" + "=" * 70, flush=True)
    print("Step 4: 执行【试验2】验证集决策阈值自动寻优 (Threshold Search)", flush=True)
    print("=" * 70, flush=True)

    # 先以默认 0.5 评估验证集
    val_res_default = evaluate_and_record_predictions(model, val_loader, sample_names=meta['val_sample_names'], device=device, threshold=0.50)
    best_th, best_f1, th_records = find_optimal_threshold(val_res_default, metric='f1')

    print(f"  [Search Result] 验证集阈值扫描完成 (0.10 ~ 0.90):", flush=True)
    print(f"     基准默认阈值: T = 0.50 | Val Acc: {val_res_default['accuracy']:.2%} | Val F1: {val_res_default['f1']:.4f} | Val AUC: {val_res_default['auc']:.4f}", flush=True)
    print(f"     搜寻最优阈值: T_opt = {best_th:.2f} | Val F1: {best_f1:.4f}", flush=True)

    # 5. 测试集多阈值对比评估
    print("\n" + "=" * 70, flush=True)
    print("Step 5: 测试集性能对比 (默认 T=0.50 vs 最优 T_opt)", flush=True)
    print("=" * 70, flush=True)

    train_res_05 = evaluate_and_record_predictions(model, train_eval_loader, sample_names=meta['train_sample_names'], device=device, threshold=0.50)
    val_res_05 = evaluate_and_record_predictions(model, val_loader, sample_names=meta['val_sample_names'], device=device, threshold=0.50)
    test_res_05 = evaluate_and_record_predictions(model, test_loader, sample_names=meta['test_sample_names'], device=device, threshold=0.50)

    test_res_opt = evaluate_and_record_predictions(model, test_loader, sample_names=meta['test_sample_names'], device=device, threshold=best_th)

    print("\n--- 默认阈值 (T=0.50) 测试集结果 ---", flush=True)
    print(f"  Test Accuracy: {test_res_05['accuracy']:.2%} | Precision: {test_res_05['precision']:.2%} | Recall: {test_res_05['recall']:.2%} | F1: {test_res_05['f1']:.4f} | AUC: {test_res_05['auc']:.4f}", flush=True)

    print(f"\n--- 最优阈值 (T_opt={best_th:.2f}) 测试集结果 ---", flush=True)
    print(f"  Test Accuracy: {test_res_opt['accuracy']:.2%} | Precision: {test_res_opt['precision']:.2%} | Recall: {test_res_opt['recall']:.2%} | F1: {test_res_opt['f1']:.4f} | AUC: {test_res_opt['auc']:.4f}", flush=True)

    smiles_res_05 = evaluate_positive_per_smiles(
        test_indices=meta['test_idx'],
        smiles_all=meta['smiles_all'],
        labels_all=meta['y'],
        probs=test_res_05['probs'],
        preds=test_res_05['preds'],
        threshold=0.50
    )

    smiles_res_opt = evaluate_positive_per_smiles(
        test_indices=meta['test_idx'],
        smiles_all=meta['smiles_all'],
        labels_all=meta['y'],
        probs=test_res_opt['probs'],
        preds=test_res_opt['preds'],
        threshold=best_th
    )

    # 6. 保存与导出结果
    print("\n" + "=" * 70, flush=True)
    print("Step 6: 保存模型与导出对比结果", flush=True)
    print("=" * 70, flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = scratch_dir / f"results_exp1_exp2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存解冻训练后的模型全权重
    checkpoint_path = run_dir / f"unfrozen_binary_classifier_{timestamp}.pt"
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'classifier_state_dict': model.classifier.state_dict(),
        'optimal_threshold': best_th,
        'history': history,
        'test_metrics_default': test_res_05,
        'test_metrics_opt': test_res_opt,
        'base_encoder_path': str(encoder_path)
    }, str(checkpoint_path))
    print(f"  [OK] 解冻模型权重已保存至: {checkpoint_path}", flush=True)

    # 导出评估表格 CSV
    export_results_to_excel(
        history=history,
        model=model,
        train_results=train_res_05,
        val_results=val_res_05,
        test_results=test_res_opt, # 使用最优阈值的测试集结果导出明细
        smiles_results=smiles_res_opt,
        meta=meta,
        output_dir=run_dir
    )

    # 导出阈值寻优明细 CSV
    with open(run_dir / 'threshold_search_records.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'f1', 'accuracy', 'precision', 'recall', 'youden_j'])
        writer.writeheader()
        writer.writerows(th_records)

    # 导出实验对比报告 CSV
    with open(run_dir / 'experiment_comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Threshold', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC', 'SMILES_Recognized'])
        writer.writerow([
            'Exp1 (Unfrozen Encoder, Default T=0.50)',
            0.50,
            test_res_05['accuracy'],
            test_res_05['precision'],
            test_res_05['recall'],
            test_res_05['f1'],
            test_res_05['auc'],
            f"{smiles_res_05['n_correct']}/{smiles_res_05['n_smiles']} ({smiles_res_05['accuracy']:.2%})" if smiles_res_05 else "N/A"
        ])
        writer.writerow([
            'Exp1+Exp2 (Unfrozen Encoder + Optimal T_opt)',
            best_th,
            test_res_opt['accuracy'],
            test_res_opt['precision'],
            test_res_opt['recall'],
            test_res_opt['f1'],
            test_res_opt['auc'],
            f"{smiles_res_opt['n_correct']}/{smiles_res_opt['n_smiles']} ({smiles_res_opt['accuracy']:.2%})" if smiles_res_opt else "N/A"
        ])

    # 绘图
    plot_comprehensive_results(history, train_res_05, val_res_05, test_res_opt, smiles_res_opt, run_dir)
    plot_confusion_matrices(train_res_05, val_res_05, test_res_opt, run_dir)
    plot_threshold_search_curve(th_records, best_th, run_dir)

    print(f"\n  [OK] 试验 1 与 试验 2 的全部结果已统一导出至:\n       --> {run_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
