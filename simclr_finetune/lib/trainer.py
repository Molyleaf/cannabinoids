import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

def safe_auc(labels, probs):
    """数值安全的 AUC 计算"""
    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.5


def train_binary_classifier(
    model, train_loader, val_loader, 
    device='cuda', epochs=100, lr=0.001, patience=15, 
    max_grad_norm=1.0, pos_weight=1.0
):
    """
    运行二分类器微调训练 (支持 CUDA、AMP 混合精度、正样本加权与梯度裁剪)
    """
    device = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    print(f"训练执行设备: {device}")
    model = model.to(device)
    
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 支持 positive class weighting 降低漏检率 (False Negative)
    if pos_weight != 1.0:
        pw_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"  [INFO] 启用正样本损失加权 pos_weight = {pos_weight:.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []
    
    for epoch in range(1, epochs + 1):
        # ===== 训练阶段 =====
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_probs_all, train_labels_all = [], []
        
        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(batch_spec)
                    loss = criterion(logits, batch_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            
            train_loss += loss.item() * batch_spec.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_spec.size(0)
            
            train_probs_all.extend(probs.detach().cpu().numpy().ravel())
            train_labels_all.extend(batch_labels.cpu().numpy().ravel())
        
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        train_auc = safe_auc(train_labels_all, train_probs_all)
        train_aucs.append(train_auc)
        
        # ===== 验证阶段 =====
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs_all, val_labels_all = [], []
        
        with torch.no_grad():
            for batch_spec, batch_labels in val_loader:
                batch_spec = batch_spec.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(batch_spec)
                        loss = criterion(logits, batch_labels)
                else:
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
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        val_auc = safe_auc(val_labels_all, val_probs_all)
        val_aucs.append(val_auc)
        
        scheduler.step(avg_val_loss)
        
        if (epoch - 1) % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n触发早停机制，停止于 epoch {epoch}", flush=True)
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'train_auc': train_aucs,
        'val_auc': val_aucs,
    }
    return model, history
