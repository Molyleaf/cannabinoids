import torch
from simclr_pretrain.lib.losses import nt_xent_loss

def train_step(model, spec, optimizer, augmenter, device, scaler, temperature):
    """进行单步 SimCLR 训练 (联合前向优化版)"""
    # 生成两个增强视图并进行拼接
    view1 = augmenter(spec)
    view2 = augmenter(spec)
    
    # 拼接为单个大批次进行一次前向传播，提高计算开销效率
    views = torch.cat([view1, view2], dim=0)
    
    # 混合精度前向传播
    if scaler is not None:
        with torch.amp.autocast('cuda'):
            _, z = model(views)
            z1, z2 = torch.chunk(z, 2, dim=0)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
        
        optimizer.zero_grad(set_to_none=True)  # 更高效
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        _, z = model(views)
        z1, z2 = torch.chunk(z, 2, dim=0)
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    return loss.item()

