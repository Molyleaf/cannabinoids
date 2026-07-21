import torch
import torch.nn.functional as F

def safe_l2_normalize(x, eps=1e-3):
    """安全的 L2 归一化，防止极小模长引发反向梯度爆炸"""
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / torch.clamp(norm, min=eps)

def nt_xent_loss(z1, z2, temperature=0.07):
    """NT-Xent 对比学习损失 (数值稳定优化版)"""
    batch_size = z1.shape[0]
    device = z1.device
    
    z1 = safe_l2_normalize(z1)
    z2 = safe_l2_normalize(z2)
    
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.T) / temperature
    
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    # 使用 -1e4 代替 float('-inf')，在 FP16 混合精度下更稳定且避免 NaN
    sim_matrix = sim_matrix.masked_fill(mask, -1e4)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

