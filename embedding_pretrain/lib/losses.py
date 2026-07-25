import torch
import torch.distributed as dist
import torch.nn.functional as F


def safe_l2_normalize(x, eps=1e-3):
    """安全的 L2 归一化，防止极小模长引发反向梯度爆炸"""
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / torch.clamp(norm, min=eps)


class GatherLayer(torch.autograd.Function):
    """
    可导出的 All-Gather 操作类，确保在 DDP 计算 NT-Xent Loss 时反向传播梯度能够传回各 GPU 卡
    """
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        world_size = dist.get_world_size()
        output = [torch.zeros_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(output, input_tensor)
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        rank = dist.get_rank()
        batch_size = input_tensor.shape[0]

        # 仅截取属于本 rank 的梯度分量
        grad_input = grad_output[rank * batch_size:(rank + 1) * batch_size]
        return grad_input


def nt_xent_loss(z1, z2, temperature=0.07, use_ddp=False):
    """
    NT-Xent 对比学习损失 (全 GPU 向量化与 All-Gather 稳定加速版)
    """
    z1 = safe_l2_normalize(z1)
    z2 = safe_l2_normalize(z2)

    if use_ddp and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size > 1:
            # 使用可导出的 All-Gather 拼接全卡表征
            z1_large = GatherLayer.apply(z1)
            z2_large = GatherLayer.apply(z2)
            
            # 当前卡 local batch 尺寸与全局 batch 尺寸
            local_batch_size = z1.shape[0]
            rank = dist.get_rank()
            
            z = torch.cat([z1_large, z2_large], dim=0)
            sim_matrix = torch.mm(torch.cat([z1, z2], dim=0), z.T) / temperature
            
            total_large = z1_large.shape[0]
            labels1 = torch.arange(local_batch_size, device=z1.device) + rank * local_batch_size
            labels2 = torch.arange(local_batch_size, device=z1.device) + rank * local_batch_size + total_large
            labels = torch.cat([labels2, labels1], dim=0)
            
            # 全向量化排除自对比掩码 (消除 Python 原生 for 循环 CPU 阻塞)
            mask = torch.zeros((2 * local_batch_size, 2 * total_large), dtype=torch.bool, device=z1.device)
            row_idx = torch.arange(local_batch_size, device=z1.device)
            mask[row_idx, rank * local_batch_size + row_idx] = True
            mask[local_batch_size + row_idx, total_large + rank * local_batch_size + row_idx] = True
                
            sim_matrix = sim_matrix.masked_fill(mask, -1e4)
            loss = F.cross_entropy(sim_matrix, labels)
            return loss

    # 单卡模式
    batch_size = z1.shape[0]
    device = z1.device

    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.T) / temperature

    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e4)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss
