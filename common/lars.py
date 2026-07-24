import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    SimCLR 官方规范的高性能 LARS (Layer-wise Adaptive Rate Scaling) 优化器。
    
    主要特性：
    1. 层级局部学习率计算 (Layer-wise LR scaling using trust coefficient eta=0.001)
    2. 针对偏置项 (Bias) 与归一化层 (LayerNorm / GroupNorm / BatchNorm) 自动排除 Weight Decay 与 LARS 缩放系数。
    """
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.9,
                 weight_decay=1e-6,
                 eta=0.001,
                 eps=1e-8,
                 exclude_bias_n_norm=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
            exclude_bias_n_norm=exclude_bias_n_norm
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            eps = group['eps']
            exclude_bias_n_norm = group['exclude_bias_n_norm']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_grad = p.grad
                
                # 判断该参数是否属于 bias 或 1D 归一化参数 (ndim <= 1)
                is_bias_or_norm = (p.ndim <= 1) and exclude_bias_n_norm

                if not is_bias_or_norm:
                    # 计算权重 L2 范数与梯度 L2 范数
                    w_norm = torch.norm(p, p=2)
                    g_norm = torch.norm(param_grad, p=2)

                    if w_norm > 0 and g_norm > 0:
                        # 计算局部学习率 Trust Ratio
                        trust_ratio = eta * w_norm / (g_norm + weight_decay * w_norm + eps)
                    else:
                        trust_ratio = 1.0

                    # 仅对非 bias/norm 参数应用 weight decay
                    if weight_decay != 0:
                        param_grad = param_grad.add(p, alpha=weight_decay)

                    # 应用局部学习率缩放
                    param_grad = param_grad.mul(trust_ratio)

                # 动量更新
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(param_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(param_grad)
                    
                    param_grad = buf

                # 变量更新
                p.add_(param_grad, alpha=-lr)

        return loss


def get_lars_optimizer(model, base_lr=0.3, global_batch_size=4096, weight_decay=1e-6, momentum=0.9, eta=0.001):
    """
    快捷构建辅助函数：自动对齐 SimCLR 学习率线性缩放并初始化 LARS 优化器
    Linear Scaling Rule: lr = base_lr * (global_batch_size / 256)
    """
    scaled_lr = base_lr * (global_batch_size / 256.0)
    
    # 区分常规参数与 Bias/Norm 参数
    regular_params = []
    bias_or_norm_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or 'bias' in name or 'norm' in name.lower() or 'bn' in name.lower() or 'gn' in name.lower():
            bias_or_norm_params.append(param)
        else:
            regular_params.append(param)

    param_groups = [
        {
            'params': regular_params,
            'weight_decay': weight_decay,
            'exclude_bias_n_norm': False
        },
        {
            'params': bias_or_norm_params,
            'weight_decay': 0.0,
            'exclude_bias_n_norm': True
        }
    ]

    optimizer = LARS(
        param_groups,
        lr=scaled_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        eta=eta
    )
    print(f"[LARS Optimizer] 全局 Batch Size: {global_batch_size} | 缩放后 Learning Rate: {scaled_lr:.4f}")
    return optimizer
