"""
SimCLR 多卡 DDP 预训练脚本 (torchrun 启动)
支持：
1. 4096 大 Batch Size 优化
2. NVIDIA DALI 混合加载 (ExternalSource + GPU PyTorch Batch Augmentation)
3. 工业级 LARS 优化器 (对齐 SimCLR 规范排除 Bias 与 GroupNorm)
4. 学习率线性缩放 + Linear Warmup 余弦退火
5. 全局 All-Gather NT-Xent 对比损失
"""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 确保根路径可导入 common 与 embedding_pretrain 模块
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.augmentation import SpectrumAugmentation
from common.lars import get_lars_optimizer
from embedding_pretrain.lib.losses import nt_xent_loss
from embedding_pretrain.lib.models import ProjectionHead, SimCLR, SpectrumEncoder
from embedding_pretrain.lib.utils import ConvergenceChecker, plot_loss_history

# 尝试导入 NVIDIA DALI
try:
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False


# ==================== DALI Pipeline 定义 ====================
if HAS_DALI:
    class ExternalInputCallable:
        def __init__(self, data, batch_size, shard_id, num_shards):
            self.data = data
            self.batch_size = batch_size
            self.shard_id = shard_id
            self.num_shards = num_shards
            
            # 计算当前 shard 的样本子集
            total_samples = len(data)
            per_shard = total_samples // num_shards
            self.start_idx = shard_id * per_shard
            self.end_idx = self.start_idx + per_shard if shard_id < num_shards - 1 else total_samples
            self.shard_data = self.data[self.start_idx:self.end_idx]
            self.indices = np.arange(len(self.shard_data))

        def __call__(self, sample_info):
            if sample_info.iteration_in_epoch == 0 and sample_info.idx_in_batch == 0:
                np.random.shuffle(self.indices)
            
            sample_idx = (sample_info.iteration_in_epoch * self.batch_size + sample_info.idx_in_batch) % len(self.shard_data)
            real_idx = self.indices[sample_idx]
            return self.shard_data[real_idx]

    @pipeline_def
    def create_dali_pipeline(external_data, batch_size, shard_id, num_shards):
        external_source = ExternalInputCallable(external_data, batch_size, shard_id, num_shards)
        spectra = fn.external_source(source=external_source, dtype=types.FLOAT, batch=False)
        return spectra


class LinearWarmupCosineAnnealingLR:
    """线性 Warmup + 余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # 线性 Warmup 阶段
            lr = self.base_lr * (self.current_epoch / max(1, self.warmup_epochs))
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def setup_ddp():
    """初始化 DDP 分布式环境"""
    is_distributed = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    if is_distributed:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    if is_distributed and not dist.is_initialized():
        backend = 'nccl' if (use_cuda and os.name != 'nt') else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')

    return rank, world_size, local_rank, device


def main():
    parser = argparse.ArgumentParser(description="SimCLR 大 Batch 多卡 DDP 预训练")
    parser.add_argument("--data_path", type=str, default=None, help="预训练 spectra.npy 路径")
    parser.add_argument("--global_batch_size", type=int, default=4096, help="全局 Batch Size (默认 4096)")
    parser.add_argument("--base_lr", type=float, default=0.3, help="基础学习率 (用于 256 Batch 下的基准)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Linear Warmup 轮数")
    parser.add_argument("--max_epochs", type=int, default=500, help="最大训练轮数")
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent 温度系数")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="LARS Weight Decay")
    parser.add_argument("--output_dir", type=str, default=None, help="模型输出保存目录")
    parser.add_argument("--use_dali", action="store_true", help="显式启用 DALI 加载器 (自动降级支持)")
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_ddp()
    is_main_process = (rank == 0)

    # 路径解析
    base_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_path) if args.data_path else base_dir / "pretrain_spectra.npy"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "pretrained_model_v2"
    
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("SimCLR 多卡 DDP 对比学习预训练 (4096 大 Batch + DALI + LARS)")
        print("=" * 60)
        print(f"Rank: {rank}/{world_size} | Device: {device}")
        print(f"数据文件: {data_path}")
        print(f"全局 Batch Size: {args.global_batch_size}")
        print(f"每卡 Local Batch Size: {args.global_batch_size // world_size}")

    # 加载数据集
    if not data_path.exists():
        raise FileNotFoundError(f"错误: 未找到预训练数据 {data_path}")
        
    spectra = np.load(data_path).astype(np.float32)
    local_batch_size = max(1, args.global_batch_size // world_size)

    # 验证是否启用 DALI
    use_dali = args.use_dali and HAS_DALI and (device.type == 'cuda')
    if is_main_process:
        if use_dali:
            print("[Data Pipeline] 成功启用 NVIDIA DALI GPU Data Loader")
        else:
            print("[Data Pipeline] 使用 PyTorch DistributedDataLoader")

    if not use_dali:
        dataset = TensorDataset(torch.from_numpy(spectra))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=2 if device.type == 'cuda' else 0,
            pin_memory=(device.type == 'cuda'),
            drop_last=True
        )

    # 初始化模型与网络结构 (保持 GroupNorm 内部特征归一化)
    input_dim = spectra.shape[1]
    encoder = SpectrumEncoder(input_dim=input_dim, hidden_dim=256).to(device)
    projection_head = ProjectionHead(input_dim=256, hidden_dim=128, output_dim=64).to(device)
    model = SimCLR(encoder, projection_head).to(device)

    if world_size > 1:
        if device.type == 'cuda':
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    # 初始化 LARS 优化器 (自动实现线性学习率缩放)
    optimizer = get_lars_optimizer(
        model=model,
        base_lr=args.base_lr,
        global_batch_size=args.global_batch_size,
        weight_decay=args.weight_decay,
        eta=0.001
    )

    scaled_lr = args.base_lr * (args.global_batch_size / 256.0)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        base_lr=scaled_lr
    )

    # 统一数据增强流水线
    augmenter = SpectrumAugmentation(mode='pretrain')
    
    # 混合精度 Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 训练主循环
    best_loss = float('inf')
    loss_history = []
    start_time = datetime.now()

    for epoch in range(1, args.max_epochs + 1):
        if not use_dali:
            sampler.set_epoch(epoch)

        current_lr = lr_scheduler.step(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_dali:
            # 动态构建 DALI Pipeline
            pipe = create_dali_pipeline(
                external_data=spectra,
                batch_size=local_batch_size,
                shard_id=rank,
                num_shards=world_size,
                num_threads=2,
                device_id=local_rank
            )
            pipe.build()
            dali_iter = DALIGenericIterator(
                pipe,
                output_map=["spectra"],
                last_batch_policy=LastBatchPolicy.DROP
            )
            batch_iterable = dali_iter
        else:
            batch_iterable = dataloader

        pbar = tqdm(batch_iterable, desc=f'Epoch {epoch:3d}/{args.max_epochs}', disable=not is_main_process)
        
        for item in pbar:
            if use_dali:
                spec_batch = item[0]["spectra"].to(device, non_blocking=True)
            else:
                spec_batch = item[0].to(device, non_blocking=True)

            # 在 GPU 上执行 Batch 特征增强
            view1 = augmenter(spec_batch)
            view2 = augmenter(spec_batch)
            views = torch.cat([view1, view2], dim=0)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    _, z = model(views)
                    z1, z2 = torch.chunk(z, 2, dim=0)
                    loss = nt_xent_loss(z1, z2, temperature=args.temperature, use_ddp=(world_size > 1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, z = model(views)
                z1, z2 = torch.chunk(z, 2, dim=0)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature, use_ddp=(world_size > 1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if is_main_process:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        avg_loss = epoch_loss / max(1, n_batches)

        # 多卡 Loss 聚合打日志
        if world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()

        if is_main_process:
            loss_history.append(avg_loss)
            elapsed = datetime.now() - start_time
            print(f'Epoch {epoch:3d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | 耗时: {str(elapsed).split(".")[0]}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                raw_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': raw_model.encoder.state_dict(),
                    'loss': avg_loss,
                }, str(output_dir / 'best_model.pt'))
                print(f'  ✓ 已更新保存最佳预训练模型 (Loss: {avg_loss:.6f})')

    if is_main_process:
        raw_model = model.module if hasattr(model, 'module') else model
        torch.save(raw_model.encoder.state_dict(), str(output_dir / 'pretrained_encoder_final_v1.pt'))
        print(f"\n[OK] 最终编码器已保存: {output_dir / 'pretrained_encoder_final_v1.pt'}")
        plot_loss_history(loss_history, best_loss, output_dir / 'pretraining_loss.png')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
