"""
SimCLR 多卡 DDP 预训练入口 (2 卡 G100 torchrun 启动)

支持功能：
1. 适应 2 卡 G100 高性能 DDP 分布式训练 (默认 2048 Batch Size, 每卡 Local Batch Size 1024)
2. 从 Parquet 数据源读取已标准化预处理的 561 维质谱特征向量 (373,330 条样本)
3. GPU 显存全量数据预加载 (零 I/O 拷贝，零 PCIe 传输与 DataLoader 阻塞)
4. GPU 向量化质谱增强流 (Sample-wise 强度抖动 + 静态位置张量缓存，保护拓扑结构与相对峰比例)
5. 工业级 LARS 优化器 (对齐 SimCLR 规范排除 Bias 与 GroupNorm/LayerNorm Weight Decay)
6. 推导的最优初始学习率 (base_lr=0.030, 8x 线性缩放后峰值 LR=0.240, 最大允许 500 Epochs / 91,000 Steps 余弦退火)
7. DDP 全局 All-Gather 向量化 NT-Xent 对比损失计算与早停收敛检测
"""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
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


def load_spectra_from_parquet(data_path: Path) -> np.ndarray:
    """直接读取已预处理好的 Parquet 质谱特征向量文件"""
    if not data_path.exists():
        raise FileNotFoundError(f"错误: 未找到预训练数据文件 {data_path}")
    
    table = pq.read_table(str(data_path), columns=['spectrum_vector'])
    raw_list = table['spectrum_vector'].to_pylist()
    spectra = np.array(raw_list, dtype=np.float32)
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
            lr = self.base_lr * (self.current_epoch / max(1, self.warmup_epochs))
        else:
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
    base_dir = Path(__file__).resolve().parent
    default_data_path = base_dir / "data_source" / "preprocessed_spectra.parquet"

    parser = argparse.ArgumentParser(description="SimCLR 2卡 G100 DDP 对比学习预训练")
    parser.add_argument("--data_path", type=str, default=str(default_data_path), help="预训练 Parquet 数据路径")
    parser.add_argument("--global_batch_size", type=int, default=2048, help="全局 Batch Size (默认 2048，适应 2 卡 G100 每卡 1024)")
    parser.add_argument("--base_lr", type=float, default=0.030, help="基础学习率 (默认 0.030, 8x 线性缩放后 Peak LR=0.240)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Linear Warmup 轮数")
    parser.add_argument("--max_epochs", type=int, default=500, help="最大允许训练轮数 (默认 500 Epochs)")
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent 温度系数")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="LARS Weight Decay")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值 (Patience)")
    parser.add_argument("--max_samples", type=int, default=None, help="限制样本数量 (用于本地测试，默认全量)")
    parser.add_argument("--output_dir", type=str, default=str(base_dir), help="模型输出保存目录")
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_ddp()
    is_main_process = (rank == 0)

    data_path = Path(args.data_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.output_dir) if args.output_dir else base_dir
    if "results_" not in base_out.name:
        output_dir = base_out / f"results_{timestamp}"
    else:
        output_dir = base_out
    
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("SimCLR 2 卡 G100 DDP 对比学习预训练 (GPU 显存全预载 + 向量化掩码 Loss + LARS)")
        print("=" * 60)
        print(f"Rank: {rank}/{world_size} | Device: {device}")
        print(f"数据文件: {data_path}")
        print(f"输出目录: {output_dir}")
        print(f"全局 Batch Size: {args.global_batch_size}")
        print(f"每卡 Local Batch Size: {args.global_batch_size // world_size}")
        print(f"最大允许 Epochs: {args.max_epochs}")

    # 从 Parquet 加载数据集
    spectra = load_spectra_from_parquet(data_path)
    if args.max_samples is not None:
        spectra = spectra[:args.max_samples]

    if is_main_process:
        print(f"成功加载质谱数据: {spectra.shape[0]:,} 个样本, 维度: {spectra.shape[1]}")

    local_batch_size = max(1, args.global_batch_size // world_size)

    # 极速优化：全量数据集直接预加载驻留 GPU 显存 (彻底消除 CPU-GPU PCIe 传输与 PyTorch DataLoader 开销)
    if device.type == 'cuda':
        if is_main_process:
            print("[Data Pipeline] 将数据集一次性预加载驻留 GPU 显存 (零 CPU 传输延迟)...")
        full_tensor = torch.from_numpy(spectra).to(device, non_blocking=True)
        # 多卡 DDP 样本切分 (Strided shard)
        rank_spectra = full_tensor[rank::world_size]
        use_gpu_direct_loader = True
    else:
        use_gpu_direct_loader = False
        dataset = TensorDataset(torch.from_numpy(spectra))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=0,
            drop_last=True
        )

    # 初始化模型结构
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

    # 统一 GPU 向量化数据增强流水线
    augmenter = SpectrumAugmentation(mode='pretrain')
    
    # 拟合收敛检测器
    convergence_checker = ConvergenceChecker(
        patience=args.patience,
        min_delta=1e-4,
        window_size=10
    )

    # 混合精度 Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 训练主循环
    best_loss = float('inf')
    loss_history = []
    start_time = datetime.now()

    for epoch in range(1, args.max_epochs + 1):
        current_lr = lr_scheduler.step(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_gpu_direct_loader:
            # 在 GPU 显存内生成零延迟打乱索引
            perm = torch.randperm(len(rank_spectra), device=device)
            shuffled_spectra = rank_spectra[perm]
            num_batches = len(shuffled_spectra) // local_batch_size
            
            pbar = tqdm(range(num_batches), desc=f'Epoch {epoch:3d}/{args.max_epochs}', disable=not is_main_process)
            
            for b_idx in pbar:
                spec_batch = shuffled_spectra[b_idx * local_batch_size : (b_idx + 1) * local_batch_size]

                # 在 GPU 上高效执行 2D Batch 矩阵特征增强
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

        else:
            if sampler is not None:
                sampler.set_epoch(epoch)

            pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d}/{args.max_epochs}', disable=not is_main_process)
            
            for (batch_data,) in pbar:
                spec_batch = batch_data.to(device, non_blocking=True)

                # 在 GPU 上高效执行 2D Batch 矩阵特征增强
                view1 = augmenter(spec_batch)
                view2 = augmenter(spec_batch)
                views = torch.cat([view1, view2], dim=0)

                optimizer.zero_grad(set_to_none=True)

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
                print(f'  [OK] 已更新保存最佳预训练模型 (Loss: {avg_loss:.6f})')

            # 早停拟合检测
            if convergence_checker.update(avg_loss):
                print(f"\n{'='*60}")
                print(f"模型在第 {epoch} 轮收敛，停止训练")
                print(f"{'='*60}")
                break

    if is_main_process:
        raw_model = model.module if hasattr(model, 'module') else model
        final_encoder_path = output_dir / 'pretrained_encoder_final_v1.pt'
        torch.save(raw_model.encoder.state_dict(), str(final_encoder_path))
        print(f"\n[OK] 最终编码器已保存至结果目录: {final_encoder_path}")

        # 同步更新共享编码器权重供后训练 (finetune) 模块直接加载
        shared_encoder_path = base_dir / 'pretrained_encoder_final_v1.pt'
        torch.save(raw_model.encoder.state_dict(), str(shared_encoder_path))
        print(f"[OK] 已同步更新后训练共享编码器: {shared_encoder_path}")

        plot_loss_history(loss_history, best_loss, output_dir / 'pretraining_loss.png')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
