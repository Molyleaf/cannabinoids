#!/bin/bash
#SBATCH --job-name=simclr_pretrain_4096
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=simclr_pretrain_%j.log
#SBATCH --error=simclr_pretrain_%j.err

echo "================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on Nodes: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "================================================="

# 分布式网络配置
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

# 设置 GPU 卡数
NUM_GPUS=${SLURM_GPUS_PER_NODE:-4}

echo "Master Address: $MASTER_ADDR"
echo "GPUs Per Node: $NUM_GPUS"

# 使用 torchrun 启动多卡并行 SimCLR 4096 Batch 预训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    simclr_pretrain/train_ddp.py \
    --global_batch_size 4096 \
    --base_lr 0.3 \
    --warmup_epochs 10 \
    --max_epochs 500 \
    --use_dali

echo "================================================="
echo "Pretraining Task Completed at $(date)"
echo "================================================="
