#!/bin/bash
#SBATCH --job-name=simclr_pretrain_4096
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/simclr_pretrain_%j.log
#SBATCH --error=logs/simclr_pretrain_%j.err

echo "================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on Nodes: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "================================================="

# 创建日志文件夹
mkdir -p logs

# 分布式网络配置
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

# 设置 GPU 卡数
NUM_GPUS=${SLURM_GPUS_PER_NODE:-4}

echo "Master Address: $MASTER_ADDR"
echo "GPUs Per Node: $NUM_GPUS"

# 使用 torchrun 启动 4 卡 A100 DDP 4096 大 Batch 预训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    embedding_pretrain/main.py \
    --data_path embedding_pretrain/data_source/preprocessed_spectra.parquet \
    --global_batch_size 4096 \
    --base_lr 0.3 \
    --warmup_epochs 10 \
    --max_epochs 500 \
    --use_dali

echo "================================================="
echo "Pretraining Task Completed at $(date)"
echo "================================================="
