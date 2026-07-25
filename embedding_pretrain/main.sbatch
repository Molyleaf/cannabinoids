#!/bin/bash
#SBATCH --job-name=simclr_pretrain_g100
#SBATCH --gpus=2
#SBATCH --output=logs/%j.out

mkdir -p logs

torchrun --nproc_per_node=2 embedding_pretrain/main.py
