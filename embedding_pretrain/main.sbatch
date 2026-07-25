#!/bin/bash
#SBATCH --job-name=simclr_pretrain
#SBATCH --gpus=4
#SBATCH --output=logs/%j.out

mkdir -p logs

torchrun --nproc_per_node=4 embedding_pretrain/main.py --use_dali
