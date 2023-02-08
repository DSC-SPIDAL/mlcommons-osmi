#!/bin/bash
#SBATCH -q debug
#SBATCH --time=30
#SBATCH --job-name=train-cosmoflow
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH -c 4
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bii_dsc
#SBATCH --constraint=a100_80gb
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
