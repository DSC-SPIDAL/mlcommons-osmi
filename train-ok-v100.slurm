#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-12:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1

module load anaconda
conda activate OSMI
pip install --user -r /project/bii_dsc_community/$USER/osmi/mlcommons-osmi/requirements-rivanna.txt

cd /project/bii_dsc_community/tma5gv/osmi/osmi-bench/model
time python train.py small_lstm
time python train.py medium_cnn
time python train.py large_tcnn