#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train-osmi
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb

## SBATCH -c 4
## SBATCH --mem=32GB

module load anaconda
conda activate OSMI
pip install --user -r /project/bii_dsc_community/$USER/osmi/mlcommons-osmi/requirements-rivanna.txt
cd /project/bii_dsc_community/$USER/osmi/osmi-bench/benchmark
singularity run --nv --home `pwd` ../serving_latest-gpu.sif tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &
# sleep 12 # check that tensorflow is running

if [[ $(lsof -i :8500) ]]; then echo "yes"; fi
nvidia-smi
python tfs_grpc_client.py -m small_lstm -b 32 -n 48 localhost:8500
