#!/usr/bin/env bash
#SBATCH --job-name=mlcommons-eq-{experiment.card_name}-{experiment.gpu_count}
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --partition={system.partition}
#SBATCH --account={system.allocation}
#SBATCH --reservation={system.reservation}
#SBATCH --constraint={system.constraint}
#SBATCH -c {experiment.cpu_num}
#SBATCH --mem={experiment.mem}
#SBATCH --time={time}
#SBATCH --gres=gpu:{experiment.card_name}:{experiment.gpu_count}
#SBATCH --mail-user=%u@virginia.edu
#SBATCH --mail-type=ALL


set -uxe

source ~/OSMI
pip install --user -r /project/bii_dsc_community/$USER/osmi/mlcommons-osmi/requirements-rivanna.txt
cd /project/bii_dsc_community/$USER/osmibench/code/osmi-bench/benchmark
singularity run --nv --home `pwd` ../tensorflow-serving_latest.sif tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &
sleep 12 # check that tensorflow is running
nvidia-smi
python tfs_grpc_client.py -m {experiment.model} -b {experiment.batch_size} -n {experiment.batches} localhost:8500

exit 0
