#!/bin/bash
#SBATCH --job-name=poisson       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=7        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=256G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=bii-gpu:8             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate myenv

#srun python myscript.py
nvidia-smi
