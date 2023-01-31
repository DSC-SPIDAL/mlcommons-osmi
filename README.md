# mlcommons-osmi

Authors: Nate, Gregor von Laszewski

## Running OSMI benchmark on rivanna

To run the OSMI benchmark, you will first need to generate the project directory with the code. We assume you are in the group `bii_dsc_community`. THis allows you access to the directory 

```/project/bii_dsc_community```

As well as the slurm partitions `gpu` and `bii_gpu`

## Set up a project directory

Firts you need to create a directory under your username in the project directory. We recommend to use your username. Follow these setps: 

```
mkdir -p /project/bii_dsc_community/$USER/osmi
cd /project/bii_dsc_community/$USER/osmi
```

## Set up Python via Miniforge and Conda

Next we recommend that you set up python. Although Conda is not our favorite development environment, we use conda here out of convenience. In future we will also document here how to set OSMI up with an environment from python.org useing vanillla python installs.

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
Source ~/.bashrc
conda create -n osmi python=3.8
Conda activate osmi
```

## Get the code

Tho get the code we clone a gitlab instance that is hosted at Oakridge National Laboratory (<https://code.ornl.gov/whb/osmi-bench>). 
To get the code, please execute

```
cd /project/bii_dsc_community/$USER/osmi
git clone https://code.ornl.gov/whb/osmi-bench.git
cd osmi-bench
```

## Interactig with Rivanna

Rivanna has two brimary modes so users can interact with it. 

* **Interactive Jobs:** The first one are interactive jobs taht allow you to 
  reseve a node on rivanna so it looks like a  login node. This interactive mode is
  usefull only during the debug phase and can serve as a conveneinet way to create 
  quickly batch scripts that are run in the second mode.

*  **Batch Jobs:** The second mode is a batch job that is controlled by a batch script. 
   We will showcase here how to set such scripts up and use them 

### Compile OSMI Models in Interactive Jobs

Once you know hwo to create jobs with a propper batch script you will likely no longer need to use interactive jobs. We keep this documentation for beginners that like to experiement in interactive mode to develop batch scripts.

We noticed that when running interactive jobs on compute node it makes writing to the files system a lot faster.
TODO: This is inprecise as its not discussed which file system ... Also you can just use git to sync

First, obtain an interactive job with 

```
ijob -c 1 -A bii_dsc_community -p standard --time=1-00:00:00
```

Next

```
cd /project/bii_dsc_community/osmibench/code/osmibench/models
```

Now Edit requirement grpc to grpcio (TODO: THis is unclear)

There are sometimes problems with installing libraries so try conda if pip doesn’t work
(TODO: This is unclear and not precise enough ther must not be any problems, ...)

The next instructions are not clear as you either do requirement or conda, also the file is called 

requirements.txt

```
pip install –user  -r ../requirements.py 
conda install –file ../requirements.py
conda install grpcio
```

TODO this is unclear ...

At this point I hat to rename .local to avoid an error

TODO: if pip or conda would have worked properly with requirements the following would already been taken care off, so this is not needed. IF additional requirements are needed, tey should be in this git repo and we need to use that requiremnts.txt filw also. Maybe it needs to be called requrements-rivanna-conda.txt or requrements-rivanna-python.txt

```
pip install –user tensorflow requests tqdm
pip install tensorflow-serving-api
python train.py small_lstm
```

Results are in small_lstm/1

```
python train.py medium_cnn
python train.py large_tcnn
cd .. 
singularity pull docker://bitnami/tensorflow-serving [for cpu]
singularity pull docker://tensorflow/serving:latest-gpu
```

Edit benchmark/models.conf to make each base_path correspond to the proper directory e.g. "/project/bii_dsc_community/osmibench/code/osmi-bench/models/small_lstm",

For this application there is no separate data

## run the client-side tests

```
singularity shell --nv --home `pwd` tensorflow-serving-gpu_latest.sif
nvidia-smi #to see if you can use gpus (on node)
Cd benchmark
tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &
Cat log //to check its working
lsof -i :8500 // to make sure it an accept incoming directions, doesn’t work on ijob so ignore
```
Edit tfs_grpc_client.py to make sure all the models use float32
python tfs_grpc_client.py -m [model, e.g. small_lstm] -b [batch size, e.g. 32] -n [# of batches, e.g. 10]  localhost:8500

simpler way

```
ijob -c 1 -A bii_dsc_community -p standard --time=1-00:00:00 --partition=bii-gpu --gres=gpu
conda activate osmi
cd /project/bii_dsc_community/osmibench/code/osmi-bench/benchmark
singularity run --nv --home `pwd` ../serving_latest-gpu.sif tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &
sleep 10
python tfs_grpc_client.py -m large_tcnn -b 128 -n 100 localhost:8500
```
run with slurm script

```
cd /project/bii_dsc_community/osmibench/code/osmi-bench/benchmark
Sbatch test_script.slurm
```

multiple gpus

Use -gres=gpu:v100:6

in benchmark directory

```
singularity exec --bind `pwd`:/home --pwd /home     ../haproxy_latest.sif haproxy -d -f haproxy-grpc.cfg >& haproxy.log &
cat haproxy.log
CUDA_VISIBLE_DEVICES=0 singularity run --home `pwd` --nv ../serving_latest-gpu.sif tensorflow_model_server --port=8500 --model_config_file=models.conf >& tfs0.log &
cat tfs0.log
CUDA_VISIBLE_DEVICES=1 singularity run --home `pwd` --nv ../serving_latest-gpu.sif tensorflow_model_server --port=8501 --model_config_file=models.conf >& tfs1.log &
cat tf
```

do this for all gpus with different ports

## References

1. Production Deployment of Machine-Learned Rotorcraft Surrogate Models on HPC, Wesley Brewer, Daniel Martinez, 
   Mathew Boyer, Dylan Jude, Andy Wissink, Ben Parsons, Junqi Yin, Valentine Anantharaj
   2021 IEEE/ACM Workshop on Machine Learning in High Performance Computing Environments (MLHPC),
   978-1-6654-1124-0/21/$31.00 ©2021 IEEE | DOI: 10.1109/MLHPC54614.2021.00008, <https://ieeexplore.ieee.org/document/9652868>
   TODO: please ask wess what the free pdf link is all gov organizations have one. for example as ornl is coauther it 
   must be on their site somewhere.
   

2. Using Rivanna for GPU ussage, Gregor von Laszewski, JP. Fleischer 
   <https://github.com/cybertraining-dsc/reu2022/blob/main/project/hpc/rivanna-introduction.md>

3. Setting up a Windows computer for research, Gregor von Laszewski, J.P Fleischer 
   <https://github.com/cybertraining-dsc/reu2022/blob/main/project/windows-configuration.md>
   
4. INitial notes to be deleted, Nate: <https://docs.google.com/document/d/1luDAAatx6ZD_9-gM5HZZLcvglLuk_OqswzAS2n_5rNA>

