# mlcommons-osmi

#setup:

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
Source ~/.bashrc
conda create -n osmi python=3.8
Conda activate osmi
//to run interactive job on compute node (makes writing to files a lot faster)
ijob -c 1 -A bii_dsc_community -p standard --time=1-00:00:00 
Cd /project/bii_dsc_community/osmibench/code/osmibench/models
//Edit requirement grpc to grpcio
//there are sometimes problems with installing libraries so try conda if pip doesn’t work
Pip install –user  -r ../requirements.py or Conda install –file ../requirements.py
Conda install grpcio
//Renamed .local to avoid the error
Pip install –user tensorflow requests tqdm
pip install tensorflow-serving-api
Python train.py small_lstm
//results are in small_lstm/1
python train.py medium_cnn
Python train.py large_tcnn
Cd .. 
singularity pull docker://bitnami/tensorflow-serving [for cpu]
singularity pull docker://tensorflow/serving:latest-gpu
//Edit benchmark/models.conf to make base_path: "/project/bii_dsc_community/osmibench/code/osmi-bench/models/small_lstm",
//open different terminal and ssh
```

To run:

```
Conda activate osmi
singularity shell --nv --home `pwd` tensorflow-serving-gpu_latest.sif
Nvidia-smi //to see if you can use gpus (on node)
Cd benchmark
tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &
Cat log //to check its working
lsof -i :8500 // to make sure it an accept incoming directions, doesn’t work on ijob so ignore
//Edit tfs_grpc_client.py to make sure all the models use float32
python tfs_grpc_client.py -m [model, e.g. small_lstm] -b [batch size, e.g. 32] -n [# of batches, e.g. 10]  localhost:8500
```
F

//simpler way
ijob -c 1 -A bii_dsc_community -p standard --time=1-00:00:00 --partition=bii-gpu --gres=gpu
conda activate osmi
cd /project/bii_dsc_community/osmibench/code/osmi-bench/benchmark
singularity run --nv --home `pwd` ../serving_latest-gpu.sif tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &
sleep 10
python tfs_grpc_client.py -m large_tcnn -b 128 -n 100 localhost:8500
//run with slurm script
Cd /project/bii_dsc_community/osmibench/code/osmi-bench/benchmark
Sbatch test_script.slurm

//multiple gpus
Use -gres=gpu:v100:6
//in benchmark directory
singularity exec --bind `pwd`:/home --pwd /home     ../haproxy_latest.sif haproxy -d -f haproxy-grpc.cfg >& haproxy.log &
Cat haproxy.log
CUDA_VISIBLE_DEVICES=0 singularity run --home `pwd` --nv ../serving_latest-gpu.sif tensorflow_model_server --port=8500 --model_config_file=models.conf >& tfs0.log &
Cat tfs0.log
CUDA_VISIBLE_DEVICES=1 singularity run --home `pwd` --nv ../serving_latest-gpu.sif tensorflow_model_server --port=8501 --model_config_file=models.conf >& tfs1.log &
Cat tf
//do this for all gpus with different ports
