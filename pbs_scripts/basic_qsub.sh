#!/bin/bash

#PBS -l select=1:ngpus=4:ncpus=16:mpiprocs=1
#PBS -l walltime=24:00:00
#PBS -o train_cifar_single_node.txt
#PBS -j oe
#PBS -N train_cifar_single_node
#PBS -q gpu

# Change to the desired working directory and set PBS workDIR
PBS_O_WORKDIR=/work/dserez/
cd ${PBS_O_WORKDIR}

# activate conda environment
source activate pytorch_base

SCRIPT='./single_node_multi_gpu.py --epochs 2'
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ${SCRIPT}

