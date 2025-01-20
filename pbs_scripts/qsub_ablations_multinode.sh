#!/bin/bash

CHUNKS=2
COMM="select=${CHUNKS}:ngpus=2:ncpus=4" # -> torchrun
# COMM="select=${CHUNKS}:ngpus=2:ncpus=4:mpiprocs=2"  # -> mpirun
QUEUE=gpu
TIME="04:00:00"

# define script args
for lr in 0.01
do

  # launch script command with args
  RUN_NAME="train_cifar_lr-${lr}"

  SCRIPT="./multi_node_multi_gpu.py --epochs 2 --lr ${lr}"

  qsub -l ${COMM} -l walltime=${TIME} -j oe -o /work/dserez/${RUN_NAME}.txt -N ${RUN_NAME} -q ${QUEUE} \
       -v ENV_NAME="pytorch_base",SCRIPT="${SCRIPT}" ./multinode_torchrun.sh
  # qsub -l ${COMM} -l walltime=${TIME} -j oe -o /work/dserez/${RUN_NAME}.txt -N ${RUN_NAME} -q ${QUEUE} \
  # -v ENV_NAME="pytorch_base",SCRIPT="${SCRIPT}" ./multinode_mpirun.sh
done


