#!/bin/bash

CHUNKS=2
COMM="select=${CHUNKS}:ngpus=1:ncpus=1:mpiprocs=1"

# Example for specific requiring specific nodes.
# COMM="select=1:host=gnode01:ngpus=1:ncpus=16:mpiprocs=1+1:host=gnode10:ngpus=1:ncpus=16:mpiprocs=1"

QUEUE=workq
TIME="48:00:00"

# working directory (all tmp files will be saved here)
WORK_DIR="/home/dserez/"

# path to python excecutable (inside conda env)
ENV_NAME=base
EXEC="${WORK_DIR}miniconda3/envs/${ENV_NAME}/bin/python"

# python script to launch (no args are added here, but you can add them if needed)
SCRIPT="${WORK_DIR}my_test.py"

qsub -l $COMM -l walltime=$TIME -j oe -N "Test" -q $QUEUE -v EXEC=$EXEC,SCRIPT="$SCRIPT",WORK_DIR=$WORK_DIR ./multinode_scripts/multinode.sh


