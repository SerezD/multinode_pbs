#!/bin/bash

CHUNKS=2  # Select number of Chunks
COMM="select=${CHUNKS}:ngpus=4:ncpus=32:mpiprocs=4"

# The following command may be used if you need specific nodes
# COMM="select=1:host=gnode06:ngpus=4:ncpus=32:mpiprocs=4+1:host=gnode07:ngpus=4:ncpus=32:mpiprocs=4"

# QUEUE NAME
QUEUE=workq

# working directory (all tmp files will be saved here)
WORK_DIR="/home/dserez/"

# path to python excecutable (inside conda env)
EXEC="${WORK_DIR}miniconda3/envs/my_env_name/bin/python"

# python script to launch
SCRIPT="${WORK_DIR}/main.py"

# define script args (depend on script)
for conf in 'conf_file_name_1' 'conf_file_name_2'
do

  # Uncomment to Resume a Training and WANDB RUN
  # ckpt="last.ckpt"
  # wandb_id="7ptext2z" # "43h7wuxk" "7ptext2z"
  ARGS="--conf $conf " # --ckpt $ckpt --wandb_id $wandb_id"

  # set output name
  NAME="${conf}"

  # launch script command with args
  SCRIPT=$SCRIPT" "$ARGS

  qsub -l $COMM -j oe -N "$NAME" -q $QUEUE -v EXEC=$EXEC,SCRIPT="$SCRIPT",WORK_DIR=$WORK_DIR ./multinode.sh
done

