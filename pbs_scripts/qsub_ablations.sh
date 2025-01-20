#!/bin/bash

CHUNKS=1
COMM="select=${CHUNKS}:ngpus=4:ncpus=8"
QUEUE=gpu
TIME="24:00:00"

# define script args
for lr in 0.01 0.001
do

  # launch script command with args
  RUN_NAME="train_cifar_lr-${lr}"

  qsub -l ${COMM} -l walltime=${TIME} -j oe -o /work/dserez/${RUN_NAME}.txt -N ${RUN_NAME} -q ${QUEUE} <<EOF

    #!/bin/bash

    # change directory to work (default is /home)
    PBS_O_WORKDIR=/work/dserez/
    cd \$PBS_O_WORKDIR

    # activate conda environment
    source activate pytorch_base

    # Define and run the Python script
    SCRIPT='./single_node_multi_gpu.py --epochs 2 --lr \${lr}'
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 \${SCRIPT}
EOF
done


