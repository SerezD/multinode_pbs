#!/bin/bash

# TODO automatic loading of correct module
# Actually not working! Manual loading below
# echo "--------------------------------"
# openmpi=`module avail | grep -o '\S*openmpi\S*gcc\S*' | sort | tail -1`
# module load $openmpi
# echo "loaded last openmpi version available on this server: $openmpi"
# echo "--------------------------------"

echo "************************************"
echo "************************************"
module load mpi/openmpi/4.1.4/gcc8-ib
echo "************************************"
echo "************************************"
# TODO

# set environment variables to speed up torch distributed communication (optional)
# https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html
BASE_ENVS="-x NCCL_NSOCKS_PERTHREAD=8 -x NCCL_SOCKET_NTHREADS=16 -x NCCL_MIN_NCHANNELS=16"

# get num nodes/gpus per node
# create nodes array (contains the name of all nodes)
# create gpus array (contains the number of gpus assigned to each node)
nodes_gpus=`sort $PBS_NODEFILE | uniq -c | xargs`
nodes_gpus=($nodes_gpus)
declare -a nodes=()
declare -a gpus=()
for item in "${nodes_gpus[@]}"
do
  # is item a number ? yes --> number of gpus | no --> node name
  if [[ $item =~ ^[0-9]+$ ]] ; then
    gpus+=($item)
  else
    nodes+=($item)
  fi
done

echo "assigned nodes: ${nodes[@]}"
echo "assigned gpus per node: ${gpus[@]}"

# env variables for distributed training
# https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_intermediate_1.html
# https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide

# world size: total number of nodes (len of lines in NODEFILE)
NODES=${#nodes[@]}
echo "NODES "$NODES

 
MASTER_ADDR=${nodes[0]}  # first node is master
echo "MASTER "$MASTER_ADDR

# find free port on $MASTER_ADDR
# comm -23 compares two files and report unique lines only of first one (-23).
# first file is a sequence of ports in the specified range, sorted alphabetically
# second file checks for all used ports on $MASTER_ADDR and sorts alphabetically
MASTER_PORT=$(ssh $MASTER_ADDR "comm -23  <(seq 4000 30000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1")
# returns also a trash string from ssh. Set as array and take last word (port number)
MASTER_PORT=($MASTER_PORT)  
MASTER_PORT=${MASTER_PORT[-1]}
echo "PORT "$MASTER_PORT

# add distributed variables
BASE_ENVS="${BASE_ENVS} -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT -x NODES=$NODES"


# cd to Project Path
cd $PPATH


# launch mpirun script on each node
for (( i=0; i<$NODES; i++ ))
do
  NODE_RANK=$i
  GPUS=${gpus[$i]}
  NODE=${nodes[$i]}
  ENVS="${BASE_ENVS} -x NODE_RANK=$NODE_RANK"
  
  mpirun $ENVS --np 1 --bind-to none --map-by slot --mca pml ob1 --mca btl ^openib --H $NODE $EXEC $SCRIPT &
done

echo "bash script here! Waiting for all jobs to finish..."
echo "#################################################################################"
wait
echo "#################################################################################"
echo "bash script here! Bye Bye"

