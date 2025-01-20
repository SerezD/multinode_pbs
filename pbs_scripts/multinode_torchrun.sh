#!/bin/bash

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

# total number of nodes (len of lines in NODEFILE)
NUM_NODES=${#nodes[@]}

MASTER_ADDR=${nodes[0]}  # first node is master
echo "MASTER "$MASTER_ADDR

# find free port on $MASTER_ADDR
# comm -23 compares two files and report unique lines only of first one (-23).
# first file is a sequence of ports in the specified range, sorted alphabetically
# second file checks for all used ports on $MASTER_ADDR and sorts alphabetically
MASTER_PORT=$(ssh $MASTER_ADDR "comm -23  <(seq 4000 30000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1")
# returns also a trash string from ssh. Set as array and take last work (port number)
MASTER_PORT=($MASTER_PORT)
MASTER_PORT=${MASTER_PORT[-1]}
echo "PORT "$MASTER_PORT

for ((i = 0; i < NUM_NODES; i++)); do
    ssh ${nodes[i]} /bin/bash << ENDSSH &
    
    echo "NODE: ${nodes[i]} Start"
    echo "Activating conda environment: ${ENV_NAME}"
    source activate ${ENV_NAME}

    # Change to the working directory
    PBS_O_WORKDIR=/work/dserez/
    cd $PBS_O_WORKDIR
    
    # Run the Python script
    torchrun --nproc_per_node=${gpus[i]} --nnodes=$NUM_NODES --node_rank=$i --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${SCRIPT}
    echo "NODE: ${nodes[i]} Done"

ENDSSH
done


echo "bash script here! Waiting for all jobs to finish..."
echo "#################################################################################"
wait
echo "#################################################################################"
echo "bash script here! Bye Bye"

