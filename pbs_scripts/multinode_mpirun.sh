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

# get WORLD_SIZE (total number of processes) and NODES_GPUS (host:n_procs, ... )
WORLD_SIZE=0
NODES_GPUS=""
for ((i = 0; i < NUM_NODES; i++)); do
    ((WORLD_SIZE += gpus[i]))
    NODES_GPUS="${NODES_GPUS}${nodes[i]}:${gpus[i]},"
done
NODES_GPUS="${NODES_GPUS%?}"

echo "NUM_NODES-> ${NUM_NODES} - WORLD_SIZE -> ${WORLD_SIZE} - NODES:GPUS -> ${NODES_GPUS}"

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

# add distributed variables
BASE_ENVS=" -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT"

# launch MPI run ONCE from master node!
MPI_COMM="mpirun ${BASE_ENVS} -np $WORLD_SIZE -H $NODES_GPUS "

echo $MPI_COMM

# Create a tmp script to run
cat << 'EOF' > /home/dserez/tmp.sh
  
#!/bin/bash

echo "Activating conda environment: ${ENV_NAME}"
source activate ${ENV_NAME}

# Change to the working directory
PBS_O_WORKDIR=/work/dserez/
cd $PBS_O_WORKDIR

# Run the Python script
python ${SCRIPT}

EOF

# Make the script executable
chmod +x /home/dserez/tmp.sh
  
# load open mpi and nccl 
module load gcc-8.5.0/ompi-4.1.4_nccl
  
# Run the script via mpirun
${MPI_COMM} /bin/bash /home/dserez/tmp.sh

# Remove the temporary script after execution
rm -f /home/dserez/tmp.sh

echo "bash script here! Waiting for all jobs to finish..."
echo "#################################################################################"
wait
echo "#################################################################################"
echo "bash script here! Bye Bye"

