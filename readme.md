## MULTINODE ON PBS CLUSTER ENVIRONMENT

This repo contains instructions to launch multinode scripts on a pbs cluster, 
setting all the `ENV_VARIABLES` needed for a multi-gpu script with pytorch or pytorch Lightning.

#### STEPS TO TAKE
1. `test.sh`: launches the pbs command and set up python script.
2. `multinode_scripts/multinode.sh`: finds the environment variables to be defined. Also launches the `mpirun` command from master node.
Check `line 75` which loades the appropriate module (on my cluster) and the `- prefix` option of the `mpirun` command at line 71 (probably need to change it on a different cluster).
3. `multinode_scripts/run.sh`: Take care of launching the final script on each node, adding the final env variable.

#### ENV VARIABLES TO BE SET

1. `MASTER_ADDR`: address of the master node.
2. `MASTER_PORT`: free communication port on the master node.
3. `WORLD_SIZE`: total number of processes used (usually `num_gpu * num_nodes`).
4. `NODE_RANK`: number rank, different for each node (master is usually 0).

#### USEFUL RESOURCES

- https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide
- https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_intermediate_1.html
- https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html


#### USEFUL `PBS` COMMANDS

- `qstat -fQ`: see permissions of Queues (e.g. max num of parallel jobs)
- `pbsnodes -aSj | grep -F 'gnode' | grep -F 'free'`: see all `free` `gnode`. 
- `qstat -wan1 -u $user`: monitor all launched jobs and requested resources by `$user`.
- `qstat -u $user | grep "$user" | cut -d"." -f1 | xargs qdel`: kill all jobs of `$user`.
- `qstat -$user | grep "R" | cut -d"." -f1 | xargs qdel`: kill all the running jobs of `$user`.
- `qstat -u $user | grep "Q" | cut -d"." -f1 | xargs qdel`: kill all the queued jobs of `$user`.