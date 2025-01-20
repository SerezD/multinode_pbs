# How to run multi-gpu and multi-node pytorch scripts on HPC cluster with PBS job scheduler

### Initial Setup

Install miniconda and (optional but suggested) set libmamba solver.

- [https://docs.anaconda.com/miniconda/install/](https://docs.anaconda.com/miniconda/install/)
- [https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)

Scripts for automatic installation and set up:   
`./initial_setup/install_miniconda.sh -d <installation_directory>`  
`./initial_setup/libmamba_solver.sh`  

If not already installed on the cluster, you can install useful things like `TMUX`, `GIT` and `HTOP` directly in 
`conda base`:

- [https://github.com/tmux/tmux/wiki](https://github.com/tmux/tmux/wiki)  
- [https://git-scm.com/](https://git-scm.com/)  
- [https://htop.dev/](https://htop.dev/)  

To conclude the setup, install your conda environment with pytorch and cuda support: 

`conda env install -f environment.yml`  

_Note: adjust cuda version as needed_

### Single Node - Multi GPU

We use `torchrun`: 
 - [https://pytorch.org/docs/stable/elastic/run.html](https://pytorch.org/docs/stable/elastic/run.html)  

`Pytorch` code example using `Distributed-Data-Parallel` in `./python_scripts/single_node_multi_gpu.py`  

Simple `PBS` script, requesting 4 gpus in `./pbs_scripts/basic_qsub.sh`  
If you need to run the same script multiple times with different parameters (ablation studies): 
`./pbs_scripts/qsub_ablations.sh`  

PBS documentation for qsub: 
- [https://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm](https://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm)



### Multi Node - Multi GPU

We need communication between nodes. In practice, we must set some environment variables:  
1. `MASTER_ADDR`: address of the master node.  
2. `MASTER_PORT`: free communication port on the master node.  
3. `WORLD_SIZE`: total number of processes used (usually `num_gpu * num_nodes`).  
4. `NODE_RANK`: number rank, different for each node (master is usually 0).  

Suggested Reading:  
- [https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide)

The same thing can be done in two ways: 
1. torchrun (similar to single node, multi gpu)
2. openmpi: [https://www.open-mpi.org/](https://www.open-mpi.org/)

All the environment variables and setup is managed in the scripts:
- `./pbs_scripts/multinode_torchrun.sh`
- `./pbs_scripts/multinode_mpirun.sh`

The former needs to `ssh` to each node and execute `torchrun` from there.  
The second loads openmpi library with `module load`: [https://modules.readthedocs.io/en/latest/](https://modules.readthedocs.io/en/latest/).  
Check both scripts and adapt as needed. 

Both scripts can be run by adapting `./pbs_scripts/qsub_ablations_multinode.sh`

### Useful `PBS` Commands to monitor Queues and Jobs

- `qstat -wan1`: monitor all jobs (in all states) and their nodes.  
- `qstat -wan1 -u $user`: monitor launched jobs and requested resources by `$user`.
- `qstat -wrn1`: monitor all running jobs.
- `qstat -wan1 | grep Q`: filter only queued jobs.
- `qstat -q`: overview on all queues.
- `qstat -fQ`: see details of queues.
- `qstat -u $user`: see all jobs submitted by $user.
- `qstat -f $jobid`: see details of specific job.
- `qstat -u $user | grep "$user" | cut -d"." -f1 | xargs qdel`: kill all jobs of `$user`.
- `qstat -$user | grep "R" | cut -d"." -f1 | xargs qdel`: kill all the running jobs of `$user`.
- `qstat -u $user | grep "Q" | cut -d"." -f1 | xargs qdel`: kill all the queued jobs of `$user`.
- `pbsnodes -aSj`: see all nodes on the cluster, the jobs running on each and free resources.
- `pbsnodes -aSj | head -n 1 && pbsnodes -aSj | grep anode`: filter only on `anodes`. 
- `pbsnodes -aSj | head -n 1 && pbsnodes -aSj | grep gnode | grep free && pbsnodes -aSj | grep gnode | grep various`: 
see all `free` or `various` `gnodes`. 