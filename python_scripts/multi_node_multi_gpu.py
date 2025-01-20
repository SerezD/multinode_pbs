import argparse
import numpy as np
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training Example")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                        help='Backend for distributed training')
    return parser.parse_args()


def setup(rank: int, world_size: int, backend: str):

    # initialize group of processes (1 gpu = 1 proc)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # seed stuff for reproducibility (same initializations, optimizers steps, sampling, etc...)
    # https://grsahagian.medium.com/what-is-random-state-42-d803402ee76b#:~:text=The%20number%2042%20is%20sort,over%20the%20period%20of%207.5
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def cleanup():
    dist.destroy_process_group()


class SimpleMLP(nn.Module):
    def __init__(self, input_size=32 * 32 * 3, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train(local_rank: int, world_size: int, world_rank: int, args: argparse.Namespace):

    setup(world_rank, world_size, args.backend)

    # Create Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if world_rank == 0:
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    dist.barrier(device_ids=[local_rank])

    if world_rank != 0:
        dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    dist.barrier(device_ids=[local_rank])
    print(f'rank: {world_rank} has passed the barrier!')

    # Dataloader with Distributed Sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Model, loss, optimizer
    model = SimpleMLP(input_size=32 * 32 * 3, num_classes=10).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):

        # set sampler epoch to use a different ordering of batches on each epoch
        sampler.set_epoch(epoch)
        ddp_model.train()

        for batch_idx, (data, target) in enumerate(dataloader):

            # load data to correct device
            data, target = data.to(local_rank), target.to(local_rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # sync loss values across devices
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size

            if batch_idx % 10 == 0 and world_rank == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}], "
                    f"Step [{batch_idx}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}")

    # Save model checkpoint
    if world_rank == 0:
        torch.save(ddp_model.state_dict(), "model_checkpoint.pth")

    cleanup()


if __name__ == "__main__":
    """
    How to RUN: 
    mpirun -np world_size -H ip_node_0:n_gpus,ip_node_1:n_gpus ... -x MASTER_ADDR=ip_master -x MASTER_PORT=1234 python train.py --args
    
    """

    if 'LOCAL_RANK' in os.environ:
        # Environment Variables created by torchrun
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # Rank of the GPU on this NODE
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # n nodes * n gpus
        WORLD_RANK = int(os.environ['RANK'])        # Rank of the GPU Globally (on all nodes)
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        # Environment Variables created by mpirun
        LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])  # Rank of the GPU on this NODE
        WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])        # n nodes * n gpus
        WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])        # Rank of the GPU Globally (on all nodes)
    else:
        raise NotImplementedError("This script must be executed with torchrun or mpirun!")

    arguments = parse_args()
    train(LOCAL_RANK, WORLD_SIZE, WORLD_RANK, arguments)
