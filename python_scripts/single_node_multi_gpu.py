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


def train(rank: int, world_size: int, args: argparse.Namespace):

    setup(rank, world_size, args.backend)

    # Create Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if rank == 0:
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    dist.barrier(device_ids=[rank])

    if rank != 0:
        dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    dist.barrier(device_ids=[rank])
    print(f'rank: {rank} has passed the barrier!')

    # Dataloader with Distributed Sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Model, loss, optimizer
    model = SimpleMLP(input_size=32 * 32 * 3, num_classes=10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):

        # set sampler epoch to use a different ordering of batches on each epoch
        sampler.set_epoch(epoch)
        ddp_model.train()

        for batch_idx, (data, target) in enumerate(dataloader):

            # load data to correct device
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}], "
                    f"Step [{batch_idx}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}")

        print(f'Rank: {rank} finished epoch: {epoch}')
        dist.barrier(device_ids=[rank])

    # Save model checkpoint
    if rank == 0:
        print('Saving model...')
        torch.save(ddp_model.state_dict(), "model_checkpoint.pth")

    cleanup()


if __name__ == "__main__":
    """
    How to RUN: 
    torchrun --nproc_per_node=ngpus --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 train.py --args
    
    Official pytorch Docs: https://pytorch.org/docs/stable/elastic/run.html
    """

    # Environment Variables created by torchrun
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # different on each process/gpu
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # total number of gpus

    arguments = parse_args()
    train(LOCAL_RANK, WORLD_SIZE, arguments)
