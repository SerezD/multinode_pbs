import os
import platform

# what is my node and conda env 
print(f'NODE: {platform.node()} - CONDA ENV: {os.environ["CONDA_PREFIX"]}')

import torch

# is cuda available ? 
print(f"[{platform.node()}]: CUDA AVAIL? {torch.cuda.is_available()}")
print(f"[{platform.node()}]: COUNT: {torch.cuda.device_count()}")

#for name, value in os.environ.items():
#    print(f"[{platform.node()}]: {name}={value}")

# Environment variables set by torch.distributed.launch
print(f"[{platform.node()}]: MASTER ADDRESS {os.environ['MASTER_ADDR']}")
print(f"[{platform.node()}]: MASTER PORT {os.environ['MASTER_PORT']}")
print(f"[{platform.node()}]: NUM PROCESSES {os.environ['WORLD_SIZE']}")
print(f"[{platform.node()}]: NODE RANK {os.environ['NODE_RANK']}")
print('#' * 25)
