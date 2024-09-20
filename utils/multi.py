import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import os

from torch.nn.parallel import DistributedDataParallel as DDP

n = torch.cuda.device_count()

rank          = int(os.environ["SLURM_PROCID"])
#gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
print(n,rank)