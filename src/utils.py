import torch
from copy import deepcopy


def clones(module, n):
    return torch.nn.ModuleList(
        [deepcopy(module) for _ in range(n)]
    )
