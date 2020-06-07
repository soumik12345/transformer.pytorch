import torch
import numpy as np
from copy import deepcopy


def clones(module, n):
    return torch.nn.ModuleList(
        [deepcopy(module) for _ in range(n)]
    )


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
