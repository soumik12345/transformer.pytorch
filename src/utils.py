import torch
from copy import deepcopy


def clones(module, n):
    return torch.nn.ModuleList(
        [deepcopy(module) for _ in range(n)]
    )


class LayerNorm(torch.nn.Module):

    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.a = torch.nn.Parameter(torch.ones(features))
        self.b = torch.nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a * (x - mean) / (std + self.epsilon) + self.b
        return x


class ResidualConnection(torch.nn.Module):

    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        sublayer_output = sublayer(self.norm(x))
        sublayer_output = self.dropout(sublayer_output)
        return x + sublayer_output
