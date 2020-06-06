import torch
from .blocks import LayerNorm
from ..utils import clones


class Encoder(torch.nn.Module):

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
