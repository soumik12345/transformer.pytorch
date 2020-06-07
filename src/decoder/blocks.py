import torch
from ..utils import clones
from ..blocks import ResidualConnection


class DecoderLayer(torch.nn.Module):

    def __init__(self, size, self_attention, source_attention, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attention = self_attention
        self.source_attention = source_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)
