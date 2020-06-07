import torch
from ..utils import clones
from .blocks import scaled_dot_product_attention


class MultiHeadedAttention(torch.nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batches = query.size(0)
        query, key, value = [
            l(x).view(batches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attention = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        x = x.transpose(1, 2).contiguous().view(batches, -1, self.h * self.d_k)
        x = self.linear_layers[-1](x)
        return x
