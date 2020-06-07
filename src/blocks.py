import torch


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


class PositionWiseFeedForward(torch.nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
