import torch
from time import time
from .utils import subsequent_mask


class Batch:

    def __init__(self, source, target=None, pad=0):
        self.source = source
        self.source_mask = (source != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.n_tokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(target, pad):
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = target_mask & torch.autograd.Variable(
            subsequent_mask(target.size(-1)).type_as(target_mask.data)
        )
        return target_mask


def train_step(data_iter, model, loss_function):
    start = time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.source, batch.target,
            batch.source_mask, batch.target_mask
        )
        loss = loss_function(
            out, batch.target_y,
            batch.n_tokens
        )
        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens
        if i % 50 == 1:
            elapsed = time() - start
            print(
                "Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                (i, loss / batch.ntokens, tokens / elapsed))
            start = time()
            tokens = 0
    return total_loss / total_tokens
