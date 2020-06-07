import torch
from .utils import subsequent_mask


class Batch:

    def __init__(self, source, target=None, pad=0):
        self.source = source
        self.src_mask = (source != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.trg_mask = self.make_std_mask(self.target, pad)
            self.n_tokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(target, pad):
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = target_mask & torch.autograd.Variable(
            subsequent_mask(target.size(-1)).type_as(target_mask.data)
        )
        return target_mask
