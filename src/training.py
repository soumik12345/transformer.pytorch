import torch
import wandb
import torchtext
from tqdm import tqdm
from time import time
from .utils import subsequent_mask


global max_source_in_batch, max_target_in_batch


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


def train_step(data_iter, model, loss_function, log_on_wandb):
    start = time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    iterable = tqdm(enumerate(data_iter)) if log_on_wandb else enumerate(data_iter)
    for i, batch in iterable:
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
        if log_on_wandb:
            wandb.log({"Loss": loss / batch.n_tokens})
        elif i % 50 == 1:
            elapsed = time() - start
            print(
                "Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                (i, loss / batch.n_tokens, tokens / elapsed))
            start = time()
            tokens = 0
    return total_loss / total_tokens


def batch_size_function(new, count, sofar):
    global max_source_in_batch, max_target_in_batch
    if count == 1:
        max_source_in_batch = 0
        max_target_in_batch = 0
    max_source_in_batch = max(max_source_in_batch, len(new.source))
    max_target_in_batch = max(max_target_in_batch, len(new.target) + 2)
    source_elements = count * max_source_in_batch
    target_elements = count * max_target_in_batch
    return max(source_elements, target_elements)


class NoamOptimizer:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_optimizer(model):
    return NoamOptimizer(
        model.source_embedding[0].d_model, 2, 4000,
        torch.optim.Adam(
            model.parameters(), lr=0,
            betas=(0.9, 0.98), eps=1e-9
        )
    )


class LabelSmoothing(torch.nn.Module):

    def __init__(self, size, padding_index, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_index = padding_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_index] = 0
        mask = torch.nonzero(target.data == self.padding_index)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(
            x, torch.autograd.Variable(true_dist, requires_grad=False)
        )


class DataIterator(torchtext.data.Iterator):

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in torchtext.data.batch(d, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(
                    self.data(), self.batch_size,
                    self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_index, batch):
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_index)
