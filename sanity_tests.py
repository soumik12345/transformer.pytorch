import torch
import wandb
import numpy as np
from matplotlib import pyplot as plt
from src.utils import subsequent_mask
from src.blocks import PositionalEncoding
from src.training import NoamOptimizer, LabelSmoothing


def test_subsequent_mask(log_on_wandb=False):
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    plt.title('Test for Subsequent Mask')
    if log_on_wandb:
        wandb.log({'Test for Subsequent Mask': plt})
    else:
        plt.show()


def test_positional_encoding(log_on_wandb=False):
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(
        torch.autograd.Variable(
            torch.zeros(1, 100, 20)
        )
    )
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title('Test for Positional Encoding')
    if log_on_wandb:
        wandb.log({'Test for Positional Encoding': plt})
    else:
        plt.show()


def test_noam_lr_policy(log_on_wandb=False):
    optimizers = [
        NoamOptimizer(512, 1, 4000, None),
        NoamOptimizer(512, 1, 8000, None),
        NoamOptimizer(256, 1, 4000, None)
    ]
    plt.plot(
        np.arange(1, 20000),
        [[opt.rate(i) for opt in optimizers] for i in range(1, 20000)]
    )
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.title('Test for Noam Learning Rate Policy')
    if log_on_wandb:
        wandb.log({'Test for Noam Learning Rate Policy': plt})
    else:
        plt.show()


def test_label_smoothing_target_distribution(log_on_wandb=False):
    criterion = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0]
        ]
    )
    v = criterion(
        torch.autograd.Variable(predict.log()),
        torch.autograd.Variable(
            torch.LongTensor([2, 1, 0])
        )
    )
    plt.imshow(criterion.true_dist)
    plt.title('Test for Label Smoothing (Target Distribution)')
    if log_on_wandb:
        wandb.log({'Test for Label Smoothing (Target Distribution)': plt})
    else:
        plt.show()


def test_label_smoothing_regularization(log_on_wandb=False):
    criterion = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        prediction = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
        return criterion(
            torch.autograd.Variable(prediction.log()),
            torch.autograd.Variable(torch.LongTensor([1]))).data.item()

    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.title('Test for Label Smoothing (Regularization)')
    if log_on_wandb:
        wandb.log({'Test for Label Smoothing (Regularization)': plt})
    else:
        plt.show()


if __name__ == '__main__':
    test_subsequent_mask()
    test_positional_encoding()
    test_noam_lr_policy()
    test_label_smoothing_target_distribution()
    test_label_smoothing_regularization()
