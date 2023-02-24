import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

class CroLoss(Module):

    def CroLoss(Y, P):
        Y = np.float_(Y)
        P = np.float_(P)
        return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :, :frame], true[i, :, :frame]))
        return torch.mean(torch.stack(loss))


class AmmLoss(Module):

    def __init__(self, gamma=2,alpha=0.25):
        super(AmmLoss, self).__init__()
        self.gamma = gamma
        self.alpha=alpha
    def forward(self, input, target):
        pt=torch.softmax(input,dim=1)
        p=pt[:,1]
        loss = -self.alpha*(1-p)**self.gamma*(target*torch.log(p))-(1-self.alpha)*p**self.gamma*((1-target)*torch.log(1-p))
        return loss.mean()


class ConLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin
    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            d = torch.dist(pred1[i, :, :frame], pred2[i, :, :frame], 2)
            if labels[i]:
                loss.append(d ** 2)
            else:
                loss.append(torch.clip(self.margin - d, min=0.) ** 2)
        return torch.mean(torch.stack(loss))
