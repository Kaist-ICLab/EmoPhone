"""GroupDRO: Distributionally Robust Optimization (Sagawa et al., 2020)."""

import copy
import operator
from collections import OrderedDict
from itertools import cycle
from numbers import Number

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ._base import DGModel
from .erm import ERM


class GroupDRO(ERM):
    """
    Group DRO (Sagawa et al., 2020)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None, **kwargs):
        device = minibatches[0][0].device

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams.get("groupdro_eta", 0.01) * losses[m].data).exp()

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


