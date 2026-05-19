"""ERM: Empirical Risk Minimization baseline."""

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


class ERM(DGModel):
    """
    Empirical Risk Minimization (Standard Training)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get('lr', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )

    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


