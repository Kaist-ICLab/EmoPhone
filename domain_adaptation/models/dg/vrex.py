"""VREx: Variance Risk Extrapolation (Krueger et al., 2021)."""

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


class VREx(ERM):
    """
    V-REx (Krueger et al., 2021)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, **kwargs):
        penalty_weight = (
            self.hparams.get('vrex_lambda', 1.0)
            if self.update_count >= self.hparams.get('vrex_penalty_anneal_iters', 0)
            else 1.0
        )

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches), device=all_x.device)

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = F.cross_entropy(logits, y)

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams.get('vrex_penalty_anneal_iters', 0):
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams.get('lr', 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': mean.item(), 'penalty': penalty.item()}


