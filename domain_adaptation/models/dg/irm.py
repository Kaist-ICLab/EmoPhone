"""IRM: Invariant Risk Minimization (Arjovsky et al., 2019)."""

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


class IRM(ERM):
    """
    Invariant Risk Minimization (Arjovsky et al., 2019)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return torch.sum(grad_1 * grad_2)

    def update(self, minibatches, unlabeled=None, **kwargs):
        penalty_weight = (
            self.hparams.get('irm_lambda', 1.0)
            if self.update_count >= self.hparams.get('irm_penalty_anneal_iters', 0)
            else 1.0
        )
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0

        for x, y in minibatches:
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + penalty_weight * penalty

        if self.update_count == self.hparams.get('irm_penalty_anneal_iters', 0):
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams.get('lr', 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


