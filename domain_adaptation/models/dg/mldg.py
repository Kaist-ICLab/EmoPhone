"""MLDG: Meta-Learning Domain Generalization (Li et al., 2018)."""

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


class MLDG(ERM):
    """
    Meta-Learning for Domain Generalization (Li et al., 2018)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.num_meta_test = self.hparams.get('n_meta_test', 1)

    def update(self, minibatches, unlabeled=None, **kwargs):
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            inner_net = copy.deepcopy(self.network)
            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            objective += inner_obj.item()

            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            objective += (self.hparams.get('mldg_beta', 1.0) * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams.get('mldg_beta', 1.0) * g_j.data / num_mb)

        objective /= len(minibatches)

        self.optimizer.step()
        return {'loss': objective}


class ParamDict(OrderedDict):
    """ParamDict from DomainBed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        if isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


