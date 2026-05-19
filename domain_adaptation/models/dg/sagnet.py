"""SagNet: style-agnostic networks (Nam et al., 2021)."""

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


class SagNet(nn.Module):
    """
    SagNet: Style Agnostic Networks (Nam et al., 2021)
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.num_classes = num_classes

        self.network_f = _build_featurizer(input_dim, self.hparams)
        self.network_c = _build_classifier(self.network_f.output_dim, num_classes, self.hparams)
        self.network_s = _build_classifier(self.network_f.output_dim, num_classes, self.hparams)

        def opt(p):
            return torch.optim.Adam(
                p,
                lr=self.hparams.get("lr", 1e-3),
                weight_decay=self.hparams.get("weight_decay", 0.0),
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = self.hparams.get("sag_w_adv", 1.0)

    def forward_c(self, x):
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1, device=x.device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {"loss_c": loss_c.item(), "loss_s": loss_s.item(), "loss_adv": loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))
