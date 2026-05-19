"""MixStyle: feature-statistic mixing (Zhou et al., 2021)."""

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


class MixStyleLayer(nn.Module):
    """
    MixStyle layer (Zhou et al., 2021). Adapted to support 2D features.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)

        if x.dim() == 2:
            mu = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((B, 1)).to(x.device)
        else:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)

        if self.mix == "random":
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            perm = torch.arange(B - 1, -1, -1)
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class MixStyle(DGModel):
    """
    MixStyle (Zhou et al., 2021). Training is ERM with MixStyle active.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.mixstyle = MixStyleLayer(
            p=self.hparams.get('mixstyle_p', 0.5),
            alpha=self.hparams.get('mixstyle_alpha', 0.1),
            eps=self.hparams.get('mixstyle_eps', 1e-6),
            mix=self.hparams.get('mixstyle_mix', 'random')
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get('lr', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )

    def forward(self, x):
        feats = self.featurizer(x)
        feats = self.mixstyle(feats)
        return self.classifier(feats)

    def predict(self, x):
        return self.forward(x)

    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


