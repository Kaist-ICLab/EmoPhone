"""Fish: gradient-matching domain generalization (Shi et al., 2022)."""

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


class WholeFish(nn.Module):
    def __init__(self, input_dim, num_classes, hparams, weights=None):
        super().__init__()
        featurizer = _build_featurizer(input_dim, hparams)
        classifier = _build_classifier(featurizer.output_dim, num_classes, hparams)
        self.net = FeatureClassifier(featurizer, classifier)
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class Fish(nn.Module):
    """
    Fish: Gradient Matching for Domain Generalization (Shi et al., 2021)
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = WholeFish(input_dim, num_classes, self.hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = WholeFish(
            self.input_dim, self.num_classes, self.hparams, weights=self.network.state_dict()
        ).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None, **kwargs):
        self.create_clone(minibatches[0][0].device)

        loss = None
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams.get("meta_lr", self.hparams.get("fish_meta_lr", 0.5)),
        )
        self.network.reset_weights(meta_weights)

        return {"loss": loss.item() if loss is not None else 0.0}

    def predict(self, x):
        return self.network(x)
