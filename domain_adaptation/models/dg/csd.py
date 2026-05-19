"""CSD: Common-Specific Decomposition (Piratla et al., 2020)."""

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


class CSDHead(nn.Module):
    def __init__(self, feature_dim, num_classes, num_domains, k=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.k = k

        self.sms = nn.Parameter(torch.normal(0, 1e-1, size=[k, feature_dim, num_classes]))
        self.sm_biases = nn.Parameter(torch.normal(0, 1e-1, size=[k, num_classes]))
        self.embs = nn.Parameter(torch.normal(mean=0.0, std=1e-4, size=[num_domains, k - 1]))
        self.cs_wt = nn.Parameter(torch.normal(mean=0.1, std=1e-4, size=[]))

    def forward(self, features, domain_onehot):
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        logits_common = torch.matmul(features, w_c) + b_c

        c_wts = torch.matmul(domain_onehot, self.embs)
        batch_size = domain_onehot.shape[0]
        c_wts = torch.cat(
            (torch.ones((batch_size, 1), device=features.device) * self.cs_wt, c_wts), 1
        )
        c_wts = torch.tanh(c_wts)

        w_d = torch.einsum("bk,kdc->bdc", c_wts, self.sms)
        b_d = torch.einsum("bk,kc->bc", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("bdc,bd->bc", w_d, features) + b_d

        return logits_specialized, logits_common


class CSD(nn.Module):
    """
    Common-Specific Decomposition (Piratla et al., 2020)
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.num_classes = num_classes
        self.num_domains = self.hparams.get("num_domains")

        self.featurizer = _build_featurizer(input_dim, self.hparams)
        num_domains = self.num_domains if self.num_domains is not None else 1
        self.csd_head = CSDHead(
            self.featurizer.output_dim, num_classes, num_domains, k=self.hparams.get("csd_k", 2)
        )

        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.csd_head.parameters()),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )

    def update(self, minibatches, unlabeled=None, domain_indices=None, **kwargs):
        if domain_indices is None:
            domain_indices = list(range(len(minibatches)))

        num_domains = (
            self.num_domains if self.num_domains is not None else (max(domain_indices) + 1)
        )
        if self.csd_head.num_domains != num_domains:
            self.csd_head = CSDHead(
                self.featurizer.output_dim,
                self.num_classes,
                num_domains,
                k=self.hparams.get("csd_k", 2),
            ).to(minibatches[0][0].device)
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.csd_head.parameters()),
                lr=self.hparams.get("lr", 1e-3),
                weight_decay=self.hparams.get("weight_decay", 0.0),
            )
        self.csd_head.num_domains = num_domains

        all_features = []
        all_y = []
        all_domain = []

        for (x, y), d_idx in zip(minibatches, domain_indices):
            feats = self.featurizer(x)
            all_features.append(feats)
            all_y.append(y)
            all_domain.append(
                torch.full((x.size(0),), int(d_idx), device=x.device, dtype=torch.long)
            )

        features = torch.cat(all_features)
        y = torch.cat(all_y)
        domain_ids = torch.cat(all_domain)
        domain_onehot = F.one_hot(domain_ids, num_classes=num_domains).float()

        logits_specialized, logits_common = self.csd_head(features, domain_onehot)

        specific_loss = F.cross_entropy(logits_specialized, y)
        class_loss = F.cross_entropy(logits_common, y)

        sms = self.csd_head.sms
        k = sms.shape[0]
        diag = torch.eye(k, device=sms.device).unsqueeze(0).repeat(self.num_classes, 1, 1)
        cps = torch.stack(
            [torch.matmul(sms[:, :, c], sms[:, :, c].t()) for c in range(self.num_classes)], dim=0
        )
        orth_loss = torch.mean((cps - diag) ** 2)

        loss = class_loss + specific_loss + self.hparams.get("csd_lambda", 1.0) * orth_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "class_loss": class_loss.item(),
            "specific_loss": specific_loss.item(),
            "orth_loss": orth_loss.item(),
        }

    def predict(self, x):
        feats = self.featurizer(x)
        w_c, b_c = self.csd_head.sms[0, :, :], self.csd_head.sm_biases[0, :]
        return torch.matmul(feats, w_c) + b_c
