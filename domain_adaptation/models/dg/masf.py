"""MASF: Model-Agnostic Domain Specifics (Dou et al., 2019)."""

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


class MASF(nn.Module):
    """
    MASF (Dou et al., 2019). PyTorch port of the official TensorFlow logic.
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.num_classes = num_classes
        self.num_domains = self.hparams.get("num_domains")

        self.featurizer = _build_featurizer(input_dim, self.hparams)
        self.classifier = _build_classifier(self.featurizer.output_dim, num_classes, self.hparams)
        self.network = FeatureClassifier(self.featurizer, self.classifier)

        metric_dim = self.hparams.get("masf_metric_dim", 128)
        self.metric_net = nn.Sequential(
            nn.Linear(self.featurizer.output_dim, metric_dim),
            nn.ReLU(),
            nn.Linear(metric_dim, metric_dim),
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        self.metric_optimizer = torch.optim.Adam(
            self.metric_net.parameters(),
            lr=self.hparams.get("masf_metric_lr", self.hparams.get("lr", 1e-3)),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )

    def _kd_loss(self, logits1, y1, logits2, y2, bool_indicator, temperature=2.0):
        # Initialize as a zero tensor tied to the autograd graph so the return
        # value is always a tensor, even when no classes contribute.
        kd_loss = logits1.sum() * 0.0 + logits2.sum() * 0.0
        eps = 1e-16
        for cls in range(self.num_classes):
            if bool_indicator[cls] < 0.5:
                continue
            mask1 = y1 == cls
            mask2 = y2 == cls
            if mask1.sum() == 0 or mask2.sum() == 0:
                continue
            act1 = logits1[mask1].mean(0)
            act2 = logits2[mask2].mean(0)
            prob1 = F.softmax(act1 / temperature, dim=0).clamp_min(1e-8)
            prob2 = F.softmax(act2 / temperature, dim=0).clamp_min(1e-8)
            kl_div = 0.5 * (
                torch.sum(prob1 * torch.log(prob1 / (prob2 + eps)))
                + torch.sum(prob2 * torch.log(prob2 / (prob1 + eps)))
            )
            kd_loss = kd_loss + kl_div
        return kd_loss / self.num_classes

    def _triplet_semihard_loss(self, embeddings, labels, margin):
        if embeddings.size(0) < 2:
            # Keep autograd graph connected even when no valid triplets exist.
            return embeddings.sum() * 0.0

        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1)
        loss_vals = []
        for i in range(embeddings.size(0)):
            anchor_label = labels[i]
            pos_mask = labels == anchor_label
            neg_mask = labels != anchor_label

            pos_mask[i] = False
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_dist = dist[i][pos_mask]
            neg_dist = dist[i][neg_mask]
            pos_dist_val = pos_dist.min()

            semihard_neg = neg_dist[neg_dist > pos_dist_val]
            if semihard_neg.numel() > 0:
                neg_dist_val = semihard_neg.min()
            else:
                neg_dist_val = neg_dist.min()

            loss_vals.append(F.relu(pos_dist_val - neg_dist_val + margin))

        if not loss_vals:
            # Keep autograd graph connected even when no valid triplets exist.
            return embeddings.sum() * 0.0

        return torch.stack(loss_vals).mean()

    def update(self, minibatches, unlabeled=None, **kwargs):
        if len(minibatches) < 3:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.network(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}

        perm = torch.randperm(len(minibatches)).tolist()
        meta_train_idx = perm[:2]
        meta_test_idx = perm[2:3]

        xa, ya = minibatches[meta_train_idx[0]]
        xa1, ya1 = minibatches[meta_train_idx[1]]
        xb, yb = minibatches[meta_test_idx[0]]

        # Step 1: source loss update
        source_loss = 0.5 * (
            F.cross_entropy(self.network(xa), ya) + F.cross_entropy(self.network(xa1), ya1)
        )
        self.optimizer.zero_grad()
        source_loss.backward()
        self.optimizer.step()

        # Inner update
        inner_net = copy.deepcopy(self.network)
        inner_opt = torch.optim.Adam(
            inner_net.parameters(),
            lr=self.hparams.get("masf_inner_lr", self.hparams.get("lr", 1e-3)),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        inner_loss = 0.5 * (
            F.cross_entropy(inner_net(xa), ya) + F.cross_entropy(inner_net(xa1), ya1)
        )
        inner_opt.zero_grad()
        inner_loss.backward()
        inner_opt.step()

        logits_a = inner_net(xa)
        logits_a1 = inner_net(xa1)
        logits_b = inner_net(xb)

        classes_b = torch.unique(yb).tolist()
        classes_a = torch.unique(ya).tolist()
        classes_a1 = torch.unique(ya1).tolist()
        bool_indicator_b_a = torch.zeros(self.num_classes, device=xa.device)
        bool_indicator_b_a1 = torch.zeros(self.num_classes, device=xa.device)
        for cls in range(self.num_classes):
            if (cls in classes_b) and (cls in classes_a):
                bool_indicator_b_a[cls] = 1.0
            if (cls in classes_b) and (cls in classes_a1):
                bool_indicator_b_a1[cls] = 1.0

        kd_temp = self.hparams.get("masf_temperature", 2.0)
        global_loss = 0.5 * (
            self._kd_loss(logits_b, yb, logits_a, ya, bool_indicator_b_a, temperature=kd_temp)
            + self._kd_loss(logits_b, yb, logits_a1, ya1, bool_indicator_b_a1, temperature=kd_temp)
        )

        part = min(xa.size(0), xa1.size(0), xb.size(0))
        input_group = torch.cat([xa[:part], xa1[:part], xb[:part]], dim=0)
        label_group = torch.cat([ya[:part], ya1[:part], yb[:part]], dim=0)

        embeddings = self.metric_net(inner_net.forward_features(input_group))
        metric_loss = self._triplet_semihard_loss(
            embeddings, label_group, margin=self.hparams.get("masf_margin", 1.0)
        )

        meta_loss = global_loss + self.hparams.get("masf_metric_weight", 0.005) * metric_loss

        # Meta update using gradients from inner_net
        meta_grads = autograd.grad(meta_loss, inner_net.parameters(), allow_unused=True)
        self.optimizer.zero_grad()
        for p, g in zip(self.network.parameters(), meta_grads):
            if g is not None:
                p.grad = g.detach()
        self.optimizer.step()

        # Metric network update (recompute to avoid graph reuse)
        self.metric_optimizer.zero_grad()
        with torch.no_grad():
            group_feats = inner_net.forward_features(input_group)
        metric_embeddings = self.metric_net(group_feats.detach())
        metric_loss_metric = self._triplet_semihard_loss(
            metric_embeddings, label_group, margin=self.hparams.get("masf_margin", 1.0)
        )
        if metric_loss_metric.requires_grad:
            metric_loss_metric.backward()
            self.metric_optimizer.step()

        return {
            "loss": (source_loss + meta_loss).item(),
            "source_loss": source_loss.item(),
            "global_loss": global_loss.item(),
            "metric_loss": metric_loss.item(),
        }

    def predict(self, x):
        return self.network(x)


# --- Training Helper ---
