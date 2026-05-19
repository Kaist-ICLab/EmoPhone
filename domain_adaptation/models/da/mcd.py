"""MCD: Maximum Classifier Discrepancy (Saito et al., 2018)."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .._da_helpers import (
    DAModel,
    EarlyStopTracker,
    _build_loaders,
    _evaluate_val,
    _infinite_iterator,
)
from ..da_tllib_losses import classifier_discrepancy, mcd_entropy


class MCD(DAModel):
    """
    Maximum Classifier Discrepancy (MCD).
    Two classifiers over a shared feature extractor.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(MCD, self).__init__(input_dim, num_classes, hparams)
        feat_dim = self.feature_extractor.output_dim
        self.classifier1 = nn.Linear(feat_dim, num_classes)
        self.classifier2 = nn.Linear(feat_dim, num_classes)
        self.classifier = None  # override

    def predict(self, x):
        feat = self.feature_extractor(x)
        o1 = self.classifier1(feat)
        o2 = self.classifier2(feat)
        return (o1 + o2) / 2.0

    def forward(self, x):
        feat = self.feature_extractor(x)
        o1 = self.classifier1(feat)
        o2 = self.classifier2(feat)
        return o1, o2


class MCDInferenceWrapper(nn.Module):
    def __init__(self, mcd_model):
        super().__init__()
        self.model = mcd_model

    def forward(self, x):
        o1, o2 = self.model(x)
        return (o1 + o2) / 2.0


def train_mcd(model, X_train, y_train, d_train, X_val, y_val, d_val,
              epochs=50, batch_size=64, lr=1e-3, patience=5,
              device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None):
    """
    MCD (TLL-style):
      A) Min CE on source (G, C1, C2)
      B) Max discrepancy on target (C1, C2) while fitting source
      C) Min discrepancy on target (G)
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("MCD requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('mcd_trade_off', 1.0)
    k_steps = model.hparams.get('mcd_k', 4)

    optimizer_g = torch.optim.Adam(model.feature_extractor.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_c = torch.optim.Adam(
        list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    tracker = EarlyStopTracker(
        patience=patience, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
    )

    epoch_iterator = tqdm(range(epochs), desc="MCD Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            # Step A: train G, C1, C2 on source
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            f_s = model.feature_extractor(X_s)
            o1_s = model.classifier1(f_s)
            o2_s = model.classifier2(f_s)
            loss_s = criterion(o1_s, y_s) + criterion(o2_s, y_s)
            loss_s.backward()
            optimizer_g.step()
            optimizer_c.step()

            # Step B: train C1, C2 to maximize discrepancy on target
            optimizer_c.zero_grad()
            f_s_det = model.feature_extractor(X_s).detach()
            f_t_det = model.feature_extractor(X_t).detach()
            o1_s = model.classifier1(f_s_det)
            o2_s = model.classifier2(f_s_det)
            o1_t = model.classifier1(f_t_det)
            o2_t = model.classifier2(f_t_det)
            loss_s = criterion(o1_s, y_s) + criterion(o2_s, y_s)
            dis = classifier_discrepancy(F.softmax(o1_t, dim=1), F.softmax(o2_t, dim=1))
            loss_c = loss_s - trade_off * dis
            loss_c.backward()
            optimizer_c.step()

            # Step C: train G to minimize discrepancy on target
            for _ in range(k_steps):
                optimizer_g.zero_grad()
                f_t = model.feature_extractor(X_t)
                o1_t = model.classifier1(f_t)
                o2_t = model.classifier2(f_t)
                dis = classifier_discrepancy(F.softmax(o1_t, dim=1), F.softmax(o2_t, dim=1))
                loss_g = trade_off * dis
                loss_g.backward()
                optimizer_g.step()

            train_loss += loss_s.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        if tracker.record(model, epoch_num=epoch + 1, train_loss=train_loss,
                          val_loss=val_loss, val_auroc=val_auroc,
                          iterator=epoch_iterator):
            break

    optimizer_label = f"{optimizer_g.__class__.__name__}+{optimizer_c.__class__.__name__}"
    return tracker.finalize(model, optimizer=optimizer_label,
                            extra_meta={"mcd_k": k_steps})
    
# --- JAN: Joint Adaptation Network ---

