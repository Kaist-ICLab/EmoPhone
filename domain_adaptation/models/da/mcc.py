"""MCC: Minimum Class Confusion (Jin et al., 2020)."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .._da_helpers import (
    DAModel,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    EarlyStopTracker,
    _build_loaders,
    _evaluate_val,
    _infinite_iterator,
)
from ..da_tllib_losses import MinimumClassConfusionLoss


class MCC(DAModel):
    """
    Minimum Class Confusion (Jin et al., 2020)
    Non-adversarial. Loss-based optimization.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(MCC, self).__init__(input_dim, num_classes, hparams)
        self.temperature = self.hparams.get('mcc_temp', 2.0)
    
    def forward(self, x):
        return self.predict(x)


# --- CORAL: Deep Correlation Alignment ---



def train_mcc(
    model,
    X_train, y_train, d_train,
    X_val, y_val, d_val,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    device: str = DEFAULT_DEVICE,
    X_target=None,
):
    """
    MCC (TLL-style): L = L_cls(source) + mu * MCC(target_logits).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("MCC requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('mcc_trade_off', 1.0)
    mcc_loss_fn = MinimumClassConfusionLoss(model.temperature)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()

    tracker = EarlyStopTracker(
        patience=patience, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
    )

    epoch_iterator = tqdm(range(epochs), desc="MCC Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            logits_s = model.predict(X_s)
            logits_t = model.predict(X_t)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = mcc_loss_fn(logits_t)
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        if tracker.record(model, epoch_num=epoch + 1, train_loss=train_loss,
                          val_loss=val_loss, val_auroc=val_auroc,
                          iterator=epoch_iterator):
            break

    return tracker.finalize(model, optimizer=optimizer)


