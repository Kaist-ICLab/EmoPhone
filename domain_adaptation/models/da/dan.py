"""DAN: Deep Adaptation Network (Long et al., 2015)."""

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
from ..da_tllib_losses import (
    GaussianKernel,
    MultipleKernelMaximumMeanDiscrepancy,
)


class DAN(DAModel):
    """
    Deep Adaptation Network (DAN).
    Uses MK-MMD to align source/target feature distributions.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DAN, self).__init__(input_dim, num_classes, hparams)


# --- Training Logic ---




def train_dan(
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
    DAN (TLL-style): L = L_cls(source) + lambda * MK-MMD(f_s, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("DAN requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('dan_trade_off', 1.0)
    kernel_num = model.hparams.get('dan_kernel_num', 5)
    linear = model.hparams.get('dan_linear', False)

    kernels = [GaussianKernel(alpha=2 ** k) for k in range(kernel_num)]
    mkmmd = MultipleKernelMaximumMeanDiscrepancy(kernels, linear=linear)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()

    tracker = EarlyStopTracker(
        patience=patience, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
    )

    epoch_iterator = tqdm(range(epochs), desc="DAN Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            logits_s = model.classifier(f_s)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = mkmmd(f_s, f_t)
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

