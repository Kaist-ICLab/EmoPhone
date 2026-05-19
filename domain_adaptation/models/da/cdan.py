"""CDAN: Conditional Domain Adversarial Network (Long et al., 2018)."""

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
    ConditionalDomainAdversarialLoss,
    DomainDiscriminator,
    WarmStartGradientReverseLayer,
)


class CDAN(DAModel):
    """
    CDAN backbone. Conditional adversarial loss handled in training.
    """
    def __init__(self, input_dim, num_classes=2, num_domains=None, hparams=None):
        super(CDAN, self).__init__(input_dim, num_classes, hparams)
        self.num_domains = num_domains



def train_cdan(
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
    CDAN (TLL-style): L = L_cls(source) + lambda * ConditionalDomainAdversarialLoss(g_s, f_s, g_t, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("CDAN requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    disc_hidden = model.hparams.get('disc_hidden', 1024)
    disc_lr = model.hparams.get('discriminator_lr', lr)
    trade_off = model.hparams.get('trade_off', 1.0)
    randomized = model.hparams.get('cdan_randomized', False)
    entropy_conditioning = model.hparams.get('cdan_entropy', True)
    randomized_dim = model.hparams.get('cdan_randomized_dim', 1024)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    feature_dim = model.feature_extractor.output_dim
    num_classes = model.classifier[-1].out_features
    disc_in_dim = randomized_dim if randomized else feature_dim * num_classes
    domain_discriminator = DomainDiscriminator(disc_in_dim, disc_hidden, batch_norm=True, sigmoid=True).to(device)
    cdan_loss = ConditionalDomainAdversarialLoss(
        domain_discriminator,
        entropy_conditioning=entropy_conditioning,
        randomized=randomized,
        num_classes=num_classes,
        features_dim=feature_dim,
        randomized_dim=randomized_dim,
        sigmoid=True,
    )
    cdan_loss.grl.max_iters = epochs * len(train_loader)

    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters()},
            {"params": model.classifier.parameters()},
            {"params": domain_discriminator.parameters(), "lr": disc_lr},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    class_criterion = nn.CrossEntropyLoss()

    tracker = EarlyStopTracker(
        patience=patience, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
    )

    epoch_iterator = tqdm(range(epochs), desc="CDAN Training")
    for epoch in epoch_iterator:
        model.train()
        domain_discriminator.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            g_s = model.classifier(f_s)
            g_t = model.classifier(f_t)

            cls_loss = class_criterion(g_s, y_s)
            transfer_loss = cdan_loss(g_s, f_s, g_t, f_t)
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

    return tracker.finalize(model, optimizer=optimizer,
                            extra_meta={"discriminator_lr": disc_lr})


