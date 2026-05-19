"""Shared building blocks for the DA training pipeline.

Holds the pieces that every train_* function in :mod:`da_models` relies on:

- :class:`DAModel`: the backbone-agnostic feature-extractor + classifier base.
- :class:`EarlyStopTracker`: standardised per-epoch bookkeeping and the
  closing :func:`_finalize_training_metadata` payload.
- :func:`_build_loaders`, :func:`_infinite_iterator`, :func:`_evaluate_val`:
  DataLoader and validation helpers used by every adaptation algorithm.
- :func:`train_standard`: thin wrapper around the supervised baseline loop.

Pulled out of the historical mega-file so each adaptation algorithm only
imports the surface it actually needs.
"""

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Make sibling top-level modules under ``basemodel-benchmarking/`` importable
# (the dash in the folder name prevents Python from treating it as a package).
_HERE = os.path.dirname(os.path.abspath(__file__))
_BMB = os.path.normpath(os.path.join(_HERE, "..", "..", "basemodel-benchmarking"))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_BMB, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer

from models import attach_training_metadata, train_torch_model

# Shared default hyperparameters used by every train_* function in this
# subpackage. The values match the historical inlined defaults exactly; the
# constants exist to make tuning discoverable and to surface the fact that
# every algorithm shares the same training-loop schedule.
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_PATIENCE: int = 5
DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class DAModel(nn.Module):
    """
    Base class for Domain Adaptation models ensuring backbone-agnostic behavior.
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DAModel, self).__init__()
        self.hparams = hparams if hparams else {}
        backbone_name = self.hparams.get("backbone", "MLP")
        dropout = self.hparams.get("dropout", 0.3)
        hidden_dim = self.hparams.get("hidden_dim", 256)
        num_layers = self.hparams.get("num_layers", 3)

        if backbone_name == "MLP":
            self.feature_extractor = MLPFeaturizer(
                input_dim,
                hidden_dim=hidden_dim,
                output_dim=128,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif backbone_name == "ResNet":
            # ResNet hparams
            num_blocks = self.hparams.get("num_blocks", 2)
            self.feature_extractor = ResNetFeaturizer(
                input_dim,
                hidden_dim=hidden_dim,
                output_dim=128,
                num_blocks=num_blocks,
                dropout=dropout,
            )
        elif backbone_name == "Transformer":
            num_layers_tr = self.hparams.get("num_layers", 2)
            nhead = self.hparams.get("nhead", 4)
            self.feature_extractor = TransformerFeaturizer(
                input_dim,
                hidden_dim=128,
                output_dim=128,
                num_layers=num_layers_tr,
                nhead=nhead,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.classifier = nn.Sequential(nn.Linear(self.feature_extractor.output_dim, num_classes))

    def predict(self, x):
        feat = self.feature_extractor(x)
        return self.classifier(feat)

    def forward(self, x):
        return self.predict(x)


def _finalize_training_metadata(
    model,
    *,
    optimizer,
    best_epoch,
    early_stopped,
    early_stop_epoch,
    epochs_ran,
    max_epochs,
    batch_size,
    patience,
    lr=None,
    weight_decay=None,
    model_selection_metric="val_auroc",
    best_metric_value=None,
    epoch_history=None,
    extra=None,
):
    payload = {
        "optimizer": optimizer,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "early_stop_epoch": early_stop_epoch,
        "epochs_ran": epochs_ran,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "model_selection_metric": model_selection_metric,
        "best_metric_value": best_metric_value,
        "epoch_history": epoch_history or [],
    }
    if extra:
        payload.update(extra)
    attach_training_metadata(model, **payload)
    return model


class EarlyStopTracker:
    """Shared training-loop bookkeeping for every train_* function in this
    module. Tracks best validation AUROC, deep-copies the best state,
    decides when patience runs out, and writes a uniform training-metadata
    payload onto the model on finalize.

    Behaviour is bit-for-bit identical to the inlined bookkeeping it
    replaces, including the zero-indexed epoch value used in the
    "Early stopping at epoch N" message (some log parsers depend on it).
    """

    def __init__(self, *, patience, epochs, batch_size, lr, weight_decay):
        self.patience = patience
        self.max_epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.best_val_score = -float("inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.best_epoch = None
        self.early_stopped = False
        self.early_stop_epoch = None
        self.epochs_ran = 0
        self.epoch_history = []

    def record(
        self, model, *, epoch_num, train_loss, val_loss, val_auroc, iterator, postfix_extra=None
    ):
        """Record one epoch's metrics. Returns True if early-stop fired."""
        self.epochs_ran = epoch_num
        if val_auroc > self.best_val_score:
            self.best_val_score = val_auroc
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.patience_counter = 0
            self.best_epoch = epoch_num
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                iterator.write(
                    f"Early stopping at epoch {epoch_num - 1} "
                    f"(Best AUROC: {self.best_val_score:.4f})"
                )
                self.early_stopped = True
                self.early_stop_epoch = epoch_num
                return True

        postfix = {
            "Loss": f"{train_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val AUC": f"{val_auroc:.4f}",
        }
        if postfix_extra:
            postfix.update(postfix_extra)
        iterator.set_postfix(postfix)

        self.epoch_history.append(
            {
                "epoch": epoch_num,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "val_auroc": round(float(val_auroc), 6),
            }
        )
        return False

    def finalize(self, model, *, optimizer, extra_meta=None):
        if isinstance(optimizer, str):
            optimizer_name = optimizer
        elif optimizer is not None:
            optimizer_name = optimizer.__class__.__name__
        else:
            optimizer_name = None
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        _finalize_training_metadata(
            model,
            optimizer=optimizer_name,
            best_epoch=self.best_epoch,
            early_stopped=self.early_stopped,
            early_stop_epoch=self.early_stop_epoch,
            epochs_ran=self.epochs_ran,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            lr=self.lr,
            weight_decay=self.weight_decay,
            best_metric_value=(
                round(float(self.best_val_score), 6) if self.best_epoch is not None else None
            ),
            epoch_history=self.epoch_history,
            extra=extra_meta,
        )
        return model


def train_standard(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    patience=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Simple supervised training wrapper (used for fallback in DA methods).
    """
    return train_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device=device,
    )


def _infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size):
    pin = torch.cuda.is_available()
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=2,
        persistent_workers=True,
    )

    target_loader = None
    if X_target is not None:
        target_dataset = torch.utils.data.TensorDataset(torch.tensor(X_target, dtype=torch.float32))
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin,
            num_workers=2,
            persistent_workers=True,
        )

    return train_loader, val_loader, target_loader


def _evaluate_val(model, val_loader, device):
    class_criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model.predict(X_batch)
            val_loss += class_criterion(logits, y_batch).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_probs.append(probs)
            val_targets.append(y_batch.cpu().numpy())
    val_loss /= len(val_loader)
    val_probs = np.concatenate(val_probs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    try:
        if val_probs.shape[1] == 2:
            val_auroc = roc_auc_score(val_targets, val_probs[:, 1])
        else:
            val_auroc = roc_auc_score(val_targets, val_probs, multi_class="ovr", average="macro")
    except ValueError:
        val_auroc = 0.5
    return val_loss, val_auroc
