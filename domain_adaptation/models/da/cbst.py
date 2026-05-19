"""CBST: Class-Balanced Self-Training (Zou et al., 2018)."""

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
from sklearn.metrics import roc_auc_score
from .._da_helpers import _finalize_training_metadata


class CBST(DAModel):
    """
    CBST: Class-Balanced Self-Training (Zou et al., 2018)
    Iterative self-training with class-balanced pseudo-label selection.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(CBST, self).__init__(input_dim, num_classes, hparams)


def train_cbst(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5,
               device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None):
    """
    CBST (official-style):
      1) Train on source.
      2) Iteratively select class-balanced pseudo-labels on target and retrain.
    Requires unlabeled target samples (X_target).
    Best checkpoint is selected on validation AUROC across pretrain epochs and
    self-training rounds; final model is restored to that checkpoint.
    """
    if X_target is None:
        raise ValueError("CBST requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Source data
    X_s = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_s = torch.tensor(y_train, dtype=torch.long).to(device)
    ds_s = torch.utils.data.TensorDataset(X_s, y_s)
    loader_s = torch.utils.data.DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)

    # Validation loader (source labels)
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=max(batch_size, 256), shuffle=False)

    # Target data (unlabeled)
    X_t = torch.tensor(X_target, dtype=torch.float32).to(device)

    best_val_score = -float('inf')
    best_model_state = None
    best_phase = None
    best_step = None
    epoch_history = []
    epochs_ran = 0

    def _track_best(phase, step):
        nonlocal best_val_score, best_model_state, best_phase, best_step
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)
        epoch_history.append({
            'phase': phase,
            'step': step,
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
        })
        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            best_phase = phase
            best_step = step
        return val_loss, val_auroc

    # 1) Pretrain on source
    pretrain_epochs = model.hparams.get('cbst_pretrain_epochs', max(1, epochs // 2))
    epoch_iterator = tqdm(range(pretrain_epochs), desc="CBST Pretrain")
    for epoch in epoch_iterator:
        model.train()
        for x, y in loader_s:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        epochs_ran += 1
        val_loss, val_auroc = _track_best('pretrain', epoch + 1)
        epoch_iterator.set_postfix({'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'})

    # 2) Iterative self-training with class-balanced thresholds
    max_iter = model.hparams.get('cbst_max_iter', 5)
    init_port = model.hparams.get('cbst_init_port', 0.2)
    port_step = model.hparams.get('cbst_port_step', 0.1)
    max_port = model.hparams.get('cbst_max_port', 0.8)
    retrain_epochs = model.hparams.get('cbst_retrain_epochs', 5)

    num_classes = model.classifier[-1].out_features

    for round_idx in range(max_iter):
        model.eval()
        with torch.no_grad():
            logits_t = model(X_t)
            probs_t = F.softmax(logits_t, dim=1)
            max_probs, preds = torch.max(probs_t, dim=1)

        current_port = min(init_port + round_idx * port_step, max_port)
        pseudo_idx = []
        pseudo_labels = []

        for c in range(num_classes):
            c_idx = (preds == c).nonzero(as_tuple=True)[0]
            if len(c_idx) == 0:
                continue
            c_probs = max_probs[c_idx]
            k = int(len(c_idx) * current_port)
            if k == 0:
                continue
            topk_vals, topk_indices = torch.topk(c_probs, k)
            global_indices = c_idx[topk_indices]
            pseudo_idx.append(global_indices)
            pseudo_labels.append(torch.full((k,), c, dtype=torch.long).to(device))

        if not pseudo_idx:
            print("CBST: No pseudo-labels selected, skipping round.")
            continue

        pseudo_idx_cat = torch.cat(pseudo_idx)
        pseudo_labels_cat = torch.cat(pseudo_labels)
        X_pseudo = X_t[pseudo_idx_cat]

        # Retrain on source + pseudo-target
        X_aug = torch.cat([X_s, X_pseudo])
        y_aug = torch.cat([y_s, pseudo_labels_cat])
        ds_aug = torch.utils.data.TensorDataset(X_aug, y_aug)
        loader_aug = torch.utils.data.DataLoader(ds_aug, batch_size=batch_size, shuffle=True, drop_last=True)

        print(f"CBST: Round {round_idx+1}/{max_iter} - Retraining with {len(X_pseudo)} pseudo-labels")
        model.train()
        for _ in range(retrain_epochs):
            for x, y in loader_aug:
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            epochs_ran += 1

        val_loss, val_auroc = _track_best('selftrain', round_idx + 1)
        print(f"CBST: Round {round_idx+1} val_auroc={val_auroc:.4f} (best={best_val_score:.4f} @ {best_phase}/{best_step})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_epochs = pretrain_epochs + max_iter * retrain_epochs
    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_step,
        early_stopped=False,
        early_stop_epoch=None,
        epochs_ran=epochs_ran,
        max_epochs=total_epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        model_selection_metric="val_auroc",
        best_metric_value=round(float(best_val_score), 6) if best_model_state is not None else None,
        epoch_history=epoch_history,
        extra={
            "cbst_pretrain_epochs": pretrain_epochs,
            "cbst_max_iter": max_iter,
            "cbst_retrain_epochs": retrain_epochs,
            "cbst_best_phase": best_phase,
        },
    )
    return model
