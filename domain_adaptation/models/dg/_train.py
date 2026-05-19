"""Shared training loop for every DG algorithm in this subpackage."""

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

from models import attach_training_metadata


def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains - num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i, j in zip(meta_train, cycle(meta_test)):
        xi, yi = minibatches[i][0], minibatches[i][1]
        xj, yj = minibatches[j][0], minibatches[j][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs




def train_dg_model(model, X_train, y_train, d_train, X_val, y_val, d_val,
                   epochs=20, batch_size=32, domains_per_batch=8, patience=20, device='cuda'):
    """
    Training loop for Domain Generalization models.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    unique_domains = np.unique(d_train)
    num_domains = len(unique_domains)

    if hasattr(model, 'num_domains'):
        model.num_domains = num_domains
    if hasattr(model, 'hparams'):
        model.hparams['num_domains'] = num_domains

    print(f"DG Training: {num_domains} domains, sampling {domains_per_batch} per batch.")

    domain_loaders = []
    domain_datasets = []
    for domain in unique_domains:
        mask = (d_train == domain)
        X_d = torch.tensor(X_train[mask], dtype=torch.float32)
        y_d = torch.tensor(y_train[mask], dtype=torch.long)
        dataset = TensorDataset(X_d, y_d)
        domain_datasets.append(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        domain_loaders.append(iter(loader))

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    steps_per_epoch = max(10, int(len(X_train) / (batch_size * domains_per_batch)))

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="DG Training")

    for epoch in epoch_iterator:
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            start_domain_idx = np.random.choice(
                num_domains,
                domains_per_batch,
                replace=(num_domains < domains_per_batch)
            )

            minibatches = []
            for d_idx in start_domain_idx:
                loader = domain_loaders[d_idx]
                try:
                    batch = next(loader)
                except StopIteration:
                    domain_loaders[d_idx] = iter(
                        DataLoader(domain_datasets[d_idx], batch_size=batch_size, shuffle=True, drop_last=False)
                    )
                    batch = next(domain_loaders[d_idx])

                x, y = batch
                minibatches.append((x.to(device), y.to(device)))

            metrics = model.update(minibatches, domain_indices=list(start_domain_idx))
            epoch_loss += metrics.get('loss', 0.0)

        model.eval()
        with torch.no_grad():
            logits = model.predict(X_val_t)
            val_loss = F.cross_entropy(logits, y_val_t).item()
            preds = logits.argmax(dim=1)
            val_acc = (preds == y_val_t).float().mean().item()

        epoch_iterator.set_postfix({
            'Loss': f'{epoch_loss/steps_per_epoch:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}'
        })
        epoch_num = epoch + 1
        epochs_ran = epoch_num
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(epoch_loss / steps_per_epoch), 6),
            'val_loss': round(float(val_loss), 6),
            'val_accuracy': round(float(val_acc), 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch}")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    attach_training_metadata(
        model,
        optimizer=getattr(getattr(model, 'optimizer', None), '__class__', type('obj', (), {})).__name__
        if getattr(model, 'optimizer', None) is not None else 'domainbed_optimizer',
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        model_selection_metric='val_loss',
        best_metric_value=round(float(best_val_loss), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        domains_per_batch=domains_per_batch,
    )
    return model
