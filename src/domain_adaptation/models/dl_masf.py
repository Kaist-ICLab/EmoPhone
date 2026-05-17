"""MASF-style domain generalization pipeline (simplified).

This module provides a PyTorch approximation of the GLOBEM `dl_masf` model.
It performs domain-aware training with a meta-train/meta-test split per step
and adds a metric-learning (triplet) regularizer on embeddings.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc
from .dl_erm import DLERMConfig, DLERMRunResult






@dataclass
class DLMASFConfig(DLERMConfig):
    meta_training_proportion: float = 0.5
    meta_testing_weight: float = 1.0
    per_domain_batch_size: int = 64
    steps_per_epoch: int = 100
    triplet_margin: float = 0.2
    local_loss_weight: float = 0.5


DLMASFRunResult = DLERMRunResult


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


def _normalization(kind: str, hidden_dim: int) -> nn.Module:
    lowered = (kind or "none").lower()
    if lowered == "batchnorm":
        return nn.BatchNorm1d(hidden_dim)
    if lowered == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if lowered == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization '{kind}'")


class _MASFNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: str,
        normalization: str,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        act = _activation(activation)
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                _normalization(normalization, hidden),
                deepcopy(act),
                nn.Dropout(dropout),
            ])
            prev = hidden
        if embedding_dim is None:
            embedding_dim = prev
        if prev != embedding_dim:
            layers.append(nn.Linear(prev, embedding_dim))
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x) if len(self.backbone) > 0 else x
        logits = self.classifier(emb).squeeze(-1)
        return logits, emb

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


class _DomainBatchSampler:
    def __init__(self, dataset: ArrayDataset, batch_size: int):
        if dataset.domains is None:
            raise ValueError("MASF requires domain identifiers in ArrayDataset.domains")
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        domains = dataset.domains.astype(np.int64)
        self.domain_to_indices: Dict[int, np.ndarray] = {}
        for domain in np.unique(domains):
            idx = np.where(domains == domain)[0]
            if idx.size > 0:
                self.domain_to_indices[int(domain)] = idx
        self.domain_ids = list(self.domain_to_indices.keys())
        if not self.domain_ids:
            raise ValueError("No domains available for MASF sampling")

    def sample(self) -> Dict[int, Dict[str, torch.Tensor]]:
        batch: Dict[int, Dict[str, torch.Tensor]] = {}
        for domain, indices in self.domain_to_indices.items():
            chosen = np.random.choice(indices, size=self.batch_size, replace=indices.size < self.batch_size)
            batch[domain] = {
                "X": torch.from_numpy(self.dataset.X[chosen]).float(),
                "y": torch.from_numpy(self.dataset.y[chosen]).float(),
            }
        return batch


class DLMASFPipeline:
    def __init__(self, config: DLMASFConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> _MASFNet:
        model = _MASFNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
        )
        return model.to(self.device)

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        dot = embeddings @ embeddings.t()
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(1) - 2.0 * dot + sq_norm.unsqueeze(0)
        return torch.clamp(dist, min=0.0)

    def _batch_hard_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.size(0) < 2:
            return torch.zeros((), device=embeddings.device)
        labels = labels.view(-1, 1)
        same = labels.eq(labels.t())
        eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
        pos_mask = same & (~eye)
        neg_mask = ~same

        dist = self._pairwise_distances(embeddings)
        dist_pos = dist.clone()
        dist_pos[~pos_mask] = -1e6
        hardest_pos = dist_pos.max(dim=1).values

        dist_neg = dist.clone()
        dist_neg[~neg_mask] = 1e6
        hardest_neg = dist_neg.min(dim=1).values

        valid = pos_mask.sum(dim=1) > 0
        if not torch.any(valid):
            return torch.zeros((), device=embeddings.device)
        losses = torch.relu(hardest_pos - hardest_neg + self.config.triplet_margin)
        return losses[valid].mean()

    def _make_loader(self, dataset: ArrayDataset, *, shuffle: bool) -> Optional[DataLoader]:
        if dataset.X.size == 0:
            return None
        tensors = (
            torch.from_numpy(dataset.X.astype(np.float32)),
            torch.from_numpy(dataset.y.astype(np.float32)),
        )
        ds = TensorDataset(*tensors)
        batch_size = min(len(ds), max(1, self.config.batch_size))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _evaluate(self, model: _MASFNet, dataset: ArrayDataset) -> Dict[str, float]:
        if dataset.X.size == 0:
            nan = float("nan")
            return {"auroc": nan, "accuracy": nan, "auprc": nan}
        loader = self._make_loader(dataset, shuffle=False)
        assert loader is not None
        preds: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits, _ = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                labels.append(yb.numpy())
        y_true = np.concatenate(labels) if labels else np.array([], dtype=np.float32)
        y_score = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
        return {
            "auroc": safe_auc(y_true, y_score),
            "accuracy": safe_accuracy(y_true, y_score),
            "auprc": safe_auprc(y_true, y_score),
        }

    def _train_stage_domain(
        self,
        model: _MASFNet,
        train_ds: ArrayDataset,
        val_ds: Optional[ArrayDataset],
        *,
        epochs: int,
        lr: float,
    ) -> Tuple[int, Dict[str, float]]:
        if train_ds.X.size == 0 or epochs <= 0:
            nan = float("nan")
            return 0, {"auroc": nan, "accuracy": nan, "auprc": nan}
        if train_ds.domains is None:
            return self._train_stage_flat(model, train_ds, val_ds, epochs=epochs, lr=lr)

        sampler = _DomainBatchSampler(train_ds, self.config.per_domain_batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        best_state = deepcopy(model.state_dict())
        best_metric = -float("inf")
        best_epoch = 0
        best_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        patience = max(1, self.config.early_stopping_patience)
        patience_counter = 0

        def _metric_value(metrics: Dict[str, float]) -> float:
            value = metrics.get("auroc")
            if value is None or np.isnan(value):
                value = metrics.get("accuracy")
            if value is None or np.isnan(value):
                value = -float("inf")
            return float(value)

        for epoch in range(epochs):
            model.train()
            for _ in range(max(1, int(self.config.steps_per_epoch))):
                batch = sampler.sample()
                domain_ids = list(batch.keys())
                np.random.shuffle(domain_ids)
                num_domains = len(domain_ids)
                meta_train = max(1, int(num_domains * max(0.0, min(1.0, self.config.meta_training_proportion))))
                if meta_train >= num_domains and num_domains > 1:
                    meta_train = num_domains - 1
                train_domains = domain_ids[:meta_train]
                test_domains = domain_ids[meta_train:] or train_domains

                def _concat(domains: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
                    xs = torch.cat([batch[d]["X"] for d in domains], dim=0).to(self.device)
                    ys = torch.cat([batch[d]["y"] for d in domains], dim=0).to(self.device)
                    return xs, ys

                x_train, y_train = _concat(train_domains)
                x_test, y_test = _concat(test_domains)
                x_all = torch.cat([x_train, x_test], dim=0)
                y_all = torch.cat([y_train, y_test], dim=0)

                optimizer.zero_grad(set_to_none=True)
                logits_train, emb_train = model(x_train)
                logits_test, emb_test = model(x_test)
                logits_all, emb_all = model(x_all)

                loss_train = self.criterion(logits_train, y_train)
                loss_test = self.criterion(logits_test, y_test)
                local_loss = self._batch_hard_triplet_loss(emb_all, y_all)
                total_loss = loss_train + self.config.meta_testing_weight * loss_test
                total_loss = total_loss + self.config.local_loss_weight * local_loss

                total_loss.backward()
                if self.config.grad_clip and self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

            eval_ds = val_ds if val_ds is not None and val_ds.X.size > 0 else train_ds
            metrics = self._evaluate(model, eval_ds)
            score = _metric_value(metrics)
            if score >= best_metric + 1e-6 or best_epoch == 0:
                best_metric = score
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = deepcopy(model.state_dict())
                best_metrics = metrics
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_state)
        return best_epoch, best_metrics

    def _train_stage_flat(
        self,
        model: _MASFNet,
        train_ds: ArrayDataset,
        val_ds: Optional[ArrayDataset],
        *,
        epochs: int,
        lr: float,
    ) -> Tuple[int, Dict[str, float]]:
        if train_ds.X.size == 0 or epochs <= 0:
            nan = float("nan")
            return 0, {"auroc": nan, "accuracy": nan, "auprc": nan}

        loader = self._make_loader(train_ds, shuffle=True)
        assert loader is not None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        best_state = deepcopy(model.state_dict())
        best_metric = -float("inf")
        best_epoch = 0
        best_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        patience = max(1, self.config.early_stopping_patience)
        patience_counter = 0

        def _metric_value(metrics: Dict[str, float]) -> float:
            value = metrics.get("auroc")
            if value is None or np.isnan(value):
                value = metrics.get("accuracy")
            if value is None or np.isnan(value):
                value = -float("inf")
            return float(value)

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits, emb = model(xb)
                loss = self.criterion(logits, yb)
                local_loss = self._batch_hard_triplet_loss(emb, yb)
                total_loss = loss + self.config.local_loss_weight * local_loss
                total_loss.backward()
                if self.config.grad_clip and self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

            eval_ds = val_ds if val_ds is not None and val_ds.X.size > 0 else train_ds
            metrics = self._evaluate(model, eval_ds)
            score = _metric_value(metrics)
            if score >= best_metric + 1e-6 or best_epoch == 0:
                best_metric = score
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = deepcopy(model.state_dict())
                best_metrics = metrics
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_state)
        return best_epoch, best_metrics

    def run(
        self,
        *,
        seed: int,
        pretrain: Optional[ArrayDataset],
        pretrain_val: Optional[ArrayDataset],
        train: ArrayDataset,
        val: ArrayDataset,
        adapt: Optional[ArrayDataset],
        evaluation: ArrayDataset,
    ) -> DLMASFRunResult:
        _seed_everything(seed)

        model = self._build_model()
        stage_durations: Dict[str, float] = {}
        stage_epochs = {
            "pretrain_epochs": 0,
            "finetune_epochs": 0,
            "adapt_epochs": 0,
        }
        pretrain_val_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}

        if pretrain is not None and pretrain.X.size > 0 and self.config.pretrain_epochs > 0:
            start = perf_counter()
            best_epoch, metrics = self._train_stage_domain(
                model,
                pretrain,
                pretrain_val,
                epochs=self.config.pretrain_epochs,
                lr=self.config.pretrain_lr,
            )
            stage_durations["pretrain_seconds"] = perf_counter() - start
            stage_epochs["pretrain_epochs"] = best_epoch
            pretrain_val_metrics = metrics

        if self.config.freeze_backbone:
            model.freeze_backbone()

        if train.X.size > 0 and self.config.finetune_epochs > 0:
            start = perf_counter()
            best_epoch, _ = self._train_stage_domain(
                model,
                train,
                val if val.X.size > 0 else None,
                epochs=self.config.finetune_epochs,
                lr=self.config.finetune_lr,
            )
            stage_durations["finetune_seconds"] = perf_counter() - start
            stage_epochs["finetune_epochs"] = best_epoch

        if adapt is not None and adapt.X.size > 0 and self.config.adapt_epochs > 0:
            start = perf_counter()
            best_epoch, _ = self._train_stage_domain(
                model,
                adapt,
                val if val.X.size > 0 else None,
                epochs=self.config.adapt_epochs,
                lr=self.config.adapt_lr,
            )
            stage_durations["adapt_seconds"] = perf_counter() - start
            stage_epochs["adapt_epochs"] = best_epoch

        train_metrics = self._evaluate(model, train)
        val_metrics = self._evaluate(model, val)
        test_metrics = self._evaluate(model, evaluation)

        return DLMASFRunResult(
            train_auroc=train_metrics["auroc"],
            val_auroc=val_metrics["auroc"],
            test_auroc=test_metrics["auroc"],
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            test_accuracy=test_metrics["accuracy"],
            train_auprc=train_metrics["auprc"],
            val_auprc=val_metrics["auprc"],
            test_auprc=test_metrics["auprc"],
            pretrain_val_auroc=pretrain_val_metrics["auroc"],
            pretrain_val_accuracy=pretrain_val_metrics["accuracy"],
            pretrain_val_auprc=pretrain_val_metrics["auprc"],
            stage_durations=stage_durations,
            stage_epochs=stage_epochs,
            state_dict=deepcopy(model.state_dict()),
        )
