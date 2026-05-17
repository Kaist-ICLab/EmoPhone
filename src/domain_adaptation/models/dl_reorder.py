"""Reorder-style multi-task pipeline for domain adaptation experiments.

This is a simplified, tabular-friendly version of the GLOBEM `dl_reorder`
model. It adds an auxiliary task that predicts whether a feature permutation
was applied, encouraging representations that are less sensitive to feature
ordering.
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
class DLReorderConfig(DLERMConfig):
    num_reorder_classes: int = 4
    reorder_rate: float = 0.5
    reorder_weight: float = 0.3
    permutation_seed: int = 13


DLReorderRunResult = DLERMRunResult


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


class _ReorderNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: str,
        normalization: str,
        num_reorder_classes: int,
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
        self.backbone = nn.Sequential(*layers)
        self.label_head = nn.Linear(prev, 1)
        self.reorder_head = nn.Linear(prev, num_reorder_classes + 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x) if len(self.backbone) > 0 else x
        label_logits = self.label_head(features).squeeze(-1)
        reorder_logits = self.reorder_head(features)
        return label_logits, reorder_logits

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


class DLReorderPipeline:
    def __init__(self, config: DLReorderConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_criterion = nn.BCEWithLogitsLoss()
        self.reorder_criterion = nn.CrossEntropyLoss()
        self._rng = np.random.RandomState(config.permutation_seed)
        self._permutations: Optional[np.ndarray] = None

    def _build_model(self) -> _ReorderNet:
        model = _ReorderNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
            num_reorder_classes=self.config.num_reorder_classes,
        )
        return model.to(self.device)

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

    def _get_permutations(self) -> np.ndarray:
        if self._permutations is None:
            perms = []
            base = np.arange(self.config.input_dim)
            for _ in range(max(1, self.config.num_reorder_classes)):
                perm = base.copy()
                self._rng.shuffle(perm)
                perms.append(perm)
            self._permutations = np.stack(perms, axis=0)
        return self._permutations

    def _apply_reorder(self, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.num_reorder_classes <= 0 or self.config.reorder_rate <= 0:
            labels = torch.zeros(xb.size(0), dtype=torch.long, device=xb.device)
            return xb, labels
        perms = self._get_permutations()
        batch_size = xb.size(0)
        reorder_labels = np.zeros(batch_size, dtype=np.int64)
        mask = self._rng.rand(batch_size) < float(self.config.reorder_rate)
        if mask.any():
            reorder_labels[mask] = self._rng.randint(1, self.config.num_reorder_classes + 1, size=int(mask.sum()))
        xb_np = xb.detach().cpu().numpy()
        for idx, label in enumerate(reorder_labels):
            if label == 0:
                continue
            perm = perms[label - 1]
            xb_np[idx] = xb_np[idx][perm]
        xb_reordered = torch.from_numpy(xb_np).to(xb.device)
        return xb_reordered, torch.from_numpy(reorder_labels).to(xb.device)

    def _predict_proba(self, model: _ReorderNet, dataset: ArrayDataset) -> np.ndarray:
        if dataset.X.size == 0:
            return np.array([], dtype=np.float32)
        loader = self._make_loader(dataset, shuffle=False)
        assert loader is not None
        preds: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits, _ = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        return np.concatenate(preds) if preds else np.array([], dtype=np.float32)

    def _evaluate(self, model: _ReorderNet, dataset: ArrayDataset) -> Dict[str, float]:
        scores = self._predict_proba(model, dataset)
        return {
            "auroc": safe_auc(dataset.y, scores),
            "accuracy": safe_accuracy(dataset.y, scores),
            "auprc": safe_auprc(dataset.y, scores),
        }

    def _train_stage(
        self,
        model: _ReorderNet,
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
                xb_reordered, reorder_labels = self._apply_reorder(xb)
                optimizer.zero_grad(set_to_none=True)
                label_logits, reorder_logits = model(xb_reordered)
                label_loss = self.label_criterion(label_logits, yb)
                reorder_loss = self.reorder_criterion(reorder_logits, reorder_labels)
                loss = label_loss + self.config.reorder_weight * reorder_loss
                loss.backward()
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
    ) -> DLReorderRunResult:
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
            best_epoch, metrics = self._train_stage(
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
            best_epoch, _ = self._train_stage(
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
            best_epoch, _ = self._train_stage(
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

        return DLReorderRunResult(
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
