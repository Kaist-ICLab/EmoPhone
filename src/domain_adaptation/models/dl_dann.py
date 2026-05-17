"""Domain-adversarial neural network (DANN) pipeline for domain adaptation."""
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


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class _FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: str,
        normalization: str,
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
        self.layers = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.layers) == 0:
            return x
        return self.layers(x)


class _DANNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: str,
        normalization: str,
        domain_hidden_dims: Sequence[int],
        domain_dropout: float,
        domain_classes: int,
    ):
        super().__init__()
        self.feature_extractor = _FeatureExtractor(input_dim, hidden_dims, dropout, activation, normalization)
        feature_dim = self.feature_extractor.output_dim if hidden_dims else input_dim
        self.label_head = nn.Linear(feature_dim, 1)

        domain_layers: list[nn.Module] = []
        prev = feature_dim
        act = _activation(activation)
        for hidden in domain_hidden_dims:
            domain_layers.extend([
                nn.Linear(prev, hidden),
                deepcopy(act),
                nn.Dropout(domain_dropout),
            ])
            prev = hidden
        domain_layers.append(nn.Linear(prev, domain_classes))
        self.domain_head = nn.Sequential(*domain_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.label_head(features).squeeze(-1)
        return logits

    def forward_with_domain(self, x: torch.Tensor, lambda_: float) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.label_head(features).squeeze(-1)
        reversed_features = _GradientReverse.apply(features, lambda_)
        domain_logits = self.domain_head(reversed_features)
        return logits, domain_logits

    def freeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


@dataclass
class DLDANNConfig(DLERMConfig):
    domain_hidden_dims: Tuple[int, ...] = (128,)
    domain_dropout: float = 0.1
    domain_loss_weight: float = 1.0
    domain_classes: int = 2
    source_label_weight: float = 1.0
    target_label_weight: float = 0.0
    reversal_lambda_max: float = 1.0
    reversal_gamma: float = 10.0
    reversal_schedule_epochs: Optional[int] = None


DLDANNRunResult = DLERMRunResult


class DLDANNPipeline:
    def __init__(self, config: DLDANNConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def _build_model(self) -> _DANNNet:
        model = _DANNNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
            domain_hidden_dims=self.config.domain_hidden_dims,
            domain_dropout=self.config.domain_dropout,
            domain_classes=self.config.domain_classes,
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

    def _make_dann_loader(
        self,
        source: ArrayDataset,
        target: Optional[ArrayDataset],
    ) -> Optional[DataLoader]:
        if source.X.size == 0:
            return None
        parts = []
        parts.append(
            (
                source.X.astype(np.float32, copy=False),
                source.y.astype(np.float32, copy=False),
                np.zeros(len(source.y), dtype=np.int64),
                np.ones(len(source.y), dtype=bool),
            )
        )
        if target is not None and target.X.size > 0:
            parts.append(
                (
                    target.X.astype(np.float32, copy=False),
                    target.y.astype(np.float32, copy=False),
                    np.ones(len(target.y), dtype=np.int64),
                    np.full(len(target.y), self.config.target_label_weight > 0.0, dtype=bool),
                )
            )
        X = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        domain = np.concatenate([p[2] for p in parts])
        label_mask = np.concatenate([p[3] for p in parts])
        tensors = (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(domain),
            torch.from_numpy(label_mask),
        )
        ds = TensorDataset(*tensors)
        batch_size = min(len(ds), max(1, self.config.batch_size))
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    def _evaluate(self, model: _DANNNet, dataset: ArrayDataset) -> Dict[str, float]:
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
                logits = model(xb)
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

    def _train_stage(
        self,
        model: _DANNNet,
        train_ds: ArrayDataset,
        val_ds: Optional[ArrayDataset],
        *,
        epochs: int,
        lr: float,
    ) -> Tuple[int, Dict[str, float]]:
        if train_ds.X.size == 0 or epochs <= 0:
            return 0, {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}

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
                logits = model(xb)
                loss = self.label_criterion(logits, yb)
                loss.backward()
                if self.config.grad_clip and self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

            metrics = self._evaluate(model, val_ds if val_ds is not None and val_ds.X.size > 0 else train_ds)
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

    def _schedule_lambda(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return float(self.config.reversal_lambda_max)
        progress = step / total_steps
        gamma = self.config.reversal_gamma
        return float(self.config.reversal_lambda_max * (2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0))

    def _train_dann_stage(
        self,
        model: _DANNNet,
        source: ArrayDataset,
        target: Optional[ArrayDataset],
        val_ds: Optional[ArrayDataset],
        *,
        epochs: int,
        lr: float,
    ) -> Tuple[int, Dict[str, float]]:
        if source.X.size == 0 or epochs <= 0:
            return 0, {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        if target is None or target.X.size == 0:
            return self._train_stage(model, source, val_ds, epochs=epochs, lr=lr)

        loader = self._make_dann_loader(source, target)
        assert loader is not None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        best_state = deepcopy(model.state_dict())
        best_metric = -float("inf")
        best_epoch = 0
        best_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        patience = max(1, self.config.early_stopping_patience)
        patience_counter = 0
        total_steps = max(1, epochs * len(loader))
        schedule_steps = total_steps
        if self.config.reversal_schedule_epochs is not None:
            schedule_steps = max(1, self.config.reversal_schedule_epochs * len(loader))
        step_counter = 0

        def _metric_value(metrics: Dict[str, float]) -> float:
            value = metrics.get("auroc")
            if value is None or np.isnan(value):
                value = metrics.get("accuracy")
            if value is None or np.isnan(value):
                value = -float("inf")
            return float(value)

        for epoch in range(epochs):
            model.train()
            for xb, yb, db, mask in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                db = db.to(self.device)
                mask = mask.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                lambda_val = self._schedule_lambda(min(step_counter, schedule_steps), schedule_steps)
                step_counter += 1
                label_logits, domain_logits = model.forward_with_domain(xb, lambda_val)
                loss = torch.tensor(0.0, device=self.device)
                label_loss = torch.tensor(0.0, device=self.device)

                source_mask = (db == 0) & mask
                if source_mask.any():
                    label_loss = label_loss + self.config.source_label_weight * self.label_criterion(label_logits[source_mask], yb[source_mask])

                if self.config.target_label_weight > 0.0:
                    target_mask = (db == 1) & mask
                    if target_mask.any():
                        label_loss = label_loss + self.config.target_label_weight * self.label_criterion(label_logits[target_mask], yb[target_mask])

                loss = loss + label_loss
                if self.config.domain_loss_weight > 0.0:
                    loss = loss + self.config.domain_loss_weight * self.domain_criterion(domain_logits, db)

                loss.backward()
                if self.config.grad_clip and self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

            metrics = self._evaluate(model, val_ds if val_ds is not None and val_ds.X.size > 0 else source)
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
    ) -> DLDANNRunResult:
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
            best_epoch, metrics = self._train_dann_stage(
                model,
                source=pretrain,
                target=train if train.X.size > 0 else None,
                val_ds=pretrain_val,
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

        return DLDANNRunResult(
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
