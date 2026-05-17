"""Common-specific decomposition (CSD) pipeline for domain adaptation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc
from .dl_erm import DLERMConfig, DLERMRunResult






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
        self.output_dim = prev if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.layers) == 0:
            return x
        return self.layers(x)


class _CSDNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: str,
        normalization: str,
        domain_count: int,
        low_rank_dim: int,
    ):
        super().__init__()
        self.feature_extractor = _FeatureExtractor(input_dim, hidden_dims, dropout, activation, normalization)
        embedding_dim = self.feature_extractor.output_dim
        self.common_head = nn.Linear(embedding_dim, 1)
        self.domain_embeddings = nn.Embedding(max(1, domain_count), low_rank_dim)
        self.weight_proj = nn.Linear(low_rank_dim, embedding_dim)
        self.bias_proj = nn.Linear(low_rank_dim, 1)

    def forward(self, x: torch.Tensor, domains: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        domain_latent = self.domain_embeddings(domains)
        specific_weight = self.weight_proj(domain_latent)
        specific_bias = self.bias_proj(domain_latent).squeeze(-1)
        specific_logits = (features * specific_weight).sum(dim=1) + specific_bias
        common_logits = self.common_head(features).squeeze(-1)
        reg = domain_latent.pow(2).mean()
        return common_logits, specific_logits, reg

    def freeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


@dataclass
class DLCSConfig(DLERMConfig):
    domain_count: int = 2
    low_rank_dim: int = 32
    common_specific_weight: float = 0.5
    reg_weight: float = 1e-4


DLCSRunResult = DLERMRunResult


class DLCSDPipeline:
    def __init__(self, config: DLCSConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> _CSDNet:
        model = _CSDNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
            domain_count=self.config.domain_count,
            low_rank_dim=self.config.low_rank_dim,
        )
        return model.to(self.device)

    def _make_loader(self, dataset: ArrayDataset, *, shuffle: bool, require_domains: bool = True) -> Optional[DataLoader]:
        if dataset.X.size == 0:
            return None
        domains = dataset.domains
        if domains is None:
            if require_domains:
                raise ValueError("dl_csd requires domain identifiers in ArrayDataset.domains")
            domains = np.zeros(len(dataset.y), dtype=np.int64)
        tensors = (
            torch.from_numpy(dataset.X.astype(np.float32)),
            torch.from_numpy(dataset.y.astype(np.float32)),
            torch.from_numpy(domains.astype(np.int64)),
        )
        ds = TensorDataset(*tensors)
        batch_size = min(len(ds), max(1, self.config.batch_size))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _evaluate(self, model: _CSDNet, dataset: ArrayDataset) -> Dict[str, float]:
        if dataset.X.size == 0:
            nan = float("nan")
            return {"auroc": nan, "accuracy": nan, "auprc": nan}
        loader = self._make_loader(dataset, shuffle=False, require_domains=False)
        assert loader is not None
        preds: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb, yb, db in loader:
                xb = xb.to(self.device)
                db = db.to(self.device)
                common_logits, specific_logits, _ = model(xb, db)
                alpha = float(self.config.common_specific_weight)
                logits = alpha * common_logits + (1.0 - alpha) * specific_logits
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
        model: _CSDNet,
        train_ds: ArrayDataset,
        val_ds: Optional[ArrayDataset],
        *,
        epochs: int,
        lr: float,
    ) -> tuple[int, Dict[str, float]]:
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

        alpha = float(self.config.common_specific_weight)
        reg_weight = float(self.config.reg_weight)

        for epoch in range(epochs):
            model.train()
            for xb, yb, db in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                db = db.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                common_logits, specific_logits, reg = model(xb, db)
                loss_common = self.criterion(common_logits, yb)
                loss_specific = self.criterion(specific_logits, yb)
                total_loss = alpha * loss_common + (1.0 - alpha) * loss_specific + reg_weight * reg
                total_loss.backward()
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
    ) -> DLCSRunResult:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = self._build_model()
        stage_durations: Dict[str, float] = {}
        stage_epochs = {
            "pretrain_epochs": 0,
            "finetune_epochs": 0,
            "adapt_epochs": 0,
        }
        pretrain_val_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}

        if pretrain is not None and pretrain.X.size > 0 and self.config.pretrain_epochs > 0:
            start_time = perf_counter()
            best_epoch, metrics = self._train_stage(
                model,
                pretrain,
                pretrain_val,
                epochs=self.config.pretrain_epochs,
                lr=self.config.pretrain_lr,
            )
            stage_durations["pretrain_seconds"] = perf_counter() - start_time
            stage_epochs["pretrain_epochs"] = best_epoch
            pretrain_val_metrics = metrics

        if self.config.freeze_backbone:
            model.freeze_backbone()

        if train.X.size > 0 and self.config.finetune_epochs > 0:
            start_time = perf_counter()
            best_epoch, _ = self._train_stage(
                model,
                train,
                val if val.X.size > 0 else None,
                epochs=self.config.finetune_epochs,
                lr=self.config.finetune_lr,
            )
            stage_durations["finetune_seconds"] = perf_counter() - start_time
            stage_epochs["finetune_epochs"] = best_epoch

        if adapt is not None and adapt.X.size > 0 and self.config.adapt_epochs > 0:
            start_time = perf_counter()
            best_epoch, _ = self._train_stage(
                model,
                adapt,
                val if val.X.size > 0 else None,
                epochs=self.config.adapt_epochs,
                lr=self.config.adapt_lr,
            )
            stage_durations["adapt_seconds"] = perf_counter() - start_time
            stage_epochs["adapt_epochs"] = best_epoch

        train_metrics = self._evaluate(model, train)
        val_metrics = self._evaluate(model, val)
        test_metrics = self._evaluate(model, evaluation)

        return DLERMRunResult(
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
