"""Clustering-augmented MLP pipeline for domain adaptation experiments.

This module implements a lightweight clustering variant of the DL ERM baseline
used in ``domain_adaptation``. It learns a global MLP classifier, then assigns
samples to KMeans clusters and optionally fine-tunes cluster-specific copies
of the global model. This keeps the interface consistent with other pipelines
while remaining dependency-light (PyTorch + scikit-learn).

For the original GLOBEM TensorFlow implementation, see
``GLOBEM/algorithm/dl_clustering.py``.
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




try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None



@dataclass
class DLClusteringConfig(DLERMConfig):
    n_clusters: int = 8
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    cluster_min_samples: int = 24
    cluster_epochs: int = 20
    cluster_lr: float = 2e-4
    cluster_use_global_init: bool = True
    cluster_source: str = "pretrain"  # choices: pretrain, train, combined


DLClusteringRunResult = DLERMRunResult


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


class _ERMNet(nn.Module):
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
        prev_dim = input_dim
        act = _activation(activation)
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                _normalization(normalization, hidden),
                deepcopy(act),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x) if len(self.backbone) > 0 else x
        return self.head(features).squeeze(-1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


class DLClusteringPipeline:
    def __init__(self, config: DLClusteringConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> _ERMNet:
        model = _ERMNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
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

    def _evaluate(self, model: _ERMNet, dataset: ArrayDataset) -> Dict[str, float]:
        scores = self._predict_proba(model, dataset)
        return {
            "auroc": safe_auc(dataset.y, scores),
            "accuracy": safe_accuracy(dataset.y, scores),
            "auprc": safe_auprc(dataset.y, scores),
        }

    def _predict_proba(self, model: _ERMNet, dataset: ArrayDataset) -> np.ndarray:
        if dataset.X.size == 0:
            return np.array([], dtype=np.float32)
        loader = self._make_loader(dataset, shuffle=False)
        assert loader is not None
        preds: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        return np.concatenate(preds) if preds else np.array([], dtype=np.float32)

    def _train_stage(
        self,
        model: _ERMNet,
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
                logits = model(xb)
                loss = self.criterion(logits, yb)
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

    def _empty_dataset(self) -> ArrayDataset:
        return ArrayDataset(
            np.empty((0, self.config.input_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    def _subset_dataset(self, dataset: ArrayDataset, mask: np.ndarray) -> ArrayDataset:
        if dataset.X.size == 0 or mask.sum() == 0:
            return self._empty_dataset()
        X = dataset.X[mask]
        y = dataset.y[mask]
        domains = dataset.domains[mask] if dataset.domains is not None else None
        return ArrayDataset(X, y, domains)

    def _merge_datasets(self, datasets: Sequence[Optional[ArrayDataset]]) -> ArrayDataset:
        parts = [ds for ds in datasets if ds is not None and ds.X.size > 0]
        if not parts:
            return self._empty_dataset()
        if len(parts) == 1:
            return parts[0]
        X = np.vstack([ds.X for ds in parts])
        y = np.concatenate([ds.y for ds in parts])
        if all(ds.domains is not None for ds in parts):
            domains = np.concatenate([ds.domains for ds in parts])
        else:
            domains = None
        return ArrayDataset(X, y, domains)

    def _select_cluster_source(
        self,
        pretrain: Optional[ArrayDataset],
        train: ArrayDataset,
    ) -> ArrayDataset:
        mode = (self.config.cluster_source or "pretrain").lower()
        if mode == "combined":
            combined = self._merge_datasets([pretrain, train])
            if combined.X.size > 0:
                return combined
        if mode == "train":
            if train.X.size > 0:
                return train
        if pretrain is not None and pretrain.X.size > 0:
            return pretrain
        return train if train.X.size > 0 else self._empty_dataset()

    def _fit_kmeans(self, dataset: ArrayDataset, *, seed: int) -> Optional["KMeans"]:
        if KMeans is None:
            raise ImportError("scikit-learn is required for DLClusteringPipeline")
        if dataset.X.size == 0:
            return None
        n_samples = dataset.X.shape[0]
        n_clusters = max(1, min(self.config.n_clusters, n_samples))
        if n_clusters <= 1:
            return None
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=self.config.kmeans_n_init,
                max_iter=self.config.kmeans_max_iter,
                random_state=seed,
            )
            kmeans.fit(dataset.X)
            return kmeans
        except Exception:
            return None

    def _assign_clusters(self, kmeans: Optional["KMeans"], dataset: ArrayDataset) -> np.ndarray:
        if dataset.X.size == 0:
            return np.array([], dtype=np.int64)
        if kmeans is None:
            return np.zeros(dataset.X.shape[0], dtype=np.int64)
        return kmeans.predict(dataset.X)

    def _predict_with_clusters(
        self,
        dataset: ArrayDataset,
        kmeans: Optional["KMeans"],
        cluster_models: Dict[int, _ERMNet],
        fallback_model: Optional[_ERMNet],
    ) -> np.ndarray:
        if dataset.X.size == 0:
            return np.array([], dtype=np.float32)
        assignments = self._assign_clusters(kmeans, dataset)
        probs = np.zeros(dataset.X.shape[0], dtype=np.float32)
        unique_clusters = np.unique(assignments) if assignments.size > 0 else np.array([0], dtype=np.int64)
        for cluster_id in unique_clusters:
            mask = assignments == cluster_id
            model = cluster_models.get(int(cluster_id), fallback_model)
            if model is None:
                probs[mask] = 0.5
                continue
            subset = self._subset_dataset(dataset, mask)
            probs[mask] = self._predict_proba(model, subset)
        return probs

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
    ) -> DLClusteringRunResult:
        _seed_everything(seed)

        model = self._build_model()
        stage_durations: Dict[str, float] = {}
        stage_epochs = {
            "pretrain_epochs": 0,
            "finetune_epochs": 0,
            "adapt_epochs": 0,
            "cluster_epochs": 0,
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

        global_state = deepcopy(model.state_dict())

        cluster_source = self._select_cluster_source(pretrain, train)
        kmeans = self._fit_kmeans(cluster_source, seed=seed) if cluster_source.X.size > 0 else None

        cluster_models: Dict[int, _ERMNet] = {}
        cluster_seconds = 0.0
        cluster_epochs_max = 0

        if kmeans is not None or cluster_source.X.size > 0:
            train_clusters = self._assign_clusters(kmeans, train)
            val_clusters = self._assign_clusters(kmeans, val)
            adapt_clusters = self._assign_clusters(kmeans, adapt) if adapt is not None else np.array([], dtype=np.int64)
            pretrain_clusters = (
                self._assign_clusters(kmeans, pretrain) if pretrain is not None else np.array([], dtype=np.int64)
            )

            n_clusters = max(1, min(self.config.n_clusters, cluster_source.X.shape[0]))
            for cluster_id in range(n_clusters):
                train_subset = self._subset_dataset(train, train_clusters == cluster_id)
                val_subset = self._subset_dataset(val, val_clusters == cluster_id)
                adapt_subset = None
                if adapt is not None and adapt_clusters.size > 0:
                    adapt_subset = self._subset_dataset(adapt, adapt_clusters == cluster_id)
                pretrain_subset = None
                if pretrain is not None and pretrain_clusters.size > 0:
                    pretrain_subset = self._subset_dataset(pretrain, pretrain_clusters == cluster_id)

                cluster_train = self._merge_datasets([train_subset, adapt_subset])
                if cluster_train.X.size == 0 and pretrain_subset is not None:
                    cluster_train = pretrain_subset

                if (
                    # cluster_train.X.shape[0] < self.config.cluster_min_samples
                    cluster_train.X.shape[0] < self.config.cluster_min_samples
                    or not cluster_train.is_valid()
                ):
                    continue

                cluster_model = self._build_model()
                if self.config.cluster_use_global_init:
                    cluster_model.load_state_dict(global_state)
                if self.config.freeze_backbone:
                    cluster_model.freeze_backbone()

                start = perf_counter()
                best_epoch, _ = self._train_stage(
                    cluster_model,
                    cluster_train,
                    val_subset if val_subset.X.size > 0 else None,
                    epochs=self.config.cluster_epochs,
                    lr=self.config.cluster_lr,
                )
                cluster_seconds += perf_counter() - start
                cluster_epochs_max = max(cluster_epochs_max, best_epoch)
                cluster_models[cluster_id] = cluster_model

        if cluster_seconds:
            stage_durations["cluster_seconds"] = cluster_seconds
        stage_epochs["cluster_epochs"] = cluster_epochs_max

        train_scores = self._predict_with_clusters(train, kmeans, cluster_models, model)
        val_scores = self._predict_with_clusters(val, kmeans, cluster_models, model)
        test_scores = self._predict_with_clusters(evaluation, kmeans, cluster_models, model)

        return DLClusteringRunResult(
            train_auroc=safe_auc(train.y, train_scores),
            val_auroc=safe_auc(val.y, val_scores),
            test_auroc=safe_auc(evaluation.y, test_scores),
            train_accuracy=safe_accuracy(train.y, train_scores),
            val_accuracy=safe_accuracy(val.y, val_scores),
            test_accuracy=safe_accuracy(evaluation.y, test_scores),
            train_auprc=safe_auprc(train.y, train_scores),
            val_auprc=safe_auprc(val.y, val_scores),
            test_auprc=safe_auprc(evaluation.y, test_scores),
            pretrain_val_auroc=pretrain_val_metrics["auroc"],
            pretrain_val_accuracy=pretrain_val_metrics["accuracy"],
            pretrain_val_auprc=pretrain_val_metrics["auprc"],
            stage_durations=stage_durations,
            stage_epochs=stage_epochs,
            state_dict=deepcopy(model.state_dict()),
        )
