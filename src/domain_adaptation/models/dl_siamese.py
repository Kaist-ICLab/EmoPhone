"""Siamese-style metric learning pipeline for domain adaptation experiments.

This is a lightweight PyTorch approximation of the GLOBEM TensorFlow
`dl_siamese` model. It learns an embedding space with a triplet loss and
predicts labels via k-NN in embedding space.
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
class DLSiameseConfig(DLERMConfig):
    embedding_dim: int = 32
    triplet_margin: float = 0.2
    knn_k: int = 3
    knn_distance_weight: bool = False
    normalize_embeddings: bool = True
    support_use_adapt: bool = True
    max_support_size: Optional[int] = 2000
    knn_batch_size: int = 512


DLSiameseRunResult = DLERMRunResult


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


class _EmbeddingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        embedding_dim: int,
        dropout: float,
        activation: str,
        normalization: str,
        normalize_embeddings: bool,
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
        layers.append(nn.Linear(prev, embedding_dim))
        self.model = nn.Sequential(*layers)
        self.normalize_embeddings = normalize_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.model(x)
        if self.normalize_embeddings:
            emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    def freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False


class DLSiamesePipeline:
    def __init__(self, config: DLSiameseConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> _EmbeddingNet:
        model = _EmbeddingNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            embedding_dim=self.config.embedding_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
            normalize_embeddings=self.config.normalize_embeddings,
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

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        dot = embeddings @ embeddings.t()
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(1) - 2.0 * dot + sq_norm.unsqueeze(0)
        return torch.clamp(dist, min=0.0)

    def _batch_hard_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.size(0) < 2:
            return embeddings.sum() * 0.0
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
            return embeddings.sum() * 0.0
        losses = torch.relu(hardest_pos - hardest_neg + self.config.triplet_margin)
        return losses[valid].mean()

    def _embed_dataset(self, model: _EmbeddingNet, dataset: ArrayDataset) -> np.ndarray:
        if dataset.X.size == 0:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)
        loader = self._make_loader(dataset, shuffle=False)
        assert loader is not None
        model.eval()
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                emb = model(xb).cpu().numpy()
                embeddings.append(emb)
        return np.vstack(embeddings) if embeddings else np.empty((0, self.config.embedding_dim), dtype=np.float32)

    def _predict_knn(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> np.ndarray:
        if train_embeddings.size == 0 or query_embeddings.size == 0:
            return np.array([], dtype=np.float32)
        k = max(1, int(self.config.knn_k))
        k = min(k, train_embeddings.shape[0])
        batch_size = max(1, int(self.config.knn_batch_size))
        probs = np.empty((query_embeddings.shape[0],), dtype=np.float32)
        for start in range(0, query_embeddings.shape[0], batch_size):
            end = min(start + batch_size, query_embeddings.shape[0])
            q_batch = query_embeddings[start:end]
            diffs = q_batch[:, None, :] - train_embeddings[None, :, :]
            dists = np.sum(diffs * diffs, axis=-1)
            idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
            selected = np.take_along_axis(dists, idx, axis=1)
            labels = train_labels[idx]
            if self.config.knn_distance_weight:
                weights = 1.0 / (selected + 1e-8)
                prob = (weights * labels).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-8)
            else:
                prob = labels.mean(axis=1)
            probs[start:end] = prob.astype(np.float32)
        return probs

    def _evaluate_knn(
        self,
        model: _EmbeddingNet,
        support: ArrayDataset,
        query: ArrayDataset,
    ) -> Dict[str, float]:
        if query.X.size == 0 or support.X.size == 0:
            nan = float("nan")
            return {"auroc": nan, "accuracy": nan, "auprc": nan}
        train_emb = self._embed_dataset(model, support)
        query_emb = self._embed_dataset(model, query)
        probs = self._predict_knn(train_emb, support.y, query_emb)
        return {
            "auroc": safe_auc(query.y, probs),
            "accuracy": safe_accuracy(query.y, probs),
            "auprc": safe_auprc(query.y, probs),
        }

    def _merge_support(self, train: ArrayDataset, adapt: Optional[ArrayDataset]) -> ArrayDataset:
        if adapt is None or adapt.X.size == 0 or not self.config.support_use_adapt:
            return train
        X = np.vstack([train.X, adapt.X]) if train.X.size else adapt.X
        y = np.concatenate([train.y, adapt.y]) if train.y.size else adapt.y
        if train.domains is not None and adapt.domains is not None:
            domains = np.concatenate([train.domains, adapt.domains])
        else:
            domains = None
        if self.config.max_support_size is not None and X.shape[0] > self.config.max_support_size:
            rng = np.random.default_rng(42)
            labels = y.astype(int)
            idx_pos = np.where(labels == 1)[0]
            idx_neg = np.where(labels == 0)[0]
            target = int(self.config.max_support_size)
            pos_target = min(len(idx_pos), max(1, target // 2))
            neg_target = min(len(idx_neg), target - pos_target)
            if len(idx_pos) == 0 or len(idx_neg) == 0:
                chosen = rng.choice(np.arange(X.shape[0]), size=target, replace=False)
            else:
                chosen_pos = rng.choice(idx_pos, size=pos_target, replace=len(idx_pos) < pos_target)
                chosen_neg = rng.choice(idx_neg, size=neg_target, replace=len(idx_neg) < neg_target)
                chosen = np.concatenate([chosen_pos, chosen_neg])
                rng.shuffle(chosen)
            X = X[chosen]
            y = y[chosen]
            if domains is not None:
                domains = domains[chosen]
        return ArrayDataset(X, y, domains)

    def _train_stage(
        self,
        model: _EmbeddingNet,
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
                emb = model(xb)
                loss = self._batch_hard_triplet_loss(emb, yb)
                loss.backward()
                if self.config.grad_clip and self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

            eval_ds = val_ds if val_ds is not None and val_ds.X.size > 0 else None
            if eval_ds is not None:
                metrics = self._evaluate_knn(model, train_ds, eval_ds)
            else:
                metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
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
    ) -> DLSiameseRunResult:
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
            print("[DLSiamesePipeline] freeze_backbone=True ignored; Siamese training requires trainable embeddings.")

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

        support = train if train.X.size > 0 else (pretrain if pretrain is not None else train)
        support = self._merge_support(support, adapt)

        train_metrics = self._evaluate_knn(model, support, train) if train.X.size else {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        val_metrics = self._evaluate_knn(model, support, val) if val.X.size else {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        test_metrics = self._evaluate_knn(model, support, evaluation) if evaluation.X.size else {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}

        return DLSiameseRunResult(
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
