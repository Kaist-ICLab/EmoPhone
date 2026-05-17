"""Meta-learning for domain generalization (MLDG) pipeline."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

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


class _FeatureNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float,
        activation: str,
        normalization: str,
    ):
        super().__init__()
        layers: List[nn.Module] = []
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
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False


@dataclass
class DLMldgConfig(DLERMConfig):
    meta_training_proportion: float = 0.5
    meta_testing_weight: float = 1.0
    inner_learning_rate: float = 1e-3
    outer_learning_rate: float = 1e-3
    inner_optimizer: str = "adam"
    outer_optimizer: str = "adam"
    per_domain_batch_size: int = 64
    steps_per_epoch: int = 100


DLMldgRunResult = DLERMRunResult


class _DomainBatchSampler:
    def __init__(self, dataset: ArrayDataset, batch_size: int):
        if dataset.domains is None:
            raise ValueError("MLDG requires domain identifiers in ArrayDataset.domains")
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
            raise ValueError("No domains available for MLDG sampling")

    def sample(self) -> Dict[int, Dict[str, torch.Tensor]]:
        batch: Dict[int, Dict[str, torch.Tensor]] = {}
        for domain, indices in self.domain_to_indices.items():
            chosen = np.random.choice(indices, size=self.batch_size, replace=indices.size < self.batch_size)
            batch[domain] = {
                "X": torch.from_numpy(self.dataset.X[chosen]).float(),
                "y": torch.from_numpy(self.dataset.y[chosen]).float(),
            }
        return batch


class DLMldgPipeline:
    def __init__(self, config: DLMldgConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> _FeatureNet:
        hidden_dims = list(self.config.hidden_dims)
        model = _FeatureNet(
            input_dim=self.config.input_dim,
            hidden_dims=hidden_dims,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization,
        )
        return model.to(self.device)

    def _evaluate(self, model: _FeatureNet, dataset: ArrayDataset) -> Dict[str, float]:
        if dataset.X.size == 0:
            nan = float("nan")
            return {"auroc": nan, "accuracy": nan, "auprc": nan}
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(dataset.X.astype(np.float32)),
                torch.from_numpy(dataset.y.astype(np.float32)),
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        preds: List[np.ndarray] = []
        labels: List[np.ndarray] = []
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

    def _make_optimizer(self, params, kind: str, lr: float):
        kind = kind.lower()
        if kind == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=self.config.weight_decay)
        if kind == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=self.config.weight_decay)
        raise ValueError(f"Unsupported optimizer '{kind}'")

    def _meta_step(
        self,
        model: _FeatureNet,
        sampler: _DomainBatchSampler,
        inner_opt: torch.optim.Optimizer,
        outer_opt: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        batch = sampler.sample()
        domain_ids = list(batch.keys())
        np.random.shuffle(domain_ids)
        num_domains = len(domain_ids)
        meta_train_domains = max(1, int(num_domains * max(0.0, min(1.0, self.config.meta_training_proportion))))
        if meta_train_domains >= num_domains and num_domains > 1:
            meta_train_domains = num_domains - 1
        if meta_train_domains <= 0:
            meta_train_domains = 1
        train_domains = domain_ids[:meta_train_domains]
        test_domains = domain_ids[meta_train_domains:]
        if not test_domains:
            test_domains = train_domains

        def _concat(dom_list: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
            xs = torch.cat([batch[d]["X"] for d in dom_list], dim=0).to(self.device)
            ys = torch.cat([batch[d]["y"] for d in dom_list], dim=0).to(self.device)
            return xs, ys

        x_train, y_train = _concat(train_domains)
        x_test, y_test = _concat(test_domains)

        state_before = deepcopy(model.state_dict())

        inner_opt.zero_grad(set_to_none=True)
        logits_train = model(x_train)
        loss_train = self.criterion(logits_train, y_train)
        loss_train.backward(retain_graph=True)
        inner_opt.step()
        inner_opt.zero_grad(set_to_none=True)

        model.load_state_dict(state_before)
        logits_test = model(x_test)
        loss_test = self.criterion(logits_test, y_test)
        total_loss = loss_train.detach() + self.config.meta_testing_weight * loss_test

        outer_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        outer_opt.step()

        return {
            "loss_train": float(loss_train.detach().cpu().item()),
            "loss_test": float(loss_test.detach().cpu().item()),
            "loss_total": float(total_loss.detach().cpu().item()),
        }

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
    ) -> DLMldgRunResult:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = self._build_model()
        if self.config.freeze_backbone:
            print("[DLMldgPipeline] freeze_backbone=True ignored; MLDG requires trainable backbone.")
        inner_opt = self._make_optimizer(model.parameters(), self.config.inner_optimizer, self.config.inner_learning_rate)
        outer_opt = self._make_optimizer(model.parameters(), self.config.outer_optimizer, self.config.outer_learning_rate)
        sampler = _DomainBatchSampler(train, self.config.per_domain_batch_size)

        history: List[Dict[str, float]] = []
        total_steps = max(1, self.config.steps_per_epoch * max(1, self.config.finetune_epochs))
        for _ in range(total_steps):
            stats = self._meta_step(model, sampler, inner_opt, outer_opt)
            history.append(stats)

        train_metrics = self._evaluate(model, train)
        val_metrics = self._evaluate(model, val)
        test_metrics = self._evaluate(model, evaluation)

        stage_durations = {"pretrain_seconds": None, "finetune_seconds": None, "adapt_seconds": None}
        stage_epochs = {"pretrain_epochs": None, "finetune_epochs": self.config.finetune_epochs, "adapt_epochs": None}

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
            pretrain_val_auroc=float("nan"),
            pretrain_val_accuracy=float("nan"),
            pretrain_val_auprc=float("nan"),
            # pretrain_val_auroc=pretrain_val_metrics["auroc"],
            # pretrain_val_accuracy=pretrain_val_metrics["accuracy"],
            # pretrain_val_auprc=pretrain_val_metrics["auprc"],
            stage_durations=stage_durations,
            stage_epochs=stage_epochs,
            state_dict=deepcopy(model.state_dict()),
        )