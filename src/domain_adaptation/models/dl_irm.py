"""Invariant risk minimization (IRM) pipeline for domain adaptation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from .common import ArrayDataset
from .dl_erm import DLERMConfig, DLERMPipeline, DLERMRunResult






@dataclass
class DLIRMConfig(DLERMConfig):
    penalty_phase1_epochs: int = 10
    penalty_phase2_epochs: int = 20
    penalty_phase1_weight: float = float(5 ** 13)
    penalty_phase2_weight: float = float(3 * 10 ** 15)
    penalty_phase3_weight: float = float(10 ** 16)


DLIRMRunResult = DLERMRunResult


class DLIRMPipeline(DLERMPipeline):
    def __init__(self, config: DLIRMConfig):
        super().__init__(config)
        self.irm_config = config

    def _penalty_weight(self, epoch: int) -> float:
        if epoch < self.irm_config.penalty_phase1_epochs:
            return float(self.irm_config.penalty_phase1_weight)
        if epoch < self.irm_config.penalty_phase2_epochs:
            return float(self.irm_config.penalty_phase2_weight)
        return float(self.irm_config.penalty_phase3_weight)

    def _irm_penalty(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scale = torch.ones(1, device=logits.device, requires_grad=True)
        scaled_loss = self.criterion(logits * scale, targets)
        grad = torch.autograd.grad(scaled_loss, [scale], create_graph=True)[0]
        return grad.pow(2)

    def _train_stage(
        self,
        model,
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

        for epoch in range(epochs):
            model.train()
            penalty_weight = torch.tensor(self._penalty_weight(epoch), device=self.device)
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = self.criterion(logits, yb)
                penalty = self._irm_penalty(logits, yb)
                total_loss = loss + penalty_weight * penalty
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
