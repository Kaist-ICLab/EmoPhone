"""TabPFN training pipeline for domain adaptation experiments."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np
from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc




try:
    from tabpfn import TabPFNClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    TabPFNClassifier = None



@dataclass
class TabPFNConfig:
    n_ensemble_configurations: int = 32
    batch_size: Optional[int] = None
    device: Optional[str] = None
    max_train_samples: Optional[int] = 4096
    subsample_seed_offset: int = 13
    overwrite_warning: bool = False


@dataclass
class TabPFNRunResult:
    train_auroc: float
    val_auroc: float
    test_auroc: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_auprc: float
    val_auprc: float
    test_auprc: float
    stage_durations: Dict[str, float]


class TabPFNPipeline:
    def __init__(self, config: TabPFNConfig):
        if TabPFNClassifier is None:
            raise ImportError("tabpfn is required to run TabPFNPipeline")
        self.config = config

    @staticmethod
    def _stack_datasets(primary: ArrayDataset, extra: Optional[ArrayDataset]) -> Tuple[np.ndarray, np.ndarray]:
        parts = []
        labels = []
        if primary.X.size > 0:
            parts.append(primary.X)
            labels.append(primary.y)
        if extra is not None and extra.X.size > 0:
            parts.append(extra.X)
            labels.append(extra.y)
        if not parts:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        X = np.vstack(parts)
        y = np.concatenate(labels)
        return X, y

    def _subsample(self, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        limit = self.config.max_train_samples
        if limit is None or limit <= 0 or X.shape[0] <= limit:
            return X, y
        rng = np.random.default_rng(seed + self.config.subsample_seed_offset)
        idx = rng.choice(X.shape[0], limit, replace=False)
        return X[idx], y[idx]

    def _build_classifier(self, seed: int) -> "TabPFNClassifier":
        kwargs = {
            "N_ensemble_configurations": self.config.n_ensemble_configurations,
            "seed": seed,
        }
        if self.config.batch_size is not None:
            kwargs["batch_size"] = self.config.batch_size
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        return TabPFNClassifier(**kwargs)

    def run(
        self,
        *,
        seed: int,
        train: ArrayDataset,
        val: ArrayDataset,
        evaluation: ArrayDataset,
        extra_train: Optional[ArrayDataset] = None,
    ) -> TabPFNRunResult:
        X_train, y_train = self._stack_datasets(train, extra_train)
        if X_train.size == 0 or len(np.unique(y_train)) < 2:
            raise ValueError("TabPFN requires at least two classes in the training data.")
        X_train, y_train = self._subsample(X_train, y_train, seed)

        classifier = self._build_classifier(seed)
        stage_durations: Dict[str, float] = {}

        start = perf_counter()
        classifier.fit(
            X_train.astype(np.float32, copy=False),
            y_train.astype(np.int64, copy=False),
            overwrite_warning=self.config.overwrite_warning,
        )
        stage_durations["train_seconds"] = perf_counter() - start

        def _predict(dataset: ArrayDataset) -> np.ndarray:
            if dataset.X.size == 0:
                return np.array([], dtype=np.float32)
            probs = classifier.predict_proba(dataset.X.astype(np.float32, copy=False))
            return probs[:, 1] if probs.ndim == 2 else probs

        train_pred = _predict(train)
        val_pred = _predict(val)
        test_pred = _predict(evaluation)

        return TabPFNRunResult(
            train_auroc=safe_auc(train.y, train_pred),
            val_auroc=safe_auc(val.y, val_pred),
            test_auroc=safe_auc(evaluation.y, test_pred),
            train_accuracy=safe_accuracy(train.y, train_pred),
            val_accuracy=safe_accuracy(val.y, val_pred),
            test_accuracy=safe_accuracy(evaluation.y, test_pred),
            train_auprc=safe_auprc(train.y, train_pred),
            val_auprc=safe_auprc(val.y, val_pred),
            test_auprc=safe_auprc(evaluation.y, test_pred),
            stage_durations=stage_durations,
        )
