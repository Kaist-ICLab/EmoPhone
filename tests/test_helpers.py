"""Unit tests for the shared training helpers.

Behavioural tests for :class:`EarlyStopTracker` and the small DataLoader
utilities -- they sit on the critical path of every train_* loop, so the
test focuses on the patience semantics, best-state restore, and the
metadata payload that downstream logging consumes.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from domain_adaptation.models._da_helpers import (
    EarlyStopTracker,
    _build_loaders,
    _evaluate_val,
    _infinite_iterator,
)


class _Tiny(nn.Module):
    """Minimal model that satisfies the .predict() / .state_dict() contract."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def predict(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.predict(x)


class _NoopIterator:
    """Stub stand-in for the tqdm iterator the tracker writes through."""

    def __init__(self):
        self.postfix = None
        self.messages = []

    def set_postfix(self, value):
        self.postfix = value

    def write(self, message):
        self.messages.append(message)


def test_tracker_updates_best_state_when_score_improves():
    tracker = EarlyStopTracker(patience=3, epochs=10, batch_size=8,
                               lr=1e-3, weight_decay=0.0)
    model = _Tiny()
    it = _NoopIterator()

    stopped = tracker.record(model, epoch_num=1, train_loss=1.0,
                             val_loss=1.2, val_auroc=0.61, iterator=it)
    assert stopped is False
    assert tracker.best_val_score == pytest.approx(0.61)
    assert tracker.best_epoch == 1
    assert tracker.best_model_state is not None


def test_tracker_fires_after_consecutive_non_improvements():
    """Patience=2 -> two consecutive epochs without improvement triggers stop."""
    tracker = EarlyStopTracker(patience=2, epochs=10, batch_size=8,
                               lr=1e-3, weight_decay=0.0)
    model = _Tiny()
    it = _NoopIterator()

    tracker.record(model, epoch_num=1, train_loss=1.0, val_loss=1.2,
                   val_auroc=0.70, iterator=it)
    tracker.record(model, epoch_num=2, train_loss=0.9, val_loss=1.1,
                   val_auroc=0.65, iterator=it)
    stopped = tracker.record(model, epoch_num=3, train_loss=0.8, val_loss=1.0,
                             val_auroc=0.60, iterator=it)

    assert stopped is True
    assert tracker.early_stopped is True
    assert tracker.early_stop_epoch == 3
    # Best snapshot stays at epoch 1, never overwritten.
    assert tracker.best_epoch == 1
    assert tracker.best_val_score == pytest.approx(0.70)
    # The recorded message uses the zero-indexed epoch (epoch_num - 1) by design.
    assert it.messages and "epoch 2" in it.messages[0]


def test_tracker_finalize_restores_best_state_and_writes_metadata():
    tracker = EarlyStopTracker(patience=2, epochs=10, batch_size=8,
                               lr=1e-3, weight_decay=0.0)
    model = _Tiny()
    it = _NoopIterator()

    tracker.record(model, epoch_num=1, train_loss=1.0, val_loss=1.0,
                   val_auroc=0.70, iterator=it)
    # Manually corrupt the live weights to simulate a worse later epoch.
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()

    tracker.finalize(model, optimizer="Adam")

    # Weights restored to the best-epoch snapshot.
    assert not torch.allclose(model.fc.weight, torch.zeros_like(model.fc.weight))

    info = getattr(model, "_training_info", {})
    assert info["best_epoch"] == 1
    assert info["optimizer"] == "Adam"
    assert info["best_metric_value"] == pytest.approx(0.70)


def test_build_loaders_and_evaluate_val_round_trip():
    X = np.random.RandomState(0).randn(32, 4).astype(np.float32)
    y = (np.random.RandomState(1).rand(32) > 0.5).astype(np.int64)

    train_loader, val_loader, target_loader = _build_loaders(
        X[:16], y[:16], X[16:24], y[16:24], X[24:], batch_size=8,
    )
    assert target_loader is not None

    val_loss, val_auroc = _evaluate_val(_Tiny(), val_loader, device="cpu")
    assert isinstance(val_loss, float)
    assert 0.0 <= val_auroc <= 1.0


def test_infinite_iterator_loops_indefinitely():
    src = [1, 2, 3]
    it = _infinite_iterator(src)
    seen = [next(it) for _ in range(7)]
    assert seen == [1, 2, 3, 1, 2, 3, 1]
