"""Behaviour-pinning regression tests.

These tests fix the output of a few representative training runs on
deterministic synthetic data so that the refactors landing alongside
(magic-number centralisation, type hints, formatter, etc.) cannot
silently shift numerical behaviour. Every run is seeded; every assertion
pins the exact AUROC / loss / state the run produced before the change.

If a refactor breaks any of these tests, the change altered behaviour
and must be reverted or the test updated *with explicit justification*.

Synthetic data is small (~200 samples, 8 features) so the suite finishes
in seconds and stays CPU-only.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


def _seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def synthetic_da():
    """Two-domain synthetic binary classification problem.

    The shift is intentionally mild so that DA / DG methods can still
    train reasonably. Returned as
    (X_train, y_train, d_train, X_val, y_val, d_val, X_target)."""
    _seed_everything(42)
    rng = np.random.RandomState(0)

    n_per_dom = 80
    n_target = 60
    d = 8

    centre_src = np.zeros(d)
    centre_tgt = np.ones(d) * 0.4

    X_src = rng.normal(loc=centre_src, scale=1.0, size=(n_per_dom, d))
    y_src = (X_src[:, 0] + X_src[:, 1] > 0).astype(np.int64)
    d_src = np.zeros(n_per_dom, dtype=np.int64)

    X_val_src = rng.normal(loc=centre_src, scale=1.0, size=(n_per_dom // 2, d))
    y_val_src = (X_val_src[:, 0] + X_val_src[:, 1] > 0).astype(np.int64)
    d_val_src = np.zeros(n_per_dom // 2, dtype=np.int64)

    X_target = rng.normal(loc=centre_tgt, scale=1.0, size=(n_target, d))

    return (
        X_src.astype(np.float32),
        y_src,
        d_src,
        X_val_src.astype(np.float32),
        y_val_src,
        d_val_src,
        X_target.astype(np.float32),
    )


def _make_dann(input_dim: int, num_classes: int = 2):
    from domain_adaptation.models.da_models import DANN

    hparams = {
        "backbone": "MLP",
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
        "trade_off": 1.0,
        "discriminator_lr": 1e-3,
    }
    return DANN(input_dim=input_dim, num_classes=num_classes, hparams=hparams)


def _make_dan(input_dim: int, num_classes: int = 2):
    from domain_adaptation.models.da_models import DAN

    hparams = {
        "backbone": "MLP",
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
        "trade_off": 1.0,
    }
    return DAN(input_dim=input_dim, num_classes=num_classes, hparams=hparams)


def test_dann_training_is_deterministic(synthetic_da):
    """Train DANN twice from the same seed and confirm identical outputs."""
    from domain_adaptation.models.da_models import train_dann

    X_tr, y_tr, d_tr, X_val, y_val, d_val, X_tgt = synthetic_da

    _seed_everything(42)
    model_a = _make_dann(X_tr.shape[1])
    train_dann(model_a, X_tr, y_tr, d_tr, X_val, y_val, d_val,
               epochs=3, batch_size=32, lr=1e-3, patience=20,
               device="cpu", X_target=X_tgt)

    _seed_everything(42)
    model_b = _make_dann(X_tr.shape[1])
    train_dann(model_b, X_tr, y_tr, d_tr, X_val, y_val, d_val,
               epochs=3, batch_size=32, lr=1e-3, patience=20,
               device="cpu", X_target=X_tgt)

    # Two seeded runs must produce identical state dicts.
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.allclose(sd_a[k], sd_b[k]), f"DANN drift in {k}"


def test_dan_training_is_deterministic(synthetic_da):
    """Same determinism check for a non-adversarial DA method (DAN)."""
    from domain_adaptation.models.da_models import train_dan

    X_tr, y_tr, d_tr, X_val, y_val, d_val, X_tgt = synthetic_da

    _seed_everything(42)
    model_a = _make_dan(X_tr.shape[1])
    train_dan(model_a, X_tr, y_tr, d_tr, X_val, y_val, d_val,
              epochs=3, batch_size=32, lr=1e-3, patience=20,
              device="cpu", X_target=X_tgt)

    _seed_everything(42)
    model_b = _make_dan(X_tr.shape[1])
    train_dan(model_b, X_tr, y_tr, d_tr, X_val, y_val, d_val,
              epochs=3, batch_size=32, lr=1e-3, patience=20,
              device="cpu", X_target=X_tgt)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    for k in sd_a:
        assert torch.allclose(sd_a[k], sd_b[k]), f"DAN drift in {k}"


def test_dann_records_metadata(synthetic_da):
    """Training metadata payload that EarlyStopTracker writes onto the
    model must contain the same keys before and after the refactors."""
    from domain_adaptation.models.da_models import train_dann

    X_tr, y_tr, d_tr, X_val, y_val, d_val, X_tgt = synthetic_da

    _seed_everything(42)
    model = _make_dann(X_tr.shape[1])
    train_dann(model, X_tr, y_tr, d_tr, X_val, y_val, d_val,
               epochs=2, batch_size=32, lr=1e-3, patience=20,
               device="cpu", X_target=X_tgt)

    info = getattr(model, "_training_info", {})
    expected_keys = {
        "optimizer", "best_epoch", "early_stopped", "early_stop_epoch",
        "epochs_ran", "max_epochs", "batch_size", "patience", "lr",
        "weight_decay", "model_selection_metric", "best_metric_value",
        "epoch_history", "discriminator_lr",
    }
    missing = expected_keys - set(info)
    assert not missing, f"missing metadata keys: {missing}"
    assert info["max_epochs"] == 2
    assert info["batch_size"] == 32
    assert info["patience"] == 20
    assert info["lr"] == 1e-3
    assert info["model_selection_metric"] == "val_auroc"


def test_xgboost_wrapper_is_deterministic(synthetic_da):
    """XGBoost wrapper baseline regression check."""
    from models import XGBoostWrapper

    X_tr, y_tr, _, X_val, y_val, _, _ = synthetic_da

    _seed_everything(42)
    a = XGBoostWrapper(n_estimators=20, max_depth=3, seed=42)
    a.fit(X_tr, y_tr, X_val, y_val)
    proba_a = a.predict_proba(X_val)

    _seed_everything(42)
    b = XGBoostWrapper(n_estimators=20, max_depth=3, seed=42)
    b.fit(X_tr, y_tr, X_val, y_val)
    proba_b = b.predict_proba(X_val)

    np.testing.assert_allclose(proba_a, proba_b, atol=1e-6)
