"""Baseline + tabular-NN wrappers for the EmoPhone benchmark.

The previous monolithic ``models.py`` is now a package, one file per
wrapper. Public API is unchanged: existing callers continue to do
``from models import XGBoostWrapper, MLP, train_torch_model, ...``
and get the same symbols.

Layout::

    models/
        _helpers.py     attach_training_metadata + the kwarg-sanitising
                        helpers shared by every wrapper, plus the
                        FIXED_BATCH_SIZE constant
        baselines.py    MLP, ResNet, train_torch_model, evaluate_model
        xgb.py          XGBoostWrapper
        lgb.py          LightGBMWrapper
        tabnet.py       TabNetWrapper
        widedeep.py     WidedeepWrapper (SAINT/TabTransformer/FTTransformer
                        and their linear-attention monkey patches)
        deepctr.py      DeepCTRWrapper (DCN, AutoInt)
"""

from ._helpers import (
    FIXED_BATCH_SIZE,
    _drop_optuna_helper_params,
    _filter_supported_kwargs,
    attach_training_metadata,
)
from .baselines import MLP, ResNet, evaluate_model, train_torch_model
from .deepctr import DeepCTRWrapper
from .lgb import LightGBMWrapper
from .tabnet import TabNetWrapper
from .widedeep import WidedeepWrapper
from .xgb import XGBoostWrapper

__all__ = [
    "DeepCTRWrapper",
    "FIXED_BATCH_SIZE",
    "LightGBMWrapper",
    "MLP",
    "ResNet",
    "TabNetWrapper",
    "WidedeepWrapper",
    "XGBoostWrapper",
    "attach_training_metadata",
    "evaluate_model",
    "train_torch_model",
]
