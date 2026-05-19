"""Backward-compatible shim.

The DA algorithm implementations have moved into the ``da/`` subpackage,
one file per algorithm. This module is preserved so that existing
imports of the form::

    from domain_adaptation.models.da_models import DANN, train_dann

continue to work unchanged.
"""

from .da import (  # noqa: F401
    ADDA,
    CBST,
    CDAN,
    CGDM,
    DAN,
    DANN,
    JAN,
    MCC,
    MCD,
    SHOT,
    DeepCORAL,
    MCDInferenceWrapper,
    train_adda,
    train_cbst,
    train_cdan,
    train_cgdm,
    train_dan,
    train_dann,
    train_deepcoral,
    train_jan,
    train_mcc,
    train_mcd,
    train_shot,
)
