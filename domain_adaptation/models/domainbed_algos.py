"""Backward-compatible shim.

The DomainBed-style DG algorithms have moved into the ``dg/`` subpackage,
one file per algorithm. This module is preserved so that existing
imports of the form::

    from domain_adaptation.models.domainbed_algos import IRM, train_dg_model

continue to work unchanged.
"""

from .dg import (  # noqa: F401
    CSD,
    DGModel,
    ERM,
    Fish,
    GroupDRO,
    IRM,
    MASF,
    MLDG,
    MixStyle,
    SagNet,
    VREx,
    train_dg_model,
)
