"""Domain generalization algorithms.

Each module under this package holds one DomainBed-style algorithm. The
:func:`train_dg_model` runner in :mod:`._train` is shared across every
algorithm. Callers should import the public surface either from here or
from the backwards-compatible :mod:`domain_adaptation.models.domainbed_algos`
shim::

    from domain_adaptation.models.domainbed_algos import IRM, train_dg_model
"""

from ._base import DGModel
from ._train import train_dg_model
from .csd import CSD
from .erm import ERM
from .fish import Fish
from .gdro import GroupDRO
from .irm import IRM
from .masf import MASF
from .mixstyle import MixStyle
from .mldg import MLDG
from .sagnet import SagNet
from .vrex import VREx

__all__ = [
    "CSD",
    "DGModel",
    "ERM",
    "Fish",
    "GroupDRO",
    "IRM",
    "MASF",
    "MixStyle",
    "MLDG",
    "SagNet",
    "VREx",
    "train_dg_model",
]
