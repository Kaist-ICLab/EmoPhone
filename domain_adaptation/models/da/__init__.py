"""Domain adaptation algorithms.

Each module here implements one DA algorithm: the model class (subclassing
:class:`DAModel`) and its ``train_*`` function. The package ``__init__``
re-exports every public symbol so that callers can continue to use::

    from domain_adaptation.models.da_models import DANN, train_dann

without caring how the file is laid out.
"""

from .adda import ADDA, train_adda
from .cbst import CBST, train_cbst
from .cdan import CDAN, train_cdan
from .cgdm import CGDM, train_cgdm
from .dan import DAN, train_dan
from .dann import DANN, train_dann
from .deepcoral import DeepCORAL, train_deepcoral
from .jan import JAN, train_jan
from .mcc import MCC, train_mcc
from .mcd import MCD, MCDInferenceWrapper, train_mcd
from .shot import SHOT, train_shot

__all__ = [
    "ADDA",
    "CBST",
    "CDAN",
    "CGDM",
    "DAN",
    "DANN",
    "DeepCORAL",
    "JAN",
    "MCC",
    "MCD",
    "MCDInferenceWrapper",
    "SHOT",
    "train_adda",
    "train_cbst",
    "train_cdan",
    "train_cgdm",
    "train_dan",
    "train_dann",
    "train_deepcoral",
    "train_jan",
    "train_mcc",
    "train_mcd",
    "train_shot",
]
