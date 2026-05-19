"""Test-suite configuration.

Adds ``basemodel-benchmarking/`` and the repo root to :data:`sys.path`
so that the module-level ``from models import ...`` and
``from domain_adaptation.models.da_models import ...`` imports used by
the benchmark codebase resolve under pytest without any additional setup.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BMB = _ROOT / "basemodel-benchmarking"

for _p in (str(_BMB), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
