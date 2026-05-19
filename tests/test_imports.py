"""Smoke test for the public benchmark surface.

Guards the per-algorithm split refactor: if a class or train_* function
goes missing or a circular import sneaks in, this test fails before
anything else does. Heavyweight model construction is *not* exercised
here -- :mod:`tests.test_helpers` covers that for the cheap pieces.
"""

import pytest


def test_baseline_wrappers_importable():
    from models import (  # noqa: F401
        DeepCTRWrapper,
        LightGBMWrapper,
        MLP,
        ResNet,
        TabNetWrapper,
        WidedeepWrapper,
        XGBoostWrapper,
        attach_training_metadata,
        evaluate_model,
        train_torch_model,
    )


def test_da_public_surface_via_shim():
    """`da_models` is the historical import path; the per-algo split must
    preserve every symbol it used to export."""
    from domain_adaptation.models.da_models import (  # noqa: F401
        ADDA,
        CBST,
        CDAN,
        CGDM,
        DAN,
        DANN,
        DeepCORAL,
        JAN,
        MCC,
        MCD,
        MCDInferenceWrapper,
        SHOT,
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


def test_dg_public_surface_via_shim():
    """Same guarantee for the DomainBed-style DG algorithms."""
    from domain_adaptation.models.domainbed_algos import (  # noqa: F401
        CSD,
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


@pytest.mark.parametrize(
    "modpath, name",
    [
        ("domain_adaptation.models.da.dann", "DANN"),
        ("domain_adaptation.models.da.cdan", "CDAN"),
        ("domain_adaptation.models.da.mcc", "MCC"),
        ("domain_adaptation.models.da.dan", "DAN"),
        ("domain_adaptation.models.da.deepcoral", "DeepCORAL"),
        ("domain_adaptation.models.da.adda", "ADDA"),
        ("domain_adaptation.models.da.mcd", "MCD"),
        ("domain_adaptation.models.da.jan", "JAN"),
        ("domain_adaptation.models.da.shot", "SHOT"),
        ("domain_adaptation.models.da.cbst", "CBST"),
        ("domain_adaptation.models.da.cgdm", "CGDM"),
        ("domain_adaptation.models.dg.erm", "ERM"),
        ("domain_adaptation.models.dg.irm", "IRM"),
        ("domain_adaptation.models.dg.vrex", "VREx"),
        ("domain_adaptation.models.dg.gdro", "GroupDRO"),
        ("domain_adaptation.models.dg.mixstyle", "MixStyle"),
        ("domain_adaptation.models.dg.mldg", "MLDG"),
        ("domain_adaptation.models.dg.fish", "Fish"),
        ("domain_adaptation.models.dg.sagnet", "SagNet"),
        ("domain_adaptation.models.dg.csd", "CSD"),
        ("domain_adaptation.models.dg.masf", "MASF"),
    ],
)
def test_per_algorithm_file_carries_its_class(modpath, name):
    """Every algorithm has its own file and the class lives there."""
    mod = __import__(modpath, fromlist=[name])
    assert hasattr(mod, name), f"{name} missing from {modpath}"
