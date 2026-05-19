"""Shared utilities for the model wrappers in this package."""

import inspect

FIXED_BATCH_SIZE = 16


def attach_training_metadata(model, **updates):
    info = dict(getattr(model, "_training_info", {}))
    for key, value in updates.items():
        if value is None and key in info:
            continue
        info[key] = value
    setattr(model, "_training_info", info)
    return model


def _drop_optuna_helper_params(kwargs):
    return {
        key: value
        for key, value in kwargs.items()
        if not (key.endswith("_is_zero") or key.endswith("_log2") or key.endswith("_log10"))
    }


def _filter_supported_kwargs(callable_obj, kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)

    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


# --- Baseline Models ---
