"""Runtime helpers for loading RouteRec and bundled baseline models in RecBole."""

from __future__ import annotations

import importlib
from typing import Callable


_PATCHED = False


_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "RouteRec": ("routerec.models.RouteRec", "RouteRec"),
    "routerec": ("routerec.models.RouteRec", "RouteRec"),
    "route_rec": ("routerec.models.RouteRec", "RouteRec"),
    "BSARec": ("routerec.models.bsarec", "BSARec"),
    "DIFSR": ("routerec.models.difsr", "DIFSR"),
    "DuoRec": ("routerec.models.duorec", "DuoRec"),
    "FAME": ("routerec.models.fame", "FAME"),
    "FDSA": ("routerec.models.fdsa", "FDSA"),
    "FEARec": ("routerec.models.fearec", "FEARec"),
    "SIGMA": ("routerec.models.sigma", "SIGMA"),
    "TiSASRec": ("routerec.models.tisasrec", "TiSASRec"),
}


def _resolve_local_model(model_name: str):
    spec = _MODEL_SPECS.get(model_name)
    if spec is None:
        return None
    module_name, class_name = spec
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _apply_numpy_compat_shims() -> None:
    """Restore NumPy aliases expected by RecBole when running on NumPy 2.x."""
    import numpy as np

    alias_map = {
        "bool_": bool,
        "bool8": np.bool_,
        "int_": int,
        "float_": np.float64,
        "complex_": np.complex128,
        "object_": object,
        "str_": str,
        "unicode_": str,
    }
    for alias_name, alias_value in alias_map.items():
        if not hasattr(np, alias_name):
            setattr(np, alias_name, alias_value)


def enable_custom_model_resolver() -> None:
    """Patch RecBole model resolver so bundled models can be loaded by name."""
    global _PATCHED
    if _PATCHED:
        return

    _apply_numpy_compat_shims()

    import recbole.config.configurator as recbole_configurator
    import recbole.data.utils as recbole_data_utils
    import recbole.utils as recbole_utils_package
    import recbole.utils.utils as recbole_utils

    try:
        import recbole.quick_start.quick_start as quick_start_module
    except ModuleNotFoundError:
        quick_start_module = None

    original_get_model: Callable[[str], object] = recbole_utils.get_model

    def patched_get_model(model_name: str):
        local = _resolve_local_model(model_name)
        if local is not None:
            return local
        return original_get_model(model_name)

    recbole_utils.get_model = patched_get_model
    recbole_utils_package.get_model = patched_get_model
    recbole_configurator.get_model = patched_get_model
    recbole_data_utils.get_model = patched_get_model
    if quick_start_module is not None:
        quick_start_module.get_model = patched_get_model
    _PATCHED = True
