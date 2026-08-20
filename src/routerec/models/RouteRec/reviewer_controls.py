"""Model-side controls for the compact camera-ready reviewer panel."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


CONTROL_MODES = ("full", "hidden_only_matched", "random_frozen")


def normalize_control_mode(value: object) -> str:
    mode = str(value or "full").strip().lower()
    if mode not in CONTROL_MODES:
        raise ValueError(f"unsupported reviewer_control_mode={value!r}; expected {CONTROL_MODES}")
    return mode


def controlled_feature_input(value: torch.Tensor, *, mode: str) -> torch.Tensor:
    normalized = normalize_control_mode(mode)
    if normalized == "hidden_only_matched":
        return torch.zeros_like(value)
    if normalized == "random_frozen":
        return value.detach()
    return value


def controlled_hidden_input(value: torch.Tensor, *, mode: str) -> torch.Tensor:
    normalized = normalize_control_mode(mode)
    return value.detach() if normalized == "random_frozen" else value


def freeze_random_router_path(modules: Iterable[nn.Module | None]) -> None:
    """Freeze only the routing path and make it deterministic under train()."""

    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        for child in module.modules():
            if isinstance(child, nn.Dropout):
                child.p = 0.0
