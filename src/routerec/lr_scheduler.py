"""Epoch-wise learning-rate schedules with an explicit continuation contract."""

from __future__ import annotations

import math
from typing import Any, Mapping


LR_SCHEDULER_TYPES = frozenset({"constant", "warmup_cosine"})


class LearningRateSchedulerError(ValueError):
    """Raised when an LR schedule is incomplete or cannot be restored safely."""


def resolve_schedule_total_epochs(config: Mapping[str, Any]) -> int:
    """Resolve the single horizon shared by model and learning-rate schedules."""

    explicit = (
        config["routerec_schedule_total_epochs"]
        if "routerec_schedule_total_epochs" in config
        else None
    )
    fallback = config["epochs"] if "epochs" in config else None
    total_epochs = int(explicit or fallback or 0)
    if total_epochs < 1:
        raise LearningRateSchedulerError("schedule requires a positive total epoch horizon")
    return total_epochs


def warmup_cosine_multiplier(
    epoch_index: int,
    *,
    total_epochs: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> float:
    """Return the exact multiplier used by the historical RouteRec trainer.

    ``epoch_index`` is zero based.  The historical implementation always used
    at least one warmup epoch, so ``warmup_ratio=0`` starts epoch zero at the
    base LR and then performs cosine decay over the remaining epochs.
    """

    total = int(total_epochs)
    if total < 1:
        raise LearningRateSchedulerError("total_epochs must be positive")
    warmup = float(warmup_ratio)
    minimum = float(min_lr_ratio)
    if not 0.0 <= warmup <= 1.0:
        raise LearningRateSchedulerError("warmup_ratio must be in [0, 1]")
    if not 0.0 <= minimum <= 1.0:
        raise LearningRateSchedulerError("min_lr_ratio must be in [0, 1]")
    warmup_epochs = max(int(round(total * warmup)), 1)
    step = int(epoch_index) + 1
    if step <= warmup_epochs:
        return max(step / float(warmup_epochs), 1e-8)
    remaining = max(total - warmup_epochs, 1)
    progress = min(max((step - warmup_epochs) / float(remaining), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (1.0 - minimum) * cosine


def scheduler_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the immutable scientific LR-schedule contract."""

    schedule_type = str(config.get("lr_scheduler_type", "constant")).strip().lower()
    if schedule_type not in LR_SCHEDULER_TYPES:
        raise LearningRateSchedulerError(
            f"unsupported lr_scheduler_type {schedule_type!r}; expected {sorted(LR_SCHEDULER_TYPES)}"
        )
    total_epochs = resolve_schedule_total_epochs(config)
    if schedule_type == "constant":
        return {
            "type": "constant",
            "step_unit": "epoch",
            "total_epochs": total_epochs,
            "warmup_ratio": 0.0,
            "min_lr_ratio": 1.0,
        }
    warmup_ratio = float(config.get("lr_scheduler_warmup_ratio", 0.0))
    min_lr_ratio = float(config.get("lr_scheduler_min_lr_ratio", 0.1))
    # Validate through the same function that will drive LambdaLR.
    warmup_cosine_multiplier(
        0,
        total_epochs=total_epochs,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    return {
        "type": "warmup_cosine",
        "step_unit": "epoch",
        "total_epochs": total_epochs,
        "warmup_ratio": warmup_ratio,
        "min_lr_ratio": min_lr_ratio,
    }


def build_lr_scheduler(optimizer: Any, config: Mapping[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    """Build the configured scheduler and return it with its strict contract."""

    import torch

    spec = scheduler_spec(config)
    if spec["type"] == "constant":
        return None, spec

    def multiplier(epoch_index: int) -> float:
        return warmup_cosine_multiplier(
            epoch_index,
            total_epochs=int(spec["total_epochs"]),
            warmup_ratio=float(spec["warmup_ratio"]),
            min_lr_ratio=float(spec["min_lr_ratio"]),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    return scheduler, spec


def current_learning_rates(optimizer: Any) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


__all__ = [
    "LR_SCHEDULER_TYPES",
    "LearningRateSchedulerError",
    "build_lr_scheduler",
    "current_learning_rates",
    "resolve_schedule_total_epochs",
    "scheduler_spec",
    "warmup_cosine_multiplier",
]
