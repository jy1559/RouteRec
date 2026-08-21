"""Training schedule controller for RouteRec."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScheduleState:
    alpha_scale: float
    mid_router_temperature: float
    micro_router_temperature: float
    stage_top_k: Optional[int]


class ScheduleController:
    """Resolve runtime alpha/temp/top-k schedule states by epoch."""

    def __init__(
        self,
        *,
        enable: bool,
        alpha_warmup_until,
        alpha_warmup_start: float,
        alpha_warmup_end: float,
        temperature_warmup_until,
        mid_router_temperature_start: float,
        mid_router_temperature_end: float,
        micro_router_temperature_start: float,
        micro_router_temperature_end: float,
        moe_top_k_policy: str,
        moe_top_k_fixed: Optional[int],
        moe_top_k_ratio: float,
        moe_top_k_min: int,
        moe_top_k_start: int,
        moe_top_k_warmup_until,
    ):
        self.enable = bool(enable)

        self.alpha_warmup_until = alpha_warmup_until
        self.alpha_warmup_start = float(alpha_warmup_start)
        self.alpha_warmup_end = float(alpha_warmup_end)

        self.temperature_warmup_until = temperature_warmup_until
        self.mid_temp_start = float(mid_router_temperature_start)
        self.mid_temp_end = float(mid_router_temperature_end)
        self.micro_temp_start = float(micro_router_temperature_start)
        self.micro_temp_end = float(micro_router_temperature_end)

        policy = str(moe_top_k_policy).lower().strip()
        if policy not in {"auto", "fixed", "half", "ratio", "dense"}:
            raise ValueError(
                "moe_top_k_policy must be one of ['auto','fixed','half','ratio','dense'], "
                f"got {moe_top_k_policy}"
            )
        self.top_k_policy = policy
        self.top_k_fixed = moe_top_k_fixed
        self.top_k_ratio = float(moe_top_k_ratio)
        self.top_k_min = max(int(moe_top_k_min), 1)
        self.top_k_start = int(moe_top_k_start)
        self.top_k_warmup_until = moe_top_k_warmup_until

    @staticmethod
    def _resolve_warmup_end_epoch(warmup_until, total_epochs: int) -> int:
        if warmup_until is None:
            return 0
        try:
            v = float(warmup_until)
        except (TypeError, ValueError):
            return 0
        if v <= 0:
            return 0
        if 0 < v <= 1:
            return max(1, int(math.ceil(float(total_epochs) * v)))
        return max(1, int(round(v)))

    @staticmethod
    def _epoch_progress(epoch_idx: int, end_epoch: int) -> float:
        if end_epoch <= 1:
            return 1.0
        e1 = max(int(epoch_idx) + 1, 1)
        return min(max((float(e1) - 1.0) / float(end_epoch - 1), 0.0), 1.0)

    @classmethod
    def _linear_warmup(cls, epoch_idx: int, end_epoch: int, start: float, end: float) -> float:
        if end_epoch <= 0:
            return float(end)
        p = cls._epoch_progress(epoch_idx=epoch_idx, end_epoch=end_epoch)
        return float(start + (end - start) * p)

    @staticmethod
    def _normalize_top_k(top_k: Optional[int], n_experts: int) -> Optional[int]:
        if top_k is None or n_experts <= 0:
            return None
        k = int(top_k)
        if k <= 0:
            return None
        k = min(k, int(n_experts))
        return None if k >= n_experts else k

    def _resolve_top_k_target(self, n_experts: int) -> Optional[int]:
        if n_experts <= 0:
            return None
        if self.top_k_policy == "dense":
            return None

        if self.top_k_policy == "half":
            target = int(math.ceil(0.5 * float(n_experts)))
        elif self.top_k_policy == "ratio":
            ratio = min(max(self.top_k_ratio, 0.0), 1.0)
            if ratio <= 0.0:
                return None
            target = int(math.ceil(ratio * float(n_experts)))
        elif self.top_k_policy == "fixed":
            if self.top_k_fixed is None:
                return None
            target = int(self.top_k_fixed)
        else:  # auto
            if self.top_k_fixed is None:
                return None
            ratio_target = int(math.ceil(min(max(self.top_k_ratio, 0.0), 1.0) * float(n_experts)))
            target = max(int(self.top_k_fixed), ratio_target)

        target = max(self.top_k_min, int(target))
        return self._normalize_top_k(target, n_experts=n_experts)

    def _scheduled_top_k(self, epoch_idx: int, total_epochs: int, n_experts: int) -> Optional[int]:
        target_top_k = self._resolve_top_k_target(n_experts=n_experts)
        if target_top_k is None:
            return None
        if not self.enable:
            return target_top_k

        end_epoch = self._resolve_warmup_end_epoch(self.top_k_warmup_until, total_epochs)
        if end_epoch <= 0:
            return target_top_k

        start_top_k = self._normalize_top_k(self.top_k_start, n_experts=n_experts)
        start_k = n_experts if start_top_k is None else int(start_top_k)
        target_k = int(target_top_k)
        if start_k == target_k:
            return target_top_k

        p = self._epoch_progress(epoch_idx=epoch_idx, end_epoch=end_epoch)
        interpolated = int(round(start_k + (target_k - start_k) * p))
        lo = min(start_k, target_k)
        hi = max(start_k, target_k)
        interpolated = min(max(interpolated, lo), hi)
        return self._normalize_top_k(interpolated, n_experts=n_experts)

    def resolve(self, *, epoch_idx: int, total_epochs: int, n_experts: int) -> ScheduleState:
        if not self.enable:
            return ScheduleState(
                alpha_scale=1.0,
                mid_router_temperature=self.mid_temp_end,
                micro_router_temperature=self.micro_temp_end,
                stage_top_k=self._resolve_top_k_target(n_experts=n_experts),
            )

        alpha_end_epoch = self._resolve_warmup_end_epoch(self.alpha_warmup_until, total_epochs)
        temp_end_epoch = self._resolve_warmup_end_epoch(self.temperature_warmup_until, total_epochs)

        alpha_scale = self._linear_warmup(
            epoch_idx=epoch_idx,
            end_epoch=alpha_end_epoch,
            start=self.alpha_warmup_start,
            end=self.alpha_warmup_end,
        )
        mid_temp = self._linear_warmup(
            epoch_idx=epoch_idx,
            end_epoch=temp_end_epoch,
            start=self.mid_temp_start,
            end=self.mid_temp_end,
        )
        micro_temp = self._linear_warmup(
            epoch_idx=epoch_idx,
            end_epoch=temp_end_epoch,
            start=self.micro_temp_start,
            end=self.micro_temp_end,
        )
        top_k = self._scheduled_top_k(
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
            n_experts=n_experts,
        )

        return ScheduleState(
            alpha_scale=alpha_scale,
            mid_router_temperature=mid_temp,
            micro_router_temperature=micro_temp,
            stage_top_k=top_k,
        )
