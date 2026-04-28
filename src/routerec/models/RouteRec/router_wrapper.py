"""Primitive and wrapper router modules for RouteRec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..RouteRecBase.routers import Router


_VALID_ROUTER_SOURCE = {"hidden", "feature", "both"}


def _normalize_top_k(top_k: Optional[int], n_slots: int) -> Optional[int]:
    if top_k is None:
        return None
    k = int(top_k)
    if k <= 0:
        return None
    k = min(k, int(n_slots))
    return None if k >= int(n_slots) else k


def _mask_logits_with_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    active_top_k = _normalize_top_k(top_k, logits.shape[-1])
    if active_top_k is None:
        return logits
    vals, idx = logits.topk(active_top_k, dim=-1)
    masked = torch.full_like(logits, -1e9)
    masked.scatter_(-1, idx, vals)
    return masked


def _safe_temperature(value: float) -> float:
    try:
        return max(float(value), 1e-6)
    except Exception:
        return 1.0


@dataclass
class PrimitiveRoutingSpec:
    source: str = "both"
    temperature: float = 1.0
    top_k: Optional[int] = None

    def normalized_source(self) -> str:
        source = str(self.source or "both").lower().strip()
        if source not in _VALID_ROUTER_SOURCE:
            raise ValueError(f"Unsupported primitive source: {self.source}")
        return source


def _primitive_payload(logits: torch.Tensor, spec: PrimitiveRoutingSpec) -> Dict[str, torch.Tensor | float | Optional[int] | str]:
    temperature = _safe_temperature(spec.temperature)
    scaled_logits = logits / temperature
    scaled_logits = _mask_logits_with_top_k(scaled_logits, spec.top_k)
    probs = F.softmax(scaled_logits, dim=-1)
    return {
        "logits": logits,
        "scaled_logits": scaled_logits,
        "probs": probs,
        "source": spec.normalized_source(),
        "temperature": temperature,
        "top_k": _normalize_top_k(spec.top_k, logits.shape[-1]),
    }


class StageJointExpertRouter(nn.Module):
    """A_JOINT: stage input -> E logits."""

    def __init__(self, *, d_in: int, n_experts: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = Router(
            d_in=int(d_in),
            n_experts=int(n_experts),
            d_hidden=int(d_hidden),
            top_k=None,
            dropout=float(dropout),
        ).net

    def forward(self, stage_input: torch.Tensor, spec: PrimitiveRoutingSpec) -> Dict[str, torch.Tensor | float | Optional[int] | str]:
        return _primitive_payload(self.net(stage_input), spec)


class StageGroupRouter(nn.Module):
    """B_GROUP: stage input -> G logits."""

    def __init__(self, *, d_in: int, n_groups: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = Router(
            d_in=int(d_in),
            n_experts=int(n_groups),
            d_hidden=int(d_hidden),
            top_k=None,
            dropout=float(dropout),
        ).net

    def forward(self, stage_input: torch.Tensor, spec: PrimitiveRoutingSpec) -> Dict[str, torch.Tensor | float | Optional[int] | str]:
        return _primitive_payload(self.net(stage_input), spec)


class StageSharedIntraRouter(nn.Module):
    """C_SHARED: stage input -> C logits (shared clone prior)."""

    def __init__(self, *, d_in: int, n_experts_per_group: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = Router(
            d_in=int(d_in),
            n_experts=int(n_experts_per_group),
            d_hidden=int(d_hidden),
            top_k=None,
            dropout=float(dropout),
        ).net

    def forward(self, stage_input: torch.Tensor, spec: PrimitiveRoutingSpec) -> Dict[str, torch.Tensor | float | Optional[int] | str]:
        return _primitive_payload(self.net(stage_input), spec)


class GroupConditionalIntraRouter(nn.Module):
    """D_COND: group inputs -> per-group C logits."""

    def __init__(self, *, d_in: int, n_groups: int, n_experts_per_group: int, d_hidden: int, dropout: float):
        super().__init__()
        self.n_groups = int(n_groups)
        self.group_heads = nn.ModuleList(
            [
                Router(
                    d_in=int(d_in),
                    n_experts=int(n_experts_per_group),
                    d_hidden=int(d_hidden),
                    top_k=None,
                    dropout=float(dropout),
                ).net
                for _ in range(self.n_groups)
            ]
        )

    def forward(self, group_inputs: torch.Tensor, spec: PrimitiveRoutingSpec) -> Dict[str, torch.Tensor | float | Optional[int] | str]:
        # group_inputs: (..., G, D)
        logits = []
        for grp_idx in range(self.n_groups):
            logits.append(self.group_heads[grp_idx](group_inputs[..., grp_idx, :]).unsqueeze(-2))
        group_logits = torch.cat(logits, dim=-2)
        return _primitive_payload(group_logits, spec)


class GroupScalarRouter(nn.Module):
    """E_SCALAR: group inputs -> one scalar per group."""

    def __init__(self, *, d_in: int, n_groups: int):
        super().__init__()
        self.n_groups = int(n_groups)
        self.group_heads = nn.ModuleList([nn.Linear(int(d_in), 1) for _ in range(self.n_groups)])

    def forward(self, group_inputs: torch.Tensor, spec: PrimitiveRoutingSpec) -> Dict[str, torch.Tensor | float | Optional[int] | str]:
        # group_inputs: (..., G, D)
        logits = []
        for grp_idx in range(self.n_groups):
            logits.append(self.group_heads[grp_idx](group_inputs[..., grp_idx, :]))
        group_logits = torch.cat(logits, dim=-1)
        return _primitive_payload(group_logits, spec)


def _group_probs_from_flat(flat_probs: torch.Tensor, *, n_groups: int, n_experts_per_group: int) -> torch.Tensor:
    return flat_probs.reshape(*flat_probs.shape[:-1], int(n_groups), int(n_experts_per_group)).sum(dim=-1)


def _intra_probs_from_flat(flat_probs: torch.Tensor, *, n_groups: int, n_experts_per_group: int) -> torch.Tensor:
    probs = flat_probs.reshape(*flat_probs.shape[:-1], int(n_groups), int(n_experts_per_group))
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs / denom


def _joint_prob_to_logit(joint: torch.Tensor) -> torch.Tensor:
    # Preserve explicit zero-probability slots from primitive top-k masking.
    return torch.where(
        joint > 0,
        torch.log(joint.clamp(min=1e-12)),
        torch.full_like(joint, -1e9),
    )


class FlatJointWrapper(nn.Module):
    alias = "w1_flat"
    name = "FlatJointWrapper"

    def forward(
        self,
        *,
        primitives: Dict[str, Dict[str, torch.Tensor | float | Optional[int] | str]],
        stage_temperature: float,
        n_groups: int,
        n_experts_per_group: int,
        params: Dict[str, object],
    ) -> Dict[str, object]:
        del params
        a = primitives["a_joint"]
        scaled_logits = a["scaled_logits"]
        flat_probs = a["probs"]
        return {
            "scaled_logits": scaled_logits,
            "raw_logits": scaled_logits * _safe_temperature(stage_temperature),
            "group_probs": _group_probs_from_flat(flat_probs, n_groups=n_groups, n_experts_per_group=n_experts_per_group),
            "intra_probs": _intra_probs_from_flat(flat_probs, n_groups=n_groups, n_experts_per_group=n_experts_per_group),
            "group_logits": None,
            "name": self.name,
            "alias": self.alias,
        }


class FlatPlusGroupIntraResidualWrapper(nn.Module):
    alias = "w2_a_plus_d"
    name = "FlatPlusGroupIntraResidualWrapper"

    def forward(
        self,
        *,
        primitives: Dict[str, Dict[str, torch.Tensor | float | Optional[int] | str]],
        stage_temperature: float,
        n_groups: int,
        n_experts_per_group: int,
        params: Dict[str, object],
    ) -> Dict[str, object]:
        alpha_d = float(params.get("alpha_d", 1.0))
        a_scaled = primitives["a_joint"]["scaled_logits"]
        d_scaled = primitives["d_cond"]["scaled_logits"].reshape(*a_scaled.shape[:-1], int(n_groups) * int(n_experts_per_group))
        scaled_logits = a_scaled + alpha_d * d_scaled
        probs = F.softmax(scaled_logits, dim=-1)
        return {
            "scaled_logits": scaled_logits,
            "raw_logits": scaled_logits * _safe_temperature(stage_temperature),
            "group_probs": _group_probs_from_flat(probs, n_groups=n_groups, n_experts_per_group=n_experts_per_group),
            "intra_probs": primitives["d_cond"]["probs"],
            "group_logits": None,
            "name": self.name,
            "alias": self.alias,
        }


class GroupSharedIntraProductWrapper(nn.Module):
    alias = "w3_bxc"
    name = "GroupSharedIntraProductWrapper"

    def forward(
        self,
        *,
        primitives: Dict[str, Dict[str, torch.Tensor | float | Optional[int] | str]],
        stage_temperature: float,
        n_groups: int,
        n_experts_per_group: int,
        params: Dict[str, object],
    ) -> Dict[str, object]:
        del params
        b_probs = primitives["b_group"]["probs"]
        c_probs = primitives["c_shared"]["probs"]
        joint = b_probs.unsqueeze(-1) * c_probs.unsqueeze(-2)
        scaled_logits = _joint_prob_to_logit(joint).reshape(*joint.shape[:-2], int(n_groups) * int(n_experts_per_group))
        return {
            "scaled_logits": scaled_logits,
            "raw_logits": scaled_logits * _safe_temperature(stage_temperature),
            "group_probs": b_probs,
            "intra_probs": c_probs.unsqueeze(-2).expand(*b_probs.shape[:-1], int(n_groups), int(n_experts_per_group)),
            "group_logits": primitives["b_group"]["scaled_logits"],
            "name": self.name,
            "alias": self.alias,
        }


class GroupConditionalProductWrapper(nn.Module):
    alias = "w4_bxd"
    name = "GroupConditionalProductWrapper"

    def forward(
        self,
        *,
        primitives: Dict[str, Dict[str, torch.Tensor | float | Optional[int] | str]],
        stage_temperature: float,
        n_groups: int,
        n_experts_per_group: int,
        params: Dict[str, object],
    ) -> Dict[str, object]:
        del params
        b_probs = primitives["b_group"]["probs"]
        d_probs = primitives["d_cond"]["probs"]
        joint = b_probs.unsqueeze(-1) * d_probs
        scaled_logits = _joint_prob_to_logit(joint).reshape(*joint.shape[:-2], int(n_groups) * int(n_experts_per_group))
        return {
            "scaled_logits": scaled_logits,
            "raw_logits": scaled_logits * _safe_temperature(stage_temperature),
            "group_probs": b_probs,
            "intra_probs": d_probs,
            "group_logits": primitives["b_group"]["scaled_logits"],
            "name": self.name,
            "alias": self.alias,
        }


class ScalarGroupConditionalProductWrapper(nn.Module):
    alias = "w5_exd"
    name = "ScalarGroupConditionalProductWrapper"

    def forward(
        self,
        *,
        primitives: Dict[str, Dict[str, torch.Tensor | float | Optional[int] | str]],
        stage_temperature: float,
        n_groups: int,
        n_experts_per_group: int,
        params: Dict[str, object],
    ) -> Dict[str, object]:
        del params
        e_probs = primitives["e_scalar"]["probs"]
        d_probs = primitives["d_cond"]["probs"]
        joint = e_probs.unsqueeze(-1) * d_probs
        scaled_logits = _joint_prob_to_logit(joint).reshape(*joint.shape[:-2], int(n_groups) * int(n_experts_per_group))
        return {
            "scaled_logits": scaled_logits,
            "raw_logits": scaled_logits * _safe_temperature(stage_temperature),
            "group_probs": e_probs,
            "intra_probs": d_probs,
            "group_logits": primitives["e_scalar"]["scaled_logits"],
            "name": self.name,
            "alias": self.alias,
        }


class GroupConditionalResidualJointWrapper(nn.Module):
    alias = "w6_bxd_plus_a"
    name = "GroupConditionalResidualJointWrapper"

    def forward(
        self,
        *,
        primitives: Dict[str, Dict[str, torch.Tensor | float | Optional[int] | str]],
        stage_temperature: float,
        n_groups: int,
        n_experts_per_group: int,
        params: Dict[str, object],
    ) -> Dict[str, object]:
        alpha_struct = float(params.get("alpha_struct", 1.0))
        alpha_a = float(params.get("alpha_a", 1.0))
        b_probs = primitives["b_group"]["probs"]
        d_probs = primitives["d_cond"]["probs"]
        z_a = primitives["a_joint"]["scaled_logits"]
        joint = b_probs.unsqueeze(-1) * d_probs
        z_struct = _joint_prob_to_logit(joint).reshape(*joint.shape[:-2], int(n_groups) * int(n_experts_per_group))
        scaled_logits = alpha_struct * z_struct + alpha_a * z_a
        probs = F.softmax(scaled_logits, dim=-1)
        return {
            "scaled_logits": scaled_logits,
            "raw_logits": scaled_logits * _safe_temperature(stage_temperature),
            "group_probs": _group_probs_from_flat(probs, n_groups=n_groups, n_experts_per_group=n_experts_per_group),
            "intra_probs": d_probs,
            "group_logits": primitives["b_group"]["scaled_logits"],
            "name": self.name,
            "alias": self.alias,
            "wrapper_internal": {
                "alpha_struct": alpha_struct,
                "alpha_a": alpha_a,
            },
        }


_WRAPPER_ALIASES = {
    "w1_flat": "w1_flat",
    "flat_joint_wrapper": "w1_flat",
    "flat_joint": "w1_flat",
    "w2_a_plus_d": "w2_a_plus_d",
    "flat_plus_group_intra_residual_wrapper": "w2_a_plus_d",
    "w3_bxc": "w3_bxc",
    "group_shared_intra_product_wrapper": "w3_bxc",
    "w4_bxd": "w4_bxd",
    "group_conditional_product_wrapper": "w4_bxd",
    "w5_exd": "w5_exd",
    "scalar_group_conditional_product_wrapper": "w5_exd",
    "w6_bxd_plus_a": "w6_bxd_plus_a",
    "group_conditional_residual_joint_wrapper": "w6_bxd_plus_a",
}

_WRAPPER_REQUIRED_PRIMITIVES = {
    "w1_flat": ("a_joint",),
    "w2_a_plus_d": ("a_joint", "d_cond"),
    "w3_bxc": ("b_group", "c_shared"),
    "w4_bxd": ("b_group", "d_cond"),
    "w5_exd": ("e_scalar", "d_cond"),
    "w6_bxd_plus_a": ("a_joint", "b_group", "d_cond"),
}


def normalize_wrapper_name(name: str) -> str:
    key = str(name or "w1_flat").lower().strip()
    if key not in _WRAPPER_ALIASES:
        raise ValueError(f"Unsupported stage_router_wrapper: {name}")
    return _WRAPPER_ALIASES[key]


def build_wrapper_module(name: str) -> nn.Module:
    wrapper = normalize_wrapper_name(name)
    if wrapper == "w1_flat":
        return FlatJointWrapper()
    if wrapper == "w2_a_plus_d":
        return FlatPlusGroupIntraResidualWrapper()
    if wrapper == "w3_bxc":
        return GroupSharedIntraProductWrapper()
    if wrapper == "w4_bxd":
        return GroupConditionalProductWrapper()
    if wrapper == "w5_exd":
        return ScalarGroupConditionalProductWrapper()
    if wrapper == "w6_bxd_plus_a":
        return GroupConditionalResidualJointWrapper()
    raise ValueError(f"Unsupported stage_router_wrapper: {name}")


def required_primitives_for_wrapper(name: str) -> Tuple[str, ...]:
    wrapper = normalize_wrapper_name(name)
    required = _WRAPPER_REQUIRED_PRIMITIVES.get(wrapper)
    if required is None:
        raise ValueError(f"Unsupported stage_router_wrapper: {name}")
    return tuple(required)
