"""
Router / Gating modules for the MoE stages.

Supports:
  - Dense softmax gating (all K experts contribute)
  - Top-k sparse gating (only top-k experts contribute; rest zeroed)
  - Rule-soft routing with ratio-bin or GLS-style fixed semantics
  - Unified backend wrapper for learned/rule-based routing
  - Load-balance regularisation loss for collapse prevention
  - Entropy regularisation (optional)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_top_k(top_k: Optional[int], n_experts: int) -> Optional[int]:
    if top_k is None:
        return None
    k = int(top_k)
    if k <= 0:
        return None
    k = min(k, int(n_experts))
    return None if k >= int(n_experts) else k


def _top_k_softmax(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k logits, re-normalise with softmax, zero the rest."""
    topk_vals, topk_idx = logits.topk(k, dim=-1)
    topk_weights = F.softmax(topk_vals, dim=-1)
    weights = torch.zeros_like(logits)
    weights.scatter_(-1, topk_idx, topk_weights)
    return weights


def _softmax_with_top_k(logits: torch.Tensor, n_experts: int, top_k: Optional[int]) -> torch.Tensor:
    active_top_k = _normalize_top_k(top_k, n_experts=n_experts)
    if active_top_k is None:
        return F.softmax(logits, dim=-1)
    return _top_k_softmax(logits, active_top_k)


def _to_ratio(values: torch.Tensor) -> torch.Tensor:
    """Convert values to [0, 1] ratio space."""
    if values.numel() == 0:
        return values

    x = values.float()
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.zeros_like(x)

    xf = x[finite]
    lo = xf.min()
    hi = xf.max()
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        ratio = x.clamp(0.0, 1.0)
        ratio[~finite] = 0.5
        return ratio

    ratio = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    ratio = ratio.clamp(0.0, 1.0)
    ratio[~finite] = 0.5
    return ratio


def _clone_centers(expert_scale: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = [(2 * i + 1) / float(2 * expert_scale) for i in range(expert_scale)]
    return torch.tensor(vals, device=device, dtype=dtype)


def _gaussian_clone_logits(
    score: torch.Tensor,
    *,
    expert_scale: int,
    sigma: float = 0.20,
) -> torch.Tensor:
    centers = _clone_centers(expert_scale, device=score.device, dtype=score.dtype)
    score_exp = score.unsqueeze(-1)
    return -((score_exp - centers) ** 2) / (2.0 * sigma * sigma)


def _compute_ratio_stats(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    ratio = _to_ratio(values)
    mean = ratio.mean(dim=-1, keepdim=True)
    std = ratio.std(dim=-1, unbiased=False, keepdim=True)
    vmax = ratio.max(dim=-1, keepdim=True).values
    vmin = ratio.min(dim=-1, keepdim=True).values
    vrange = vmax - vmin
    peak = vmax - mean
    zero_frac = (ratio <= 0.10).float().mean(dim=-1, keepdim=True)
    low_frac = (ratio <= 0.25).float().mean(dim=-1, keepdim=True)
    mid_frac = ((ratio >= 0.30) & (ratio <= 0.70)).float().mean(dim=-1, keepdim=True)
    high_frac = (ratio >= 0.75).float().mean(dim=-1, keepdim=True)
    spike_frac = (ratio >= 0.90).float().mean(dim=-1, keepdim=True)
    mean_abs_dev = (ratio - mean).abs().mean(dim=-1, keepdim=True)
    return {
        "ratio": ratio,
        "mean": mean,
        "std": std,
        "vmax": vmax,
        "vmin": vmin,
        "vrange": vrange,
        "peak": peak,
        "zero_frac": zero_frac,
        "low_frac": low_frac,
        "mid_frac": mid_frac,
        "high_frac": high_frac,
        "spike_frac": spike_frac,
        "mean_abs_dev": mean_abs_dev,
    }


def _group_local_stat_clone_logits(stats: Dict[str, torch.Tensor], sharpness: float) -> torch.Tensor:
    mean = stats["mean"]
    std = stats["std"]
    vrange = stats["vrange"]
    peak = stats["peak"]
    zero_frac = stats["zero_frac"]
    low_frac = stats["low_frac"]
    mid_frac = stats["mid_frac"]
    high_frac = stats["high_frac"]
    spike_frac = stats["spike_frac"]
    sharp = float(sharpness)
    low = (
        1.1 * zero_frac
        + 0.7 * low_frac
        - sharp * (mean - 0.18).pow(2)
        - 1.1 * std
        - 0.4 * peak
        - 0.3 * high_frac
    )
    mid = (
        0.8 * mid_frac
        - sharp * (mean - 0.50).pow(2)
        - 0.5 * std
        - 0.2 * vrange
        - 0.2 * peak
    )
    high = (
        1.2 * high_frac
        + 0.9 * spike_frac
        + 0.7 * peak
        + 0.5 * std
        + 0.3 * vrange
        - sharp * (mean - 0.78).pow(2)
        - 0.3 * zero_frac
    )
    return torch.cat([low, mid, high], dim=-1)


def _group_local_activity_logit(stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"]
    std = stats["std"]
    vrange = stats["vrange"]
    peak = stats["peak"]
    zero_frac = stats["zero_frac"]
    high_frac = stats["high_frac"]
    spike_frac = stats["spike_frac"]
    return (
        0.9 * mean
        + 0.7 * high_frac
        + 0.4 * spike_frac
        + 0.5 * peak
        + 0.35 * std
        + 0.20 * vrange
        - 0.80 * zero_frac
    )


def build_group_local_stat_logits_12way(
    rule_features: torch.Tensor,
    *,
    feature_groups: Sequence[Sequence[int]],
    expert_scale: int,
    stat_sharpness: float = 16.0,
) -> torch.Tensor:
    if rule_features.size(-1) <= 0:
        raise ValueError("rule_features last dimension must be > 0 for GLS rule routing.")
    if int(expert_scale) <= 0:
        raise ValueError(f"expert_scale must be > 0, got {expert_scale}")

    per_group_logits = []
    for feat_idx in feature_groups:
        idx_list = [int(i) for i in feat_idx if 0 <= int(i) < rule_features.size(-1)]
        if not idx_list:
            idx_list = [0]
        idx = torch.tensor(idx_list, device=rule_features.device, dtype=torch.long)
        group_feat = rule_features.index_select(-1, idx)
        stats = _compute_ratio_stats(group_feat)
        if int(expert_scale) == 3:
            clone_logits = _group_local_stat_clone_logits(stats, sharpness=stat_sharpness)
        else:
            clone_logits = _gaussian_clone_logits(stats["mean"], expert_scale=int(expert_scale))
        per_group_logits.append(_group_local_activity_logit(stats) + clone_logits)
    return torch.cat(per_group_logits, dim=-1)


class Router(nn.Module):
    """Gating network for one MoE stage.

    Parameters
    ----------
    d_in : int
        Dimension of the router input (= concat of stage features + prev stage
        output, when applicable).
    n_experts : int
        Number of experts (K).
    d_hidden : int
        Hidden size of the 2-layer MLP.
    top_k : int or None
        If None -> dense softmax (all experts).
        If int  -> top-k sparse gating (only top-k experts, rest weight=0).
    dropout : float
        Dropout in router MLP.
    """

    def __init__(
        self,
        d_in: int,
        n_experts: int = 4,
        d_hidden: int = 64,
        top_k: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_experts),
        )

    def forward(
        self,
        router_input: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating weights.

        Args:
            router_input: [*, d_in]
            temperature: Softmax temperature for gating logits.
            top_k: Optional runtime override for sparse top-k routing.
                - ``None``: use module default ``self.top_k``.
                - ``<=0``: dense softmax over all experts.
                - ``>0``: sparse top-k softmax.
        Returns:
            weights : [*, K]   - soft gating weights (sum=1 per sample).
            logits  : [*, K]   - raw logits (for auxiliary losses / logging).
        """
        logits = self.net(router_input)
        scale = max(float(temperature), 1e-6)
        scaled_logits = logits / scale

        active_top_k = self.top_k if top_k is None else top_k
        weights = _softmax_with_top_k(scaled_logits, n_experts=self.n_experts, top_k=active_top_k)
        return weights, scaled_logits


class RuleSoftRouter(nn.Module):
    """Rule-based soft router using fixed feature semantics."""

    def __init__(
        self,
        n_experts: int,
        n_stage_features: int,
        selected_feature_indices: Optional[List[List[int]]],
        feature_names: List[str],
        n_bins: int = 5,
        expert_bias: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        variant: str = "ratio_bins",
        group_feature_indices: Optional[List[List[int]]] = None,
        expert_scale: Optional[int] = None,
        stat_sharpness: float = 16.0,
    ):
        super().__init__()
        if n_experts <= 0:
            raise ValueError(f"n_experts must be > 0, got {n_experts}")
        if n_stage_features <= 0:
            raise ValueError(f"n_stage_features must be > 0, got {n_stage_features}")
        if int(n_bins) < 2:
            raise ValueError(f"rule_router.n_bins must be >= 2, got {n_bins}")
        if len(feature_names) != int(n_stage_features):
            raise ValueError(
                f"feature_names length must match n_stage_features, got {len(feature_names)} vs {n_stage_features}"
            )

        self.n_experts = int(n_experts)
        self.n_stage_features = int(n_stage_features)
        self.n_bins = int(n_bins)
        self.top_k = top_k
        self.feature_names = list(feature_names)
        self.variant = str(variant).lower().strip()
        self.stat_sharpness = float(stat_sharpness)
        self.expert_scale = int(expert_scale) if expert_scale is not None else 0
        if self.variant not in {"ratio_bins", "teacher_gls"}:
            raise ValueError(
                f"rule_router.variant must be one of ['ratio_bins','teacher_gls'], got {variant}"
            )

        self.group_feature_indices: List[List[int]] = []
        if self.variant == "teacher_gls":
            if self.expert_scale <= 0:
                raise ValueError("teacher_gls router requires expert_scale > 0.")
            cleaned_groups: List[List[int]] = []
            for idxs in list(group_feature_indices or []):
                valid = [int(i) for i in idxs if 0 <= int(i) < self.n_stage_features]
                cleaned_groups.append(valid or [0])
            if len(cleaned_groups) * self.expert_scale != self.n_experts:
                raise ValueError(
                    "teacher_gls router expects len(group_feature_indices) * expert_scale == n_experts, "
                    f"got {len(cleaned_groups)} * {self.expert_scale} vs {self.n_experts}"
                )
            self.group_feature_indices = cleaned_groups

        if self.variant == "ratio_bins":
            feature_rows = list(selected_feature_indices or [])
            if len(feature_rows) != int(n_experts):
                raise ValueError(
                    "selected_feature_indices length must match n_experts, "
                    f"got {len(feature_rows)} vs {n_experts}"
                )

            max_sel = 1
            for idxs in feature_rows:
                if idxs:
                    max_sel = max(max_sel, len(idxs))

            padded_idx = []
            mask = []
            for idxs in feature_rows:
                valid = [int(i) for i in idxs if 0 <= int(i) < self.n_stage_features]
                if not valid:
                    valid = [0]
                row_idx = list(valid)
                row_mask = [1.0] * len(valid)
                while len(row_idx) < max_sel:
                    row_idx.append(row_idx[-1])
                    row_mask.append(0.0)
                padded_idx.append(row_idx)
                mask.append(row_mask)

            self.register_buffer(
                "selected_idx",
                torch.tensor(padded_idx, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "selected_mask",
                torch.tensor(mask, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "selected_count",
                self.selected_mask.sum(dim=-1).clamp(min=1.0),
                persistent=False,
            )
        else:
            self.register_buffer(
                "selected_idx",
                torch.zeros((1, 1), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "selected_mask",
                torch.zeros((1, 1), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "selected_count",
                torch.ones(1, dtype=torch.float32),
                persistent=False,
            )

        if expert_bias is None:
            bias = torch.zeros(self.n_experts, dtype=torch.float32)
        else:
            b = list(expert_bias)
            if len(b) != self.n_experts:
                raise ValueError(
                    "rule_router.expert_bias length must match n_experts, "
                    f"got {len(b)} vs {self.n_experts}"
                )
            bias = torch.tensor([float(v) for v in b], dtype=torch.float32)
        self.register_buffer("expert_bias", bias, persistent=False)

    def _compute_logits(self, rule_features: torch.Tensor) -> torch.Tensor:
        if rule_features.size(-1) != self.n_stage_features:
            raise ValueError(
                f"RuleSoftRouter expected last dim {self.n_stage_features}, got {rule_features.size(-1)}"
            )

        if self.variant == "teacher_gls":
            return build_group_local_stat_logits_12way(
                rule_features,
                feature_groups=self.group_feature_indices,
                expert_scale=self.expert_scale,
                stat_sharpness=self.stat_sharpness,
            )

        original_shape = rule_features.shape[:-1]
        flat = rule_features.reshape(-1, self.n_stage_features)
        n_flat = flat.size(0)
        if n_flat == 0:
            return flat.new_zeros(*original_shape, self.n_experts)

        idx_flat = self.selected_idx.reshape(-1)
        gathered = flat[:, idx_flat].reshape(n_flat, self.n_experts, -1)

        ratio = _to_ratio(gathered)
        bins = torch.floor(ratio * float(self.n_bins)).clamp(0, self.n_bins - 1)
        bin_center = (bins + 0.5) / float(self.n_bins)

        mask = self.selected_mask.unsqueeze(0)
        mean_score = (bin_center * mask).sum(dim=-1) / self.selected_count.unsqueeze(0)
        logits = mean_score + self.expert_bias.unsqueeze(0)
        return logits.reshape(*original_shape, self.n_experts)

    def compute_logits(self, rule_features: torch.Tensor) -> torch.Tensor:
        return self._compute_logits(rule_features)

    def forward(
        self,
        rule_features: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(rule_features)
        scale = max(float(temperature), 1e-6)
        scaled_logits = logits / scale
        active_top_k = self.top_k if top_k is None else top_k
        weights = _softmax_with_top_k(scaled_logits, n_experts=self.n_experts, top_k=active_top_k)
        return weights, scaled_logits


class RouterBackend(nn.Module):
    """Common backend wrapper for learned/rule-soft stage routers."""

    def __init__(
        self,
        *,
        impl: str,
        n_experts: int,
        top_k: Optional[int],
        d_in: Optional[int] = None,
        d_hidden: int = 64,
        dropout: float = 0.1,
        rule_soft_kwargs: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        impl_key = str(impl).lower().strip()
        if impl_key not in {"learned", "rule_soft"}:
            raise ValueError(f"router_impl must be one of ['learned','rule_soft'], got {impl}")
        self.impl = impl_key
        self.n_experts = int(n_experts)
        self.top_k = top_k

        if self.impl == "learned":
            if d_in is None or int(d_in) <= 0:
                raise ValueError(f"learned router requires d_in>0, got {d_in}")
            self.backend = Router(
                d_in=int(d_in),
                n_experts=self.n_experts,
                d_hidden=int(d_hidden),
                top_k=top_k,
                dropout=float(dropout),
            )
        else:
            kwargs = dict(rule_soft_kwargs or {})
            kwargs.setdefault("n_experts", self.n_experts)
            kwargs.setdefault("top_k", top_k)
            self.backend = RuleSoftRouter(**kwargs)

    def forward(
        self,
        *,
        router_input: Optional[torch.Tensor] = None,
        rule_features: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.impl == "learned":
            if router_input is None:
                raise ValueError("RouterBackend(learned) requires router_input.")
            return self.backend(router_input, temperature=temperature, top_k=top_k)
        if rule_features is None:
            raise ValueError("RouterBackend(rule_soft) requires rule_features.")
        return self.backend(rule_features, temperature=temperature, top_k=top_k)


# -----------------------------------------------------------------------
# Auxiliary losses
# -----------------------------------------------------------------------

def load_balance_loss(
    gate_weights: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Compute load-balance loss: encourages uniform expert utilisation.

    L_balance = Sum_k (mean(g_{:,k}) - 1/K)^2

    Args:
        gate_weights: [B, (T,) K]  - gating weights from the router.
        n_experts: K
    Returns:
        Scalar loss.
    """
    flat = gate_weights.reshape(-1, n_experts)
    mean_load = flat.mean(dim=0)
    target = 1.0 / n_experts
    return ((mean_load - target) ** 2).sum()


def router_entropy_loss(
    gate_weights: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Negative entropy of the mean gating distribution - encourages diversity.

    Maximising entropy <-> minimising negative entropy.

    Args:
        gate_weights: [B, (T,) K]
        n_experts: K
    Returns:
        Scalar loss (lower = more uniform distribution).
    """
    flat = gate_weights.reshape(-1, n_experts)
    mean_dist = flat.mean(dim=0)
    mean_dist = mean_dist.clamp(min=1e-8)
    entropy = -(mean_dist * mean_dist.log()).sum()
    return -entropy
