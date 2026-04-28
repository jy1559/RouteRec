"""Auxiliary losses for RouteRecV2."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..RouteRecBase.routers import load_balance_loss


def _masked_load_balance(gate_weights: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
    if item_seq_len is None:
        return load_balance_loss(gate_weights, n_experts=gate_weights.shape[-1])

    _, tlen, n_experts = gate_weights.shape
    lens = item_seq_len.to(device=gate_weights.device).long()
    valid = torch.arange(tlen, device=gate_weights.device).unsqueeze(0) < lens.unsqueeze(1)
    if not valid.any():
        return torch.tensor(0.0, device=gate_weights.device)
    return load_balance_loss(gate_weights[valid], n_experts=n_experts)


def _to_ratio(values: torch.Tensor) -> torch.Tensor:
    """Map features to [0,1] ratio space for stable variance regularization."""
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


def compute_expert_aux_loss(
    weights: Dict[str, torch.Tensor],
    item_seq_len: Optional[torch.Tensor],
    balance_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    if not weights:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for w in weights.values():
        total = total + _masked_load_balance(w, item_seq_len=item_seq_len)
    return float(balance_lambda) * total


def compute_stage_merge_aux_loss(
    stage_merge_weights: Optional[torch.Tensor],
    *,
    item_seq_len: Optional[torch.Tensor],
    balance_lambda: float,
    enabled: bool,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    if not enabled or stage_merge_weights is None:
        return torch.tensor(0.0, device=device)
    stage_aux = _masked_load_balance(stage_merge_weights, item_seq_len=item_seq_len)
    return float(balance_lambda) * float(scale) * stage_aux


def compute_feature_specialization_aux_loss(
    *,
    weights: Dict[str, torch.Tensor],
    feat: Optional[torch.Tensor],
    stage_feature_indices: Dict[str, Sequence[int]],
    selected_stages: Iterable[str],
    item_seq_len: Optional[torch.Tensor],
    min_tokens_per_expert: float,
    aux_lambda: float,
    enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    """Encourage feature-consistent routing by minimizing within-expert feature dispersion."""
    if (not enabled) or feat is None or (not weights):
        return torch.tensor(0.0, device=device)
    if float(aux_lambda) <= 0:
        return torch.tensor(0.0, device=device)

    stage_whitelist = {str(s).strip().lower() for s in selected_stages if str(s).strip()}
    if not stage_whitelist:
        return torch.tensor(0.0, device=device)

    bsz, tlen, _ = feat.shape
    if item_seq_len is not None:
        lens = item_seq_len.to(device=feat.device).long()
        valid_mask = torch.arange(tlen, device=feat.device).unsqueeze(0) < lens.unsqueeze(1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)
    else:
        valid_mask = torch.ones(bsz, tlen, dtype=torch.bool, device=feat.device)

    stage_losses = []
    min_tokens = max(float(min_tokens_per_expert), 0.0)
    for stage_key, gate_w in weights.items():
        base_stage = str(stage_key).split("@", 1)[0].strip().lower()
        if base_stage not in stage_whitelist:
            continue

        feat_idx = list(stage_feature_indices.get(base_stage, ()))
        if not feat_idx:
            continue

        idx = torch.tensor(feat_idx, device=feat.device, dtype=torch.long)
        x = feat.index_select(-1, idx)
        x = _to_ratio(x)

        w_flat = gate_w[valid_mask]
        x_flat = x[valid_mask]
        if w_flat.numel() == 0:
            continue

        sum_w = w_flat.sum(dim=0)
        valid_expert = sum_w >= min_tokens
        if not valid_expert.any():
            continue

        # mu_k = sum_n w_nk * x_n / sum_n w_nk
        mu = (w_flat.transpose(0, 1) @ x_flat) / sum_w.clamp(min=1e-8).unsqueeze(-1)

        # var_k = sum_n w_nk * ||x_n - mu_k||^2 / sum_n w_nk
        sq_dist = (x_flat.unsqueeze(1) - mu.unsqueeze(0)).pow(2).sum(dim=-1)
        var = (w_flat * sq_dist).sum(dim=0) / sum_w.clamp(min=1e-8)
        stage_losses.append(var[valid_expert].mean())

    if not stage_losses:
        return torch.tensor(0.0, device=device)

    return float(aux_lambda) * torch.stack(stage_losses).mean()


def compute_router_distill_aux_loss(
    *,
    teacher_group_logits: Dict[str, torch.Tensor],
    student_group_logits: Dict[str, torch.Tensor],
    item_seq_len: Optional[torch.Tensor],
    aux_lambda: float,
    distill_temperature: float,
    enabled: bool,
    progress: float,
    until: float,
    device: torch.device,
) -> torch.Tensor:
    if not enabled or float(aux_lambda) <= 0:
        return torch.tensor(0.0, device=device)
    if not teacher_group_logits or not student_group_logits:
        return torch.tensor(0.0, device=device)

    until = float(until)
    if until <= 0:
        return torch.tensor(0.0, device=device)

    progress = max(float(progress), 0.0)
    if progress >= until:
        return torch.tensor(0.0, device=device)

    tau = max(float(distill_temperature), 1e-6)
    weight = float(aux_lambda) * max(0.0, 1.0 - (progress / until))
    stage_losses = []

    common_keys = sorted(set(teacher_group_logits) & set(student_group_logits))
    for stage_key in common_keys:
        teacher = teacher_group_logits[stage_key]
        student = student_group_logits[stage_key]
        if teacher.shape != student.shape or teacher.ndim != 3:
            continue

        _, tlen, _ = teacher.shape
        if item_seq_len is not None:
            lens = item_seq_len.to(device=teacher.device).long()
            valid_mask = torch.arange(tlen, device=teacher.device).unsqueeze(0) < lens.unsqueeze(1)
            if not valid_mask.any():
                continue
            teacher = teacher[valid_mask]
            student = student[valid_mask]
        else:
            teacher = teacher.reshape(-1, teacher.shape[-1])
            student = student.reshape(-1, student.shape[-1])

        if teacher.numel() == 0:
            continue

        teacher_prob = F.softmax(teacher / tau, dim=-1)
        student_log_prob = F.log_softmax(student / tau, dim=-1)
        stage_kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        stage_losses.append((tau * tau) * stage_kl)

    if not stage_losses:
        return torch.tensor(0.0, device=device)

    return weight * torch.stack(stage_losses).mean()
