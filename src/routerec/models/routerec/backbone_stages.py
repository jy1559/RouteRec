"""Stage modules shared by the RouteRec backbone."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_features import STAGE_ALL_FEATURES
from .feature_bank import _to_ratio
from .routers import Router, RuleSoftRouter
from .sparse_stages import MoEStage as SparseMoEStage
from .sparse_stages import _scaled_expert_lists, _scaled_expert_names
from .transformer import PositionwiseFFN, TransformerEncoder


def _normalize_top_k(top_k: Optional[int], n_experts: int) -> Optional[int]:
    if top_k is None:
        return None
    k = int(top_k)
    if k <= 0:
        return None
    k = min(k, int(n_experts))
    return None if k >= int(n_experts) else k


def _softmax_with_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    active_top_k = _normalize_top_k(top_k, logits.shape[-1])
    if active_top_k is None:
        return F.softmax(logits, dim=-1)
    topk_vals, topk_idx = logits.topk(active_top_k, dim=-1)
    topk_weights = F.softmax(topk_vals, dim=-1)
    weights = torch.zeros_like(logits)
    weights.scatter_(-1, topk_idx, topk_weights)
    return weights


@dataclass
class StageRuntimeConfig:
    stage_name: str
    pass_layers: int
    moe_blocks: int
    d_model: int
    d_ff: int
    n_heads: int
    dropout: float
    attn_dropout: float
    d_expert_hidden: int
    d_router_hidden: int
    expert_depth: int
    expert_scale: int
    feature_bank_dim: int
    top_k: Optional[int]
    router_use_hidden: bool
    router_use_feature: bool
    expert_use_hidden: bool
    expert_use_feature: bool
    macro_routing_scope: str
    macro_session_pooling: str
    mid_router_temperature: float
    micro_router_temperature: float
    mid_router_feature_dropout: float
    micro_router_feature_dropout: float
    use_valid_ratio_gating: bool
    col2idx: Dict[str, int]
    expert_feature_lists: list[list[str]]
    expert_names: list[str]
    router_impl: str
    rule_router_cfg: Dict[str, Any]
    rule_bias_scale: float
    inter_layer_style: str
    router_group_feature_mode: str
    moe_block_variant: str


class _IdentityInterBlock(nn.Module):
    def forward(self, hidden: torch.Tensor, item_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        return hidden


class _ResidualNonlinearInterBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(int(d_model))
        self.proj = nn.Linear(int(d_model), int(d_model))
        self.drop = nn.Dropout(float(dropout))
        self.resid_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden: torch.Tensor, item_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = self.drop(F.gelu(self.proj(self.ln(hidden))))
        return hidden + self.resid_scale * delta


class _ResidualFFNInterBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(int(d_model))
        self.ffn = PositionwiseFFN(int(d_model), int(d_ff), float(dropout))
        self.resid_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden: torch.Tensor, item_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        return hidden + self.resid_scale * self.ffn(self.ln(hidden))


def _build_inter_block(style: str, cfg: StageRuntimeConfig) -> nn.Module:
    key = str(style or "attn").lower().strip()
    if key == "attn":
        return TransformerEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=1,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            ffn_moe=False,
        )
    if key == "identity":
        return _IdentityInterBlock()
    if key == "nonlinear":
        return _ResidualNonlinearInterBlock(d_model=cfg.d_model, dropout=cfg.dropout)
    if key == "ffn":
        return _ResidualFFNInterBlock(d_model=cfg.d_model, d_ff=cfg.d_ff, dropout=cfg.dropout)
    raise ValueError(
        "stage_inter_layer_style must be one of ['attn','identity','nonlinear','ffn'], "
        f"got {style}"
    )


def _build_moe_replacement_block(variant: str, cfg: StageRuntimeConfig) -> nn.Module:
    key = str(variant or "moe").lower().strip()
    if key == "dense_ffn":
        return _ResidualFFNInterBlock(d_model=cfg.d_model, d_ff=cfg.d_ff, dropout=cfg.dropout)
    if key == "nonlinear":
        return _ResidualNonlinearInterBlock(d_model=cfg.d_model, dropout=cfg.dropout)
    if key == "identity":
        return _IdentityInterBlock()
    raise ValueError(
        "moe_block_variant must be one of ['moe','dense_ffn','nonlinear','identity'], "
        f"got {variant}"
    )


class NExpertMLP(nn.Module):
    """Shallow expert MLP over shared transformed features."""

    def __init__(
        self,
        *,
        d_feat_in: int,
        d_model: int,
        d_hidden: int,
        d_out: int,
        depth: int,
        use_hidden: bool,
        use_feature: bool,
        dropout: float,
    ):
        super().__init__()
        if not (use_hidden or use_feature):
            raise ValueError("NExpertMLP requires at least one input source.")

        self.use_hidden = bool(use_hidden)
        self.use_feature = bool(use_feature)
        input_dim = (int(d_model) if self.use_hidden else 0) + (int(d_feat_in) if self.use_feature else 0)

        hidden_dim = max(int(d_hidden), 1)
        n_hidden = max(int(depth), 0)
        layers = []
        prev_dim = input_dim
        for _ in range(n_hidden):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, int(d_out)))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, feat_flat: torch.Tensor) -> torch.Tensor:
        pieces = []
        if self.use_hidden:
            pieces.append(hidden)
        if self.use_feature:
            pieces.append(feat_flat)
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        return self.net(x)


class NExpertGroup(nn.Module):
    """Per-stage expert group using shared transformed feature slices."""

    def __init__(
        self,
        *,
        expert_feature_dims: list[int],
        d_model: int,
        d_hidden: int,
        d_out: int,
        depth: int,
        use_hidden: bool,
        use_feature: bool,
        dropout: float,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                NExpertMLP(
                    d_feat_in=d_feat_in,
                    d_model=d_model,
                    d_hidden=d_hidden,
                    d_out=d_out,
                    depth=depth,
                    use_hidden=use_hidden,
                    use_feature=use_feature,
                    dropout=dropout,
                )
                for d_feat_in in expert_feature_dims
            ]
        )

    def forward(self, hidden: torch.Tensor, expert_inputs: list[torch.Tensor]) -> torch.Tensor:
        outputs = [
            expert(hidden, expert_input)
            for expert, expert_input in zip(self.experts, expert_inputs)
        ]
        return torch.stack(outputs, dim=-2)


class NMoEStage(nn.Module):
    """Single lightweight residual MoE stage with optional rule-soft prior/bias."""

    def __init__(self, cfg: StageRuntimeConfig):
        super().__init__()
        self.stage_name = cfg.stage_name
        self.router_impl = str(cfg.router_impl).lower().strip()
        if self.router_impl not in {"learned", "rule_soft"}:
            raise ValueError(
                "router_impl must be one of ['learned','rule_soft'], "
                f"got {cfg.router_impl}"
            )
        self.router_use_hidden = bool(cfg.router_use_hidden)
        self.router_use_feature = bool(cfg.router_use_feature)
        self.router_group_feature_mode = str(cfg.router_group_feature_mode or "none").lower().strip()
        if self.router_group_feature_mode not in {"none", "mean", "mean_std"}:
            raise ValueError(
                "router_group_feature_mode must be one of ['none','mean','mean_std'], "
                f"got {cfg.router_group_feature_mode}"
            )
        self.expert_use_hidden = bool(cfg.expert_use_hidden)
        self.expert_use_feature = bool(cfg.expert_use_feature)
        if not (self.router_use_hidden or self.router_use_feature or self.router_group_feature_mode != "none"):
            raise ValueError(
                "router inputs cannot all be disabled. Enable hidden, feature, or group-feature mode."
            )
        if not (self.expert_use_hidden or self.expert_use_feature):
            raise ValueError("expert_use_hidden and expert_use_feature cannot both be false.")

        self.expert_scale = int(cfg.expert_scale)
        self.feature_bank_dim = int(cfg.feature_bank_dim)
        self.rule_router_cfg = dict(cfg.rule_router_cfg or {})
        variant = str(self.rule_router_cfg.get("variant", "ratio_bins")).lower().strip()
        if variant != "ratio_bins":
            raise ValueError(
                "RouteRec only supports rule_router.variant='ratio_bins'. "
                f"Got {variant}"
            )
        self.rule_router_cfg["variant"] = "ratio_bins"
        self.rule_bias_scale = float(cfg.rule_bias_scale)
        self.current_top_k = _normalize_top_k(cfg.top_k, n_experts=1_000_000)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.alpha_scale = 1.0
        self.pre_ln = nn.LayerNorm(int(cfg.d_model))
        self.resid_drop = nn.Dropout(float(cfg.dropout))

        stage_all_features = list(STAGE_ALL_FEATURES[cfg.stage_name])
        stage_idx = [int(cfg.col2idx[name]) for name in stage_all_features if name in cfg.col2idx]
        self.register_buffer(
            "stage_feat_idx",
            torch.tensor(stage_idx, dtype=torch.long),
            persistent=False,
        )
        self.stage_all_features = [stage_all_features[i] for i in range(len(stage_idx))]
        stage_col2local = {name: idx for idx, name in enumerate(self.stage_all_features)}
        self._router_group_local_idx: list[torch.Tensor] = []

        self.base_names = list(cfg.expert_names)
        self.group_names = list(self.base_names)
        self.expert_names = _scaled_expert_names(self.base_names, self.expert_scale)
        scaled_lists = _scaled_expert_lists(cfg.expert_feature_lists, self.expert_scale)
        self.n_experts = len(scaled_lists)

        expert_bank_indices = []
        expert_bank_dims = []
        for expert_feats in scaled_lists:
            raw_idx = [
                int(cfg.col2idx[name])
                for name in expert_feats
                if name in cfg.col2idx
            ]
            if not raw_idx:
                raw_idx = [int(self.stage_feat_idx[0].item())]
            expert_bank_indices.append(raw_idx)
            expert_bank_dims.append(len(raw_idx) * self.feature_bank_dim)

        for group_idx, expert_feats in enumerate(cfg.expert_feature_lists):
            local_idx = [
                int(stage_col2local[name])
                for name in expert_feats
                if name in stage_col2local
            ]
            if not local_idx and len(self.stage_all_features) > 0:
                local_idx = [0]
            tensor_idx = torch.tensor(local_idx, dtype=torch.long)
            self.register_buffer(
                f"router_group_local_idx_{group_idx}",
                tensor_idx,
                persistent=False,
            )
            self._router_group_local_idx.append(tensor_idx)

        for expert_idx, raw_idx in enumerate(expert_bank_indices):
            self.register_buffer(
                f"expert_bank_idx_{expert_idx}",
                torch.tensor(raw_idx, dtype=torch.long),
                persistent=False,
            )

        self.expert_group = NExpertGroup(
            expert_feature_dims=expert_bank_dims,
            d_model=int(cfg.d_model),
            d_hidden=int(cfg.d_expert_hidden),
            d_out=int(cfg.d_model),
            depth=int(cfg.expert_depth),
            use_hidden=self.expert_use_hidden,
            use_feature=self.expert_use_feature,
            dropout=float(cfg.dropout),
        )

        router_in_dim = 0
        if self.router_use_hidden:
            router_in_dim += int(cfg.d_model)
        if self.router_use_feature:
            router_in_dim += len(self.stage_all_features) * self.feature_bank_dim
        if self.router_group_feature_mode == "mean":
            router_in_dim += len(self._router_group_local_idx)
        elif self.router_group_feature_mode == "mean_std":
            router_in_dim += 2 * len(self._router_group_local_idx)
        self.learned_router = None
        if self.router_impl == "learned":
            self.learned_router = Router(
                d_in=router_in_dim,
                n_experts=self.n_experts,
                d_hidden=int(cfg.d_router_hidden),
                top_k=None,
                dropout=float(cfg.dropout),
            )

        feature_per_expert = max(int(self.rule_router_cfg.get("feature_per_expert", 4)), 1)
        selected_indices, _ = SparseMoEStage._resolve_rule_feature_selection(
            stage_name=self.stage_name,
            base_names=self.base_names,
            scaled_names=self.expert_names,
            scaled_feature_lists=scaled_lists,
            expert_scale=self.expert_scale,
            stage_col2local=stage_col2local,
            feature_per_expert=feature_per_expert,
            rule_router_cfg=self.rule_router_cfg,
        )
        expert_bias = SparseMoEStage._resolve_rule_expert_bias(
            stage_name=self.stage_name,
            base_names=self.base_names,
            scaled_names=self.expert_names,
            expert_scale=self.expert_scale,
            n_experts=self.n_experts,
            rule_router_cfg=self.rule_router_cfg,
        )
        rule_kwargs = {
            "n_experts": self.n_experts,
            "n_stage_features": len(self.stage_all_features),
            "selected_feature_indices": selected_indices,
            "feature_names": self.stage_all_features,
            "n_bins": int(self.rule_router_cfg.get("n_bins", 5)),
            "expert_bias": expert_bias,
            "top_k": None,
            "variant": "ratio_bins",
        }
        self.rule_router = RuleSoftRouter(**rule_kwargs)
        self.rule_prior_router = self.rule_router if self.rule_bias_scale > 0 else None

        router_mode = "token"
        session_pooling = "mean"
        router_temperature = 1.0
        router_feature_dropout = 0.0
        reliability_feature_name = None

        if cfg.stage_name == "macro":
            router_mode = str(cfg.macro_routing_scope).lower().strip()
            session_pooling = str(cfg.macro_session_pooling).lower().strip()
            router_temperature = 1.0
        elif cfg.stage_name == "mid":
            router_temperature = float(cfg.mid_router_temperature)
            router_feature_dropout = float(cfg.mid_router_feature_dropout)
            reliability_feature_name = "mid_valid_r" if cfg.use_valid_ratio_gating else None
        elif cfg.stage_name == "micro":
            router_temperature = float(cfg.micro_router_temperature)
            router_feature_dropout = float(cfg.micro_router_feature_dropout)
            reliability_feature_name = "mic_valid_r" if cfg.use_valid_ratio_gating else None

        if router_mode not in {"token", "session"}:
            raise ValueError(f"Unsupported router mode for RouteRec stage: {router_mode}")
        if session_pooling not in {"mean", "last"}:
            raise ValueError(
                "RouteRec supports macro_session_pooling in ['mean','last'] only. "
                f"Got {session_pooling}"
            )
        self.router_mode = router_mode
        self.session_pooling = session_pooling
        self.current_router_temperature = max(float(router_temperature), 1e-6)
        self.router_feat_drop = nn.Dropout(max(float(router_feature_dropout), 0.0))
        self.reliability_feat_idx = cfg.col2idx.get(reliability_feature_name)
        self.last_router_aux: Dict[str, torch.Tensor] = {}

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if alpha_scale is not None:
            self.alpha_scale = float(alpha_scale)
        if router_temperature is not None:
            self.current_router_temperature = max(float(router_temperature), 1e-6)
        if top_k is not None:
            self.current_top_k = _normalize_top_k(top_k, n_experts=self.n_experts)

    @staticmethod
    def _build_valid_mask(
        batch_size: int,
        seq_len: int,
        item_seq_len: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if item_seq_len is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        lens = item_seq_len.to(device=device).long()
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        return arange < lens.unsqueeze(1)

    def _pool_sequence(
        self,
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.session_pooling == "mean":
            weight = valid_mask.float().unsqueeze(-1)
            denom = weight.sum(dim=1).clamp(min=1.0)
            return (seq * weight).sum(dim=1) / denom

        if item_seq_len is None:
            idx = torch.full(
                (seq.size(0),),
                fill_value=max(seq.size(1) - 1, 0),
                dtype=torch.long,
                device=seq.device,
            )
        else:
            idx = item_seq_len.to(device=seq.device).long().clamp(min=1, max=seq.size(1)) - 1
        return seq[torch.arange(seq.size(0), device=seq.device), idx]

    def _apply_reliability(self, stage_raw_feat: torch.Tensor, stage_bank_flat: torch.Tensor, feat: torch.Tensor):
        if self.reliability_feat_idx is None:
            return stage_raw_feat, stage_bank_flat
        reliability = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0)
        stage_raw_feat = stage_raw_feat * reliability.unsqueeze(-1)
        # Reliability is a per-token scalar gate, so it should broadcast across
        # the flattened shared-bank dimension regardless of bank_dim.
        stage_bank_flat = stage_bank_flat * reliability.unsqueeze(-1)
        return stage_raw_feat, stage_bank_flat

    def _stage_bank_flat(self, feat_bank: torch.Tensor) -> torch.Tensor:
        stage_bank = feat_bank.index_select(2, self.stage_feat_idx)
        return stage_bank.reshape(stage_bank.size(0), stage_bank.size(1), -1)

    def _group_router_features(self, stage_raw_feat: torch.Tensor) -> Optional[torch.Tensor]:
        if self.router_group_feature_mode == "none" or not self._router_group_local_idx:
            return None

        pieces = []
        for idx in self._router_group_local_idx:
            idx = idx.to(device=stage_raw_feat.device)
            group_feat = stage_raw_feat.index_select(-1, idx)
            mean_feat = group_feat.mean(dim=-1, keepdim=True)
            if self.router_group_feature_mode == "mean":
                pieces.append(mean_feat)
                continue
            std_feat = group_feat.std(dim=-1, keepdim=True, unbiased=False)
            pieces.extend([mean_feat, std_feat])
        return torch.cat(pieces, dim=-1) if pieces else None

    def _resolve_session_pool_inputs(
        self,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        routing_item_seq_len: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.router_mode != "session" or routing_item_seq_len is None:
            return valid_mask, item_seq_len
        session_valid_mask = self._build_valid_mask(
            valid_mask.size(0),
            valid_mask.size(1),
            routing_item_seq_len,
            valid_mask.device,
        )
        return session_valid_mask, routing_item_seq_len

    def _build_router_inputs(
        self,
        *,
        h_norm: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        routing_item_seq_len: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_raw_feat = feat.index_select(-1, self.stage_feat_idx)
        stage_bank_flat = self._stage_bank_flat(feat_bank)
        stage_raw_feat, stage_bank_flat = self._apply_reliability(stage_raw_feat, stage_bank_flat, feat)
        stage_bank_flat = self.router_feat_drop(stage_bank_flat)
        group_router_feat = self._group_router_features(stage_raw_feat)
        session_valid_mask, session_item_seq_len = self._resolve_session_pool_inputs(
            valid_mask,
            item_seq_len,
            routing_item_seq_len,
        )

        if self.router_mode == "session":
            pooled_parts = []
            if self.router_use_hidden:
                pooled_parts.append(self._pool_sequence(h_norm, session_valid_mask, session_item_seq_len))
            if self.router_use_feature:
                pooled_parts.append(self._pool_sequence(stage_bank_flat, session_valid_mask, session_item_seq_len))
            if group_router_feat is not None:
                pooled_parts.append(self._pool_sequence(group_router_feat, session_valid_mask, session_item_seq_len))
            router_input = pooled_parts[0] if len(pooled_parts) == 1 else torch.cat(pooled_parts, dim=-1)
            rule_features = self._pool_sequence(stage_raw_feat, session_valid_mask, session_item_seq_len)
            return router_input, rule_features

        router_parts = []
        if self.router_use_hidden:
            router_parts.append(h_norm)
        if self.router_use_feature:
            router_parts.append(stage_bank_flat)
        if group_router_feat is not None:
            router_parts.append(group_router_feat)
        router_input = router_parts[0] if len(router_parts) == 1 else torch.cat(router_parts, dim=-1)
        return router_input, stage_raw_feat

    def _compute_router_outputs(
        self,
        *,
        h_norm: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        routing_item_seq_len: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_input, rule_features = self._build_router_inputs(
            h_norm=h_norm,
            feat=feat,
            feat_bank=feat_bank,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
            routing_item_seq_len=routing_item_seq_len,
        )

        if self.router_impl == "rule_soft":
            raw_logits = self.rule_router.compute_logits(rule_features)
        else:
            if self.learned_router is None:
                raise RuntimeError("learned router is not initialized.")
            raw_logits = self.learned_router.net(router_input)
            if self.rule_prior_router is not None:
                raw_logits = raw_logits + self.rule_bias_scale * self.rule_prior_router.compute_logits(rule_features)

        scaled_logits = raw_logits / max(float(self.current_router_temperature), 1e-6)
        weights = _softmax_with_top_k(scaled_logits, self.current_top_k)
        return weights, scaled_logits

    def _gather_expert_inputs(self, feat_bank: torch.Tensor) -> list[torch.Tensor]:
        inputs = []
        for expert_idx in range(self.n_experts):
            idx = getattr(self, f"expert_bank_idx_{expert_idx}")
            expert_feat = feat_bank.index_select(2, idx).reshape(feat_bank.size(0), feat_bank.size(1), -1)
            inputs.append(expert_feat)
        return inputs

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
        routing_item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = hidden.shape
        h_norm = self.pre_ln(hidden)
        valid_mask = self._build_valid_mask(batch_size, seq_len, item_seq_len, hidden.device)

        gate_weights, gate_logits = self._compute_router_outputs(
            h_norm=h_norm,
            feat=feat,
            feat_bank=feat_bank,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
            routing_item_seq_len=routing_item_seq_len,
        )
        if self.router_mode == "session":
            gate_weights = gate_weights.unsqueeze(1).expand(-1, seq_len, -1)
            gate_logits = gate_logits.unsqueeze(1).expand(-1, seq_len, -1)

        expert_inputs = self._gather_expert_inputs(feat_bank)
        expert_out = self.expert_group(h_norm, expert_inputs)
        stage_out = (gate_weights.unsqueeze(-1) * expert_out).sum(dim=-2)

        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_out)
        self.last_router_aux = {}
        return next_hidden, stage_out, gate_weights, gate_logits, {}


class StageBranchRunner(nn.Module):
    """Run one explicit pass/MoE branch for RouteRec."""

    def __init__(self, cfg: StageRuntimeConfig):
        super().__init__()
        self.stage_name = cfg.stage_name
        self.pass_layers = int(cfg.pass_layers)
        self.moe_blocks = int(cfg.moe_blocks)
        self.inter_layer_style = str(cfg.inter_layer_style).lower().strip()
        self.moe_block_variant = str(cfg.moe_block_variant or "moe").lower().strip()
        if self.moe_block_variant not in {"moe", "dense_ffn", "nonlinear", "identity"}:
            raise ValueError(
                "moe_block_variant must be one of ['moe','dense_ffn','nonlinear','identity'], "
                f"got {cfg.moe_block_variant}"
            )
        self.pass_transformer = None
        self.pass_blocks = nn.ModuleList()

        if self.pass_layers > 0:
            if self.inter_layer_style == "attn":
                self.pass_transformer = TransformerEncoder(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    n_layers=self.pass_layers,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    ffn_moe=False,
                )
            else:
                self.pass_blocks = nn.ModuleList(
                    [_build_inter_block(self.inter_layer_style, cfg) for _ in range(self.pass_layers)]
                )

        if self.moe_blocks > 0:
            self.moe_pre_blocks = nn.ModuleList(
                [_build_inter_block(self.inter_layer_style, cfg) for _ in range(self.moe_blocks)]
            )
            if self.moe_block_variant == "moe":
                self.stage_module = NMoEStage(cfg)
                self.moe_replacement_blocks = nn.ModuleList()
            else:
                self.stage_module = None
                self.moe_replacement_blocks = nn.ModuleList(
                    [_build_moe_replacement_block(self.moe_block_variant, cfg) for _ in range(self.moe_blocks)]
                )
        else:
            self.moe_pre_blocks = nn.ModuleList()
            self.stage_module = None
            self.moe_replacement_blocks = nn.ModuleList()

    @property
    def n_experts(self) -> int:
        return 0 if self.stage_module is None else int(self.stage_module.n_experts)

    @property
    def expert_names(self) -> list[str]:
        return [] if self.stage_module is None else list(self.stage_module.expert_names)

    @property
    def group_names(self) -> list[str]:
        return [] if self.stage_module is None else list(self.stage_module.group_names)

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if self.stage_module is None:
            return
        self.stage_module.set_schedule_state(
            alpha_scale=alpha_scale,
            router_temperature=router_temperature,
            top_k=top_k,
        )

    def _run_pass_layers(self, hidden: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        if self.pass_transformer is None:
            out = hidden
            for block in self.pass_blocks:
                if isinstance(block, TransformerEncoder):
                    out, _ = block(out, item_seq)
                else:
                    out = block(out, item_seq)
            return out
        out, _ = self.pass_transformer(hidden, item_seq)
        return out

    def forward_serial(
        self,
        *,
        hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
        routing_item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        out = self._run_pass_layers(hidden, item_seq)

        weights: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.stage_module is None and len(self.moe_replacement_blocks) == 0:
            return out, weights, logits, router_aux

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            if isinstance(pre_block, TransformerEncoder):
                out, _ = pre_block(out, item_seq)
            else:
                out = pre_block(out, item_seq)
            if self.stage_module is not None:
                out, _delta, w, l, aux = self.stage_module(
                    out,
                    feat,
                    feat_bank,
                    item_seq_len=item_seq_len,
                    routing_item_seq_len=routing_item_seq_len,
                )
                key = f"{self.stage_name}@{idx}"
                weights[key] = w
                logits[key] = l
                for aux_key, aux_tensor in aux.items():
                    router_aux.setdefault(aux_key, {})[key] = aux_tensor
            else:
                out = self.moe_replacement_blocks[idx - 1](out, item_seq)
        return out, weights, logits, router_aux

    def forward_parallel(
        self,
        *,
        base_hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
        routing_item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        out = self._run_pass_layers(base_hidden, item_seq)

        weights: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.stage_module is None and len(self.moe_replacement_blocks) == 0:
            return out, (out - base_hidden), weights, logits, router_aux

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            if isinstance(pre_block, TransformerEncoder):
                out, _ = pre_block(out, item_seq)
            else:
                out = pre_block(out, item_seq)
            if self.stage_module is not None:
                out, _delta, w, l, aux = self.stage_module(
                    out,
                    feat,
                    feat_bank,
                    item_seq_len=item_seq_len,
                    routing_item_seq_len=routing_item_seq_len,
                )
                key = f"{self.stage_name}@{idx}"
                weights[key] = w
                logits[key] = l
                for aux_key, aux_tensor in aux.items():
                    router_aux.setdefault(aux_key, {})[key] = aux_tensor
            else:
                out = self.moe_replacement_blocks[idx - 1](out, item_seq)

        return out, (out - base_hidden), weights, logits, router_aux
