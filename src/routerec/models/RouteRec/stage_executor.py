"""layer_layout-based executor for RouteRec."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reviewer_controls import (
    controlled_feature_input,
    controlled_hidden_input,
    freeze_random_router_path,
    normalize_control_mode,
)
from .stage_modules import (
    N3StageBlock,
    SASRecStyleAttentionBlock,
    SASRecStyleFFNBlock,
    SASRecStyleLayerBlock,
    StageRuntimeConfigN3,
)


_PLAIN_TOKENS = {"layer", "attn", "ffn"}
_STAGE_NAMES = ("macro", "mid", "micro")
_STAGE_TOKENS = {
    "macro",
    "macro_ffn",
    "mid",
    "mid_ffn",
    "micro",
    "micro_ffn",
}
_VALID_TOKENS = _PLAIN_TOKENS | _STAGE_TOKENS
_BUNDLE_AGGS = {"sum", "mean", "learned", "router"}


def _parse_bundle_token(token: str) -> Optional[tuple[list[str], str]]:
    text = str(token or "").lower().strip()
    if not text.startswith("bundle_"):
        return None
    parts = [p for p in text.split("_") if p]
    if len(parts) < 4:
        return None
    agg = parts[-1]
    if agg not in _BUNDLE_AGGS:
        return None
    stage_names = parts[1:-1]
    if len(stage_names) not in {2, 3}:
        return None
    if any(stage not in _STAGE_NAMES for stage in stage_names):
        return None
    if len(set(stage_names)) != len(stage_names):
        return None
    return list(stage_names), agg


class _BundleMergeRouter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_features: int,
        d_hidden: int,
        n_branches: int,
        dropout: float,
    ):
        super().__init__()
        d_in = int(d_model) + int(max(n_features, 1))
        d_hid = int(max(d_hidden, 8))
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_hid, int(n_branches)),
        )

    def forward(self, *, hidden: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden, feat], dim=-1)
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights, logits


class StageExecutorN3(nn.Module):
    """Execute explicit N3 layer_layout tokens left-to-right."""

    def __init__(
        self,
        *,
        layer_layout: List[str],
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
        hidden_act: str,
        layer_norm_eps: float,
        d_feat_emb: int,
        d_expert_hidden: int,
        d_router_hidden: int,
        expert_depth_by_stage: Dict[str, int],
        expert_hidden_by_stage: Dict[str, int],
        expert_scale: int,
        stage_top_k: Optional[int],
        macro_session_pooling: str,
        stage_router_granularity: Dict[str, str],
        stage_all_features: Dict[str, List[str]],
        stage_family_features: Dict[str, Dict[str, List[str]]],
        stage_feature_encoder_mode: Dict[str, str],
        stage_compute_mode: Dict[str, str],
        stage_router_mode: Dict[str, str],
        stage_router_source: Dict[str, str],
        stage_feature_injection: Dict[str, str],
        rule_router_cfg: Dict[str, object],
        rule_bias_scale: float,
        feature_group_bias_lambda: float,
        feature_group_prior_temperature: float,
        stage_router_wrapper: Dict[str, str],
        stage_router_primitives: Dict[str, Dict[str, object]],
        mid_router_temperature: float,
        micro_router_temperature: float,
        dense_hidden_scale: float,
        stage_residual_mode: Dict[str, str],
        residual_alpha_fixed: Dict[str, float],
        residual_alpha_init: Dict[str, float],
        residual_shared_ffn_scale: float,
        stage_family_dropout_prob: Dict[str, float],
        stage_feature_dropout_prob: Dict[str, float],
        stage_feature_dropout_scope: Dict[str, str],
        col2idx: Dict[str, int],
        intra_group_bias_mode: Optional[Dict[str, str]] = None,
        intra_group_bias_scale: Optional[Dict[str, float]] = None,
        reviewer_control_mode: str = "full",
    ):
        super().__init__()
        self.reviewer_control_mode = normalize_control_mode(reviewer_control_mode)
        self.layer_layout = [str(token).lower().strip() for token in list(layer_layout or [])]
        if not self.layer_layout:
            raise ValueError("layer_layout cannot be empty.")

        unknown = []
        for token in self.layer_layout:
            if token in _VALID_TOKENS:
                continue
            if _parse_bundle_token(token) is not None:
                continue
            unknown.append(token)
        if unknown:
            raise ValueError(f"Unsupported layer_layout tokens: {unknown}")

        self.layer_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.stage_attn_blocks = nn.ModuleList()
        self.stage_blocks = nn.ModuleDict()
        self.bundle_learned_logits = nn.ParameterList()
        self.bundle_router_modules = nn.ModuleList()
        self.global_residual_alpha = nn.Parameter(torch.tensor(0.0))
        self.compiled_ops: List[dict] = []

        self._stage_counts = defaultdict(int)
        self._n_all_features = int(max(len(col2idx), 1))
        self._stage_all_features = {stage: list(stage_all_features.get(stage, [])) for stage in _STAGE_NAMES}
        self._stage_family_features = {
            stage: {name: list(cols) for name, cols in dict(stage_family_features.get(stage, {}) or {}).items()}
            for stage in _STAGE_NAMES
        }
        intra_group_bias_mode = dict(intra_group_bias_mode or {})
        intra_group_bias_scale = dict(intra_group_bias_scale or {})

        for stage_name in _STAGE_NAMES:
            if stage_name not in self.stage_blocks:
                router_temperature = 1.0
                if stage_name == "mid":
                    router_temperature = float(mid_router_temperature)
                elif stage_name == "micro":
                    router_temperature = float(micro_router_temperature)
                feature_names = self._stage_all_features.get(stage_name, [])
                cfg = StageRuntimeConfigN3(
                    stage_name=stage_name,
                    d_model=d_model,
                    d_ff=d_ff,
                    d_feat_emb=d_feat_emb,
                    d_expert_hidden=int(expert_hidden_by_stage.get(stage_name, d_expert_hidden)),
                    d_router_hidden=d_router_hidden,
                    expert_depth=int(expert_depth_by_stage.get(stage_name, 1)),
                    expert_scale=expert_scale,
                    top_k=stage_top_k,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    stage_feature_indices=tuple(
                        int(col2idx[name]) for name in feature_names if name in col2idx
                    ),
                    stage_feature_names=tuple(name for name in feature_names if name in col2idx),
                    stage_family_features=self._stage_family_features.get(stage_name, {}),
                    stage_feature_encoder_mode=str(stage_feature_encoder_mode.get(stage_name, "linear")),
                    stage_compute_mode=str(stage_compute_mode.get(stage_name, "moe")),
                    stage_router_mode=str(stage_router_mode.get(stage_name, "learned")),
                    stage_router_source=str(stage_router_source.get(stage_name, "both")),
                    stage_feature_injection=str(stage_feature_injection.get(stage_name, "none")),
                    routing_granularity=str(stage_router_granularity.get(stage_name, "token")),
                    session_pooling=str(macro_session_pooling),
                    rule_router_cfg=dict(rule_router_cfg or {}),
                    rule_bias_scale=float(rule_bias_scale),
                    feature_group_bias_lambda=float(feature_group_bias_lambda),
                    feature_group_prior_temperature=float(feature_group_prior_temperature),
                    stage_router_wrapper=str(stage_router_wrapper.get(stage_name, "w1_flat")),
                    stage_router_primitives=dict(stage_router_primitives.get(stage_name, {}) or {}),
                    intra_group_bias_mode=str(intra_group_bias_mode.get(stage_name, "none")),
                    intra_group_bias_scale=float(intra_group_bias_scale.get(stage_name, 0.0)),
                    reviewer_control_mode=str(reviewer_control_mode),
                    router_temperature=router_temperature,
                    dense_hidden_scale=float(dense_hidden_scale),
                    stage_residual_mode=str(stage_residual_mode.get(stage_name, "base")),
                    residual_alpha_fixed=float(residual_alpha_fixed.get(stage_name, 0.5)),
                    residual_alpha_init=float(residual_alpha_init.get(stage_name, 0.0)),
                    shared_ffn_scale=float(residual_shared_ffn_scale),
                    stage_family_dropout_prob=float(stage_family_dropout_prob.get(stage_name, 0.0)),
                    stage_feature_dropout_prob=float(stage_feature_dropout_prob.get(stage_name, 0.0)),
                    stage_feature_dropout_scope=str(stage_feature_dropout_scope.get(stage_name, "token")),
                )
                self.stage_blocks[stage_name] = N3StageBlock(cfg)

        op_idx = 0
        for token in self.layer_layout:
            bundle = _parse_bundle_token(token)
            if bundle is not None:
                op_idx += 1
                stage_names, agg = bundle
                stage_entries = []
                for stage_name in stage_names:
                    self._stage_counts[stage_name] += 1
                    stage_entries.append(
                        {
                            "stage": stage_name,
                            "stage_key": f"{stage_name}@{self._stage_counts[stage_name]}",
                        }
                    )

                learned_index = None
                router_index = None
                if agg == "learned":
                    self.bundle_learned_logits.append(nn.Parameter(torch.zeros(len(stage_names), dtype=torch.float32)))
                    learned_index = len(self.bundle_learned_logits) - 1
                elif agg == "router":
                    self.bundle_router_modules.append(
                        _BundleMergeRouter(
                            d_model=d_model,
                            n_features=self._n_all_features,
                            d_hidden=d_router_hidden,
                            n_branches=len(stage_names),
                            dropout=dropout,
                        )
                    )
                    router_index = len(self.bundle_router_modules) - 1

                self.compiled_ops.append(
                    {
                        "kind": "bundle",
                        "agg": agg,
                        "token": token,
                        "name": f"{op_idx:02d}_{token}.bundle",
                        "stage_entries": stage_entries,
                        "learned_index": learned_index,
                        "router_index": router_index,
                    }
                )
                continue

            if token == "layer":
                op_idx += 1
                self.layer_blocks.append(
                    SASRecStyleLayerBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        hidden_dropout_prob=dropout,
                        attn_dropout_prob=attn_dropout,
                        hidden_act=hidden_act,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append({"kind": "layer", "index": len(self.layer_blocks) - 1, "name": f"{op_idx:02d}_layer"})
                continue
            if token == "attn":
                op_idx += 1
                self.attn_blocks.append(
                    SASRecStyleAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        hidden_dropout_prob=dropout,
                        attn_dropout_prob=attn_dropout,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append({"kind": "attn", "index": len(self.attn_blocks) - 1, "name": f"{op_idx:02d}_attn"})
                continue
            if token == "ffn":
                op_idx += 1
                self.ffn_blocks.append(
                    SASRecStyleFFNBlock(
                        d_model=d_model,
                        d_ff=d_ff,
                        hidden_dropout_prob=dropout,
                        hidden_act=hidden_act,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append({"kind": "ffn", "index": len(self.ffn_blocks) - 1, "name": f"{op_idx:02d}_ffn"})
                continue

            stage_name = token.replace("_ffn", "")
            if token == stage_name:
                op_idx += 1
                self.stage_attn_blocks.append(
                    SASRecStyleAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        hidden_dropout_prob=dropout,
                        attn_dropout_prob=attn_dropout,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append(
                    {
                        "kind": "stage_attn",
                        "index": len(self.stage_attn_blocks) - 1,
                        "stage": stage_name,
                        "name": f"{op_idx:02d}_{token}.attn",
                    }
                )
            self._stage_counts[stage_name] += 1
            self.compiled_ops.append(
                {
                    "kind": "stage",
                    "stage": stage_name,
                    "stage_key": f"{stage_name}@{self._stage_counts[stage_name]}",
                    "name": f"{op_idx:02d}_{token}.ffn",
                }
            )

        self.requires_features = False
        self.supports_diagnostics = False
        for op in self.compiled_ops:
            if op["kind"] == "stage":
                block = self.stage_blocks[op["stage"]]
                self.requires_features = self.requires_features or bool(block.requires_features)
                self.supports_diagnostics = self.supports_diagnostics or bool(block.supports_diagnostics)
                continue
            if op["kind"] == "bundle":
                for ent in op["stage_entries"]:
                    block = self.stage_blocks[ent["stage"]]
                    self.requires_features = self.requires_features or bool(block.requires_features)
                    self.supports_diagnostics = self.supports_diagnostics or bool(block.supports_diagnostics)

        if self.reviewer_control_mode == "random_frozen":
            freeze_random_router_path(self.bundle_router_modules)
            for parameter in self.bundle_learned_logits:
                parameter.requires_grad_(False)

    def stage_n_experts(self) -> int:
        return max((int(self.stage_blocks[name].n_experts) for name in _STAGE_NAMES), default=0)

    def stage_expert_names(self) -> Dict[str, list[str]]:
        return {
            stage_name: list(self.stage_blocks[stage_name].expert_names)
            for stage_name in _STAGE_NAMES
            if self.stage_blocks[stage_name].supports_diagnostics
        }

    def stage_group_names(self) -> Dict[str, list[str]]:
        return {
            stage_name: list(self.stage_blocks[stage_name].group_names)
            for stage_name in _STAGE_NAMES
            if self.stage_blocks[stage_name].supports_diagnostics
        }

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        stage_temperatures: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        _ = alpha_scale
        stage_temperatures = dict(stage_temperatures or {})
        for stage_name in _STAGE_NAMES:
            self.stage_blocks[stage_name].set_schedule_state(
                alpha_scale=alpha_scale,
                router_temperature=stage_temperatures.get(stage_name),
                top_k=top_k,
            )

    def _merge_bundle(
        self,
        *,
        base_hidden: torch.Tensor,
        feat: Optional[torch.Tensor],
        branch_hiddens: list[torch.Tensor],
        agg: str,
        learned_index: Optional[int],
        router_index: Optional[int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not branch_hiddens:
            return base_hidden, None, None
        if len(branch_hiddens) == 1:
            ones = torch.ones(
                base_hidden.size(0),
                base_hidden.size(1),
                1,
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            )
            zeros = torch.zeros_like(ones)
            return branch_hiddens[0], ones, zeros

        deltas = torch.stack([h - base_hidden for h in branch_hiddens], dim=-2)

        if agg == "sum":
            merged = base_hidden + deltas.sum(dim=-2)
            return merged, None, None

        if agg == "mean":
            merged = base_hidden + deltas.mean(dim=-2)
            return merged, None, None

        if agg == "learned":
            if learned_index is None:
                raise RuntimeError("learned bundle op missing learned_index")
            raw = self.bundle_learned_logits[learned_index]
            w = F.softmax(raw, dim=-1)
            weights = w.view(1, 1, -1).expand(base_hidden.size(0), base_hidden.size(1), -1)
            logits = raw.view(1, 1, -1).expand_as(weights)
            merged = base_hidden + (weights.unsqueeze(-1) * deltas).sum(dim=-2)
            return merged, weights, logits

        if agg == "router":
            if router_index is None:
                raise RuntimeError("router bundle op missing router_index")
            if feat is None:
                feat_in = base_hidden.new_zeros(base_hidden.size(0), base_hidden.size(1), self._n_all_features)
            else:
                feat_in = feat
                if feat_in.size(-1) < self._n_all_features:
                    pad = feat_in.new_zeros(feat_in.size(0), feat_in.size(1), self._n_all_features - feat_in.size(-1))
                    feat_in = torch.cat([feat_in, pad], dim=-1)
                elif feat_in.size(-1) > self._n_all_features:
                    feat_in = feat_in[..., : self._n_all_features]
            weights, logits = self.bundle_router_modules[router_index](
                hidden=controlled_hidden_input(base_hidden, mode=self.reviewer_control_mode),
                feat=controlled_feature_input(feat_in, mode=self.reviewer_control_mode),
            )
            merged = base_hidden + (weights.unsqueeze(-1) * deltas).sum(dim=-2)
            return merged, weights, logits

        raise RuntimeError(f"Unsupported bundle agg: {agg}")

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        feat: Optional[torch.Tensor],
        item_seq_len: Optional[torch.Tensor],
        routing_item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, object]],
        Dict[str, Dict[str, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        out = hidden
        # Apply the reviewer intervention once at the executor boundary so no
        # cue path can bypass it (including bundle merge routers and future
        # feature-injection branches). Stage blocks repeat the same control as
        # a local fail-safe.
        controlled_feat = (
            None
            if feat is None
            else controlled_feature_input(feat, mode=self.reviewer_control_mode)
        )
        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, object]] = {}
        dense_aux: Dict[str, Dict[str, torch.Tensor]] = {}

        for op in self.compiled_ops:
            kind = op["kind"]
            if kind == "layer":
                out = self.layer_blocks[op["index"]](out, attention_mask)
                continue
            if kind == "attn":
                out = self.attn_blocks[op["index"]](out, attention_mask)
                continue
            if kind == "ffn":
                out = self.ffn_blocks[op["index"]](out)
                continue
            if kind == "stage_attn":
                out = self.stage_attn_blocks[op["index"]](out, attention_mask)
                continue

            if kind == "bundle":
                branch_hiddens = []
                for entry in op["stage_entries"]:
                    stage_key = entry["stage_key"]
                    next_hidden, weights, logits, stage_router_aux, stage_dense_aux = self.stage_blocks[entry["stage"]](
                        out,
                        controlled_feat,
                        item_seq_len=item_seq_len,
                        routing_item_seq_len=routing_item_seq_len,
                        alpha_override=self.global_residual_alpha,
                    )
                    branch_hiddens.append(next_hidden)
                    if weights is not None:
                        gate_weights[stage_key] = weights
                    if logits is not None:
                        gate_logits[stage_key] = logits
                    for aux_name, aux_tensor in dict(stage_router_aux or {}).items():
                        router_aux.setdefault(aux_name, {})[stage_key] = aux_tensor
                    if stage_dense_aux:
                        dense_aux[stage_key] = dict(stage_dense_aux)

                merged_hidden, merge_w, merge_l = self._merge_bundle(
                    base_hidden=out,
                    feat=controlled_feat,
                    branch_hiddens=branch_hiddens,
                    agg=str(op["agg"]),
                    learned_index=op.get("learned_index"),
                    router_index=op.get("router_index"),
                )
                out = merged_hidden
                if merge_w is not None:
                    bundle_key = str(op.get("name", "bundle"))
                    router_aux.setdefault("bundle_merge_weights", {})[bundle_key] = merge_w
                    if merge_l is not None:
                        router_aux.setdefault("bundle_merge_logits", {})[bundle_key] = merge_l
                    entropy = -(merge_w.clamp(min=1e-8) * merge_w.clamp(min=1e-8).log()).sum(dim=-1)
                    dense_aux[bundle_key] = {
                        "merge_weight_max": merge_w.max(dim=-1).values.detach(),
                        "merge_entropy": entropy.detach(),
                    }
                continue

            stage_key = op["stage_key"]
            next_hidden, weights, logits, stage_router_aux, stage_dense_aux = self.stage_blocks[op["stage"]](
                out,
                controlled_feat,
                item_seq_len=item_seq_len,
                routing_item_seq_len=routing_item_seq_len,
                alpha_override=self.global_residual_alpha,
            )
            out = next_hidden
            if weights is not None:
                gate_weights[stage_key] = weights
            if logits is not None:
                gate_logits[stage_key] = logits
            for aux_name, aux_tensor in dict(stage_router_aux or {}).items():
                router_aux.setdefault(aux_name, {})[stage_key] = aux_tensor
            if stage_dense_aux:
                dense_aux[stage_key] = dict(stage_dense_aux)

        return out, gate_weights, gate_logits, router_aux, dense_aux, None, None
