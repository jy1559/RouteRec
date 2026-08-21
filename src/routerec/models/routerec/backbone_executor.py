"""Stage executor shared by the RouteRec backbone."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone_stages import StageBranchRunner, StageRuntimeConfig
from .layout import LayoutSpec
from .merge_router import StageMergeRouter


_STAGE_NAMES = ("macro", "mid", "micro")


class BackboneStageExecutor(nn.Module):
    """Execute RouteRec backbone branches in serial or parallel mode."""

    def __init__(
        self,
        *,
        layout: LayoutSpec,
        d_model: int,
        n_features: int,
        d_feat_emb: int,
        d_expert_hidden: int,
        d_router_hidden: int,
        expert_depth_by_stage: Dict[str, int],
        expert_hidden_by_stage: Dict[str, int],
        expert_scale: int,
        feature_bank_dim: int,
        stage_top_k: Optional[int],
        dropout: float,
        attn_dropout: float,
        n_heads: int,
        d_ff: int,
        col2idx: Dict[str, int],
        stage_expert_lists: Dict[str, list[list[str]]],
        stage_expert_names: Dict[str, list[str]],
        router_impl: str,
        router_impl_by_stage: Optional[Dict[str, str]],
        rule_router_cfg: Optional[Dict[str, Any]],
        rule_bias_scale: float,
        router_use_hidden: bool,
        router_use_feature: bool,
        expert_use_hidden: bool,
        expert_use_feature: bool,
        macro_routing_scope: str,
        macro_session_pooling: str,
        mid_router_temperature: float,
        micro_router_temperature: float,
        mid_router_feature_dropout: float,
        micro_router_feature_dropout: float,
        use_valid_ratio_gating: bool,
        parallel_stage_gate_top_k: Optional[int],
        parallel_stage_gate_temperature: float,
        inter_layer_style: str,
        router_group_feature_mode: str,
        moe_block_variant: str,
    ):
        super().__init__()
        self.execution = layout.execution
        self.parallel_stage_gate_temperature = float(parallel_stage_gate_temperature)
        self.stage_order = [stage_name for stage_name in _STAGE_NAMES if stage_name in layout.stages]
        self.router_impl = str(router_impl).lower().strip()
        self.router_impl_by_stage = {
            str(key): str(val).lower().strip()
            for key, val in dict(router_impl_by_stage or {}).items()
        }
        self.rule_router_cfg = dict(rule_router_cfg or {})
        self.rule_bias_scale = float(rule_bias_scale)

        self.branches = nn.ModuleDict()
        for stage_name in self.stage_order:
            spec = layout.stages[stage_name]
            stage_router_impl = self.router_impl_by_stage.get(stage_name, self.router_impl)
            cfg = StageRuntimeConfig(
                stage_name=stage_name,
                pass_layers=spec.pass_layers,
                moe_blocks=spec.moe_blocks,
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                d_expert_hidden=int(expert_hidden_by_stage.get(stage_name, d_expert_hidden)),
                d_router_hidden=d_router_hidden,
                expert_depth=int(expert_depth_by_stage.get(stage_name, 1)),
                expert_scale=expert_scale,
                feature_bank_dim=feature_bank_dim,
                top_k=stage_top_k,
                router_use_hidden=router_use_hidden,
                router_use_feature=router_use_feature,
                expert_use_hidden=expert_use_hidden,
                expert_use_feature=expert_use_feature,
                macro_routing_scope=macro_routing_scope,
                macro_session_pooling=macro_session_pooling,
                mid_router_temperature=mid_router_temperature,
                micro_router_temperature=micro_router_temperature,
                mid_router_feature_dropout=mid_router_feature_dropout,
                micro_router_feature_dropout=micro_router_feature_dropout,
                use_valid_ratio_gating=use_valid_ratio_gating,
                col2idx=col2idx,
                expert_feature_lists=stage_expert_lists[stage_name],
                expert_names=stage_expert_names[stage_name],
                router_impl=stage_router_impl,
                rule_router_cfg=self.rule_router_cfg,
                rule_bias_scale=self.rule_bias_scale,
                inter_layer_style=str(inter_layer_style),
                router_group_feature_mode=str(router_group_feature_mode),
                moe_block_variant=str(moe_block_variant),
            )
            self.branches[stage_name] = StageBranchRunner(cfg)

        if self.execution == "parallel":
            self.merge_stage_names = [stage_name for stage_name in self.stage_order if self._branch_has_effect(stage_name)]
            if len(self.merge_stage_names) >= 2:
                self.stage_merge_router = StageMergeRouter(
                    n_stages=len(self.merge_stage_names),
                    n_features=n_features,
                    d_model=d_model,
                    d_feat_emb=d_feat_emb,
                    d_router_hidden=d_router_hidden,
                    dropout=dropout,
                    top_k=parallel_stage_gate_top_k,
                    use_hidden=router_use_hidden,
                    use_feature=router_use_feature,
                )
            else:
                self.stage_merge_router = None
        else:
            self.merge_stage_names = []
            self.stage_merge_router = None

    def _branch_has_effect(self, stage_name: str) -> bool:
        branch = self.branches[stage_name]
        return bool(branch.pass_layers > 0 or branch.moe_blocks > 0)

    def stage_n_experts(self) -> int:
        for stage_name in self.stage_order:
            n_experts = self.branches[stage_name].n_experts
            if n_experts > 0:
                return int(n_experts)
        return 0

    def stage_expert_names(self) -> Dict[str, list[str]]:
        return {
            stage_name: list(self.branches[stage_name].expert_names)
            for stage_name in self.stage_order
            if self.branches[stage_name].expert_names
        }

    def stage_group_names(self) -> Dict[str, list[str]]:
        return {
            stage_name: list(self.branches[stage_name].group_names)
            for stage_name in self.stage_order
            if self.branches[stage_name].group_names
        }

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        stage_temperatures: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        stage_temperatures = stage_temperatures or {}
        for stage_name in self.stage_order:
            self.branches[stage_name].set_schedule_state(
                alpha_scale=alpha_scale,
                router_temperature=stage_temperatures.get(stage_name),
                top_k=top_k,
            )

    def _merge_parallel(
        self,
        *,
        base_hidden: torch.Tensor,
        feat: torch.Tensor,
        stage_deltas: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        active_names = [stage_name for stage_name in self.merge_stage_names if stage_name in stage_deltas]
        if not active_names:
            return base_hidden, None, None

        if len(active_names) == 1:
            stage_name = active_names[0]
            ones = torch.ones(
                base_hidden.shape[0],
                base_hidden.shape[1],
                1,
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            )
            zeros = torch.zeros_like(ones)
            return base_hidden + stage_deltas[stage_name], ones, zeros

        if self.stage_merge_router is None:
            stacked = torch.stack([stage_deltas[name] for name in active_names], dim=-2)
            weights = torch.full(
                (base_hidden.shape[0], base_hidden.shape[1], len(active_names)),
                fill_value=1.0 / float(len(active_names)),
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            )
            logits = torch.log(weights.clamp(min=1e-8))
            merged_delta = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
            return base_hidden + merged_delta, weights, logits

        weights, logits = self.stage_merge_router(
            hidden=base_hidden,
            feat=feat,
            temperature=self.parallel_stage_gate_temperature,
            top_k=self.stage_merge_router.top_k,
        )

        stacked = torch.stack([stage_deltas[name] for name in active_names], dim=-2)
        merged_delta = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return base_hidden + merged_delta, weights, logits

    def forward(
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
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.execution == "serial":
            out = hidden
            for stage_name in self.stage_order:
                out, w, l, aux = self.branches[stage_name].forward_serial(
                    hidden=out,
                    item_seq=item_seq,
                    feat=feat,
                    feat_bank=feat_bank,
                    item_seq_len=item_seq_len,
                    routing_item_seq_len=routing_item_seq_len,
                )
                gate_weights.update(w)
                gate_logits.update(l)
                for aux_key, aux_map in aux.items():
                    router_aux.setdefault(aux_key, {}).update(aux_map)
            return out, gate_weights, gate_logits, router_aux, None, None

        base_hidden = hidden
        stage_deltas: Dict[str, torch.Tensor] = {}
        for stage_name in self.stage_order:
            _stage_out, delta, w, l, aux = self.branches[stage_name].forward_parallel(
                base_hidden=base_hidden,
                item_seq=item_seq,
                feat=feat,
                feat_bank=feat_bank,
                item_seq_len=item_seq_len,
                routing_item_seq_len=routing_item_seq_len,
            )
            if delta.abs().sum().item() > 0:
                stage_deltas[stage_name] = delta
            gate_weights.update(w)
            gate_logits.update(l)
            for aux_key, aux_map in aux.items():
                router_aux.setdefault(aux_key, {}).update(aux_map)

        out, stage_merge_weights, stage_merge_logits = self._merge_parallel(
            base_hidden=base_hidden,
            feat=feat,
            stage_deltas=stage_deltas,
        )
        return out, gate_weights, gate_logits, router_aux, stage_merge_weights, stage_merge_logits
