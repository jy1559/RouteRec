"""Stage modules for RouteRecV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..RouteRecBase.moe_stages import MoEStage
from ..RouteRecBase.transformer import TransformerEncoder
from .feature_config import STAGE_ALL_FEATURES


@dataclass
class StageRuntimeConfig:
    stage_name: str
    pass_layers: int
    moe_blocks: int
    d_model: int
    d_ff: int
    n_heads: int
    dropout: float
    d_feat_emb: int
    d_expert_hidden: int
    d_router_hidden: int
    expert_scale: int
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
    router_design: str
    group_top_k: int
    expert_top_k: int
    router_distill_enable: bool


class MoEStageV2(nn.Module):
    """Thin wrapper over v1 MoEStage to expose stage delta."""

    def __init__(self, cfg: StageRuntimeConfig):
        super().__init__()

        router_mode = "token"
        session_pooling = "query"
        router_temperature = 1.0
        router_feature_dropout = 0.0
        reliability_feature_name = None

        if cfg.stage_name == "macro":
            router_mode = cfg.macro_routing_scope
            session_pooling = cfg.macro_session_pooling
            router_temperature = 1.0
        elif cfg.stage_name == "mid":
            router_mode = "token"
            session_pooling = "query"
            router_temperature = cfg.mid_router_temperature
            router_feature_dropout = cfg.mid_router_feature_dropout
            reliability_feature_name = "mid_valid_r" if cfg.use_valid_ratio_gating else None
        elif cfg.stage_name == "micro":
            router_mode = "token"
            session_pooling = "query"
            router_temperature = cfg.micro_router_temperature
            router_feature_dropout = cfg.micro_router_feature_dropout
            reliability_feature_name = "mic_valid_r" if cfg.use_valid_ratio_gating else None

        self.stage = MoEStage(
            stage_name=cfg.stage_name,
            expert_feature_lists=cfg.expert_feature_lists,
            stage_all_features=STAGE_ALL_FEATURES[cfg.stage_name],
            col2idx=cfg.col2idx,
            d_model=cfg.d_model,
            d_feat_emb=cfg.d_feat_emb,
            d_expert_hidden=cfg.d_expert_hidden,
            d_router_hidden=cfg.d_router_hidden,
            expert_scale=cfg.expert_scale,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
            router_use_hidden=cfg.router_use_hidden,
            router_use_feature=cfg.router_use_feature,
            expert_use_hidden=cfg.expert_use_hidden,
            expert_use_feature=cfg.expert_use_feature,
            expert_names=cfg.expert_names,
            router_impl=cfg.router_impl,
            rule_router_cfg=cfg.rule_router_cfg,
            router_design=cfg.router_design,
            group_top_k=cfg.group_top_k,
            expert_top_k=cfg.expert_top_k,
            router_distill_enable=cfg.router_distill_enable,
            router_mode=router_mode,
            session_pooling=session_pooling,
            router_temperature=router_temperature,
            router_feature_dropout=router_feature_dropout,
            reliability_feature_name=reliability_feature_name,
        )

    @property
    def n_experts(self) -> int:
        return int(self.stage.n_experts)

    @property
    def expert_names(self) -> list[str]:
        return list(self.stage.expert_names)

    @property
    def group_names(self) -> list[str]:
        return list(self.stage.group_names)

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.stage.set_schedule_state(
            alpha_scale=alpha_scale,
            router_temperature=router_temperature,
            top_k=top_k,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        next_hidden, gate_w, gate_l = self.stage(hidden, feat, item_seq_len=item_seq_len)
        stage_delta = next_hidden - hidden
        return next_hidden, stage_delta, gate_w, gate_l, dict(self.stage.last_router_aux)


class StageBranchRunner(nn.Module):
    """Run one stage branch with explicit non-MoE pass and repeated MoE blocks."""

    def __init__(self, cfg: StageRuntimeConfig):
        super().__init__()
        self.stage_name = cfg.stage_name
        self.pass_layers = int(cfg.pass_layers)
        self.moe_blocks = int(cfg.moe_blocks)

        if self.pass_layers > 0:
            self.pass_transformer = TransformerEncoder(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=self.pass_layers,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                ffn_moe=False,
            )
        else:
            self.pass_transformer = None

        if self.moe_blocks > 0:
            self.moe_pre_blocks = nn.ModuleList(
                [
                    TransformerEncoder(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        n_layers=1,
                        d_ff=cfg.d_ff,
                        dropout=cfg.dropout,
                        ffn_moe=False,
                    )
                    for _ in range(self.moe_blocks)
                ]
            )
            self.stage_module = MoEStageV2(cfg)
        else:
            self.moe_pre_blocks = nn.ModuleList()
            self.stage_module = None

    @property
    def n_experts(self) -> int:
        if self.stage_module is None:
            return 0
        return self.stage_module.n_experts

    @property
    def expert_names(self) -> list[str]:
        if self.stage_module is None:
            return []
        return self.stage_module.expert_names

    @property
    def group_names(self) -> list[str]:
        if self.stage_module is None:
            return []
        return self.stage_module.group_names

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
            return hidden
        out, _ = self.pass_transformer(hidden, item_seq)
        return out

    def forward_serial(
        self,
        *,
        hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
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
        if self.stage_module is None:
            return out, weights, logits, router_aux

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            out, _ = pre_block(out, item_seq)
            out, _delta, w, l, aux = self.stage_module(out, feat, item_seq_len=item_seq_len)
            key = f"{self.stage_name}@{idx}"
            weights[key] = w
            logits[key] = l
            for aux_key, aux_tensor in aux.items():
                router_aux.setdefault(aux_key, {})[key] = aux_tensor
        return out, weights, logits, router_aux

    def forward_parallel(
        self,
        *,
        base_hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
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
        if self.stage_module is None:
            return out, (out - base_hidden), weights, logits, router_aux

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            out, _ = pre_block(out, item_seq)
            out, _delta, w, l, aux = self.stage_module(out, feat, item_seq_len=item_seq_len)
            key = f"{self.stage_name}@{idx}"
            weights[key] = w
            logits[key] = l
            for aux_key, aux_tensor in aux.items():
                router_aux.setdefault(aux_key, {})[key] = aux_tensor

        return out, (out - base_hidden), weights, logits, router_aux
