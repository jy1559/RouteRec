"""Lightweight RouteRecN model."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from ..RouteRecBase.logging_utils import MoELogger
from ..RouteRecBase.routers import load_balance_loss
from ..RouteRecBase.transformer import TransformerEncoder
from ..RouteRecV2.config_schema import ConfigResolver, parse_layout_catalog_from_config
from ..RouteRecV2.feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGES,
    feature_list_field,
    build_column_to_index,
)
from ..RouteRecV2.layout_schema import (
    LayoutSpec,
    parse_layout_catalog,
    stage_boundary_summary,
    total_stage_moe_blocks,
)
from ..RouteRecV2.losses import compute_expert_aux_loss, compute_stage_merge_aux_loss
from ..RouteRecV2.schedule import ScheduleController
from .feature_bank import SharedFeatureBank
from .stage_executor import StageExecutorN

logger = logging.getLogger(__name__)


def _is_none_like(value: Optional[object]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


class RouteRecN(SequentialRecommender):
    """Lightweight RouteRec_N: simple flat router + shared feature bank."""

    input_type = "point"

    @staticmethod
    def _parse_stage_int_map(raw_value, *, default_value: int) -> Dict[str, int]:
        out = {stage_name: int(default_value) for stage_name in ("macro", "mid", "micro")}
        if raw_value is None:
            return out
        if not isinstance(raw_value, dict):
            raise ValueError("stage int map must be a dict when provided.")
        for stage_name, value in raw_value.items():
            key = str(stage_name).lower().strip()
            if key not in out:
                raise ValueError(f"Unsupported stage key in stage int map: {stage_name}")
            out[key] = int(value)
        return out

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        resolver = ConfigResolver(config)
        resolver.assert_removed_keys()
        resolver.assert_embedding_only_dimension()
        if not _is_none_like(resolver.get("teacher_design", None)):
            raise ValueError("RouteRecN does not support 'teacher_design'.")
        if not _is_none_like(resolver.get("teacher_delivery", None)):
            raise ValueError("RouteRecN does not support 'teacher_delivery'.")
        if bool(resolver.get("router_distill_enable", False)):
            raise ValueError("RouteRecN does not support router_distill_enable.")
        if bool(resolver.get("routerec_v2_feature_spec_aux_enable", False)):
            raise ValueError("RouteRecN does not support feature specialization auxiliary loss.")

        self.n_items = dataset.item_num
        self.d_model = int(resolver.get("embedding_size", 128))
        self.d_feat_emb = int(resolver.get("d_feat_emb", 16))
        self.d_expert_hidden = int(resolver.get("d_expert_hidden", 128))
        self.d_router_hidden = int(resolver.get("d_router_hidden", 64))
        self.expert_scale = int(resolver.get("expert_scale", 1))
        if self.expert_scale < 1:
            raise ValueError(f"expert_scale must be >= 1, got {self.expert_scale}")

        self.n_heads = int(resolver.get("num_heads", 8))
        self.d_ff = int(resolver.get("d_ff", 0) or (2 * self.d_model))
        self.dropout = float(resolver.get("hidden_dropout_prob", 0.1))
        self.attn_dropout = float(resolver.get("attn_dropout_prob", self.dropout))
        self.max_seq_length = int(resolver.get("MAX_ITEM_LIST_LENGTH", 10))

        raw_catalog = parse_layout_catalog_from_config(resolver)
        self.layout_catalog = parse_layout_catalog(raw_catalog)
        self.layout_id = int(resolver.get("routerec_v2_layout_id", resolver.get("arch_layout_id", 0)))
        if not (0 <= self.layout_id < len(self.layout_catalog)):
            raise ValueError(
                f"routerec_v2_layout_id out of range: id={self.layout_id}, catalog_size={len(self.layout_catalog)}"
            )
        selected_layout = self.layout_catalog[self.layout_id]
        execution_override = str(resolver.get("routerec_stage_execution_mode", selected_layout.execution)).lower().strip()
        if execution_override not in {"serial", "parallel"}:
            raise ValueError(
                f"routerec_stage_execution_mode must be one of ['serial','parallel'], got {execution_override}"
            )
        self.layout = LayoutSpec(
            layout_id=selected_layout.layout_id,
            execution=execution_override,
            global_pre_layers=selected_layout.global_pre_layers,
            global_post_layers=selected_layout.global_post_layers,
            stages=selected_layout.stages,
        )

        self.router_use_hidden = bool(resolver.get("router_use_hidden", True))
        self.router_use_feature = bool(resolver.get("router_use_feature", True))
        self.expert_use_hidden = bool(resolver.get("expert_use_hidden", True))
        self.expert_use_feature = bool(resolver.get("expert_use_feature", True))
        self.router_impl = str(resolver.get("router_impl", "learned")).lower().strip()
        if self.router_impl not in {"learned", "rule_soft"}:
            raise ValueError(
                f"router_impl must be one of ['learned','rule_soft'], got {self.router_impl}"
            )
        raw_router_impl_by_stage = resolver.get("router_impl_by_stage", {}) or {}
        if not isinstance(raw_router_impl_by_stage, dict):
            raise ValueError("router_impl_by_stage must be a dict when provided.")
        self.router_impl_by_stage = {}
        for stage_name, impl_value in raw_router_impl_by_stage.items():
            key = str(stage_name).lower().strip()
            val = str(impl_value).lower().strip()
            if key not in {"macro", "mid", "micro"}:
                raise ValueError(f"Unsupported router_impl_by_stage key: {stage_name}")
            if val not in {"learned", "rule_soft"}:
                raise ValueError(
                    "router_impl_by_stage values must be one of ['learned','rule_soft'], "
                    f"got {impl_value}"
                )
            self.router_impl_by_stage[key] = val

        self.rule_router_cfg = resolver.get("rule_router", {}) or {}
        if not isinstance(self.rule_router_cfg, dict):
            raise ValueError("rule_router must be a dict when provided.")
        rule_variant = str(self.rule_router_cfg.get("variant", "ratio_bins")).lower().strip()
        if rule_variant != "ratio_bins":
            raise ValueError(
                "RouteRecN supports rule_router.variant='ratio_bins' only."
            )

        self.router_design = str(resolver.get("router_design", "simple_flat")).lower().strip()
        if self.router_design != "simple_flat":
            raise ValueError("RouteRecN only supports router_design='simple_flat'.")
        self.rule_bias_scale = float(resolver.get("rule_bias_scale", 0.0))
        self.use_valid_ratio_gating = bool(resolver.get("use_valid_ratio_gating", True))

        raw_top_k = resolver.get("moe_top_k", 1)
        self.moe_top_k = None if int(raw_top_k) <= 0 else int(raw_top_k)
        self.moe_top_k_policy = str(resolver.get("moe_top_k_policy", "fixed")).lower().strip()
        self.moe_top_k_ratio = float(resolver.get("moe_top_k_ratio", 0.5))
        self.moe_top_k_min = int(resolver.get("moe_top_k_min", 1))
        self.moe_top_k_start = int(resolver.get("moe_top_k_start", self.moe_top_k or 0))
        self.moe_top_k_warmup_until = resolver.get("moe_top_k_warmup_until", 0)

        self.routerec_schedule_enable = bool(resolver.get("routerec_schedule_enable", True))
        self.alpha_warmup_until = resolver.get("alpha_warmup_until", 0)
        self.alpha_warmup_start = float(resolver.get("alpha_warmup_start", 0.0))
        self.alpha_warmup_end = float(resolver.get("alpha_warmup_end", 1.0))
        self.temperature_warmup_until = resolver.get("temperature_warmup_until", 0.3)
        self.mid_router_temperature = float(resolver.get("mid_router_temperature", 1.2))
        self.micro_router_temperature = float(resolver.get("micro_router_temperature", 1.2))
        self.mid_router_temperature_start = float(
            resolver.get("mid_router_temperature_start", max(self.mid_router_temperature, 1.6))
        )
        self.micro_router_temperature_start = float(
            resolver.get("micro_router_temperature_start", max(self.micro_router_temperature, 1.6))
        )
        self.mid_router_feature_dropout = float(resolver.get("mid_router_feature_dropout", 0.0))
        self.micro_router_feature_dropout = float(resolver.get("micro_router_feature_dropout", 0.0))

        self.parallel_stage_gate_top_k = int(resolver.get("routerec_v2_parallel_stage_gate_top_k", 0))
        self.parallel_stage_gate_top_k = None if self.parallel_stage_gate_top_k <= 0 else self.parallel_stage_gate_top_k
        self.parallel_stage_gate_temperature = float(
            resolver.get("routerec_v2_parallel_stage_gate_temperature", 1.0)
        )

        self.use_aux_loss = bool(resolver.get("use_aux_loss", True))
        self.balance_loss_lambda = float(resolver.get("balance_loss_lambda", 0.002))
        self.stage_merge_aux_enable = bool(resolver.get("routerec_v2_stage_merge_aux_enable", False))
        self.stage_merge_aux_lambda_scale = float(resolver.get("routerec_v2_stage_merge_aux_lambda_scale", 1.0))
        self.z_loss_lambda = float(resolver.get("z_loss_lambda", 0.0))
        self.gate_entropy_lambda = float(resolver.get("gate_entropy_lambda", 0.0))
        self.gate_entropy_until = float(resolver.get("gate_entropy_until", 0.0))

        self.ffn_moe = bool(resolver.get("ffn_moe", False))
        self.n_ffn_experts = int(resolver.get("n_ffn_experts", 4))
        raw_ffn_top_k = int(resolver.get("ffn_top_k", 0))
        self.ffn_top_k = None if raw_ffn_top_k <= 0 else raw_ffn_top_k
        self.stage_inter_layer_style = str(resolver.get("stage_inter_layer_style", "attn")).lower().strip()
        if self.stage_inter_layer_style not in {"attn", "identity", "nonlinear", "ffn"}:
            raise ValueError(
                "stage_inter_layer_style must be one of ['attn','identity','nonlinear','ffn'], "
                f"got {self.stage_inter_layer_style}"
            )
        self.moe_block_variant = str(resolver.get("moe_block_variant", "moe")).lower().strip()
        if self.moe_block_variant not in {"moe", "dense_ffn", "nonlinear", "identity"}:
            raise ValueError(
                "moe_block_variant must be one of ['moe','dense_ffn','nonlinear','identity'], "
                f"got {self.moe_block_variant}"
            )
        self.router_group_feature_mode = str(resolver.get("router_group_feature_mode", "none")).lower().strip()
        if self.router_group_feature_mode not in {"none", "mean", "mean_std"}:
            raise ValueError(
                "router_group_feature_mode must be one of ['none','mean','mean_std'], "
                f"got {self.router_group_feature_mode}"
            )

        self.routerec_debug_logging = bool(resolver.get("routerec_debug_logging", False))
        self.routerec_special_logging = bool(resolver.get("routerec_special_logging", True))
        self.routerec_schedule_log_every_epoch = max(int(resolver.get("routerec_schedule_log_every_epoch", 1)), 1)
        self._last_logged_top_k = None
        self.history_input_mode = str(resolver.get("history_input_mode", "session_only")).lower().strip()
        self.current_session_item_length_field = str(
            resolver.get("current_session_item_length_field", "current_session_item_length")
            or "current_session_item_length"
        )
        self.routerec_session_router_context_scope = str(
            resolver.get("routerec_session_router_context_scope", "auto")
        ).lower().strip()
        if self.routerec_session_router_context_scope not in {"auto", "full_sequence", "current_session"}:
            raise ValueError(
                "routerec_session_router_context_scope must be one of ['auto','full_sequence','current_session'], "
                f"got {self.routerec_session_router_context_scope}"
            )

        self.feature_encoder_mode = str(resolver.get("feature_encoder_mode", "linear")).lower().strip()
        raw_patterns = resolver.get("feature_encoder_sinusoidal_features", None)
        if raw_patterns is None:
            raw_patterns = ["*time*", "*gap*", "*int*", "*pop*", "*valid_r*"]
        elif isinstance(raw_patterns, str):
            raw_patterns = [raw_patterns]
        elif not isinstance(raw_patterns, (list, tuple)):
            raise ValueError("feature_encoder_sinusoidal_features must be a list or string.")
        self.feature_encoder_sinusoidal_features = [str(v) for v in raw_patterns]
        self.feature_encoder_sinusoidal_n_freqs = int(resolver.get("feature_encoder_sinusoidal_n_freqs", 4))

        self.feature_fields = [feature_list_field(col) for col in ALL_FEATURE_COLUMNS]
        self.n_features = len(self.feature_fields)
        self.feature_encoder = SharedFeatureBank(
            feature_names=ALL_FEATURE_COLUMNS,
            mode=self.feature_encoder_mode,
            sinusoidal_patterns=self.feature_encoder_sinusoidal_features,
            sinusoidal_n_freqs=self.feature_encoder_sinusoidal_n_freqs,
        )

        self.item_embedding = nn.Embedding(self.n_items, self.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        self.input_ln = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(self.dropout)

        if self.layout.global_pre_layers > 0:
            self.pre_transformer = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.layout.global_pre_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                ffn_moe=False,
            )
        else:
            self.pre_transformer = None

        self.post_transformer = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.layout.global_post_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            ffn_moe=self.ffn_moe,
            n_ffn_experts=self.n_ffn_experts,
            ffn_top_k=self.ffn_top_k,
        )

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        stage_expert_lists = {stage_name: list(expert_dict.values()) for stage_name, expert_dict in STAGES}
        stage_expert_names = {stage_name: list(expert_dict.keys()) for stage_name, expert_dict in STAGES}
        expert_hidden_by_stage = self._parse_stage_int_map(
            resolver.get("expert_hidden_by_stage", {}),
            default_value=self.d_expert_hidden,
        )
        expert_depth_by_stage = self._parse_stage_int_map(
            resolver.get("expert_depth_by_stage", {}),
            default_value=1,
        )
        self.stage_executor = StageExecutorN(
            layout=self.layout,
            d_model=self.d_model,
            n_features=self.n_features,
            d_feat_emb=self.d_feat_emb,
            d_expert_hidden=self.d_expert_hidden,
            d_router_hidden=self.d_router_hidden,
            expert_depth_by_stage=expert_depth_by_stage,
            expert_hidden_by_stage=expert_hidden_by_stage,
            expert_scale=self.expert_scale,
            feature_bank_dim=self.feature_encoder.bank_dim,
            stage_top_k=self.moe_top_k,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            col2idx=col2idx,
            stage_expert_lists=stage_expert_lists,
            stage_expert_names=stage_expert_names,
            router_impl=self.router_impl,
            router_impl_by_stage=self.router_impl_by_stage,
            rule_router_cfg=self.rule_router_cfg,
            rule_bias_scale=self.rule_bias_scale,
            router_use_hidden=self.router_use_hidden,
            router_use_feature=self.router_use_feature,
            expert_use_hidden=self.expert_use_hidden,
            expert_use_feature=self.expert_use_feature,
            macro_routing_scope=str(resolver.get("macro_routing_scope", "session")).lower(),
            macro_session_pooling=str(resolver.get("macro_session_pooling", "mean")).lower(),
            mid_router_temperature=self.mid_router_temperature,
            micro_router_temperature=self.micro_router_temperature,
            mid_router_feature_dropout=self.mid_router_feature_dropout,
            micro_router_feature_dropout=self.micro_router_feature_dropout,
            use_valid_ratio_gating=self.use_valid_ratio_gating,
            parallel_stage_gate_top_k=self.parallel_stage_gate_top_k,
            parallel_stage_gate_temperature=self.parallel_stage_gate_temperature,
            inter_layer_style=self.stage_inter_layer_style,
            router_group_feature_mode=self.router_group_feature_mode,
            moe_block_variant=self.moe_block_variant,
        )

        self._stage_n_experts = self.stage_executor.stage_n_experts()
        self._stage_expert_names = self.stage_executor.stage_expert_names()
        self._stage_group_names = self.stage_executor.stage_group_names()
        self.moe_logger = MoELogger(self._stage_expert_names)
        self.group_logger = MoELogger(self._stage_group_names)
        self._reset_router_epoch_stats()

        self.schedule = ScheduleController(
            enable=self.routerec_schedule_enable,
            alpha_warmup_until=self.alpha_warmup_until,
            alpha_warmup_start=self.alpha_warmup_start,
            alpha_warmup_end=self.alpha_warmup_end,
            temperature_warmup_until=self.temperature_warmup_until,
            mid_router_temperature_start=self.mid_router_temperature_start,
            mid_router_temperature_end=self.mid_router_temperature,
            micro_router_temperature_start=self.micro_router_temperature_start,
            micro_router_temperature_end=self.micro_router_temperature,
            moe_top_k_policy=self.moe_top_k_policy,
            moe_top_k_fixed=self.moe_top_k,
            moe_top_k_ratio=self.moe_top_k_ratio,
            moe_top_k_min=self.moe_top_k_min,
            moe_top_k_start=self.moe_top_k_start,
            moe_top_k_warmup_until=self.moe_top_k_warmup_until,
        )

        self._schedule_epoch = 0
        # Keep the model schedule invariant across exact-continuation rungs.
        from routerec.lr_scheduler import resolve_schedule_total_epochs

        self._schedule_total_epochs = resolve_schedule_total_epochs(config)

        self.arch_layout_id = int(self.layout_id)
        self.n_pre_layer = int(self.layout.global_pre_layers)
        self.n_pre_macro = int(self.layout.stages.get("macro").pass_layers if "macro" in self.layout.stages else 0)
        self.n_pre_mid = int(self.layout.stages.get("mid").pass_layers if "mid" in self.layout.stages else 0)
        self.n_pre_micro = int(self.layout.stages.get("micro").pass_layers if "micro" in self.layout.stages else 0)
        self.n_post_layer = int(self.layout.global_post_layers)
        self.n_total_attn_layers = (
            self.n_pre_layer
            + self.n_post_layer
            + sum(int(spec.pass_layers + spec.moe_blocks) for spec in self.layout.stages.values())
        )
        self.num_layers = int(self.n_total_attn_layers)

        try:
            config["routerec_v2_layout_id"] = int(self.layout_id)
            config["routerec_stage_execution_mode"] = self.layout.execution
            config["routerec_v2_layout_summary"] = stage_boundary_summary(self.layout)
            config["router_impl"] = self.router_impl
            config["router_impl_by_stage"] = dict(self.router_impl_by_stage)
            config["router_design"] = self.router_design
            config["rule_bias_scale"] = float(self.rule_bias_scale)
            config["feature_encoder_mode"] = self.feature_encoder_mode
            config["feature_encoder_sinusoidal_features"] = list(self.feature_encoder_sinusoidal_features)
            config["feature_encoder_sinusoidal_n_freqs"] = int(self.feature_encoder_sinusoidal_n_freqs)
            config["routerec_special_logging"] = bool(self.routerec_special_logging)
            config["stage_inter_layer_style"] = str(self.stage_inter_layer_style)
            config["moe_block_variant"] = str(self.moe_block_variant)
            config["router_group_feature_mode"] = str(self.router_group_feature_mode)
            config["z_loss_lambda"] = float(self.z_loss_lambda)
            config["gate_entropy_lambda"] = float(self.gate_entropy_lambda)
            config["gate_entropy_until"] = float(self.gate_entropy_until)
            config["expert_hidden_by_stage"] = dict(expert_hidden_by_stage)
            config["expert_depth_by_stage"] = dict(expert_depth_by_stage)
        except Exception:
            pass

        self.set_schedule_epoch(epoch_idx=0, max_epochs=self._schedule_total_epochs, log_now=True)
        self.apply(self._init_weights)

        logger.info(
            "RouteRecN init: layout_id=%s execution=%s boundaries=%s total_moe_blocks=%s "
            "router_impl=%s router_impl_by_stage=%s rule_bias_scale=%.4f feature_encoder_mode=%s "
            "feature_bank_dim=%s expert_scale=%s top_k=%s balance_lambda=%.6f "
            "z_loss_lambda=%.6f gate_entropy_lambda=%.6f gate_entropy_until=%.4f "
            "inter_layer_style=%s moe_block_variant=%s router_group_feature_mode=%s special_logging=%s",
            self.layout_id,
            self.layout.execution,
            stage_boundary_summary(self.layout),
            total_stage_moe_blocks(self.layout),
            self.router_impl,
            self.router_impl_by_stage,
            self.rule_bias_scale,
            self.feature_encoder_mode,
            self.feature_encoder.bank_dim,
            self.expert_scale,
            self.moe_top_k,
            self.balance_loss_lambda,
            self.z_loss_lambda,
            self.gate_entropy_lambda,
            self.gate_entropy_until,
            self.stage_inter_layer_style,
            self.moe_block_variant,
            self.router_group_feature_mode,
            self.routerec_special_logging,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _gather_features(self, interaction) -> torch.Tensor:
        feat_list = []
        item_seq = interaction[self.ITEM_SEQ]
        batch_size, seq_len = item_seq.shape
        for field in self.feature_fields:
            if field in interaction:
                feat_list.append(interaction[field].float())
            else:
                feat_list.append(torch.zeros(batch_size, seq_len, device=item_seq.device))
                logger.warning("Feature field '%s' not found - using zeros.", field)
        return torch.stack(feat_list, dim=-1)

    @staticmethod
    def _sequence_valid_mask(item_seq_len: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        lens = item_seq_len.to(device=device).long().clamp(min=1, max=seq_len)
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        return arange < lens.unsqueeze(1)

    def _use_current_session_router_context(self) -> bool:
        if self.routerec_session_router_context_scope == "current_session":
            return True
        if self.routerec_session_router_context_scope == "full_sequence":
            return False
        return self.history_input_mode == "full_history_session_targets"

    def _resolve_routing_item_seq_len(
        self,
        interaction,
        item_seq_len: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        if not self._use_current_session_router_context():
            return item_seq_len

        field_name = self.current_session_item_length_field
        if field_name not in interaction:
            return item_seq_len

        routing_item_seq_len = interaction[field_name]
        if torch.is_tensor(routing_item_seq_len):
            routing_item_seq_len = routing_item_seq_len.to(device=item_seq_len.device)
        else:
            routing_item_seq_len = torch.as_tensor(routing_item_seq_len, device=item_seq_len.device)
        return routing_item_seq_len.long().clamp(min=1, max=seq_len)

    def _resolve_forward_lengths(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        routing_item_seq_len = self._resolve_routing_item_seq_len(
            interaction,
            item_seq_len,
            item_seq.size(1),
        )
        return item_seq, item_seq_len, routing_item_seq_len

    def _aux_until_active(self, until: float) -> float:
        limit = float(until or 0.0)
        if limit <= 0:
            return 0.0
        epoch_num = float(self._schedule_epoch + 1)
        if limit <= 1.0:
            cutoff = max(int(round(limit * max(self._schedule_total_epochs, 1))), 1)
            return 1.0 if epoch_num <= cutoff else 0.0
        return 1.0 if epoch_num <= limit else 0.0

    def _compute_router_z_loss(self, gate_logits: Dict[str, torch.Tensor], item_seq_len: torch.Tensor) -> torch.Tensor:
        if not gate_logits:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for logits in gate_logits.values():
            mask = self._sequence_valid_mask(item_seq_len, logits.size(1), logits.device).float()
            z = torch.logsumexp(logits, dim=-1).pow(2)
            denom = mask.sum().clamp(min=1.0)
            losses.append((z * mask).sum() / denom)
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_gate_entropy_reg(self, gate_weights: Dict[str, torch.Tensor], item_seq_len: torch.Tensor) -> torch.Tensor:
        if not gate_weights:
            return self.item_embedding.weight.new_tensor(0.0)
        entropies = []
        for weights in gate_weights.values():
            mask = self._sequence_valid_mask(item_seq_len, weights.size(1), weights.device).float()
            entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(dim=-1)
            denom = mask.sum().clamp(min=1.0)
            entropies.append((entropy * mask).sum() / denom)
        return torch.stack(entropies).mean() if entropies else self.item_embedding.weight.new_tensor(0.0)

    def _reset_router_epoch_stats(self) -> None:
        self._router_group_entropy_sum = defaultdict(float)
        self._router_group_entropy_count = defaultdict(int)

    def set_schedule_epoch(self, epoch_idx: int, max_epochs: Optional[int] = None, log_now: bool = False) -> None:
        self._schedule_epoch = max(int(epoch_idx), 0)
        if max_epochs is not None:
            self._schedule_total_epochs = max(int(max_epochs), 1)

        state = self.schedule.resolve(
            epoch_idx=self._schedule_epoch,
            total_epochs=self._schedule_total_epochs,
            n_experts=self._stage_n_experts,
        )
        runtime_top_k = -1 if state.stage_top_k is None else int(state.stage_top_k)
        self.stage_executor.set_schedule_state(
            alpha_scale=state.alpha_scale,
            stage_temperatures={
                "mid": state.mid_router_temperature,
                "micro": state.micro_router_temperature,
            },
            top_k=runtime_top_k,
        )

        should_log = log_now or (self._schedule_epoch % self.routerec_schedule_log_every_epoch == 0)
        if self._last_logged_top_k != state.stage_top_k:
            should_log = True
        if should_log:
            logger.info(
                "RouteRec_N schedule epoch=%s/%s alpha_scale=%.4f mid_temp=%.4f micro_temp=%.4f top_k=%s",
                self._schedule_epoch + 1,
                self._schedule_total_epochs,
                state.alpha_scale,
                state.mid_router_temperature,
                state.micro_router_temperature,
                ("dense" if state.stage_top_k is None else state.stage_top_k),
            )
            self._last_logged_top_k = state.stage_top_k

    def forward(self, item_seq, item_seq_len, feat=None, routing_item_seq_len=None):
        batch_size, seq_len = item_seq.shape

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)

        if self.pre_transformer is not None:
            tokens, _ = self.pre_transformer(tokens, item_seq)

        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        stage_merge_weights = None
        stage_merge_logits = None

        if feat is not None:
            feat_bank = self.feature_encoder(feat)
            tokens, gate_weights, gate_logits, router_aux, stage_merge_weights, stage_merge_logits = self.stage_executor(
                hidden=tokens,
                item_seq=item_seq,
                feat=feat,
                feat_bank=feat_bank,
                item_seq_len=item_seq_len,
                routing_item_seq_len=routing_item_seq_len,
            )

        hidden, ffn_moe_weights = self.post_transformer(tokens, item_seq)
        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = hidden.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "stage_merge_weights": stage_merge_weights,
            "stage_merge_logits": stage_merge_logits,
            "ffn_moe_weights": ffn_moe_weights,
            "router_aux": router_aux,
        }
        return seq_output, aux_data

    def calculate_loss(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, aux_data = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )
        logits = seq_output @ self.item_embedding.weight.T
        ce_loss = F.cross_entropy(logits, pos_items)

        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.use_aux_loss and self.balance_loss_lambda > 0:
            aux_loss = aux_loss + compute_expert_aux_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len=item_seq_len,
                balance_lambda=self.balance_loss_lambda,
                device=ce_loss.device,
            )
            aux_loss = aux_loss + compute_stage_merge_aux_loss(
                aux_data.get("stage_merge_weights"),
                item_seq_len=item_seq_len,
                balance_lambda=self.balance_loss_lambda,
                enabled=self.stage_merge_aux_enable,
                scale=self.stage_merge_aux_lambda_scale,
                device=ce_loss.device,
            )
            if self.ffn_moe and aux_data.get("ffn_moe_weights"):
                for stage_weights in aux_data["ffn_moe_weights"].values():
                    aux_loss = aux_loss + self.balance_loss_lambda * load_balance_loss(
                        stage_weights,
                        self.n_ffn_experts,
                    )
        if self.z_loss_lambda > 0:
            aux_loss = aux_loss + self.z_loss_lambda * self._compute_router_z_loss(
                aux_data.get("gate_logits", {}),
                item_seq_len,
            )
        entropy_scale = self._aux_until_active(self.gate_entropy_until)
        if self.gate_entropy_lambda > 0 and entropy_scale > 0:
            aux_loss = aux_loss - (self.gate_entropy_lambda * entropy_scale) * self._compute_gate_entropy_reg(
                aux_data.get("gate_weights", {}),
                item_seq_len,
            )

        if self.routerec_debug_logging and self.training and aux_data.get("gate_weights"):
            self.moe_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                item_seq_len=item_seq_len,
            )

        return ce_loss + aux_loss

    def predict(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)
        test_item = interaction[self.ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )
        test_item_emb = self.item_embedding(test_item)
        return (seq_output * test_item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )
        return seq_output @ self.item_embedding.weight.T

    def get_epoch_log_summary(self) -> Dict:
        summary = self.moe_logger.get_and_reset()
        summary["group_stages"] = {}
        summary["router"] = {}
        self._reset_router_epoch_stats()
        return summary
