"""RouteRecV2 model.

Highlights:
- Object-based layout with explicit non-MoE pass / MoE repeat boundaries.
- Stage execution modes: serial, parallel, and parallel+repeat via layout moe_blocks.
- Optional parallel stage-merge auxiliary loss.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from .config_schema import ConfigResolver, parse_layout_catalog_from_config
from .feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGES,
    STAGE_ALL_FEATURES,
    feature_list_field,
    build_column_to_index,
)
from .layout_schema import LayoutSpec, parse_layout_catalog, stage_boundary_summary, total_stage_moe_blocks
from .losses import (
    compute_expert_aux_loss,
    compute_feature_specialization_aux_loss,
    compute_router_distill_aux_loss,
    compute_stage_merge_aux_loss,
)
from .schedule import ScheduleController
from .stage_executor import StageExecutorV2
from ..RouteRecBase.logging_utils import MoELogger
from ..RouteRecBase.routers import load_balance_loss
from ..RouteRecBase.transformer import TransformerEncoder

logger = logging.getLogger(__name__)


class RouteRecBase_V2(SequentialRecommender):
    """RouteRecV2: boundary-explicit Stage-MoE sequential recommender."""

    input_type = "point"

    @staticmethod
    def _parse_stage_list(raw_value, default: list[str]) -> list[str]:
        if raw_value is None:
            return list(default)
        if isinstance(raw_value, (list, tuple)):
            vals = [str(v).strip().lower() for v in raw_value if str(v).strip()]
            return vals if vals else list(default)
        if isinstance(raw_value, str):
            txt = raw_value.strip()
            if txt.startswith("[") and txt.endswith("]"):
                txt = txt[1:-1]
            vals = [tok.strip().lower() for tok in txt.split(",") if tok.strip()]
            return vals if vals else list(default)
        return list(default)

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        resolver = ConfigResolver(config)
        resolver.assert_removed_keys()
        resolver.assert_embedding_only_dimension()

        # --- Core dimensions ---
        self.n_items = dataset.item_num
        self.d_model = int(resolver.get("embedding_size", 128))
        self.d_feat_emb = int(resolver.get("d_feat_emb", 16))
        self.d_expert_hidden = int(resolver.get("d_expert_hidden", 256))
        self.d_router_hidden = int(resolver.get("d_router_hidden", 64))
        self.expert_scale = int(resolver.get("expert_scale", 3))
        if self.expert_scale < 1:
            raise ValueError(f"expert_scale must be >=1, got {self.expert_scale}")

        self.n_heads = int(resolver.get("num_heads", 8))
        self.d_ff = int(resolver.get("d_ff", 0) or (4 * self.d_model))
        self.dropout = float(resolver.get("hidden_dropout_prob", 0.1))

        self.max_seq_length = int(resolver.get("MAX_ITEM_LIST_LENGTH", 10))

        # --- Layout ---
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
        if execution_override != selected_layout.execution:
            logger.warning(
                "Layout execution override applied: layout.execution=%s -> routerec_stage_execution_mode=%s",
                selected_layout.execution,
                execution_override,
            )
        self.layout = LayoutSpec(
            layout_id=selected_layout.layout_id,
            execution=execution_override,
            global_pre_layers=selected_layout.global_pre_layers,
            global_post_layers=selected_layout.global_post_layers,
            stages=selected_layout.stages,
        )

        # --- MoE toggles ---
        self.router_use_hidden = bool(resolver.get("router_use_hidden", True))
        self.router_use_feature = bool(resolver.get("router_use_feature", True))
        self.expert_use_hidden = bool(resolver.get("expert_use_hidden", True))
        self.expert_use_feature = bool(resolver.get("expert_use_feature", True))
        self.router_impl = str(resolver.get("router_impl", "learned")).lower().strip()
        if self.router_impl not in {"learned", "rule_soft"}:
            raise ValueError(
                f"router_impl must be one of ['learned','rule_soft'], got {self.router_impl}"
            )
        raw_router_impl_by_stage = resolver.get("router_impl_by_stage", {})
        if raw_router_impl_by_stage is None:
            raw_router_impl_by_stage = {}
        if not isinstance(raw_router_impl_by_stage, dict):
            raise ValueError("router_impl_by_stage must be a dict when provided.")
        self.router_impl_by_stage = {}
        for stage_key, impl_value in raw_router_impl_by_stage.items():
            stage_name = str(stage_key).lower().strip()
            impl_name = str(impl_value).lower().strip()
            if stage_name not in {"macro", "mid", "micro"}:
                raise ValueError(
                    f"router_impl_by_stage has unsupported stage key: {stage_key}"
                )
            if impl_name not in {"learned", "rule_soft"}:
                raise ValueError(
                    "router_impl_by_stage values must be one of ['learned','rule_soft'], "
                    f"got {impl_value} for stage {stage_key}"
                )
            self.router_impl_by_stage[stage_name] = impl_name
        self.rule_router_cfg = resolver.get("rule_router", {}) or {}
        if not isinstance(self.rule_router_cfg, dict):
            raise ValueError("rule_router must be a dict when provided.")
        self.router_design = str(
            resolver.get("router_design", "group_factorized_interaction")
        ).lower().strip()
        if self.router_design not in {"flat_legacy", "group_factorized_interaction"}:
            raise ValueError(
                "router_design must be one of ['flat_legacy','group_factorized_interaction'], "
                f"got {self.router_design}"
            )
        self.group_top_k = int(resolver.get("group_top_k", 0))
        self.expert_top_k = int(resolver.get("expert_top_k", 1))
        self.router_distill_enable = bool(resolver.get("router_distill_enable", False))
        self.router_distill_lambda = float(resolver.get("router_distill_lambda", 0.0))
        self.router_distill_temperature = float(resolver.get("router_distill_temperature", 1.5))
        self.router_distill_until = float(resolver.get("router_distill_until", 0.2))

        if self.router_impl == "learned" and not (self.router_use_hidden or self.router_use_feature):
            raise ValueError("router_use_hidden and router_use_feature cannot both be false")
        if not (self.expert_use_hidden or self.expert_use_feature):
            raise ValueError("expert_use_hidden and expert_use_feature cannot both be false")

        # --- Routing / schedule ---
        raw_top_k = resolver.get("moe_top_k", 0)
        self.moe_top_k = None if int(raw_top_k) <= 0 else int(raw_top_k)
        self.moe_top_k_policy = str(resolver.get("moe_top_k_policy", "auto")).lower().strip()
        self.moe_top_k_ratio = float(resolver.get("moe_top_k_ratio", 0.5))
        self.moe_top_k_min = int(resolver.get("moe_top_k_min", 1))
        self.moe_top_k_start = int(resolver.get("moe_top_k_start", 0))
        self.moe_top_k_warmup_until = resolver.get("moe_top_k_warmup_until", 0)

        self.routerec_schedule_enable = bool(resolver.get("routerec_schedule_enable", False))
        self.alpha_warmup_until = resolver.get("alpha_warmup_until", 0)
        self.alpha_warmup_start = float(resolver.get("alpha_warmup_start", 0.0))
        self.alpha_warmup_end = float(resolver.get("alpha_warmup_end", 1.0))
        self.temperature_warmup_until = resolver.get("temperature_warmup_until", 0)
        self.mid_router_temperature = float(resolver.get("mid_router_temperature", 1.3))
        self.micro_router_temperature = float(resolver.get("micro_router_temperature", 1.3))
        self.mid_router_temperature_start = float(
            resolver.get("mid_router_temperature_start", self.mid_router_temperature)
        )
        self.micro_router_temperature_start = float(
            resolver.get("micro_router_temperature_start", self.micro_router_temperature)
        )

        self.mid_router_feature_dropout = float(resolver.get("mid_router_feature_dropout", 0.1))
        self.micro_router_feature_dropout = float(resolver.get("micro_router_feature_dropout", 0.1))
        self.use_valid_ratio_gating = bool(resolver.get("use_valid_ratio_gating", True))

        self.parallel_stage_gate_top_k = int(resolver.get("routerec_v2_parallel_stage_gate_top_k", 0))
        self.parallel_stage_gate_top_k = None if self.parallel_stage_gate_top_k <= 0 else self.parallel_stage_gate_top_k
        self.parallel_stage_gate_temperature = float(
            resolver.get("routerec_v2_parallel_stage_gate_temperature", 1.0)
        )

        # --- Loss ---
        self.use_aux_loss = bool(resolver.get("use_aux_loss", True))
        self.balance_loss_lambda = float(resolver.get("balance_loss_lambda", 0.003))
        self.stage_merge_aux_enable = bool(resolver.get("routerec_v2_stage_merge_aux_enable", False))
        self.stage_merge_aux_lambda_scale = float(resolver.get("routerec_v2_stage_merge_aux_lambda_scale", 1.0))
        self.feature_spec_aux_enable = bool(resolver.get("routerec_v2_feature_spec_aux_enable", True))
        self.feature_spec_aux_lambda = float(resolver.get("routerec_v2_feature_spec_aux_lambda", 3e-4))
        self.feature_spec_aux_stages = self._parse_stage_list(
            resolver.get("routerec_v2_feature_spec_stages", ["mid"]),
            default=["mid"],
        )
        self.feature_spec_aux_min_tokens = float(resolver.get("routerec_v2_feature_spec_min_tokens", 8))

        # --- Optional FFN-MoE ---
        self.ffn_moe = bool(resolver.get("ffn_moe", False))
        self.n_ffn_experts = int(resolver.get("n_ffn_experts", 4))
        raw_ffn_top_k = int(resolver.get("ffn_top_k", 0))
        self.ffn_top_k = None if raw_ffn_top_k <= 0 else raw_ffn_top_k

        # --- Logging ---
        self.routerec_debug_logging = bool(resolver.get("routerec_debug_logging", False))
        self.routerec_schedule_log_every_epoch = max(int(resolver.get("routerec_schedule_log_every_epoch", 1)), 1)
        self._last_logged_top_k = None

        # --- Embeddings ---
        self.item_embedding = nn.Embedding(self.n_items, self.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        self.input_ln = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(self.dropout)

        # --- Global pre/post ---
        if self.layout.global_pre_layers > 0:
            self.pre_transformer = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.layout.global_pre_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
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
            ffn_moe=self.ffn_moe,
            n_ffn_experts=self.n_ffn_experts,
            ffn_top_k=self.ffn_top_k,
        )

        # --- Features ---
        self.feature_fields = [feature_list_field(col) for col in ALL_FEATURE_COLUMNS]
        self.n_features = len(self.feature_fields)

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.stage_feature_indices = {
            stage_name: [int(col2idx[f]) for f in feat_names if f in col2idx]
            for stage_name, feat_names in STAGE_ALL_FEATURES.items()
        }
        stage_expert_lists = {stage_name: list(expert_dict.values()) for stage_name, expert_dict in STAGES}
        stage_expert_names = {stage_name: list(expert_dict.keys()) for stage_name, expert_dict in STAGES}

        self.stage_executor = StageExecutorV2(
            layout=self.layout,
            d_model=self.d_model,
            n_features=self.n_features,
            d_feat_emb=self.d_feat_emb,
            d_expert_hidden=self.d_expert_hidden,
            d_router_hidden=self.d_router_hidden,
            expert_scale=self.expert_scale,
            stage_top_k=self.moe_top_k,
            dropout=self.dropout,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            col2idx=col2idx,
            stage_expert_lists=stage_expert_lists,
            stage_expert_names=stage_expert_names,
            router_impl=self.router_impl,
            router_impl_by_stage=self.router_impl_by_stage,
            rule_router_cfg=self.rule_router_cfg,
            router_design=self.router_design,
            group_top_k=self.group_top_k,
            expert_top_k=self.expert_top_k,
            router_distill_enable=self.router_distill_enable,
            router_use_hidden=self.router_use_hidden,
            router_use_feature=self.router_use_feature,
            expert_use_hidden=self.expert_use_hidden,
            expert_use_feature=self.expert_use_feature,
            macro_routing_scope=str(resolver.get("macro_routing_scope", "session")).lower(),
            macro_session_pooling=str(resolver.get("macro_session_pooling", "query")).lower(),
            mid_router_temperature=self.mid_router_temperature,
            micro_router_temperature=self.micro_router_temperature,
            mid_router_feature_dropout=self.mid_router_feature_dropout,
            micro_router_feature_dropout=self.micro_router_feature_dropout,
            use_valid_ratio_gating=self.use_valid_ratio_gating,
            parallel_stage_gate_top_k=self.parallel_stage_gate_top_k,
            parallel_stage_gate_temperature=self.parallel_stage_gate_temperature,
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

        # Keep runtime fields aligned for logging consumers.
        try:
            config["routerec_v2_layout_id"] = int(self.layout_id)
            config["routerec_stage_execution_mode"] = self.layout.execution
            config["routerec_v2_layout_summary"] = stage_boundary_summary(self.layout)
            config["router_impl"] = self.router_impl
            config["router_impl_by_stage"] = dict(self.router_impl_by_stage)
            config["router_design"] = self.router_design
            config["group_top_k"] = int(self.group_top_k)
            config["expert_top_k"] = int(self.expert_top_k)
            config["router_distill_enable"] = bool(self.router_distill_enable)
            config["router_distill_lambda"] = float(self.router_distill_lambda)
            config["router_distill_temperature"] = float(self.router_distill_temperature)
            config["router_distill_until"] = float(self.router_distill_until)
            config["routerec_v2_feature_spec_aux_enable"] = bool(self.feature_spec_aux_enable)
            config["routerec_v2_feature_spec_aux_lambda"] = float(self.feature_spec_aux_lambda)
            config["routerec_v2_feature_spec_stages"] = list(self.feature_spec_aux_stages)
            config["routerec_v2_feature_spec_min_tokens"] = float(self.feature_spec_aux_min_tokens)
        except Exception:
            pass

        self.set_schedule_epoch(epoch_idx=0, max_epochs=self._schedule_total_epochs, log_now=True)
        self.apply(self._init_weights)

        logger.info(
            "RouteRecV2 init: layout_id=%s execution=%s global_pre=%s global_post=%s boundaries=%s total_moe_blocks=%s "
            "d_model=%s d_feat=%s d_exp=%s d_router=%s expert_scale=%s top_k=%s aux=%s merge_aux=%s "
            "feature_spec_aux=%s feature_spec_lambda=%s feature_spec_stages=%s feature_spec_min_tokens=%s "
            "router_impl=%s router_impl_by_stage=%s router_design=%s group_top_k=%s expert_top_k=%s "
            "router_distill_enable=%s router_distill_lambda=%s router_distill_temperature=%s router_distill_until=%s "
            "rule_router=%s",
            self.layout_id,
            self.layout.execution,
            self.layout.global_pre_layers,
            self.layout.global_post_layers,
            stage_boundary_summary(self.layout),
            total_stage_moe_blocks(self.layout),
            self.d_model,
            self.d_feat_emb,
            self.d_expert_hidden,
            self.d_router_hidden,
            self.expert_scale,
            self.moe_top_k,
            self.use_aux_loss,
            self.stage_merge_aux_enable,
            self.feature_spec_aux_enable,
            self.feature_spec_aux_lambda,
            self.feature_spec_aux_stages,
            self.feature_spec_aux_min_tokens,
            self.router_impl,
            self.router_impl_by_stage,
            self.router_design,
            self.group_top_k,
            self.expert_top_k,
            self.router_distill_enable,
            self.router_distill_lambda,
            self.router_distill_temperature,
            self.router_distill_until,
            self.rule_router_cfg,
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
                "RouteRec_v2 schedule epoch=%s/%s alpha_scale=%.4f mid_temp=%.4f micro_temp=%.4f top_k=%s",
                self._schedule_epoch + 1,
                self._schedule_total_epochs,
                state.alpha_scale,
                state.mid_router_temperature,
                state.micro_router_temperature,
                ("dense" if state.stage_top_k is None else state.stage_top_k),
            )
            self._last_logged_top_k = state.stage_top_k

    def _gather_features(self, interaction) -> torch.Tensor:
        feat_list = []
        item_seq = interaction[self.ITEM_SEQ]
        bsz, tlen = item_seq.shape
        for field in self.feature_fields:
            if field in interaction:
                feat_list.append(interaction[field].float())
            else:
                feat_list.append(torch.zeros(bsz, tlen, device=item_seq.device))
                logger.warning("Feature field '%s' not found - using zeros.", field)
        return torch.stack(feat_list, dim=-1)

    def _reset_router_epoch_stats(self) -> None:
        self._router_group_entropy_sum = defaultdict(float)
        self._router_group_entropy_count = defaultdict(int)
        self._router_active_clone_sum = {}
        self._router_active_clone_count = defaultdict(int)

    def _accumulate_router_stats(
        self,
        *,
        group_weights: Dict[str, torch.Tensor],
        intra_group_weights: Dict[str, torch.Tensor],
        item_seq_len: Optional[torch.Tensor],
    ) -> None:
        for stage_key, group_w in group_weights.items():
            _, tlen, _ = group_w.shape
            if item_seq_len is not None:
                lens = item_seq_len.to(device=group_w.device).long()
                valid = torch.arange(tlen, device=group_w.device).unsqueeze(0) < lens.unsqueeze(1)
                flat_group = group_w[valid]
            else:
                flat_group = group_w.reshape(-1, group_w.shape[-1])
            if flat_group.numel() == 0:
                continue
            entropy = -(flat_group.clamp(min=1e-8) * flat_group.clamp(min=1e-8).log()).sum(dim=-1)
            self._router_group_entropy_sum[stage_key] += float(entropy.sum().item())
            self._router_group_entropy_count[stage_key] += int(entropy.numel())

        for stage_key, intra_w in intra_group_weights.items():
            _, tlen, _, _ = intra_w.shape
            if item_seq_len is not None:
                lens = item_seq_len.to(device=intra_w.device).long()
                valid = torch.arange(tlen, device=intra_w.device).unsqueeze(0) < lens.unsqueeze(1)
                flat_intra = intra_w[valid]
            else:
                flat_intra = intra_w.reshape(-1, intra_w.shape[-2], intra_w.shape[-1])
            if flat_intra.numel() == 0:
                continue
            active = (flat_intra > 0).sum(dim=-1).float()
            active_sum = active.sum(dim=0).detach().cpu()
            if stage_key not in self._router_active_clone_sum:
                self._router_active_clone_sum[stage_key] = torch.zeros_like(active_sum)
            self._router_active_clone_sum[stage_key] += active_sum
            self._router_active_clone_count[stage_key] += int(active.shape[0])

    def forward(self, item_seq, item_seq_len, feat=None):
        bsz, tlen = item_seq.shape

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(tlen, device=item_seq.device).unsqueeze(0).expand(bsz, -1)
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
            tokens, gate_weights, gate_logits, router_aux, stage_merge_weights, stage_merge_logits = self.stage_executor(
                hidden=tokens,
                item_seq=item_seq,
                feat=feat,
                item_seq_len=item_seq_len,
            )

        hidden, ffn_moe_weights = self.post_transformer(tokens, item_seq)

        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = hidden.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "group_weights": router_aux.get("group_weights", {}),
            "group_logits": router_aux.get("group_logits", {}),
            "group_logits_raw": router_aux.get("group_logits_raw", {}),
            "intra_group_weights": router_aux.get("intra_group_weights", {}),
            "intra_group_logits": router_aux.get("intra_group_logits", {}),
            "intra_group_logits_raw": router_aux.get("intra_group_logits_raw", {}),
            "teacher_group_logits": router_aux.get("teacher_group_logits", {}),
            "stage_merge_weights": stage_merge_weights,
            "stage_merge_logits": stage_merge_logits,
            "ffn_moe_weights": ffn_moe_weights,
        }
        return seq_output, aux_data

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, aux_data = self.forward(item_seq, item_seq_len, feat)

        logits = seq_output @ self.item_embedding.weight.T
        ce_loss = F.cross_entropy(logits, pos_items)

        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.use_aux_loss:
            if self.balance_loss_lambda > 0:
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
                    for lw in aux_data["ffn_moe_weights"].values():
                        aux_loss = aux_loss + self.balance_loss_lambda * load_balance_loss(
                            lw,
                            self.n_ffn_experts,
                        )

            aux_loss = aux_loss + compute_feature_specialization_aux_loss(
                weights=aux_data.get("gate_weights", {}),
                feat=feat,
                stage_feature_indices=self.stage_feature_indices,
                selected_stages=self.feature_spec_aux_stages,
                item_seq_len=item_seq_len,
                min_tokens_per_expert=self.feature_spec_aux_min_tokens,
                aux_lambda=self.feature_spec_aux_lambda,
                enabled=self.feature_spec_aux_enable,
                device=ce_loss.device,
            )
            aux_loss = aux_loss + compute_router_distill_aux_loss(
                teacher_group_logits=aux_data.get("teacher_group_logits", {}),
                student_group_logits=aux_data.get("group_logits_raw", {}),
                item_seq_len=item_seq_len,
                aux_lambda=self.router_distill_lambda,
                distill_temperature=self.router_distill_temperature,
                enabled=self.router_distill_enable,
                progress=float(self._schedule_epoch) / float(max(self._schedule_total_epochs - 1, 1)),
                until=self.router_distill_until,
                device=ce_loss.device,
            )

        if self.routerec_debug_logging and self.training:
            if aux_data.get("gate_weights"):
                self.moe_logger.accumulate(
                    gate_weights=aux_data["gate_weights"],
                    item_seq_len=item_seq_len,
                )
            if aux_data.get("group_weights"):
                self.group_logger.accumulate(
                    gate_weights=aux_data["group_weights"],
                    item_seq_len=item_seq_len,
                )
                self._accumulate_router_stats(
                    group_weights=aux_data.get("group_weights", {}),
                    intra_group_weights=aux_data.get("intra_group_weights", {}),
                    item_seq_len=item_seq_len,
                )

        return ce_loss + aux_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(item_seq, item_seq_len, feat)

        test_item_emb = self.item_embedding(test_item)
        return (seq_output * test_item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(item_seq, item_seq_len, feat)

        return seq_output @ self.item_embedding.weight.T

    def get_epoch_log_summary(self) -> Dict:
        summary = self.moe_logger.get_and_reset()
        group_summary = self.group_logger.get_and_reset()
        summary["group_stages"] = group_summary.get("stages", {})
        summary["router"] = {}

        all_stage_keys = set(self._router_group_entropy_sum) | set(self._router_active_clone_sum)
        for stage_key in sorted(all_stage_keys):
            base_stage = stage_key.split("@", 1)[0]
            entropy_count = max(self._router_group_entropy_count.get(stage_key, 0), 1)
            active_count = max(self._router_active_clone_count.get(stage_key, 0), 1)
            active_sum = self._router_active_clone_sum.get(stage_key)
            if active_sum is None:
                active_list = []
            else:
                active_list = (active_sum / float(active_count)).tolist()
            summary["router"][stage_key] = {
                "group_names": list(self._stage_group_names.get(base_stage, [])),
                "group_entropy": float(self._router_group_entropy_sum.get(stage_key, 0.0)) / float(entropy_count),
                "active_clones_per_group": active_list,
            }

        self._reset_router_epoch_stats()
        return summary
