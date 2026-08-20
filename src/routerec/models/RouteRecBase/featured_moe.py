"""
RouteRecBase: 3-Stage Hierarchical Mixture-of-Experts Sequential Recommender.

RecBole-compatible SequentialRecommender that fuses engineered features
(Macro / Mid / Micro) through a hierarchical MoE into a Transformer
encoder backbone for next-item prediction.

Architecture:
    Item emb + PosEmb
        ↓
    [Global Pre Transformer] (n_pre_layer)
        ↓
    Macro stage: [optional pre-attn x n_pre_macro] -> [MoE stage or bypass]
        ↓
    Mid stage:   [optional pre-attn x n_pre_mid]   -> [MoE stage or bypass]
        ↓
    Micro stage: [optional pre-attn x n_pre_micro] -> [MoE stage or bypass]
        ↓
    [Global Post Transformer] (n_post_layer)
        ↓
    Last-position hidden -> dot product logits -> CE loss

Layout control (layout-first priority):
    arch_layout_catalog: list of [n_pre_layer, n_pre_macro, n_pre_mid, n_pre_micro, n_post_layer]
    arch_layout_id: index into catalog

Stage depth semantics:
    -1 = bypass (skip stage entirely)
     0 = MoE-only (no stage-pre attention)
    >=1 = stage-pre attention blocks + stage MoE

Budget rule:
    num_layers >= 0  -> assert every catalog layout has same total attention count.
    num_layers < 0   -> skip assert; runtime num_layers is overwritten by selected layout sum.

SHIFT INVARIANT:
    RecBole's SequentialDataset handles the shift: ``item_id_list`` = [1..T-1],
    ``item_id`` (POS_ITEM_ID) = i_T.  Feature ``*_list`` fields are aligned
    the same way by the data pipeline in recbole_patch.py.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from .feature_config import (
    ALL_FEATURE_COLUMNS,
    feature_list_field,
    build_column_to_index,
)
from .moe_stages import HierarchicalMoE
from .routers import load_balance_loss
from .transformer import TransformerEncoder
from .logging_utils import MoELogger
from .analysis_logger import ExpertAnalysisLogger

logger = logging.getLogger(__name__)


_LAYOUT_KEYS = (
    "n_pre_layer",
    "n_pre_macro",
    "n_pre_mid",
    "n_pre_micro",
    "n_post_layer",
)
_STAGE_NAMES = ("macro", "mid", "micro")
_DEFAULT_LAYOUT = (2, 0, 0, 0, 0)


def _layout_attn_sum(layout: Sequence[int]) -> int:
    return int(layout[0]) + sum(max(0, int(layout[i])) for i in (1, 2, 3)) + int(layout[4])


def _parse_layout_catalog(raw_catalog) -> list[Tuple[int, int, int, int, int]]:
    if raw_catalog is None:
        return [tuple(_DEFAULT_LAYOUT)]
    if not isinstance(raw_catalog, (list, tuple)) or len(raw_catalog) == 0:
        raise ValueError("arch_layout_catalog must be a non-empty list of 5-int layouts")

    parsed: list[Tuple[int, int, int, int, int]] = []
    for idx, layout in enumerate(raw_catalog):
        if not isinstance(layout, (list, tuple)) or len(layout) != 5:
            raise ValueError(f"arch_layout_catalog[{idx}] must be length-5 list/tuple, got: {layout}")
        vals = tuple(int(v) for v in layout)
        if vals[0] < 0 or vals[4] < 0:
            raise ValueError(
                f"arch_layout_catalog[{idx}] invalid global depth: n_pre_layer/n_post_layer must be >=0, got {vals}"
            )
        for sid in (1, 2, 3):
            if vals[sid] < -1:
                raise ValueError(
                    f"arch_layout_catalog[{idx}] invalid stage depth: stage depths must be >=-1, got {vals}"
                )
        parsed.append(vals)
    return parsed


class RouteRecBase(SequentialRecommender):
    """3-Stage Hierarchical MoE Sequential Recommender with layout-based depth control.

    Config keys (with defaults):
        # Dimensions
        embedding_size      : 128
        hidden_size         : 128   # alias, synced to embedding_size
        d_feat_emb          : 64
        d_expert_hidden     : 64
        d_router_hidden     : 64
        expert_scale        : 1

        # Architecture layout
        arch_layout_catalog : [[2, 0, 0, 0, 0]]
        arch_layout_id      : 0
        n_pre_layer         : 2      # exposed/logged; resolved from layout
        n_pre_macro         : 0      # exposed/logged; resolved from layout
        n_pre_mid           : 0      # exposed/logged; resolved from layout
        n_pre_micro         : 0      # exposed/logged; resolved from layout
        n_post_layer        : 0      # exposed/logged; resolved from layout
        num_layers          : 2      # budget/effective total attention layer count

        # Transformer
        num_heads           : 4
        d_ff                : 0      # 0 = 4 * d_model
        hidden_dropout_prob : 0.1

        # MoE input toggles
        router_use_hidden   : True
        router_use_feature  : True
        expert_use_hidden   : True
        expert_use_feature  : True

        # MoE gating
        moe_top_k           : 0    # 0 = dense softmax; 2 = top-2 sparse
        macro_routing_scope : session  # session | token
        macro_session_pooling: query   # query | mean | last
        mid_router_temperature   : 1.3
        micro_router_temperature : 1.3
        mid_router_feature_dropout   : 0.1
        micro_router_feature_dropout : 0.1
        use_valid_ratio_gating : true  # use mid/mic valid ratio as router reliability

        # Auxiliary loss
        use_aux_loss        : True
        balance_loss_lambda : 0.01

        # FFN MoE (optional Transformer-internal MoE)
        ffn_moe             : False
        n_ffn_experts       : 4
        ffn_top_k           : 0
        stage_moe_repeat_after_pre_layer : False  # repeat [pre-attn -> stage MoE] for depth>0

        # Logging
        routerec_debug_logging  : False
        log_expert_weights  : False  # legacy fallback
        log_expert_analysis : False  # legacy fallback
    """

    input_type = "point"

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(key, default):
            return config[key] if key in config else default

        # ---- Core dimensions ----
        self.n_items = dataset.item_num
        self.d_model = config["embedding_size"]
        self.d_feat_emb = _cfg("d_feat_emb", 64)
        self.d_expert_hidden = _cfg("d_expert_hidden", 64)
        self.d_router_hidden = _cfg("d_router_hidden", 64)
        self.expert_scale = int(_cfg("expert_scale", 1))
        if self.expert_scale not in (1, 2, 3):
            raise ValueError(f"expert_scale must be one of [1,2,3], got {self.expert_scale}")
        self.n_heads = _cfg("num_heads", 4)
        self.d_ff = _cfg("d_ff", 0) or (4 * self.d_model)
        self.dropout = _cfg("hidden_dropout_prob", 0.1)
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]

        # ---- Layout parse (layout-first priority) ----
        self.arch_layout_catalog = _parse_layout_catalog(_cfg("arch_layout_catalog", [list(_DEFAULT_LAYOUT)]))
        self.arch_layout_id = int(_cfg("arch_layout_id", 0))
        if not (0 <= self.arch_layout_id < len(self.arch_layout_catalog)):
            raise ValueError(
                f"arch_layout_id out of range: id={self.arch_layout_id}, "
                f"catalog_size={len(self.arch_layout_catalog)}"
            )

        selected_layout = self.arch_layout_catalog[self.arch_layout_id]
        resolved_from_layout = {k: int(selected_layout[i]) for i, k in enumerate(_LAYOUT_KEYS)}
        user_depth_values = {k: int(_cfg(k, resolved_from_layout[k])) for k in _LAYOUT_KEYS}
        logger.warning(
            "[RouteRecBase Layout] arch_layout_id=%d selected_layout=%s input_depth=%s order=%s",
            self.arch_layout_id,
            [resolved_from_layout[k] for k in _LAYOUT_KEYS],
            [user_depth_values[k] for k in _LAYOUT_KEYS],
            list(_LAYOUT_KEYS),
        )

        mismatch_keys = [k for k in _LAYOUT_KEYS if user_depth_values[k] != resolved_from_layout[k]]
        if mismatch_keys:
            mismatch_detail = ", ".join(
                f"{k}:input={user_depth_values[k]}->layout={resolved_from_layout[k]}"
                for k in mismatch_keys
            )
            logger.warning(
                "Layout priority applied (arch_layout_id=%d). Overriding depth keys by layout: %s",
                self.arch_layout_id,
                mismatch_detail,
            )

        self.n_pre_layer = resolved_from_layout["n_pre_layer"]
        self.n_pre_macro = resolved_from_layout["n_pre_macro"]
        self.n_pre_mid = resolved_from_layout["n_pre_mid"]
        self.n_pre_micro = resolved_from_layout["n_pre_micro"]
        self.n_post_layer = resolved_from_layout["n_post_layer"]

        # Keep post-layer alias for compatibility with older internal naming.
        self.n_layers = self.n_post_layer

        # Stage enabled/bypass state
        self.stage_depths = {
            "macro": self.n_pre_macro,
            "mid": self.n_pre_mid,
            "micro": self.n_pre_micro,
        }
        self.stage_enabled = {s: self.stage_depths[s] >= 0 for s in _STAGE_NAMES}
        self.any_moe = any(self.stage_enabled.values())

        # ---- Attention budget validation / effective num_layers ----
        requested_num_layers = int(_cfg("num_layers", 2))
        if requested_num_layers >= 0:
            for lid, layout in enumerate(self.arch_layout_catalog):
                layout_sum = _layout_attn_sum(layout)
                if layout_sum != requested_num_layers:
                    raise ValueError(
                        "num_layers budget mismatch: "
                        f"num_layers={requested_num_layers}, "
                        f"arch_layout_catalog[{lid}]={list(layout)} has total_attn={layout_sum}. "
                        "All catalog layouts must have identical total attn when num_layers>=0."
                    )

        self.n_total_attn_layers = _layout_attn_sum(selected_layout)
        self.num_layers = self.n_total_attn_layers if requested_num_layers < 0 else requested_num_layers

        # Keep runtime config aligned so downstream logging (e.g., wandb/config dumps)
        # consistently sees effective attention depth.
        try:
            config["num_layers"] = int(self.num_layers)
            config["n_pre_layer"] = int(self.n_pre_layer)
            config["n_pre_macro"] = int(self.n_pre_macro)
            config["n_pre_mid"] = int(self.n_pre_mid)
            config["n_pre_micro"] = int(self.n_pre_micro)
            config["n_post_layer"] = int(self.n_post_layer)
            config["n_total_attn_layers"] = int(self.n_total_attn_layers)
        except Exception:
            pass

        # ---- MoE input toggles ----
        self.router_use_hidden = bool(_cfg("router_use_hidden", True))
        self.router_use_feature = bool(_cfg("router_use_feature", True))
        self.expert_use_hidden = bool(_cfg("expert_use_hidden", True))
        self.expert_use_feature = bool(_cfg("expert_use_feature", True))

        # ---- MoE config ----
        raw_top_k = _cfg("moe_top_k", 0)
        self.moe_top_k = None if raw_top_k == 0 else int(raw_top_k)
        self.macro_routing_scope = str(_cfg("macro_routing_scope", "session")).lower()
        self.macro_session_pooling = str(_cfg("macro_session_pooling", "query")).lower()
        self.mid_router_temperature = float(_cfg("mid_router_temperature", 1.3))
        self.micro_router_temperature = float(_cfg("micro_router_temperature", 1.3))
        self.mid_router_feature_dropout = float(_cfg("mid_router_feature_dropout", 0.1))
        self.micro_router_feature_dropout = float(_cfg("micro_router_feature_dropout", 0.1))
        self.use_valid_ratio_gating = bool(_cfg("use_valid_ratio_gating", True))

        # ---- Training schedules (optional; default OFF for backward compatibility) ----
        self.routerec_schedule_enable = bool(_cfg("routerec_schedule_enable", False))
        # Warm-up endpoint spec:
        #   - int >= 2  : reach target at that epoch (1-based)
        #   - 0 < float <= 1 : reach target at ratio * total_epochs
        # Legacy *_steps keys are still accepted as fallback aliases.
        self.alpha_warmup_until = _cfg("alpha_warmup_until", _cfg("alpha_warmup_steps", 0))
        self.alpha_warmup_start = float(_cfg("alpha_warmup_start", 0.0))
        self.alpha_warmup_end = float(_cfg("alpha_warmup_end", 1.0))

        self.mid_router_temperature_start = float(
            _cfg("mid_router_temperature_start", self.mid_router_temperature)
        )
        self.micro_router_temperature_start = float(
            _cfg("micro_router_temperature_start", self.micro_router_temperature)
        )
        self.temperature_warmup_until = _cfg(
            "temperature_warmup_until", _cfg("temperature_warmup_steps", 0)
        )

        # Top-k routing policy:
        #   fixed: use moe_top_k exactly
        #   half : use ceil(K/2)
        #   ratio: use ceil(K * moe_top_k_ratio)
        #   auto : if moe_top_k>0, use max(moe_top_k, ceil(K*ratio)); else dense
        #   dense: always dense softmax
        self.moe_top_k_policy = str(_cfg("moe_top_k_policy", "auto")).lower()
        if self.moe_top_k_policy not in ("auto", "fixed", "half", "ratio", "dense"):
            raise ValueError(
                "moe_top_k_policy must be one of ['auto','fixed','half','ratio','dense'], "
                f"got {self.moe_top_k_policy}"
            )
        self.moe_top_k_ratio = float(_cfg("moe_top_k_ratio", 0.5))
        self.moe_top_k_min = max(int(_cfg("moe_top_k_min", 1)), 1)
        # Optional non-dense warm-up start (legacy-compatible). <=0 means dense.
        self.moe_top_k_start = int(_cfg("moe_top_k_start", 0))
        self.moe_top_k_warmup_until = _cfg(
            "moe_top_k_warmup_until", _cfg("moe_top_k_warmup_steps", 0)
        )

        self.routerec_schedule_log_every_epoch = max(
            int(_cfg("routerec_schedule_log_every_epoch", _cfg("routerec_schedule_log_every", 1))),
            1,
        )
        self._schedule_epoch = 0
        # Continuation rungs train to a smaller current budget, but every
        # schedule surface must see the immutable full-horizon contract.
        from routerec.lr_scheduler import resolve_schedule_total_epochs

        self._schedule_total_epochs = resolve_schedule_total_epochs(config)
        self._last_logged_top_k: Optional[int] = None
        self.history_input_mode = str(_cfg("history_input_mode", "session_only")).lower().strip()
        self.current_session_item_length_field = str(
            _cfg("current_session_item_length_field", "current_session_item_length")
            or "current_session_item_length"
        )
        self.routerec_session_router_context_scope = str(
            _cfg("routerec_session_router_context_scope", "auto")
        ).lower().strip()
        if self.routerec_session_router_context_scope not in {"auto", "full_sequence", "current_session"}:
            raise ValueError(
                "routerec_session_router_context_scope must be one of ['auto','full_sequence','current_session'], "
                f"got {self.routerec_session_router_context_scope}"
            )

        # ---- Aux loss ----
        self.use_aux_loss = _cfg("use_aux_loss", True)
        self.balance_loss_lambda = _cfg("balance_loss_lambda", 0.01)
        self.stage_moe_repeat_after_pre_layer = bool(_cfg("stage_moe_repeat_after_pre_layer", False))

        # ---- FFN MoE ----
        self.ffn_moe = _cfg("ffn_moe", False)
        self.n_ffn_experts = _cfg("n_ffn_experts", 4)
        raw_ffn_top_k = _cfg("ffn_top_k", 0)
        self.ffn_top_k = None if raw_ffn_top_k == 0 else int(raw_ffn_top_k)

        # ---- Logging ----
        raw_debug_logging = _cfg("routerec_debug_logging", None)
        if raw_debug_logging is None:
            legacy_weights = _cfg("log_expert_weights", False)
            legacy_analysis = _cfg("log_expert_analysis", False)
            self.routerec_debug_logging = bool(legacy_weights or legacy_analysis)
        else:
            self.routerec_debug_logging = bool(raw_debug_logging)

        self.log_expert_weights = self.routerec_debug_logging
        self.log_expert_analysis = self.routerec_debug_logging
        self.analysis_sample_rate = _cfg("analysis_sample_rate", 0.2)
        self.analysis_n_bins = _cfg("analysis_n_bins", 5)

        # ---- Feature field names ----
        self.feature_fields = [feature_list_field(c) for c in ALL_FEATURE_COLUMNS]
        self.n_features = len(ALL_FEATURE_COLUMNS)

        # ================================================================
        # Layers
        # ================================================================

        self.item_embedding = nn.Embedding(self.n_items, self.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        self.input_ln = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(self.dropout)

        # Global pre attention
        if self.n_pre_layer > 0:
            self.pre_transformer = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_pre_layer,
                d_ff=self.d_ff,
                dropout=self.dropout,
                ffn_moe=False,
            )
        else:
            self.pre_transformer = None

        # Stage-pre attention blocks:
        # - default path: depth-many attention blocks, then stage MoE once.
        # - repeat path: apply [1-layer attention -> stage MoE] depth times.
        self.stage_pre_transformers = nn.ModuleDict()
        self.stage_pre_repeat_blocks = nn.ModuleDict()
        for stage_name in _STAGE_NAMES:
            depth = self.stage_depths[stage_name]
            if depth <= 0:
                continue
            if self.stage_moe_repeat_after_pre_layer:
                self.stage_pre_repeat_blocks[stage_name] = nn.ModuleList([
                    TransformerEncoder(
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                        n_layers=1,
                        d_ff=self.d_ff,
                        dropout=self.dropout,
                        ffn_moe=False,
                    )
                    for _ in range(depth)
                ])
            else:
                self.stage_pre_transformers[stage_name] = TransformerEncoder(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_layers=depth,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    ffn_moe=False,
                )

        # Hierarchical stage MoE modules (stage on/off derived from depth >= 0)
        if self.any_moe:
            self.hierarchical_moe = HierarchicalMoE(
                d_model=self.d_model,
                d_feat_emb=self.d_feat_emb,
                d_expert_hidden=self.d_expert_hidden,
                d_router_hidden=self.d_router_hidden,
                expert_scale=self.expert_scale,
                top_k=self.moe_top_k,
                dropout=self.dropout,
                use_macro=self.stage_enabled["macro"],
                use_mid=self.stage_enabled["mid"],
                use_micro=self.stage_enabled["micro"],
                router_use_hidden=self.router_use_hidden,
                router_use_feature=self.router_use_feature,
                expert_use_hidden=self.expert_use_hidden,
                expert_use_feature=self.expert_use_feature,
                macro_routing_scope=self.macro_routing_scope,
                macro_session_pooling=self.macro_session_pooling,
                mid_router_temperature=self.mid_router_temperature,
                micro_router_temperature=self.micro_router_temperature,
                mid_router_feature_dropout=self.mid_router_feature_dropout,
                micro_router_feature_dropout=self.micro_router_feature_dropout,
                use_valid_ratio_gating=self.use_valid_ratio_gating,
            )
        else:
            self.hierarchical_moe = None

        self._stage_n_experts = 0
        if self.hierarchical_moe is not None:
            for stage_name in _STAGE_NAMES:
                if self.hierarchical_moe.has_stage(stage_name):
                    self._stage_n_experts = int(
                        getattr(self.hierarchical_moe, f"{stage_name}_stage").n_experts
                    )
                    break

        # Global post attention
        self.post_transformer = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_post_layer,
            d_ff=self.d_ff,
            dropout=self.dropout,
            ffn_moe=self.ffn_moe,
            n_ffn_experts=self.n_ffn_experts,
            ffn_top_k=self.ffn_top_k,
        )

        # ---- MoE loggers ----
        active_expert_names = self.hierarchical_moe.expert_names if self.hierarchical_moe is not None else {}
        self.moe_logger = MoELogger(active_expert_names)

        if self.log_expert_analysis and self.any_moe:
            col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
            self.analysis_logger = ExpertAnalysisLogger(
                expert_names=active_expert_names,
                col2idx=col2idx,
                n_bins=self.analysis_n_bins,
                sample_rate=self.analysis_sample_rate,
            )
        else:
            self.analysis_logger = None

        # Initialize runtime schedule states once at epoch=1 start.
        self.set_schedule_epoch(epoch_idx=0, max_epochs=self._schedule_total_epochs, log_now=True)

        self.apply(self._init_weights)

        active_stages = [s for s in _STAGE_NAMES if self.stage_enabled[s]]
        logger.info(
            "RouteRecBase: d_model=%s, d_feat_emb=%s, expert_scale=%s, "
            "layout_id=%s, layout=%s, effective_num_layers=%s, n_total_attn_layers=%s, "
            "n_pre_layer=%s, n_pre_macro=%s, n_pre_mid=%s, n_pre_micro=%s, n_post_layer=%s, "
            "heads=%s, active_stages=%s, moe_top_k=%s, use_aux_loss=%s, "
            "macro_scope=%s, macro_pool=%s, mid_temp=%s, micro_temp=%s, "
            "mid_feat_drop=%s, micro_feat_drop=%s, use_valid_ratio_gating=%s, "
            "ffn_moe=%s, stage_moe_repeat_after_pre_layer=%s, n_features=%s, routerec_debug_logging=%s, "
            "schedule_enable=%s, alpha_warmup_until=%s, temp_warmup_until=%s, "
            "top_k_policy=%s, top_k_ratio=%s, top_k_warmup_until=%s",
            self.d_model,
            self.d_feat_emb,
            self.expert_scale,
            self.arch_layout_id,
            list(selected_layout),
            self.num_layers,
            self.n_total_attn_layers,
            self.n_pre_layer,
            self.n_pre_macro,
            self.n_pre_mid,
            self.n_pre_micro,
            self.n_post_layer,
            self.n_heads,
            active_stages,
            self.moe_top_k,
            self.use_aux_loss,
            self.macro_routing_scope,
            self.macro_session_pooling,
            self.mid_router_temperature,
            self.micro_router_temperature,
            self.mid_router_feature_dropout,
            self.micro_router_feature_dropout,
            self.use_valid_ratio_gating,
            self.ffn_moe,
            self.stage_moe_repeat_after_pre_layer,
            self.n_features,
            self.routerec_debug_logging,
            self.routerec_schedule_enable,
            self.alpha_warmup_until,
            self.temperature_warmup_until,
            self.moe_top_k_policy,
            self.moe_top_k_ratio,
            self.moe_top_k_warmup_until,
        )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_warmup_end_epoch(
        warmup_until,
        total_epochs: int,
    ) -> int:
        """Resolve warm-up endpoint (1-based epoch index)."""
        if warmup_until is None:
            return 0
        try:
            v = float(warmup_until)
        except (TypeError, ValueError):
            return 0
        if v <= 0:
            return 0
        # ratio mode: 0 < v <= 1
        if 0 < v <= 1:
            return max(1, int(math.ceil(float(total_epochs) * v)))
        # epoch mode: v >= 2 (or any >1 value)
        return max(1, int(round(v)))

    @staticmethod
    def _epoch_progress(epoch_idx: int, end_epoch: int) -> float:
        """Warm-up progress where epoch 1 starts at progress=0 and end_epoch reaches 1."""
        if end_epoch <= 1:
            return 1.0
        e1 = max(int(epoch_idx) + 1, 1)  # 1-based epoch
        return min(max((float(e1) - 1.0) / float(end_epoch - 1), 0.0), 1.0)

    @classmethod
    def _linear_warmup(
        cls,
        epoch_idx: int,
        end_epoch: int,
        start: float,
        end: float,
    ) -> float:
        if end_epoch <= 0:
            return float(end)
        progress = cls._epoch_progress(epoch_idx=epoch_idx, end_epoch=end_epoch)
        return float(start + (end - start) * progress)

    @staticmethod
    def _normalize_top_k(top_k: Optional[int], n_experts: int) -> Optional[int]:
        """Clamp top-k to valid sparse range; return None for dense routing."""
        if n_experts <= 0 or top_k is None:
            return None
        k = int(top_k)
        if k <= 0:
            return None
        k = min(k, int(n_experts))
        return None if k >= n_experts else k

    def _resolve_top_k_target(self, n_experts: int) -> Optional[int]:
        if n_experts <= 0:
            return None
        policy = self.moe_top_k_policy
        if policy == "dense":
            return None
        if policy == "half":
            target = int(math.ceil(0.5 * float(n_experts)))
        elif policy == "ratio":
            ratio = min(max(self.moe_top_k_ratio, 0.0), 1.0)
            if ratio <= 0.0:
                return None
            target = int(math.ceil(ratio * float(n_experts)))
        elif policy == "fixed":
            if self.moe_top_k is None:
                return None
            target = int(self.moe_top_k)
        else:  # auto
            if self.moe_top_k is None:
                return None
            ratio_target = int(math.ceil(min(max(self.moe_top_k_ratio, 0.0), 1.0) * float(n_experts)))
            target = max(int(self.moe_top_k), ratio_target)

        target = max(self.moe_top_k_min, int(target))
        return self._normalize_top_k(target, n_experts=n_experts)

    def _scheduled_top_k(self, epoch_idx: int, total_epochs: int, n_experts: int) -> Optional[int]:
        target_top_k = self._resolve_top_k_target(n_experts=n_experts)
        if target_top_k is None:
            return None

        if not self.routerec_schedule_enable:
            return target_top_k

        end_epoch = self._resolve_warmup_end_epoch(
            warmup_until=self.moe_top_k_warmup_until,
            total_epochs=total_epochs,
        )
        if end_epoch <= 0:
            return target_top_k

        # Legacy start knob: <=0 means dense start.
        start_top_k = self._normalize_top_k(self.moe_top_k_start, n_experts=n_experts)
        start_k = n_experts if start_top_k is None else int(start_top_k)
        target_k = int(target_top_k)

        if start_k == target_k:
            return target_top_k

        progress = self._epoch_progress(epoch_idx=epoch_idx, end_epoch=end_epoch)
        interpolated = int(round(start_k + (target_k - start_k) * progress))
        lower = min(start_k, target_k)
        upper = max(start_k, target_k)
        interpolated = min(max(interpolated, lower), upper)
        return self._normalize_top_k(interpolated, n_experts=n_experts)

    def _apply_training_schedule(self, log_now: bool = False) -> None:
        if not self.any_moe or self.hierarchical_moe is None:
            return

        if not self.routerec_schedule_enable:
            stage_temps = {
                "mid": self.mid_router_temperature,
                "micro": self.micro_router_temperature,
            }
            static_top_k = self._resolve_top_k_target(n_experts=self._stage_n_experts)
            runtime_top_k = -1 if static_top_k is None else int(static_top_k)
            self.hierarchical_moe.set_schedule_state(
                alpha_scale=1.0,
                stage_temperatures=stage_temps,
                top_k=runtime_top_k,
            )
            return

        epoch_idx = int(self._schedule_epoch)
        total_epochs = int(self._schedule_total_epochs)
        alpha_end_epoch = self._resolve_warmup_end_epoch(
            warmup_until=self.alpha_warmup_until,
            total_epochs=total_epochs,
        )
        temp_end_epoch = self._resolve_warmup_end_epoch(
            warmup_until=self.temperature_warmup_until,
            total_epochs=total_epochs,
        )
        alpha_scale = self._linear_warmup(
            epoch_idx=epoch_idx,
            end_epoch=alpha_end_epoch,
            start=self.alpha_warmup_start,
            end=self.alpha_warmup_end,
        )
        mid_temp = self._linear_warmup(
            epoch_idx=epoch_idx,
            end_epoch=temp_end_epoch,
            start=self.mid_router_temperature_start,
            end=self.mid_router_temperature,
        )
        micro_temp = self._linear_warmup(
            epoch_idx=epoch_idx,
            end_epoch=temp_end_epoch,
            start=self.micro_router_temperature_start,
            end=self.micro_router_temperature,
        )
        top_k = self._scheduled_top_k(
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
            n_experts=self._stage_n_experts,
        )
        runtime_top_k = -1 if top_k is None else int(top_k)

        self.hierarchical_moe.set_schedule_state(
            alpha_scale=alpha_scale,
            stage_temperatures={"mid": mid_temp, "micro": micro_temp},
            top_k=runtime_top_k,
        )

        should_log = log_now or (epoch_idx % self.routerec_schedule_log_every_epoch == 0)
        if self._last_logged_top_k != top_k:
            should_log = True

        if should_log:
            logger.info(
                "RouteRec schedule epoch=%s/%s alpha_scale=%.4f mid_temp=%.4f micro_temp=%.4f top_k=%s",
                epoch_idx + 1,
                total_epochs,
                alpha_scale,
                mid_temp,
                micro_temp,
                ("dense" if top_k is None else top_k),
            )
            self._last_logged_top_k = top_k

    def set_schedule_epoch(
        self,
        epoch_idx: int,
        max_epochs: Optional[int] = None,
        log_now: bool = False,
    ) -> None:
        """Set runtime schedule state using epoch index (0-based)."""
        self._schedule_epoch = max(int(epoch_idx), 0)
        if max_epochs is not None:
            self._schedule_total_epochs = max(int(max_epochs), 1)
        self._apply_training_schedule(log_now=log_now)

    # ------------------------------------------------------------------
    # Feature gathering
    # ------------------------------------------------------------------

    def _gather_features(self, interaction) -> Optional[torch.Tensor]:
        """Stack all feature sequence columns into a single tensor.

        Returns None if no MoE stages are active (skip gathering entirely).
        """
        if not self.any_moe:
            return None

        feat_list = []
        for field in self.feature_fields:
            if field in interaction:
                f = interaction[field]
                feat_list.append(f.float())
            else:
                bsz = interaction[self.ITEM_SEQ].shape[0]
                tlen = interaction[self.ITEM_SEQ].shape[1]
                feat_list.append(
                    torch.zeros(bsz, tlen, device=interaction[self.ITEM_SEQ].device)
                )
                logger.warning("Feature field '%s' not found - using zeros.", field)

        return torch.stack(feat_list, dim=-1)

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, item_seq, item_seq_len, feat=None, routing_item_seq_len=None):
        """Encode input and return hidden state at last valid position.

        Args:
            item_seq     : [B, T]
            item_seq_len : [B]
            feat         : [B, T, n_features] or None
        Returns:
            seq_output   : [B, d_model]
            aux_data     : dict
        """
        bsz, tlen = item_seq.shape

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(tlen, device=item_seq.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)

        if self.pre_transformer is not None:
            tokens, _ = self.pre_transformer(tokens, item_seq)

        gate_weights, gate_logits = {}, {}
        if self.any_moe and feat is not None and self.hierarchical_moe is not None:
            for stage_name in _STAGE_NAMES:
                if not self.stage_enabled[stage_name]:
                    continue

                if (
                    self.stage_moe_repeat_after_pre_layer
                    and stage_name in self.stage_pre_repeat_blocks
                ):
                    for ridx, stage_pre_block in enumerate(self.stage_pre_repeat_blocks[stage_name], start=1):
                        tokens, _ = stage_pre_block(tokens, item_seq)
                        tokens, w, l = self.hierarchical_moe.forward_stage(
                            stage_name,
                            tokens,
                            feat,
                            item_seq_len=item_seq_len,
                            routing_item_seq_len=routing_item_seq_len,
                        )
                        stage_key = f"{stage_name}@{ridx}"
                        gate_weights[stage_key] = w
                        gate_logits[stage_key] = l
                    continue

                if stage_name in self.stage_pre_transformers:
                    tokens, _ = self.stage_pre_transformers[stage_name](tokens, item_seq)

                tokens, w, l = self.hierarchical_moe.forward_stage(
                    stage_name,
                    tokens,
                    feat,
                    item_seq_len=item_seq_len,
                    routing_item_seq_len=routing_item_seq_len,
                )
                gate_weights[stage_name] = w
                gate_logits[stage_name] = l

        hidden, ffn_moe_weights = self.post_transformer(tokens, item_seq)

        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = hidden.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "ffn_moe_weights": ffn_moe_weights,
        }
        return seq_output, aux_data

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        """CE loss + optional MoE balance loss."""
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
            if aux_data["gate_weights"] and self.hierarchical_moe is not None:
                aux_loss = self.hierarchical_moe.compute_aux_loss(
                    aux_data["gate_weights"],
                    item_seq_len=item_seq_len,
                    balance_lambda=self.balance_loss_lambda,
                )
            if self.ffn_moe and aux_data["ffn_moe_weights"]:
                for lw in aux_data["ffn_moe_weights"].values():
                    aux_loss = aux_loss + self.balance_loss_lambda * load_balance_loss(
                        lw, self.n_ffn_experts,
                    )

        total_loss = ce_loss + aux_loss

        if self.log_expert_weights and self.training and aux_data["gate_weights"]:
            self.moe_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                item_seq_len=item_seq_len,
            )

        if self.analysis_logger is not None and self.training and aux_data["gate_weights"]:
            self.analysis_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                feat=feat,
                logits=logits,
                pos_items=pos_items,
                item_seq_len=item_seq_len,
            )

        return total_loss

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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
        scores = (seq_output * test_item_emb).sum(dim=-1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )

        scores = seq_output @ self.item_embedding.weight.T
        return scores

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def get_epoch_log_summary(self) -> Dict:
        summary = self.moe_logger.get_and_reset()
        if self.analysis_logger is not None:
            summary["analysis"] = self.analysis_logger.get_and_reset()
        return summary
