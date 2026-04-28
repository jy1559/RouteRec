"""RouteRec model with layer_layout backbone."""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..RouteRecBase.logging_utils import MoELogger
from ..RouteRecBase.routers import load_balance_loss
from ..RouteRecN.routerec_n import RouteRecN
from ..RouteRecV2.config_schema import ConfigResolver
from ..RouteRecV2.losses import compute_expert_aux_loss, compute_stage_merge_aux_loss
from .diagnostics import N3DiagnosticCollector
from .feature_config import (
    ALL_FEATURE_COLUMNS,
    GROUP_ORDER,
    STAGE_NAMES,
    build_column_to_index,
    build_stage_feature_spec,
    feature_list_field,
    load_feature_meta_v3,
    validate_feature_meta_v3,
)
from .stage_executor import StageExecutorN3

logger = logging.getLogger(__name__)


_PLAIN_TOKENS = {"layer", "attn", "ffn"}
_LEGACY_ROUTER_KEYS = (
    "stage_router_type",
    "stage_factored_group_router_source",
    "stage_factored_group_logit_scale",
    "stage_factored_intra_logit_scale",
    "stage_factored_combine_mode",
)
_FEATURE_PERTURB_MODES = {
    "none",
    "zero",
    "shuffle",
    "global_permute",
    "batch_permute",
    "family_permute",
    "position_shift",
    "role_swap",
    "stage_mismatch",
}
_FEATURE_PERTURB_APPLIES = {"none", "train", "eval", "both"}


def _is_effectively_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _set_config_key(config_obj, key: str, value) -> None:
    try:
        config_obj[key] = value
    except Exception:
        pass
    final_cfg = getattr(config_obj, "final_config_dict", None)
    if isinstance(final_cfg, dict):
        final_cfg[key] = value


def _parse_layer_layout(raw_value) -> List[str]:
    if raw_value is None:
        return ["layer", "layer", "layer"]
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip().strip("'\"") for part in text.split(",")]
        out = [part for part in parts if part]
        return out or ["layer", "layer", "layer"]
    if isinstance(raw_value, (list, tuple)):
        out = [str(v).strip().lower() for v in raw_value if str(v).strip()]
        return out or ["layer", "layer", "layer"]
    return ["layer", "layer", "layer"]


def _parse_stage_str_map(raw_value, *, default_value: str) -> Dict[str, str]:
    default = {stage: str(default_value).lower().strip() for stage in STAGE_NAMES}
    if raw_value is None:
        return default
    if isinstance(raw_value, str):
        value = str(raw_value).lower().strip()
        return {stage: value for stage in STAGE_NAMES}
    if not isinstance(raw_value, dict):
        return default
    out = dict(default)
    for stage in STAGE_NAMES:
        value = raw_value.get(stage, raw_value.get(stage.capitalize(), default_value))
        out[stage] = str(value).lower().strip()
    return out


def _parse_stage_float_map(raw_value, *, default_value: float) -> Dict[str, float]:
    default = {stage: float(default_value) for stage in STAGE_NAMES}
    if raw_value is None:
        return default
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        return {stage: value for stage in STAGE_NAMES}
    if not isinstance(raw_value, dict):
        return default
    out = dict(default)
    for stage in STAGE_NAMES:
        value = raw_value.get(stage, raw_value.get(stage.capitalize(), default_value))
        try:
            out[stage] = float(value)
        except Exception:
            out[stage] = float(default_value)
    return out


def _parse_stage_nested_map(raw_value, *, default_value: Optional[dict] = None) -> Dict[str, dict]:
    base_default = dict(default_value or {})
    out = {stage: copy.deepcopy(base_default) for stage in STAGE_NAMES}
    if raw_value is None or not isinstance(raw_value, dict):
        return out
    global_default = raw_value.get("_default")
    if isinstance(global_default, dict):
        for stage in STAGE_NAMES:
            out[stage] = copy.deepcopy(global_default)
    for stage in STAGE_NAMES:
        value = raw_value.get(stage, raw_value.get(stage.capitalize(), None))
        if isinstance(value, dict):
            out[stage] = copy.deepcopy(value)
    return out


def _parse_family_tokens(raw_value) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = str(raw_value).strip()
        if not text:
            return []
        parts = [tok.strip() for tok in text.split(",") if tok.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        parts = [str(tok).strip() for tok in raw_value if str(tok).strip()]
    else:
        parts = [str(raw_value).strip()] if str(raw_value).strip() else []

    valid = {str(name).lower(): str(name).lower() for name in GROUP_ORDER}
    out: List[str] = []
    for token in parts:
        key = token.lower()
        if key in valid and key not in out:
            out.append(key)
    return out


def _parse_keyword_tokens(raw_value) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = str(raw_value).strip()
        if not text:
            return []
        parts = [tok.strip() for tok in text.split(",") if tok.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        parts = [str(tok).strip() for tok in raw_value if str(tok).strip()]
    else:
        parts = [str(raw_value).strip()] if str(raw_value).strip() else []

    out: List[str] = []
    for token in parts:
        lowered = str(token).lower()
        if lowered and lowered not in out:
            out.append(lowered)
    return out


class RouteRec(RouteRecN):
    """N3 branch with explicit layer_layout and stage-wise controls."""

    @staticmethod
    def _ensure_layout_catalog(config) -> None:
        catalog = [
            {
                "id": "DSL0",
                "execution": "serial",
                "global_pre_layers": 0,
                "global_post_layers": 0,
                "stages": {
                    "macro": {"pass_layers": 0, "moe_blocks": 0},
                    "mid": {"pass_layers": 0, "moe_blocks": 0},
                    "micro": {"pass_layers": 0, "moe_blocks": 0},
                },
            }
        ]
        _set_config_key(config, "routerec_v2_layout_catalog", copy.deepcopy(catalog))
        _set_config_key(config, "routerec_v2_layout_id", 0)
        _set_config_key(config, "routerec_stage_execution_mode", "serial")

    @staticmethod
    def _assert_no_legacy_router_keys(config) -> None:
        cfg = getattr(config, "final_config_dict", None)
        if not isinstance(cfg, dict):
            return
        found = []
        for key in _LEGACY_ROUTER_KEYS:
            if key not in cfg:
                continue
            val = cfg.get(key)
            if _is_effectively_empty(val):
                continue
            found.append(key)
        if found:
            joined = ", ".join(found)
            raise ValueError(
                "Legacy router config keys are not supported in RouteRec. "
                f"Use stage_router_wrapper + stage_router_primitives only. Found: {joined}"
            )

    def __init__(self, config, dataset):
        self._assert_no_legacy_router_keys(config)
        self._ensure_layout_catalog(config)
        super().__init__(config, dataset)

        resolver = ConfigResolver(config)
        self.hidden_act = str(resolver.get("hidden_act", "gelu")).lower().strip()
        self.layer_norm_eps = float(resolver.get("layer_norm_eps", 1e-12))
        self.layer_layout = _parse_layer_layout(resolver.get("layer_layout", ["layer", "layer", "layer"]))
        self.macro_history_window = int(resolver.get("macro_history_window", 5))
        self.stage_router_granularity = _parse_stage_str_map(
            resolver.get("stage_router_granularity", None),
            default_value="session",
        )
        self.stage_router_granularity["micro"] = "token"
        self.stage_feature_encoder_mode = _parse_stage_str_map(
            resolver.get("stage_feature_encoder_mode", None),
            default_value="linear",
        )
        self.stage_compute_mode = _parse_stage_str_map(
            resolver.get("stage_compute_mode", None),
            default_value="moe",
        )
        self.stage_router_mode = _parse_stage_str_map(
            resolver.get("stage_router_mode", None),
            default_value="learned",
        )
        self.stage_router_source = _parse_stage_str_map(
            resolver.get("stage_router_source", None),
            default_value="both",
        )
        self.stage_feature_injection = _parse_stage_str_map(
            resolver.get("stage_feature_injection", None),
            default_value="none",
        )
        self.stage_router_wrapper = _parse_stage_str_map(
            resolver.get("stage_router_wrapper", None),
            default_value="w1_flat",
        )
        self.stage_router_primitives = _parse_stage_nested_map(
            resolver.get("stage_router_primitives", None),
            default_value={},
        )
        self.stage_residual_mode = _parse_stage_str_map(
            resolver.get("stage_residual_mode", None),
            default_value="base",
        )
        self.residual_alpha_fixed = _parse_stage_float_map(
            resolver.get("residual_alpha_fixed", None),
            default_value=0.5,
        )
        self.residual_alpha_init = _parse_stage_float_map(
            resolver.get("residual_alpha_init", None),
            default_value=0.0,
        )
        self.residual_shared_ffn_scale = float(resolver.get("residual_shared_ffn_scale", 1.0))
        self.stage_feature_family_mask = resolver.get("stage_feature_family_mask", {}) or {}
        self.stage_feature_family_topk = resolver.get("stage_feature_family_topk", {}) or {}
        self.stage_feature_family_custom = resolver.get("stage_feature_family_custom", {}) or {}
        self.stage_feature_drop_keywords = resolver.get("stage_feature_drop_keywords", []) or []
        self.stage_family_dropout_prob = _parse_stage_float_map(
            resolver.get("stage_family_dropout_prob", None),
            default_value=0.0,
        )
        self.stage_feature_dropout_prob = _parse_stage_float_map(
            resolver.get("stage_feature_dropout_prob", None),
            default_value=0.0,
        )
        self.stage_feature_dropout_scope = _parse_stage_str_map(
            resolver.get("stage_feature_dropout_scope", None),
            default_value="token",
        )
        self.dense_hidden_scale = float(resolver.get("dense_hidden_scale", 1.0))
        self.rule_agreement_lambda = float(resolver.get("rule_agreement_lambda", 0.0))
        self.group_coverage_lambda = float(resolver.get("group_coverage_lambda", 0.0))
        self.group_prior_align_lambda = float(resolver.get("group_prior_align_lambda", 0.0))
        self.feature_group_bias_lambda = float(resolver.get("feature_group_bias_lambda", 0.0))
        self.feature_group_prior_temperature = float(resolver.get("feature_group_prior_temperature", 1.0))
        self.factored_group_balance_lambda = float(resolver.get("factored_group_balance_lambda", 0.0))
        self.route_smoothness_lambda = float(resolver.get("route_smoothness_lambda", 0.0))
        self.route_smoothness_stage_weight = _parse_stage_float_map(
            resolver.get("route_smoothness_stage_weight", None),
            default_value=1.0,
        )
        self.route_consistency_lambda = float(resolver.get("route_consistency_lambda", 0.0))
        self.route_consistency_pairs = max(int(resolver.get("route_consistency_pairs", 4)), 1)
        self.route_consistency_min_sim = float(resolver.get("route_consistency_min_sim", -1.0))
        self.route_separation_lambda = float(resolver.get("route_separation_lambda", 0.0))
        self.route_separation_pairs = max(int(resolver.get("route_separation_pairs", 4)), 1)
        self.route_separation_max_sim = float(resolver.get("route_separation_max_sim", 0.2))
        self.route_separation_margin = float(resolver.get("route_separation_margin", 0.05))
        self.route_sharpness_lambda = float(resolver.get("route_sharpness_lambda", 0.0))
        self.route_monopoly_lambda = float(resolver.get("route_monopoly_lambda", 0.0))
        self.route_monopoly_tau = float(resolver.get("route_monopoly_tau", 0.0))
        self.route_prior_lambda = float(resolver.get("route_prior_lambda", 0.0))
        self.route_prior_bias_scale = float(resolver.get("route_prior_bias_scale", 0.5))
        self.intra_group_bias_mode = _parse_stage_str_map(
            resolver.get("intra_group_bias_mode", None),
            default_value="none",
        )
        self.intra_group_bias_scale = _parse_stage_float_map(
            resolver.get("intra_group_bias_scale", None),
            default_value=0.0,
        )
        self.routerec_diag_logging = bool(resolver.get("routerec_diag_logging", True))
        self.routerec_diag_sample_sessions = max(int(resolver.get("routerec_diag_sample_sessions", 256)), 0)
        self.feature_perturb_mode = str(resolver.get("feature_perturb_mode", "none")).lower().strip()
        if self.feature_perturb_mode not in _FEATURE_PERTURB_MODES:
            self.feature_perturb_mode = "none"
        self.feature_perturb_apply = str(resolver.get("feature_perturb_apply", "none")).lower().strip()
        if self.feature_perturb_apply not in _FEATURE_PERTURB_APPLIES:
            self.feature_perturb_apply = "none"
        self.feature_perturb_family = _parse_family_tokens(resolver.get("feature_perturb_family", ""))
        self.feature_perturb_keywords = _parse_keyword_tokens(resolver.get("feature_perturb_keywords", []))
        try:
            self.feature_perturb_shift = int(resolver.get("feature_perturb_shift", 1))
        except Exception:
            self.feature_perturb_shift = 1
        self.feature_perturb_shift = max(self.feature_perturb_shift, 1)

        self._feature_ablation_mode = "none"
        self._feature_ablation_family: Optional[str] = None
        self._diag_collector: Optional[N3DiagnosticCollector] = None

        meta = load_feature_meta_v3(
            data_path=str(resolver.get("data_path", "")),
            dataset=str(resolver.get("dataset", "")),
        )
        self.feature_meta_v3 = validate_feature_meta_v3(meta) if meta else {}
        requested_feature_mode = str(resolver.get("feature_mode", "")).strip().lower()
        self.feature_spec = build_stage_feature_spec(
            macro_history_window=self.macro_history_window,
            stage_feature_family_mask=self.stage_feature_family_mask,
            stage_feature_family_topk=self.stage_feature_family_topk,
            stage_feature_family_custom=self.stage_feature_family_custom,
            stage_feature_drop_keywords=self.stage_feature_drop_keywords,
        )

        # If feature_meta_v3 is unavailable (common on datasets without v3 engineering),
        # disable engineered feature columns for this run to avoid noisy missing-field fallbacks.
        meta_feature_list = list((self.feature_meta_v3 or {}).get("all_features", []) or [])
        if meta_feature_list:
            meta_feature_set = set(str(name) for name in meta_feature_list)
            effective_feature_columns = [col for col in ALL_FEATURE_COLUMNS if col in meta_feature_set]
        else:
            effective_feature_columns = list(ALL_FEATURE_COLUMNS)
            if requested_feature_mode == "full_v3":
                effective_feature_columns = []
                logger.warning(
                    "feature_meta_v3 not found for dataset=%s (feature_mode=%s) - "
                    "engineered feature columns disabled for this run.",
                    str(resolver.get("dataset", "")),
                    requested_feature_mode,
                )

        # Keep feature tensors aligned with stage feature spec for structural portability ablations.
        active_stage_features = set()
        for stage_name in STAGE_NAMES:
            for col_name in list(self.feature_spec.get("stage_all_features", {}).get(stage_name, []) or []):
                active_stage_features.add(str(col_name))
        if active_stage_features:
            effective_feature_columns = [
                col for col in effective_feature_columns
                if str(col) in active_stage_features
            ]

        self.feature_base_fields = list(effective_feature_columns)
        self.feature_fields = [feature_list_field(col) for col in self.feature_base_fields]
        self.n_features = len(self.feature_fields)
        self._missing_feature_field_once: set[str] = set()
        self._scalar_fallback_field_once: set[str] = set()

        col2idx = build_column_to_index(self.feature_base_fields)
        self._feature_col2idx = dict(col2idx)
        self._feature_family_ablation_indices = {}
        for family_name in GROUP_ORDER:
            family_cols = []
            for stage_name in STAGE_NAMES:
                family_cols.extend(
                    list((self.feature_spec["stage_family_features"].get(stage_name, {}) or {}).get(family_name, []) or [])
                )
            self._feature_family_ablation_indices[family_name.lower()] = sorted(
                {int(col2idx[col]) for col in family_cols if col in col2idx}
            )
        self._feature_stage_indices = {}
        self._feature_stage_family_indices = {}
        for stage_name in STAGE_NAMES:
            stage_cols = list((self.feature_spec["stage_all_features"].get(stage_name, {}) or []) or [])
            self._feature_stage_indices[stage_name] = [int(col2idx[col]) for col in stage_cols if col in col2idx]
            fam_map = {}
            for family_name in GROUP_ORDER:
                fam_cols = list(
                    (self.feature_spec["stage_family_features"].get(stage_name, {}) or {}).get(family_name, []) or []
                )
                fam_map[family_name.lower()] = [int(col2idx[col]) for col in fam_cols if col in col2idx]
            self._feature_stage_family_indices[stage_name] = fam_map
        expert_hidden_by_stage = self._parse_stage_int_map(
            resolver.get("expert_hidden_by_stage", {}),
            default_value=self.d_expert_hidden,
        )
        expert_depth_by_stage = self._parse_stage_int_map(
            resolver.get("expert_depth_by_stage", {}),
            default_value=1,
        )

        self.pre_transformer = None
        self.post_transformer = None
        self.stage_executor = StageExecutorN3(
            layer_layout=self.layer_layout,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            d_feat_emb=self.d_feat_emb,
            d_expert_hidden=self.d_expert_hidden,
            d_router_hidden=self.d_router_hidden,
            expert_depth_by_stage=expert_depth_by_stage,
            expert_hidden_by_stage=expert_hidden_by_stage,
            expert_scale=self.expert_scale,
            stage_top_k=self.moe_top_k,
            macro_session_pooling=str(resolver.get("macro_session_pooling", "mean")).lower(),
            stage_router_granularity=self.stage_router_granularity,
            stage_all_features=self.feature_spec["stage_all_features"],
            stage_family_features=self.feature_spec["stage_family_features"],
            stage_feature_encoder_mode=self.stage_feature_encoder_mode,
            stage_compute_mode=self.stage_compute_mode,
            stage_router_mode=self.stage_router_mode,
            stage_router_source=self.stage_router_source,
            stage_feature_injection=self.stage_feature_injection,
            rule_router_cfg=self.rule_router_cfg,
            rule_bias_scale=self.rule_bias_scale,
            feature_group_bias_lambda=self.feature_group_bias_lambda,
            feature_group_prior_temperature=self.feature_group_prior_temperature,
            stage_router_wrapper=self.stage_router_wrapper,
            stage_router_primitives=self.stage_router_primitives,
            intra_group_bias_mode=self.intra_group_bias_mode,
            intra_group_bias_scale=self.intra_group_bias_scale,
            mid_router_temperature=self.mid_router_temperature,
            micro_router_temperature=self.micro_router_temperature,
            dense_hidden_scale=self.dense_hidden_scale,
            stage_residual_mode=self.stage_residual_mode,
            residual_alpha_fixed=self.residual_alpha_fixed,
            residual_alpha_init=self.residual_alpha_init,
            residual_shared_ffn_scale=self.residual_shared_ffn_scale,
            stage_family_dropout_prob=self.stage_family_dropout_prob,
            stage_feature_dropout_prob=self.stage_feature_dropout_prob,
            stage_feature_dropout_scope=self.stage_feature_dropout_scope,
            col2idx=col2idx,
        )
        self.stage_executor.apply(self._init_weights)
        self._stage_n_experts = self.stage_executor.stage_n_experts()
        self._stage_expert_names = self.stage_executor.stage_expert_names()
        self._stage_group_names = self.stage_executor.stage_group_names()
        self.moe_logger = MoELogger(self._stage_expert_names)
        self.group_logger = MoELogger(self._stage_group_names)
        self._reset_router_epoch_stats()
        self.set_schedule_epoch(epoch_idx=self._schedule_epoch, max_epochs=self._schedule_total_epochs, log_now=True)

        self.n_pre_layer = 0
        self.n_pre_macro = 0
        self.n_pre_mid = 0
        self.n_pre_micro = 0
        self.n_post_layer = 0
        self.arch_layout_id = -1
        self.num_layers = int(sum(token in {"layer", "macro", "mid", "micro", "attn"} for token in self.layer_layout))
        self.n_total_attn_layers = int(sum(token in {"layer", "macro", "mid", "micro", "attn"} for token in self.layer_layout))
        self._uses_stage_tokens = any(token not in _PLAIN_TOKENS for token in self.layer_layout)
        self._requires_features = bool(self.stage_executor.requires_features)
        self._supports_diag = bool(self.stage_executor.supports_diagnostics)

        for key, value in {
            "layer_layout": list(self.layer_layout),
            "stage_router_granularity": dict(self.stage_router_granularity),
            "stage_feature_encoder_mode": dict(self.stage_feature_encoder_mode),
            "stage_compute_mode": dict(self.stage_compute_mode),
            "stage_router_mode": dict(self.stage_router_mode),
            "stage_router_source": dict(self.stage_router_source),
            "stage_feature_injection": dict(self.stage_feature_injection),
            "stage_router_wrapper": dict(self.stage_router_wrapper),
            "stage_router_primitives": dict(self.stage_router_primitives),
            "stage_residual_mode": dict(self.stage_residual_mode),
            "residual_alpha_fixed": dict(self.residual_alpha_fixed),
            "residual_alpha_init": dict(self.residual_alpha_init),
            "residual_shared_ffn_scale": self.residual_shared_ffn_scale,
            "macro_history_window": self.macro_history_window,
            "stage_feature_family_mask": self.stage_feature_family_mask,
            "stage_feature_family_topk": self.stage_feature_family_topk,
            "stage_feature_family_custom": self.stage_feature_family_custom,
            "stage_feature_drop_keywords": self.stage_feature_drop_keywords,
            "stage_family_dropout_prob": dict(self.stage_family_dropout_prob),
            "stage_feature_dropout_prob": dict(self.stage_feature_dropout_prob),
            "stage_feature_dropout_scope": dict(self.stage_feature_dropout_scope),
            "dense_hidden_scale": self.dense_hidden_scale,
            "group_prior_align_lambda": self.group_prior_align_lambda,
            "feature_group_bias_lambda": self.feature_group_bias_lambda,
            "feature_group_prior_temperature": self.feature_group_prior_temperature,
            "factored_group_balance_lambda": self.factored_group_balance_lambda,
            "route_smoothness_lambda": self.route_smoothness_lambda,
            "route_smoothness_stage_weight": dict(self.route_smoothness_stage_weight),
            "route_consistency_lambda": self.route_consistency_lambda,
            "route_consistency_pairs": self.route_consistency_pairs,
            "route_consistency_min_sim": self.route_consistency_min_sim,
            "route_separation_lambda": self.route_separation_lambda,
            "route_separation_pairs": self.route_separation_pairs,
            "route_separation_max_sim": self.route_separation_max_sim,
            "route_separation_margin": self.route_separation_margin,
            "route_sharpness_lambda": self.route_sharpness_lambda,
            "route_monopoly_lambda": self.route_monopoly_lambda,
            "route_monopoly_tau": self.route_monopoly_tau,
            "route_prior_lambda": self.route_prior_lambda,
            "route_prior_bias_scale": self.route_prior_bias_scale,
            "intra_group_bias_mode": dict(self.intra_group_bias_mode),
            "intra_group_bias_scale": dict(self.intra_group_bias_scale),
            "hidden_act": self.hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "routerec_special_logging": bool(self.routerec_special_logging),
            "routerec_diag_logging": bool(self.routerec_diag_logging),
            "feature_perturb_mode": self.feature_perturb_mode,
            "feature_perturb_apply": self.feature_perturb_apply,
            "feature_perturb_family": list(self.feature_perturb_family),
            "feature_perturb_keywords": list(self.feature_perturb_keywords),
            "feature_perturb_shift": self.feature_perturb_shift,
        }.items():
            _set_config_key(config, key, value)

        logger.info(
            "RouteRec init: layer_layout=%s compute=%s router_mode=%s router_source=%s "
            "feature_injection=%s routing=%s feature_groups=%s uses_stage=%s diag=%s "
            "perturb=(mode=%s apply=%s family=%s keywords=%s shift=%s) meta_loaded=%s",
            self.layer_layout,
            self.stage_compute_mode,
            self.stage_router_mode,
            self.stage_router_source,
            self.stage_feature_injection,
            self.stage_router_granularity,
            self.feature_spec["stage_family_mask"],
            self._uses_stage_tokens,
            self._supports_diag,
            self.feature_perturb_mode,
            self.feature_perturb_apply,
            self.feature_perturb_family,
            self.feature_perturb_keywords,
            self.feature_perturb_shift,
            bool(self.feature_meta_v3),
        )

    def set_feature_ablation_mode(self, mode: str = "none", family: Optional[str] = None) -> None:
        normalized = str(mode or "none").lower().strip()
        if normalized not in {"none", "zero", "shuffle"}:
            normalized = "none"
        self._feature_ablation_mode = normalized
        family_name = str(family or "").strip().lower()
        self._feature_ablation_family = family_name if family_name in self._feature_family_ablation_indices else None

    def _feature_ablation_tag(self) -> str:
        if self._feature_ablation_mode == "none":
            base_tag = "none"
        elif self._feature_ablation_family:
            base_tag = f"{self._feature_ablation_mode}:{self._feature_ablation_family}"
        else:
            base_tag = self._feature_ablation_mode
        perturb_tag = self._feature_perturb_tag()
        if perturb_tag == "none":
            return base_tag
        if base_tag == "none":
            return perturb_tag
        return f"{base_tag}|{perturb_tag}"

    def _feature_perturb_tag(self) -> str:
        if self.feature_perturb_mode == "none" or self.feature_perturb_apply == "none":
            return "none"
        family_tag = ""
        if self.feature_perturb_family:
            family_tag = ":" + "+".join(self.feature_perturb_family)
        keyword_tag = ""
        if self.feature_perturb_keywords:
            keyword_tag = ":kw=" + "+".join(self.feature_perturb_keywords)
        return (
            f"perturb:{self.feature_perturb_mode}"
            f"@{self.feature_perturb_apply}"
            f"{family_tag}"
            f"{keyword_tag}"
            f"#shift{int(self.feature_perturb_shift)}"
        )

    def _should_apply_feature_perturb(self) -> bool:
        if self.feature_perturb_mode == "none":
            return False
        if self.feature_perturb_apply == "none":
            return False
        if self.feature_perturb_apply == "both":
            return True
        if self.feature_perturb_apply == "train":
            return bool(self.training)
        if self.feature_perturb_apply == "eval":
            return not bool(self.training)
        return False

    def _selected_perturb_families(self) -> List[str]:
        if self.feature_perturb_family:
            return [name for name in self.feature_perturb_family if name in self._feature_family_ablation_indices]
        return [str(name).lower() for name in GROUP_ORDER]

    def _selected_feature_indices_by_keywords(self, feature_dim: int) -> List[int]:
        if feature_dim <= 0:
            return []
        if not self.feature_perturb_keywords:
            return []
        selected = []
        for col_name, col_idx in dict(self._feature_col2idx or {}).items():
            col_text = str(col_name).lower()
            if any(keyword in col_text for keyword in self.feature_perturb_keywords):
                selected.append(int(col_idx))
        return sorted({int(idx) for idx in selected if 0 <= int(idx) < int(feature_dim)})

    def _selected_feature_indices_for_perturb(self, feature_dim: int) -> List[int]:
        if feature_dim <= 0:
            return []
        if self.feature_perturb_keywords:
            # Explicit keyword targeting has priority over family-based selection.
            return self._selected_feature_indices_by_keywords(feature_dim)
        families = self._selected_perturb_families()
        selected = []
        for family in families:
            selected.extend(list(self._feature_family_ablation_indices.get(family, []) or []))
        if selected:
            return sorted({int(idx) for idx in selected if 0 <= int(idx) < int(feature_dim)})
        return list(range(int(feature_dim)))

    @staticmethod
    def _copy_indices(
        out: torch.Tensor,
        src: torch.Tensor,
        *,
        dst_indices: List[int],
        src_indices: List[int],
    ) -> None:
        if not dst_indices or not src_indices:
            return
        width = min(len(dst_indices), len(src_indices))
        if width <= 0:
            return
        out[..., dst_indices[:width]] = src[..., src_indices[:width]]

    def _apply_role_swap(self, feat: torch.Tensor) -> torch.Tensor:
        out = feat.clone()
        if not self.feature_perturb_family:
            # Canonical default swap pair for full-family role sanity.
            pairs = [("tempo", "exposure"), ("focus", "memory")]
        else:
            families = self._selected_perturb_families()
            if len(families) == 1:
                fam = str(families[0]).lower()
                pair_lookup = {
                    "tempo": "exposure",
                    "exposure": "tempo",
                    "focus": "memory",
                    "memory": "focus",
                }
                mate = pair_lookup.get(fam, "")
                pairs = [(fam, mate)] if mate else []
            else:
                pairs = []
                cursor = 0
                while cursor + 1 < len(families):
                    pairs.append((str(families[cursor]).lower(), str(families[cursor + 1]).lower()))
                    cursor += 2

        for left, right in pairs:
            left_idx = list(self._feature_family_ablation_indices.get(left, []) or [])
            right_idx = list(self._feature_family_ablation_indices.get(right, []) or [])
            width = min(len(left_idx), len(right_idx))
            if width <= 0:
                continue
            li = left_idx[:width]
            ri = right_idx[:width]
            out[..., li] = feat[..., ri]
            out[..., ri] = feat[..., li]
        return out

    def _apply_stage_mismatch(self, feat: torch.Tensor) -> torch.Tensor:
        out = feat.clone()
        selected_families = set(self._selected_perturb_families())
        rotate_map = {"macro": "mid", "mid": "micro", "micro": "macro"}
        for dst_stage, src_stage in rotate_map.items():
            if selected_families:
                dst_idx = []
                src_idx = []
                for fam in selected_families:
                    dst_idx.extend(list((self._feature_stage_family_indices.get(dst_stage, {}) or {}).get(fam, []) or []))
                    src_idx.extend(list((self._feature_stage_family_indices.get(src_stage, {}) or {}).get(fam, []) or []))
            else:
                dst_idx = list(self._feature_stage_indices.get(dst_stage, []) or [])
                src_idx = list(self._feature_stage_indices.get(src_stage, []) or [])
            self._copy_indices(out, feat, dst_indices=dst_idx, src_indices=src_idx)
        return out

    def _apply_feature_perturb(self, feat: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(feat):
            return feat
        if feat.ndim != 3:
            return feat
        if feat.size(-1) <= 0:
            return feat
        if not self._should_apply_feature_perturb():
            return feat

        mode = self.feature_perturb_mode
        selected_idx = self._selected_feature_indices_for_perturb(int(feat.size(-1)))
        if mode == "zero":
            if not selected_idx:
                return torch.zeros_like(feat)
            out = feat.clone()
            out[..., selected_idx] = 0.0
            return out
        if mode == "shuffle":
            if feat.size(0) <= 1:
                return feat
            perm = torch.randperm(feat.size(0), device=feat.device)
            if not selected_idx:
                return feat.index_select(0, perm)
            out = feat.clone()
            shuffled = feat.index_select(0, perm)
            out[..., selected_idx] = shuffled[..., selected_idx]
            return out
        if mode == "global_permute":
            flat = feat.reshape(-1, feat.size(-1))
            if flat.size(0) <= 1:
                return feat
            perm = torch.randperm(flat.size(0), device=feat.device)
            if not selected_idx:
                return flat.index_select(0, perm).view_as(feat)
            out = feat.clone().reshape(-1, feat.size(-1))
            src = flat.index_select(0, perm)
            out[:, selected_idx] = src[:, selected_idx]
            return out.view_as(feat)
        if mode == "batch_permute":
            if feat.size(1) <= 1:
                return feat
            out = feat.clone()
            for batch_idx in range(feat.size(0)):
                perm = torch.randperm(feat.size(1), device=feat.device)
                if not selected_idx:
                    out[batch_idx] = feat[batch_idx].index_select(0, perm)
                else:
                    out[batch_idx, :, selected_idx] = feat[batch_idx].index_select(0, perm)[:, selected_idx]
            return out
        if mode == "family_permute":
            if feat.size(0) <= 1:
                return feat
            out = feat.clone()
            for family in self._selected_perturb_families():
                fam_idx = list(self._feature_family_ablation_indices.get(family, []) or [])
                if not fam_idx:
                    continue
                perm = torch.randperm(feat.size(0), device=feat.device)
                out[..., fam_idx] = feat.index_select(0, perm)[..., fam_idx]
            return out
        if mode == "position_shift":
            shift = int(self.feature_perturb_shift)
            if feat.size(1) <= 1 or shift <= 0:
                return feat
            if not selected_idx:
                return torch.roll(feat, shifts=shift, dims=1)
            out = feat.clone()
            out[..., selected_idx] = torch.roll(feat[..., selected_idx], shifts=shift, dims=1)
            return out
        if mode == "role_swap":
            return self._apply_role_swap(feat)
        if mode == "stage_mismatch":
            return self._apply_stage_mismatch(feat)
        return feat

    @property
    def feature_family_names(self) -> List[str]:
        return list(GROUP_ORDER)

    def begin_diagnostic_eval(self, *, split_name: str) -> None:
        if not self.routerec_diag_logging or not self._supports_diag:
            self._diag_collector = None
            return
        self._diag_collector = N3DiagnosticCollector(
            split_name=split_name,
            stage_family_features=self.feature_spec["stage_family_features"],
            stage_expert_names=self._stage_expert_names,
            stage_router_granularity=self.stage_router_granularity,
            all_feature_columns=self.feature_base_fields,
            max_positions=int(self.max_seq_length),
            feature_mode=self._feature_ablation_tag(),
            consistency_pairs=int(self.route_consistency_pairs),
        )

    def end_diagnostic_eval(self) -> Optional[dict]:
        if self._diag_collector is None:
            return None
        payload = self._diag_collector.finalize()
        self._diag_collector = None
        return payload

    def _apply_feature_ablation(self, feat: torch.Tensor) -> torch.Tensor:
        family_indices = []
        if self._feature_ablation_family:
            family_indices = list(self._feature_family_ablation_indices.get(self._feature_ablation_family, []) or [])
            if not family_indices:
                return feat
        if self._feature_ablation_mode == "zero":
            if family_indices:
                out = feat.clone()
                out[..., family_indices] = 0.0
                return out
            return torch.zeros_like(feat)
        if self._feature_ablation_mode == "shuffle" and feat.size(0) > 1:
            perm = torch.randperm(feat.size(0), device=feat.device)
            if family_indices:
                out = feat.clone()
                shuffled = feat.index_select(0, perm)
                out[..., family_indices] = shuffled[..., family_indices]
                return out
            return feat.index_select(0, perm)
        return feat

    def _gather_features(self, interaction) -> torch.Tensor:
        feat_list = []
        item_seq = interaction[self.ITEM_SEQ]
        batch_size, seq_len = item_seq.shape
        device = item_seq.device

        for base_field, seq_field in zip(self.feature_base_fields, self.feature_fields):
            if seq_field in interaction:
                values = interaction[seq_field].float()
            elif base_field in interaction:
                values = interaction[base_field].float()
                if values.dim() == 1:
                    values = values.unsqueeze(1)
                if values.dim() == 2 and values.size(1) == 1:
                    values = values.expand(-1, seq_len)
                elif values.dim() == 2 and values.size(1) != seq_len:
                    values = values[:, :1].expand(-1, seq_len)
                if base_field not in self._scalar_fallback_field_once:
                    logger.info("Feature field '%s' using scalar fallback broadcast.", base_field)
                    self._scalar_fallback_field_once.add(base_field)
            else:
                values = torch.zeros(batch_size, seq_len, device=device)
                if seq_field not in self._missing_feature_field_once:
                    logger.warning("Feature field '%s' not found - using zeros.", seq_field)
                    self._missing_feature_field_once.add(seq_field)

            if values.dim() == 1:
                values = values.unsqueeze(1).expand(-1, seq_len)
            elif values.dim() == 3 and values.size(-1) == 1:
                values = values.squeeze(-1)
            feat_list.append(values.to(device=device))

        feat = torch.stack(feat_list, dim=-1)
        feat = self._apply_feature_ablation(feat)
        feat = self._apply_feature_perturb(feat)
        return feat

    def _update_eval_diagnostics(self, *, interaction, feat, item_seq_len, aux_data) -> None:
        if self.training or self._diag_collector is None:
            return
        try:
            self._diag_collector.update(
                interaction=interaction,
                feat=feat,
                item_seq_len=item_seq_len,
                aux_data=aux_data,
            )
        except Exception:
            pass

    def _compute_rule_agreement_loss(
        self,
        gate_logits: Dict[str, torch.Tensor],
        router_aux: Dict[str, Dict[str, object]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        target_map = dict((router_aux or {}).get("rule_target_logits", {}) or {})
        if not target_map:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for stage_key, target_logits in target_map.items():
            student_logits = gate_logits.get(stage_key)
            if student_logits is None:
                continue
            mask = self._sequence_valid_mask(item_seq_len, student_logits.size(1), student_logits.device).float()
            teacher = F.softmax(target_logits.detach(), dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            kl = F.kl_div(student_log_probs, teacher, reduction="none").sum(dim=-1)
            denom = mask.sum().clamp(min=1.0)
            losses.append((kl * mask).sum() / denom)
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_group_coverage_reward(
        self,
        router_aux: Dict[str, Dict[str, object]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        group_map = dict((router_aux or {}).get("group_weights", {}) or {})
        if not group_map:
            return self.item_embedding.weight.new_tensor(0.0)
        rewards = []
        for group_weights in group_map.values():
            if group_weights.size(-1) <= 1:
                continue
            mask = self._sequence_valid_mask(item_seq_len, group_weights.size(1), group_weights.device).float().unsqueeze(-1)
            denom = mask.sum().clamp(min=1.0)
            usage = (group_weights * mask).sum(dim=(0, 1)) / denom
            usage = usage / usage.sum().clamp(min=1e-8)
            entropy = -(usage.clamp(min=1e-8) * usage.clamp(min=1e-8).log()).sum()
            rewards.append(entropy / math.log(float(group_weights.size(-1))))
        return torch.stack(rewards).mean() if rewards else self.item_embedding.weight.new_tensor(0.0)

    def _compute_group_prior_alignment_loss(
        self,
        router_aux: Dict[str, Dict[str, object]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        group_map = dict((router_aux or {}).get("group_weights", {}) or {})
        prior_map = dict((router_aux or {}).get("group_prior", {}) or {})
        if not group_map or not prior_map:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for stage_key in sorted(set(group_map) & set(prior_map)):
            student = group_map.get(stage_key)
            teacher = prior_map.get(stage_key)
            if student is None or teacher is None:
                continue
            if (not torch.is_tensor(student)) or (not torch.is_tensor(teacher)):
                continue
            if student.size(-1) <= 1 or teacher.size(-1) != student.size(-1):
                continue

            teacher_prob = teacher.detach().to(device=student.device, dtype=student.dtype)
            # group_prior can be either [B, G] (session-level) or [B, S, G] (token-level).
            # Align to student [B, S, G] (or [B, G]) shape using safe broadcasting.
            while teacher_prob.ndim < student.ndim:
                teacher_prob = teacher_prob.unsqueeze(1)
            if teacher_prob.ndim != student.ndim:
                continue
            try:
                teacher_prob = torch.broadcast_to(teacher_prob, student.shape)
            except RuntimeError:
                continue

            teacher_prob = teacher_prob / teacher_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            student_prob = student / student.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kl = (teacher_prob * (teacher_prob.clamp(min=1e-8).log() - student_prob.clamp(min=1e-8).log())).sum(dim=-1)
            if student.ndim >= 3:
                mask = self._sequence_valid_mask(item_seq_len, student.size(1), student.device).float()
            else:
                mask = torch.ones_like(kl, dtype=kl.dtype, device=kl.device)
            denom = mask.sum().clamp(min=1.0)
            losses.append((kl * mask).sum() / denom)
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_factored_group_balance_loss(
        self,
        router_aux: Dict[str, Dict[str, object]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Load balance loss applied to factored group router logits.

        Penalizes collapse of group-level routing (when the factored router always
        sends everything to one group). Applied only when factored_group_logits are present.
        Uses the same mean-prob * mean-prob-softmax formulation as standard balance loss.
        """
        fgl_map = dict((router_aux or {}).get("factored_group_logits", {}) or {})
        if not fgl_map:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for stage_key, fgl in fgl_map.items():
            if not torch.is_tensor(fgl) or fgl.size(-1) <= 1:
                continue
            group_probs = F.softmax(fgl, dim=-1)  # (..., n_groups)
            seq_len = fgl.size(1) if fgl.ndim == 3 else 1
            mask = self._sequence_valid_mask(item_seq_len, seq_len, fgl.device).float()
            if fgl.ndim == 3:
                valid_unsq = mask.unsqueeze(-1)
                denom = mask.sum().clamp(min=1.0)
                mean_prob = (group_probs * valid_unsq).sum(dim=(0, 1)) / denom
            else:
                mean_prob = group_probs.mean(dim=0)
            n_grps = float(mean_prob.size(-1))
            # balance_loss = n_groups * sum(mean_prob_i * mean_prob_i)  [encourages uniform distribution]
            losses.append(n_grps * (mean_prob * mean_prob).sum())
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    @staticmethod
    def _base_stage_name(stage_key: str) -> str:
        text = str(stage_key or "")
        if "@" in text:
            text = text.split("@", 1)[0]
        if "." in text:
            text = text.split(".", 1)[0]
        return text

    def _session_gate_prob(
        self,
        stage_weights: torch.Tensor,
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._sequence_valid_mask(item_seq_len, stage_weights.size(1), stage_weights.device).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (stage_weights * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _compute_route_smoothness_loss(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for stage_key, weights in (gate_weights or {}).items():
            if not torch.is_tensor(weights) or weights.ndim != 3 or weights.size(1) <= 1:
                continue
            stage_name = self._base_stage_name(stage_key)
            stage_w = float(self.route_smoothness_stage_weight.get(stage_name, 1.0))
            if stage_w <= 0:
                continue
            valid = self._sequence_valid_mask(item_seq_len, weights.size(1), weights.device).float()
            pair_mask = valid[:, 1:] * valid[:, :-1]
            if pair_mask.sum() <= 0:
                continue
            delta = (weights[:, 1:, :] - weights[:, :-1, :]).abs().sum(dim=-1)
            losses.append(stage_w * ((delta * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)))
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_route_consistency_loss(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: torch.Tensor,
        feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feat is None or not torch.is_tensor(feat) or feat.ndim != 3:
            return self.item_embedding.weight.new_tensor(0.0)
        batch_size = int(feat.size(0))
        if batch_size <= 1:
            return self.item_embedding.weight.new_tensor(0.0)
        feat_repr = feat.mean(dim=1)
        feat_norm = F.normalize(feat_repr, p=2, dim=-1)
        sim = feat_norm @ feat_norm.transpose(0, 1)
        sim.fill_diagonal_(-1e9)
        k = min(int(self.route_consistency_pairs), batch_size - 1)
        if k <= 0:
            return self.item_embedding.weight.new_tensor(0.0)
        neigh_pack = sim.topk(k=k, dim=1)
        neigh_idx = neigh_pack.indices
        neigh_sim = neigh_pack.values

        losses = []
        for weights in (gate_weights or {}).values():
            if not torch.is_tensor(weights) or weights.ndim != 3:
                continue
            session_prob = self._session_gate_prob(weights, item_seq_len).clamp(min=1e-8)
            pi = session_prob.unsqueeze(1).expand(-1, k, -1)
            pj = session_prob.index_select(0, neigh_idx.reshape(-1)).reshape(batch_size, k, -1).clamp(min=1e-8)
            m = 0.5 * (pi + pj)
            js = 0.5 * ((pi * (pi.log() - m.log())).sum(dim=-1) + (pj * (pj.log() - m.log())).sum(dim=-1))
            min_sim = float(self.route_consistency_min_sim)
            if min_sim > -1.0:
                pair_mask = neigh_sim >= min_sim
                if bool(pair_mask.any()):
                    losses.append(js[pair_mask].mean())
            else:
                losses.append(js.mean())
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_route_separation_loss(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: torch.Tensor,
        feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feat is None or not torch.is_tensor(feat) or feat.ndim != 3:
            return self.item_embedding.weight.new_tensor(0.0)
        batch_size = int(feat.size(0))
        if batch_size <= 1:
            return self.item_embedding.weight.new_tensor(0.0)

        feat_repr = feat.mean(dim=1)
        feat_norm = F.normalize(feat_repr, p=2, dim=-1)
        sim = feat_norm @ feat_norm.transpose(0, 1)
        sim.fill_diagonal_(1e9)
        k = min(int(self.route_separation_pairs), batch_size - 1)
        if k <= 0:
            return self.item_embedding.weight.new_tensor(0.0)

        far_pack = (-sim).topk(k=k, dim=1)
        far_idx = far_pack.indices
        far_sim = sim.gather(1, far_idx)
        max_sim = float(self.route_separation_max_sim)
        margin = max(float(self.route_separation_margin), 0.0)

        losses = []
        for weights in (gate_weights or {}).values():
            if not torch.is_tensor(weights) or weights.ndim != 3:
                continue
            session_prob = self._session_gate_prob(weights, item_seq_len).clamp(min=1e-8)
            pi = session_prob.unsqueeze(1).expand(-1, k, -1)
            pj = session_prob.index_select(0, far_idx.reshape(-1)).reshape(batch_size, k, -1).clamp(min=1e-8)
            m = 0.5 * (pi + pj)
            js = 0.5 * ((pi * (pi.log() - m.log())).sum(dim=-1) + (pj * (pj.log() - m.log())).sum(dim=-1))

            pair_loss = torch.relu(margin - js)
            if max_sim < 1.0:
                pair_mask = far_sim <= max_sim
                if bool(pair_mask.any()):
                    losses.append(pair_loss[pair_mask].mean())
            else:
                losses.append(pair_loss.mean())

        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_route_sharpness_loss(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for weights in (gate_weights or {}).values():
            if not torch.is_tensor(weights) or weights.ndim != 3:
                continue
            mask = self._sequence_valid_mask(item_seq_len, weights.size(1), weights.device).float()
            entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(dim=-1)
            losses.append((entropy * mask).sum() / mask.sum().clamp(min=1.0))
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_route_monopoly_loss(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for weights in (gate_weights or {}).values():
            if not torch.is_tensor(weights) or weights.ndim != 3:
                continue
            mask = self._sequence_valid_mask(item_seq_len, weights.size(1), weights.device)
            top1 = weights.argmax(dim=-1)
            active = top1[mask]
            if active.numel() <= 0:
                continue
            n_exp = int(weights.size(-1))
            usage = torch.bincount(active, minlength=n_exp).float() / float(active.numel())
            tau = float(self.route_monopoly_tau)
            if tau <= 0:
                tau = min(0.45, 2.5 / max(float(n_exp), 1.0))
            losses.append(torch.relu(usage - tau).pow(2).sum())
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_route_prior_loss(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: torch.Tensor,
        feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feat is None or not torch.is_tensor(feat) or feat.ndim != 3:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        feat_mean = feat.mean(dim=1)
        for stage_key, weights in (gate_weights or {}).items():
            if not torch.is_tensor(weights) or weights.ndim != 3:
                continue
            stage_name = self._base_stage_name(stage_key)
            family_map = dict(self.feature_spec.get("stage_family_features", {}).get(stage_name, {}) or {})
            expert_names = list(self._stage_expert_names.get(stage_name, []) or [])
            if not family_map or not expert_names:
                continue

            family_scores = []
            family_names = []
            for family_name, columns in family_map.items():
                idx = [self._feature_col2idx[c] for c in columns if c in self._feature_col2idx]
                if not idx:
                    continue
                sel = feat_mean.index_select(-1, torch.tensor(idx, device=feat_mean.device))
                family_scores.append(sel.mean(dim=-1))
                family_names.append(family_name)
            if not family_scores:
                continue
            fam_logits = torch.stack(family_scores, dim=-1)
            fam_prob = F.softmax(fam_logits / max(float(self.feature_group_prior_temperature), 1e-6), dim=-1)

            prior = torch.zeros(weights.size(0), weights.size(-1), device=weights.device, dtype=weights.dtype)
            for fam_idx, fam_name in enumerate(family_names):
                matched = [i for i, en in enumerate(expert_names) if fam_name.lower() in str(en).lower()]
                if not matched:
                    continue
                share = fam_prob[:, fam_idx] / float(len(matched))
                for ex_idx in matched:
                    prior[:, ex_idx] = prior[:, ex_idx] + share
            prior = prior + 1e-8
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            student = self._session_gate_prob(weights, item_seq_len).clamp(min=1e-8)
            if self.route_prior_bias_scale > 0:
                blended = student + float(self.route_prior_bias_scale) * prior
                student = blended / blended.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            losses.append((student * (student.log() - prior.log())).sum(dim=-1).mean())
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def forward(self, item_seq, item_seq_len, feat=None, routing_item_seq_len=None):
        batch_size, seq_len = item_seq.shape
        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)
        attention_mask = self.get_attention_mask(item_seq)

        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, object]] = {}
        dense_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        stage_merge_weights = None
        stage_merge_logits = None

        tokens, gate_weights, gate_logits, router_aux, dense_aux, stage_merge_weights, stage_merge_logits = self.stage_executor(
            hidden=tokens,
            attention_mask=attention_mask,
            feat=feat,
            item_seq_len=item_seq_len,
            routing_item_seq_len=routing_item_seq_len,
        )

        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = tokens.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "stage_merge_weights": stage_merge_weights,
            "stage_merge_logits": stage_merge_logits,
            "ffn_moe_weights": {},
            "router_aux": router_aux,
            "dense_aux": dense_aux,
        }
        return seq_output, aux_data

    def calculate_loss(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        feat = self._gather_features(interaction) if self._requires_features else None
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

        router_aux = aux_data.get("router_aux", {}) or {}
        z_logit_map = dict(router_aux.get("learned_gate_logits", {}) or {})
        if not z_logit_map:
            z_logit_map = aux_data.get("gate_logits", {})
        if self.z_loss_lambda > 0:
            aux_loss = aux_loss + self.z_loss_lambda * self._compute_router_z_loss(z_logit_map, item_seq_len)

        entropy_scale = self._aux_until_active(self.gate_entropy_until)
        if self.gate_entropy_lambda > 0 and entropy_scale > 0:
            aux_loss = aux_loss - (self.gate_entropy_lambda * entropy_scale) * self._compute_gate_entropy_reg(
                aux_data.get("gate_weights", {}),
                item_seq_len,
            )
        if self.rule_agreement_lambda > 0:
            aux_loss = aux_loss + self.rule_agreement_lambda * self._compute_rule_agreement_loss(
                aux_data.get("gate_logits", {}),
                router_aux,
                item_seq_len,
            )
        if self.group_coverage_lambda > 0:
            aux_loss = aux_loss - self.group_coverage_lambda * self._compute_group_coverage_reward(
                router_aux,
                item_seq_len,
            )
        if self.group_prior_align_lambda > 0:
            aux_loss = aux_loss + self.group_prior_align_lambda * self._compute_group_prior_alignment_loss(
                router_aux,
                item_seq_len,
            )
        if self.factored_group_balance_lambda > 0:
            aux_loss = aux_loss + self.factored_group_balance_lambda * self._compute_factored_group_balance_loss(
                router_aux,
                item_seq_len,
            )
        if self.route_smoothness_lambda > 0:
            aux_loss = aux_loss + self.route_smoothness_lambda * self._compute_route_smoothness_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len,
            )
        if self.route_consistency_lambda > 0:
            aux_loss = aux_loss + self.route_consistency_lambda * self._compute_route_consistency_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len,
                feat,
            )
        if self.route_separation_lambda > 0:
            aux_loss = aux_loss + self.route_separation_lambda * self._compute_route_separation_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len,
                feat,
            )
        if self.route_sharpness_lambda > 0:
            aux_loss = aux_loss + self.route_sharpness_lambda * self._compute_route_sharpness_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len,
            )
        if self.route_monopoly_lambda > 0:
            aux_loss = aux_loss + self.route_monopoly_lambda * self._compute_route_monopoly_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len,
            )
        if self.route_prior_lambda > 0:
            aux_loss = aux_loss + self.route_prior_lambda * self._compute_route_prior_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len,
                feat,
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
        feat = self._gather_features(interaction) if self._requires_features else None
        seq_output, aux_data = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )
        self._update_eval_diagnostics(
            interaction=interaction,
            feat=feat,
            item_seq_len=item_seq_len,
            aux_data=aux_data,
        )
        test_item_emb = self.item_embedding(test_item)
        return (seq_output * test_item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)
        feat = self._gather_features(interaction) if self._requires_features else None
        seq_output, aux_data = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )
        self._update_eval_diagnostics(
            interaction=interaction,
            feat=feat,
            item_seq_len=item_seq_len,
            aux_data=aux_data,
        )
        return seq_output @ self.item_embedding.weight.T
