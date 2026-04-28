"""Feature configuration helpers for RouteRec and feature_added_v3."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ...datasets import resolve_dataset_dir


GROUP_ORDER: Tuple[str, ...] = ("Tempo", "Focus", "Memory", "Exposure")
STAGE_NAMES: Tuple[str, ...] = ("macro", "mid", "micro")
MACRO_SCOPE_BY_WINDOW = {
    5: "macro5",
    10: "macro10",
}

FAMILIES: Dict[str, Dict[str, List[str]]] = {
    "macro5": {
        "Tempo": ["mac5_ctx_valid_r", "mac5_gap_last", "mac5_pace_mean", "mac5_pace_trend"],
        "Focus": ["mac5_theme_ent_mean", "mac5_theme_top1_mean", "mac5_theme_repeat_r", "mac5_theme_shift_r"],
        "Memory": ["mac5_repeat_mean", "mac5_adj_cat_overlap_mean", "mac5_adj_item_overlap_mean", "mac5_repeat_trend"],
        "Exposure": ["mac5_pop_mean", "mac5_pop_std_mean", "mac5_pop_ent_mean", "mac5_pop_trend"],
    },
    "macro10": {
        "Tempo": ["mac10_ctx_valid_r", "mac10_gap_last", "mac10_pace_mean", "mac10_pace_trend"],
        "Focus": ["mac10_theme_ent_mean", "mac10_theme_top1_mean", "mac10_theme_repeat_r", "mac10_theme_shift_r"],
        "Memory": ["mac10_repeat_mean", "mac10_adj_cat_overlap_mean", "mac10_adj_item_overlap_mean", "mac10_repeat_trend"],
        "Exposure": ["mac10_pop_mean", "mac10_pop_std_mean", "mac10_pop_ent_mean", "mac10_pop_trend"],
    },
    "mid": {
        "Tempo": ["mid_valid_r", "mid_int_mean", "mid_int_std", "mid_sess_age"],
        "Focus": ["mid_cat_ent", "mid_cat_top1", "mid_cat_switch_r", "mid_cat_uniq_r"],
        "Memory": ["mid_item_uniq_r", "mid_repeat_r", "mid_novel_r", "mid_max_run_i"],
        "Exposure": ["mid_pop_mean", "mid_pop_std", "mid_pop_ent", "mid_pop_trend"],
    },
    "micro": {
        "Tempo": ["mic_valid_r", "mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid"],
        "Focus": ["mic_cat_switch_now", "mic_last_cat_mismatch_r", "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r"],
        "Memory": ["mic_is_recons", "mic_suffix_recons_r", "mic_suffix_uniq_i", "mic_suffix_max_run_i"],
        "Exposure": ["mic_last_pop", "mic_suffix_pop_std", "mic_suffix_pop_ent", "mic_pop_delta_vs_mid"],
    },
}


def _unique_preserve_order(rows: Iterable[Iterable[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


ALL_FEATURE_COLUMNS: List[str] = _unique_preserve_order(
    [group_cols for scope in ("macro5", "macro10", "mid", "micro") for group_cols in FAMILIES[scope].values()]
)


def feature_list_field(col: str) -> str:
    return f"{col}_list"


def build_column_to_index(columns: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(columns)}


def normalize_macro_window(raw_value) -> int:
    try:
        window = int(raw_value)
    except (TypeError, ValueError):
        window = 5
    return 10 if window == 10 else 5


def default_stage_family_mask() -> Dict[str, List[str]]:
    return {stage: list(GROUP_ORDER) for stage in STAGE_NAMES}


def normalize_stage_family_mask(raw_value) -> Dict[str, List[str]]:
    default_mask = default_stage_family_mask()
    if raw_value is None:
        return default_mask
    if isinstance(raw_value, (list, tuple)):
        selected = [str(v).strip() for v in raw_value if str(v).strip() in GROUP_ORDER]
        selected = selected or list(GROUP_ORDER)
        return {stage: list(selected) for stage in STAGE_NAMES}
    if not isinstance(raw_value, dict):
        return default_mask

    out = {}
    for stage in STAGE_NAMES:
        raw_stage = raw_value.get(stage, raw_value.get(stage.capitalize(), None))
        if raw_stage is None:
            out[stage] = list(default_mask[stage])
            continue
        if isinstance(raw_stage, str):
            values = [raw_stage]
        else:
            values = list(raw_stage)
        selected = [str(v).strip() for v in values if str(v).strip() in GROUP_ORDER]
        out[stage] = selected or list(default_mask[stage])
    return out


def _parse_positive_int(raw_value: Any) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def normalize_stage_family_topk(raw_value) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {stage: {} for stage in STAGE_NAMES}
    if raw_value is None:
        return out

    # Allow scalar shorthand: apply same top-k to all stages/families.
    if isinstance(raw_value, (int, float, str)):
        topk = _parse_positive_int(raw_value)
        if topk <= 0:
            return out
        return {
            stage: {family: int(topk) for family in GROUP_ORDER}
            for stage in STAGE_NAMES
        }

    if not isinstance(raw_value, dict):
        return out

    # Allow flat family map shorthand: {"Tempo": 2, "Focus": 1, ...}
    flat_family_map = {}
    for family in GROUP_ORDER:
        val = raw_value.get(family)
        if val is None:
            continue
        topk = _parse_positive_int(val)
        if topk > 0:
            flat_family_map[family] = int(topk)
    if flat_family_map:
        for stage in STAGE_NAMES:
            out[stage] = dict(flat_family_map)
        return out

    for stage in STAGE_NAMES:
        stage_raw = raw_value.get(stage, raw_value.get(stage.capitalize(), None))
        if stage_raw is None:
            continue
        if isinstance(stage_raw, (int, float, str)):
            topk = _parse_positive_int(stage_raw)
            if topk > 0:
                out[stage] = {family: int(topk) for family in GROUP_ORDER}
            continue
        if not isinstance(stage_raw, dict):
            continue
        stage_map: Dict[str, int] = {}
        for family in GROUP_ORDER:
            val = stage_raw.get(family)
            if val is None:
                continue
            topk = _parse_positive_int(val)
            if topk > 0:
                stage_map[family] = int(topk)
        out[stage] = stage_map
    return out


def normalize_stage_family_custom(raw_value) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {stage: {} for stage in STAGE_NAMES}
    if raw_value is None or not isinstance(raw_value, dict):
        return out

    # Allow flat family map shorthand: {"Tempo": [...], "Focus": [...], ...}
    flat_family_map: Dict[str, List[str]] = {}
    for family in GROUP_ORDER:
        val = raw_value.get(family)
        if val is None:
            continue
        if isinstance(val, str):
            selected = [val.strip()] if val.strip() else []
        else:
            try:
                selected = [str(v).strip() for v in list(val) if str(v).strip()]
            except Exception:
                selected = []
        if selected:
            flat_family_map[family] = selected
    if flat_family_map:
        for stage in STAGE_NAMES:
            out[stage] = {family: list(cols) for family, cols in flat_family_map.items()}
        return out

    for stage in STAGE_NAMES:
        stage_raw = raw_value.get(stage, raw_value.get(stage.capitalize(), None))
        if stage_raw is None or not isinstance(stage_raw, dict):
            continue
        stage_map: Dict[str, List[str]] = {}
        for family in GROUP_ORDER:
            val = stage_raw.get(family)
            if val is None:
                continue
            if isinstance(val, str):
                selected = [val.strip()] if val.strip() else []
            else:
                try:
                    selected = [str(v).strip() for v in list(val) if str(v).strip()]
                except Exception:
                    selected = []
            if selected:
                stage_map[family] = selected
        out[stage] = stage_map
    return out


def normalize_feature_drop_keywords(raw_value) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        tokens = [tok.strip().lower() for tok in raw_value.split(",") if tok.strip()]
        return tokens
    if isinstance(raw_value, (list, tuple, set)):
        return [str(tok).strip().lower() for tok in raw_value if str(tok).strip()]
    return []


def stage_scope_map(macro_history_window: int) -> Dict[str, str]:
    macro_window = normalize_macro_window(macro_history_window)
    return {
        "macro": MACRO_SCOPE_BY_WINDOW[macro_window],
        "mid": "mid",
        "micro": "micro",
    }


def build_stage_feature_spec(
    *,
    macro_history_window: int,
    stage_feature_family_mask,
    stage_feature_family_topk=None,
    stage_feature_family_custom=None,
    stage_feature_drop_keywords=None,
) -> Dict[str, object]:
    family_mask = normalize_stage_family_mask(stage_feature_family_mask)
    family_topk = normalize_stage_family_topk(stage_feature_family_topk)
    family_custom = normalize_stage_family_custom(stage_feature_family_custom)
    drop_keywords = normalize_feature_drop_keywords(stage_feature_drop_keywords)
    scopes = stage_scope_map(macro_history_window)

    stage_all_features: Dict[str, List[str]] = {}
    stage_family_features: Dict[str, Dict[str, List[str]]] = {}
    stage_experts: Dict[str, OrderedDict[str, List[str]]] = {}

    for stage in STAGE_NAMES:
        scope = scopes[stage]
        scope_families = FAMILIES[scope]
        active_groups = set(family_mask[stage])
        stage_custom = dict(family_custom.get(stage, {}) or {})
        stage_topk = dict(family_topk.get(stage, {}) or {})
        family_features: Dict[str, List[str]] = {}
        for group_name in GROUP_ORDER:
            if group_name not in active_groups:
                family_features[group_name] = []
                continue
            cols = list(scope_families[group_name])
            custom_cols = list(stage_custom.get(group_name, []) or [])
            if custom_cols:
                custom_set = set(custom_cols)
                cols = [name for name in cols if name in custom_set]
            if drop_keywords:
                cols = [
                    name for name in cols
                    if not any(keyword in str(name).lower() for keyword in drop_keywords)
                ]
            topk = _parse_positive_int(stage_topk.get(group_name))
            if topk > 0:
                cols = cols[:topk]
            family_features[group_name] = cols
        stage_family_features[stage] = family_features
        stage_all_features[stage] = _unique_preserve_order(
            family_features[group_name] for group_name in GROUP_ORDER if family_features[group_name]
        )
        stage_experts[stage] = OrderedDict(
            (group_name, list(family_features[group_name]))
            for group_name in GROUP_ORDER
        )

    return {
        "macro_history_window": normalize_macro_window(macro_history_window),
        "stage_scopes": scopes,
        "stage_family_mask": family_mask,
        "stage_family_topk": family_topk,
        "stage_family_custom": family_custom,
        "stage_feature_drop_keywords": list(drop_keywords),
        "stage_all_features": stage_all_features,
        "stage_family_features": stage_family_features,
        "stage_experts": stage_experts,
        "all_feature_columns": list(ALL_FEATURE_COLUMNS),
    }


def load_feature_meta_v3(*, data_path: str, dataset: str) -> dict:
    dataset_dir = resolve_dataset_dir(dataset=dataset, data_path=data_path)
    candidates = []
    if dataset_dir is not None:
        candidates.append(dataset_dir / "feature_meta_v3.json")

    data_root = Path(str(data_path))
    candidates.extend(
        [
            data_root / str(dataset) / "feature_meta_v3.json",
            data_root / "feature_meta_v3.json",
        ]
    )

    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def validate_feature_meta_v3(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {}
    windows = meta.get("macro_windows", [])
    families = meta.get("families", {})
    ok_windows = set(int(v) for v in windows if str(v).isdigit())
    if not {5, 10}.issubset(ok_windows):
        raise ValueError(f"feature_meta_v3.macro_windows must contain 5 and 10, got {windows}")
    for family_name in ("macro5", "macro10", "mid", "micro"):
        groups = families.get(family_name, {})
        if not isinstance(groups, dict):
            raise ValueError(f"feature_meta_v3.families.{family_name} missing or invalid")
        for group_name in GROUP_ORDER:
            if group_name not in groups:
                raise ValueError(
                    f"feature_meta_v3.families.{family_name} missing group {group_name}"
                )
    return meta
