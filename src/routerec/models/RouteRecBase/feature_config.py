"""
Feature configuration: column names → stage/expert assignment.

Each stage (Macro/Mid/Micro) defines 4 base experts. Runtime expert count can
be expanded by ``expert_scale`` (cloned experts with identical feature subsets).
This module defines the
canonical mapping and provides helpers to build index tensors for efficient
gathering at runtime.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Stage → Expert → Feature columns
# ---------------------------------------------------------------------------

MACRO_EXPERTS: OrderedDict[str, List[str]] = OrderedDict(
    M1_Retention=[
        "mac_user_level", "mac_sess_gap", "mac_is_new", "mac_hist_len",
    ],
    M2_Rhythm=[
        "mac_time_sin", "mac_time_cos", "mac_is_weekend", "mac_hist_speed",
        "mac_sess_gap",
    ],
    M3_Spectrum=[
        "mac_hist_ent", "mac_hist_uniq_r", "mac_hist_top1", "mac_hist_switch_r",
    ],
    M4_Mainstream=[
        "mac_hist_pop_avg", "mac_hist_pop_std", "mac_hist_ent", "mac_hist_speed",
    ],
)

MID_EXPERTS: OrderedDict[str, List[str]] = OrderedDict(
    m2_1_DeepDiver=[
        "mid_win_ent", "mid_win_top1", "mid_int_std", "mid_sess_time",
    ],
    m2_2_MoodSurfer=[
        "mid_pop_avg", "mid_pop_drift", "mid_ent_drift", "mid_win_switch",
    ],
    m2_3_Pacer=[
        "mid_int_avg", "mid_accel", "mid_last_dev", "mid_valid_r",
    ],
    m2_4_Repeat=[
        "mid_novel_r", "mid_uniq_item_r", "mid_uniq_cat_r", "mid_max_run",
    ],
)

MICRO_EXPERTS: OrderedDict[str, List[str]] = OrderedDict(
    m3_1_Impulse=[
        "mic_last_int", "mic_short_avg", "mic_short_std", "mic_valid_r",
    ],
    m3_2_Hopper=[
        "mic_switch", "mic_uniq_c", "mic_cat_ent", "mic_cat_top1",
    ],
    m3_3_Binger=[
        "mic_is_recons", "mic_uniq_i", "mic_recons_r", "mic_max_run_i",
    ],
    m3_4_Mainstream=[
        "mic_pop_avg", "mic_pop_std", "mic_pop_drift", "mic_pop_ent",
    ],
)

# Convenience list of all stage definitions in order (Macro → Mid → Micro)
STAGES: List[Tuple[str, OrderedDict[str, List[str]]]] = [
    ("macro", MACRO_EXPERTS),
    ("mid",   MID_EXPERTS),
    ("micro", MICRO_EXPERTS),
]

# Stage names
STAGE_NAMES: List[str] = ["macro", "mid", "micro"]

# ---------------------------------------------------------------------------
# Derived: flattened feature lists per stage (union over experts, preserving
# the order of first appearance — used for router input construction).
# ---------------------------------------------------------------------------

def _unique_ordered(lists: List[List[str]]) -> List[str]:
    """Merge multiple feature lists preserving first-appearance order."""
    seen: set = set()
    out: list = []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def get_stage_all_features(stage_experts: OrderedDict[str, List[str]]) -> List[str]:
    """Return union of all feature names used by any expert in a stage."""
    return _unique_ordered(list(stage_experts.values()))


# Pre-computed per-stage feature unions
MACRO_ALL_FEATURES: List[str] = get_stage_all_features(MACRO_EXPERTS)
MID_ALL_FEATURES:   List[str] = get_stage_all_features(MID_EXPERTS)
MICRO_ALL_FEATURES: List[str] = get_stage_all_features(MICRO_EXPERTS)

STAGE_ALL_FEATURES: Dict[str, List[str]] = {
    "macro": MACRO_ALL_FEATURES,
    "mid":   MID_ALL_FEATURES,
    "micro": MICRO_ALL_FEATURES,
}

# Complete ordered list of ALL feature columns loaded from .inter
ALL_FEATURE_COLUMNS: List[str] = _unique_ordered(
    [MACRO_ALL_FEATURES, MID_ALL_FEATURES, MICRO_ALL_FEATURES]
)

# ---------------------------------------------------------------------------
# Index mapping helpers
# ---------------------------------------------------------------------------

def build_column_to_index(columns: List[str]) -> Dict[str, int]:
    """Map column name → position index in the concatenated feature tensor."""
    return {c: i for i, c in enumerate(columns)}


def build_expert_indices(
    stage_experts: OrderedDict[str, List[str]],
    col2idx: Dict[str, int],
) -> List[List[int]]:
    """For each expert in *stage_experts*, return list of indices into the
    concat feature tensor (ordered by ``col2idx``)."""
    return [
        [col2idx[c] for c in feats]
        for feats in stage_experts.values()
    ]


def build_stage_indices(
    stage_all_features: List[str],
    col2idx: Dict[str, int],
) -> List[int]:
    """Indices for the full feature set of a stage (used as router input)."""
    return [col2idx[c] for c in stage_all_features]


# ---------------------------------------------------------------------------
# RecBole field names (with _list suffix for sequence versions)
# ---------------------------------------------------------------------------

def feature_list_field(col: str) -> str:
    """Return the RecBole sequence-field name for a feature column."""
    return col + "_list"


def all_feature_list_fields() -> List[str]:
    """Return sequence-field names for all feature columns."""
    return [feature_list_field(c) for c in ALL_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Expert name helpers (for logging)
# ---------------------------------------------------------------------------

def get_expert_names() -> Dict[str, List[str]]:
    """Return {stage_name: [expert_name, ...]} for all stages."""
    return {
        "macro": list(MACRO_EXPERTS.keys()),
        "mid":   list(MID_EXPERTS.keys()),
        "micro": list(MICRO_EXPERTS.keys()),
    }
