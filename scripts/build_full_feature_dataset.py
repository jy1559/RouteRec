#!/usr/bin/env python3
"""Build a leakage-correct, current-schema full dataset from basic files.

Reads from:
  Datasets/processed/basic/{dataset}/{dataset}.inter
  Datasets/processed/basic/{dataset}/{dataset}.item

Writes to:
  Datasets/processed/final_dataset/{dataset}/{dataset}.inter
  Datasets/processed/final_dataset/{dataset}/{dataset}.item
  Datasets/processed/final_dataset/{dataset}/{dataset}.train.inter
  Datasets/processed/final_dataset/{dataset}/{dataset}.valid.inter
  Datasets/processed/final_dataset/{dataset}/{dataset}.test.inter
  Datasets/processed/final_dataset/{dataset}/feature_meta_v3.json

Feature structure (64 engineered features plus four identity columns):
  macro (mac5 + mac10): 16 each — Tempo / Focus / Memory / Exposure
  mid:                  16      — session-level (session_constant_last)
  micro (mic):          16      — recent-5-interaction level

This is a versioned reconstruction, not an exact reproduction of the lost v3
builder.  It follows the feature contract reverse-engineered from the current
release, fixes the sampled release's train-boundary and ``session_constant_last``
leakage, and records those differences in the output metadata.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASIC_ROOT = REPO_ROOT / "Datasets" / "processed" / "basic"
DEFAULT_OUT_ROOT   = REPO_ROOT / "Datasets" / "processed" / "final_dataset"


# ---------------------------------------------------------------------------
# Feature / normalization definitions (identical to build_beauty_feature_v3)
# ---------------------------------------------------------------------------

FAMILIES = {
    "macro5": {
        "Tempo":    ["mac5_ctx_valid_r", "mac5_gap_last", "mac5_pace_mean", "mac5_pace_trend"],
        "Focus":    ["mac5_theme_ent_mean", "mac5_theme_top1_mean", "mac5_theme_repeat_r", "mac5_theme_shift_r"],
        "Memory":   ["mac5_repeat_mean", "mac5_adj_cat_overlap_mean", "mac5_adj_item_overlap_mean", "mac5_repeat_trend"],
        "Exposure": ["mac5_pop_mean", "mac5_pop_std_mean", "mac5_pop_ent_mean", "mac5_pop_trend"],
    },
    "macro10": {
        "Tempo":    ["mac10_ctx_valid_r", "mac10_gap_last", "mac10_pace_mean", "mac10_pace_trend"],
        "Focus":    ["mac10_theme_ent_mean", "mac10_theme_top1_mean", "mac10_theme_repeat_r", "mac10_theme_shift_r"],
        "Memory":   ["mac10_repeat_mean", "mac10_adj_cat_overlap_mean", "mac10_adj_item_overlap_mean", "mac10_repeat_trend"],
        "Exposure": ["mac10_pop_mean", "mac10_pop_std_mean", "mac10_pop_ent_mean", "mac10_pop_trend"],
    },
    "mid": {
        "Tempo":    ["mid_valid_r", "mid_int_mean", "mid_int_std", "mid_sess_age"],
        "Focus":    ["mid_cat_ent", "mid_cat_top1", "mid_cat_switch_r", "mid_cat_uniq_r"],
        "Memory":   ["mid_item_uniq_r", "mid_repeat_r", "mid_novel_r", "mid_max_run_i"],
        "Exposure": ["mid_pop_mean", "mid_pop_std", "mid_pop_ent", "mid_pop_trend"],
    },
    "micro": {
        "Tempo":    ["mic_valid_r", "mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid"],
        "Focus":    ["mic_cat_switch_now", "mic_last_cat_mismatch_r", "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r"],
        "Memory":   ["mic_is_recons", "mic_suffix_recons_r", "mic_suffix_uniq_i", "mic_suffix_max_run_i"],
        "Exposure": ["mic_last_pop", "mic_suffix_pop_std", "mic_suffix_pop_ent", "mic_pop_delta_vs_mid"],
    },
}

ALL_FEATURES: List[str] = []
for _scope in ("macro5", "macro10", "mid", "micro"):
    for _fam in ("Tempo", "Focus", "Memory", "Exposure"):
        ALL_FEATURES.extend(FAMILIES[_scope][_fam])

MID_FEATURES = [n for n in ALL_FEATURES if n.startswith("mid_")]

CONTINUOUS_FEATURES = {
    "mac5_gap_last", "mac5_pace_mean",
    "mac10_gap_last", "mac10_pace_mean",
    "mid_int_mean", "mid_int_std", "mid_sess_age",
    "mic_last_gap", "mic_gap_mean",
    "mac5_pop_mean", "mac5_pop_std_mean",
    "mac10_pop_mean", "mac10_pop_std_mean",
    "mid_pop_mean", "mid_pop_std",
    "mic_last_pop", "mic_suffix_pop_std",
}

CONTINUOUS_PAIR_BASE = {
    "mac5_pace_trend": "mac5_pace_mean",
    "mac5_pop_trend": "mac5_pop_mean",
    "mac10_pace_trend": "mac10_pace_mean",
    "mac10_pop_trend": "mac10_pop_mean",
    "mid_pop_trend": "mid_pop_mean",
    "mic_gap_delta_vs_mid": "mid_int_mean",
    "mic_pop_delta_vs_mid": "mid_pop_mean",
}

BOUNDED_PAIR_FEATURES = {"mac5_repeat_trend", "mac10_repeat_trend"}

# Default missing value for each feature
DEFAULT_VALUE: Dict[str, float] = {n: 0.5 for n in ALL_FEATURES}
for _n in (
    "mac5_ctx_valid_r", "mac10_ctx_valid_r",
    "mid_valid_r", "mic_valid_r",
    "mic_cat_switch_now", "mic_last_cat_mismatch_r",
    "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r",
    "mic_is_recons", "mic_suffix_recons_r",
    "mic_suffix_uniq_i", "mic_suffix_max_run_i",
    "mic_suffix_pop_ent",
):
    DEFAULT_VALUE[_n] = 0.0


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basic-root",      type=Path, default=DEFAULT_BASIC_ROOT)
    p.add_argument("--output-root",     type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--source-dataset",  type=str,  required=True)
    p.add_argument("--target-dataset",  type=str,  required=True)
    p.add_argument("--fit-session-ratio", type=float, default=0.7)
    p.add_argument("--timestamp-unit", choices=("s", "ms"), required=True)
    p.add_argument("--micro-window",    type=int,  default=5)
    p.add_argument("--mid-valid-cap",   type=int,  default=10)
    p.add_argument("--n-pop-bins",      type=int,  default=10)
    p.add_argument("--split-ratios",    type=str,  default="0.7,0.15,0.15")
    p.add_argument(
        "--tail-unseen-policy",
        choices=("drop_non_target", "keep", "precleaned_frozen"),
        default="drop_non_target",
        help=(
            "Apply the old v4 unseen-context rule before feature computation. "
            "The target row is always retained."
        ),
    )
    p.add_argument(
        "--frozen-basic-splits",
        action="store_true",
        help=(
            "Consume source .train/.valid/.test files as authoritative session membership. "
            "Requires --tail-unseen-policy=precleaned_frozen and exact combined/split identity."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Event:
    row_idx: int
    item:    str
    ts_ms:   float
    user:    str
    # Keep the exact frozen basic token for byte/order-stable base-four output.
    # A float round trip is not lossless for raw KuaiRec timestamps such as
    # ``1593051862.7030001`` even though it is numerically equivalent.
    timestamp_raw: str | None = None


def timestamp_scale_to_seconds(unit: str) -> float:
    """Return the divisor that converts an input timestamp delta to seconds."""
    if unit == "s":
        return 1.0
    if unit == "ms":
        return 1000.0
    raise ValueError(f"unsupported timestamp unit: {unit!r}")


@dataclass
class SessionSummary:
    start_ts:    float
    end_ts:      float
    pace_mean:   float | None
    cat_ent:     float
    cat_top1:    float
    cat_switch:  float
    cat_repeat:  float
    item_repeat: float
    pop_mean:    float
    pop_std:     float
    pop_ent:     float
    item_set:    set[str]
    cat_set:     set[str]


# ---------------------------------------------------------------------------
# Helper math functions
# ---------------------------------------------------------------------------

def safe_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    return (sum(vals) / len(vals)) if vals else None


def safe_std(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    n = len(vals)
    if n <= 1:
        return None
    mu = sum(vals) / n
    return math.sqrt(max(0.0, sum((x - mu) ** 2 for x in vals) / n))


def entropy_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    n = len(vals)
    if n <= 1:
        return 0.0
    c = Counter(vals)
    h = -sum((cnt / n) * math.log(cnt / n) for cnt in c.values() if cnt > 0)
    denom = math.log(max(2, len(c)))
    return 0.0 if denom <= 0 else min(1.0, max(0.0, h / denom))


def entropy_ratio_from_counts(counts: Counter[str], total: int) -> float:
    """Compute ``entropy_ratio`` from counters without rebuilding a list."""
    if total <= 1:
        return 0.0
    h = math.log(total) - sum(
        count * math.log(count) for count in counts.values() if count > 0
    ) / total
    denom = math.log(max(2, len(counts)))
    return 0.0 if denom <= 0 else clip01(h / denom)


def top1_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    return max(Counter(vals).values()) / len(vals) if vals else 0.0


def uniq_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    return len(set(vals)) / len(vals) if vals else 0.0


def switch_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    if len(vals) <= 1:
        return 0.0
    return sum(1 for i in range(1, len(vals)) if vals[i] != vals[i - 1]) / (len(vals) - 1)


def max_run_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    if not vals:
        return 0.0
    best = run = 1
    for i in range(1, len(vals)):
        run = (run + 1) if vals[i] == vals[i - 1] else 1
        best = max(best, run)
    return best / len(vals)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    den = len(a | b)
    return len(a & b) / den if den else 0.0


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def log_delta(lhs: float, rhs: float) -> float:
    """Difference on the log1p scale used to fit continuous statistics."""
    return math.log1p(max(0.0, float(lhs))) - math.log1p(max(0.0, float(rhs)))


def quantile_sorted(sv: List[float], q: float) -> float:
    if not sv:
        return 0.0
    if q <= 0:
        return sv[0]
    if q >= 1:
        return sv[-1]
    pos = (len(sv) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sv[lo] if lo == hi else sv[lo] * (1 - (pos - lo)) + sv[hi] * (pos - lo)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def strip_type(col: str) -> str:
    clean = col.lstrip("\ufeff")
    return clean.split(":", 1)[0] if ":" in clean else clean


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_timestamp(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, ".15g")


def load_basic_data(
    inter_path: Path, item_path: Path
) -> tuple[Dict[str, List[Event]], Dict[str, List[str]], List[str], Dict[str, str]]:
    by_sid: Dict[str, List[Event]] = defaultdict(list)
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        hdr = [strip_type(c) for c in (reader.fieldnames or [])]
        for row_idx, row in enumerate(reader):
            # map stripped names
            plain = {strip_type(k): v for k, v in row.items()}
            by_sid[plain["session_id"]].append(Event(
                row_idx=row_idx,
                item=plain["item_id"],
                ts_ms=float(plain["timestamp"]),
                user=plain["user_id"],
                timestamp_raw=plain["timestamp"],
            ))

    user_by_sid: Dict[str, str] = {}
    start_ts:    Dict[str, float] = {}
    first_row:   Dict[str, int]   = {}
    for sid, evts in by_sid.items():
        evts.sort(key=lambda e: (e.ts_ms, e.row_idx))
        by_sid[sid] = evts
        user_by_sid[sid] = evts[0].user
        start_ts[sid]    = evts[0].ts_ms
        first_row[sid]   = evts[0].row_idx

    session_order = sorted(by_sid.keys(), key=lambda s: (start_ts[s], first_row[s], s))
    sessions_by_user: Dict[str, List[str]] = defaultdict(list)
    for sid in session_order:
        sessions_by_user[user_by_sid[sid]].append(sid)

    item_cat: Dict[str, str] = {}
    with item_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            plain = {strip_type(k): v for k, v in row.items()}
            item_cat[plain["item_id"]] = plain["category"]

    return by_sid, sessions_by_user, session_order, item_cat


def parse_split_ratios(raw: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("--split-ratios requires three positive values")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"--split-ratios must sum to one, got {values}")
    return values  # type: ignore[return-value]


def split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train = int(math.floor(total * ratios[0]))
    valid = int(math.floor(total * ratios[1]))
    test = total - train - valid
    if total >= 3 and min(train, valid, test) <= 0:
        raise ValueError(f"split would be empty for total={total}, ratios={ratios}")
    return train, valid, test


def load_frozen_split_membership(
    source_dir: Path,
    source_dataset: str,
    session_order: List[str],
) -> tuple[dict[str, str], dict[str, int], dict[str, list[str]]]:
    """Load exact source split membership and reject identity ambiguity."""

    membership: dict[str, str] = {}
    rows = {split: 0 for split in ("train", "valid", "test")}
    ordered_by_split = {split: [] for split in ("train", "valid", "test")}
    for split in ("train", "valid", "test"):
        path = source_dir / f"{source_dataset}.{split}.inter"
        if not path.is_file():
            raise ValueError(f"missing frozen basic split: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            names = {strip_type(name): name for name in (reader.fieldnames or [])}
            if "session_id" not in names:
                raise ValueError(f"frozen basic split lacks session_id: {path}")
            previous_sid: str | None = None
            for row in reader:
                sid = str(row[names["session_id"]])
                previous = membership.setdefault(sid, split)
                if previous != split:
                    raise ValueError(f"session {sid!r} appears in both {previous} and {split}")
                if sid != previous_sid:
                    ordered_by_split[split].append(sid)
                    previous_sid = sid
                rows[split] += 1
    expected = set(session_order)
    actual = set(membership)
    if actual != expected:
        raise ValueError(
            "frozen basic split membership differs from combined source: "
            f"missing={len(expected - actual)} unexpected={len(actual - expected)}"
        )
    if any(len(set(values)) != len(values) for values in ordered_by_split.values()):
        raise ValueError("a frozen basic split contains a non-contiguous re-entering session")
    return membership, rows, ordered_by_split


def filter_tail_unseen_context(
    by_sid: Dict[str, List[Event]],
    session_order: List[str],
    train_session_count: int,
) -> dict[str, int]:
    """Apply the v4 cold-context rule before deriving any feature.

    Every training row is retained.  In validation/test sessions, items that
    never occur in training are removed from the context, while the final
    target event is retained even when cold.  Performing this before summaries
    and prefix features prevents those features from encoding removed rows.
    """
    train_items = {
        event.item
        for sid in session_order[:train_session_count]
        for event in by_sid[sid]
    }
    dropped = 0
    cold_targets = 0
    fallback_contexts = 0
    for sid in session_order[train_session_count:]:
        events = by_sid[sid]
        if not events:
            continue
        target = events[-1]
        if target.item not in train_items:
            cold_targets += 1
        kept = [event for event in events[:-1] if event.item in train_items]
        # A sequential example needs at least one context event plus target.
        # If filtering would leave only the target, retain the latest context
        # even when cold.  This avoids an undocumented denominator change when
        # the frozen converter skips length-one sessions.
        if not kept and len(events) >= 2:
            kept = [events[-2]]
            fallback_contexts += 1
        kept.append(target)
        dropped += len(events) - len(kept)
        by_sid[sid] = kept
    return {
        "train_item_count": len(train_items),
        "dropped_non_target_unseen_rows": dropped,
        "cold_target_sessions_retained": cold_targets,
        "fallback_latest_cold_context_rows_retained": fallback_contexts,
    }


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------

def build_popularity(
    by_sid: Dict[str, List[Event]],
    fit_sessions: set[str],
    n_bins: int,
) -> tuple[Dict[str, float], Dict[str, int], List[float]]:
    item_counts: Counter = Counter()
    for sid in fit_sessions:
        for e in by_sid[sid]:
            item_counts[e.item] += 1
    if not item_counts:
        return {}, {}, [0.0] * (n_bins + 1)
    pop_score = {item: float(cnt) for item, cnt in item_counts.items()}
    log_counts = sorted(math.log1p(float(cnt)) for cnt in item_counts.values())
    edges = [quantile_sorted(log_counts, i / float(n_bins)) for i in range(n_bins + 1)]
    pop_bin: Dict[str, int] = {}
    for item, count in pop_score.items():
        value = math.log1p(count)
        bucket = 0
        while bucket + 1 < len(edges) and value >= edges[bucket + 1]:
            bucket += 1
        pop_bin[item] = min(n_bins - 1, bucket)
    return pop_score, pop_bin, edges


# ---------------------------------------------------------------------------
# Session summaries
# ---------------------------------------------------------------------------

def summarize_sessions(
    by_sid: Dict[str, List[Event]],
    item_cat: Dict[str, str],
    pop_score: Dict[str, float],
    pop_bin: Dict[str, int],
    n_bins: int,
    timestamp_divisor: float,
) -> Dict[str, SessionSummary]:
    out: Dict[str, SessionSummary] = {}
    for sid, evts in by_sid.items():
        items = [e.item for e in evts]
        cats  = [item_cat.get(it, "Unknown") for it in items]
        pops  = [pop_score.get(it, 0.0) for it in items]
        bins  = [pop_bin.get(it, 0) for it in items]
        gaps  = [
            max(0.0, (evts[i].ts_ms - evts[i - 1].ts_ms) / timestamp_divisor)
            for i in range(1, len(evts))
        ]
        out[sid] = SessionSummary(
            start_ts=evts[0].ts_ms,
            end_ts=evts[-1].ts_ms,
            pace_mean=safe_mean(gaps),
            cat_ent=entropy_ratio(cats),
            cat_top1=top1_ratio(cats),
            cat_switch=switch_ratio(cats),
            cat_repeat=1.0 - uniq_ratio(cats),
            item_repeat=1.0 - uniq_ratio(items),
            pop_mean=safe_mean(pops) or 0.0,
            pop_std=safe_std(pops)   or 0.0,
            pop_ent=entropy_ratio([str(b) for b in bins]),
            item_set=set(items),
            cat_set=set(cats),
        )
    return out


# ---------------------------------------------------------------------------
# Macro features per session
# ---------------------------------------------------------------------------

def macro_features(
    *,
    current_start: float,
    ctx_sids: List[str],
    summaries: Dict[str, SessionSummary],
    window: int,
    prefix: str,
    timestamp_divisor: float,
) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {
        f"{prefix}_ctx_valid_r": min(len(ctx_sids), window) / float(window),
        f"{prefix}_gap_last": None,
        f"{prefix}_pace_mean": None, f"{prefix}_pace_trend": None,
        f"{prefix}_theme_ent_mean": None, f"{prefix}_theme_top1_mean": None,
        f"{prefix}_theme_repeat_r": None, f"{prefix}_theme_shift_r": None,
        f"{prefix}_repeat_mean": None,
        f"{prefix}_adj_cat_overlap_mean": None, f"{prefix}_adj_item_overlap_mean": None,
        f"{prefix}_repeat_trend": None,
        f"{prefix}_pop_mean": None, f"{prefix}_pop_std_mean": None,
        f"{prefix}_pop_ent_mean": None, f"{prefix}_pop_trend": None,
    }
    if not ctx_sids:
        return out
    ctx = [summaries[s] for s in ctx_sids]
    last, first = ctx[-1], ctx[0]
    out[f"{prefix}_gap_last"]       = max(0.0, (current_start - last.end_ts) / timestamp_divisor)
    out[f"{prefix}_pace_mean"]      = safe_mean([x.pace_mean for x in ctx if x.pace_mean is not None])
    out[f"{prefix}_pace_trend"]     = log_delta(last.pace_mean, first.pace_mean) if (len(ctx) >= 2 and first.pace_mean is not None and last.pace_mean is not None) else 0.0
    out[f"{prefix}_theme_ent_mean"] = safe_mean([x.cat_ent   for x in ctx])
    out[f"{prefix}_theme_top1_mean"]= safe_mean([x.cat_top1  for x in ctx])
    out[f"{prefix}_theme_repeat_r"] = safe_mean([x.cat_repeat for x in ctx])
    out[f"{prefix}_theme_shift_r"]  = safe_mean([x.cat_switch for x in ctx])
    out[f"{prefix}_repeat_mean"]    = safe_mean([x.item_repeat for x in ctx])
    out[f"{prefix}_repeat_trend"]   = (ctx[-1].item_repeat - ctx[0].item_repeat) if len(ctx) >= 2 else 0.0
    adj_cat  = [jaccard(ctx[i-1].cat_set,  ctx[i].cat_set)  for i in range(1, len(ctx))]
    adj_item = [jaccard(ctx[i-1].item_set, ctx[i].item_set) for i in range(1, len(ctx))]
    out[f"{prefix}_adj_cat_overlap_mean"]  = safe_mean(adj_cat)  if adj_cat  else None
    out[f"{prefix}_adj_item_overlap_mean"] = safe_mean(adj_item) if adj_item else None
    out[f"{prefix}_pop_mean"]      = safe_mean([x.pop_mean for x in ctx])
    out[f"{prefix}_pop_std_mean"]  = safe_mean([x.pop_std  for x in ctx])
    out[f"{prefix}_pop_ent_mean"]  = safe_mean([x.pop_ent  for x in ctx])
    out[f"{prefix}_pop_trend"]     = log_delta(ctx[-1].pop_mean, ctx[0].pop_mean) if len(ctx) >= 2 else 0.0
    return out


def build_macro_by_session(
    sessions_by_user: Dict[str, List[str]],
    summaries: Dict[str, SessionSummary],
    timestamp_divisor: float,
) -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for _user, sids in sessions_by_user.items():
        for i, sid in enumerate(sids):
            start = summaries[sid].start_ts
            f5  = macro_features(current_start=start, ctx_sids=sids[max(0, i-5):i],  summaries=summaries, window=5,  prefix="mac5", timestamp_divisor=timestamp_divisor)
            f10 = macro_features(current_start=start, ctx_sids=sids[max(0, i-10):i], summaries=summaries, window=10, prefix="mac10", timestamp_divisor=timestamp_divisor)
            merged = dict(f5)
            merged.update(f10)
            out[sid] = merged
    return out


# ---------------------------------------------------------------------------
# Per-row feature computation (mid + micro)
# ---------------------------------------------------------------------------

def _compute_session_rows_reference(
    *,
    events:        List[Event],
    macro:         Dict[str, float | None],
    item_cat:      Dict[str, str],
    pop_score:     Dict[str, float],
    pop_bin:       Dict[str, int],
    micro_window:  int,
    mid_valid_cap: int,
    n_pop_bins:    int,
    timestamp_divisor: float,
) -> List[Dict[str, float | None]]:
    rows: List[Dict[str, float | None]] = []
    items = [e.item for e in events]
    cats  = [item_cat.get(it, "Unknown") for it in items]
    pops  = [pop_score.get(it, 0.0) for it in items]
    bins  = [pop_bin.get(it, 0)     for it in items]
    ts    = [e.ts_ms for e in events]

    for i in range(len(events)):
        feat = dict(macro)
        # Strict prefix: the feature attached to event i cannot inspect event i
        # or any later event.  This also avoids the old session-constant-last
        # postprocess, which leaked future content into early prefixes.
        prefix_items = items[:i]
        prefix_cats  = cats[:i]
        prefix_pops  = pops[:i]
        prefix_bins  = bins[:i]

        # ── mid features ────────────────────────────────────────────────────
        feat["mid_valid_r"]    = min(i, mid_valid_cap) / float(mid_valid_cap)
        gaps = [max(0.0, (ts[j] - ts[j-1]) / timestamp_divisor) for j in range(1, i)]
        feat["mid_int_mean"]   = safe_mean(gaps)
        feat["mid_int_std"]    = safe_std(gaps)
        feat["mid_sess_age"]   = max(0.0, (ts[i-1] - ts[0]) / timestamp_divisor) if i else None
        feat["mid_cat_ent"]    = entropy_ratio(prefix_cats)
        feat["mid_cat_top1"]   = top1_ratio(prefix_cats)
        feat["mid_cat_switch_r"] = switch_ratio(prefix_cats)
        feat["mid_cat_uniq_r"] = uniq_ratio(prefix_cats)
        feat["mid_item_uniq_r"]= uniq_ratio(prefix_items)
        feat["mid_repeat_r"]   = (1.0 - feat["mid_item_uniq_r"]) if prefix_items else None
        feat["mid_novel_r"]    = feat["mid_item_uniq_r"] if prefix_items else None
        feat["mid_max_run_i"]  = max_run_ratio(prefix_items)
        feat["mid_pop_mean"]   = safe_mean(prefix_pops)
        feat["mid_pop_std"]    = safe_std(prefix_pops)
        feat["mid_pop_ent"]    = entropy_ratio([str(b) for b in prefix_bins])
        feat["mid_pop_trend"]  = log_delta(prefix_pops[-1], sum(prefix_pops[:-1]) / len(prefix_pops[:-1])) if len(prefix_pops) > 1 else 0.0

        # ── micro features ───────────────────────────────────────────────────
        win      = items[max(0, i - micro_window):i]
        win_cats = cats[max(0,  i - micro_window):i]
        win_pops = pops[max(0,  i - micro_window):i]
        win_bins = bins[max(0,  i - micro_window):i]
        win_ts   = ts[max(0,    i - micro_window):i]
        w = len(win)

        feat["mic_valid_r"] = w / float(max(1, micro_window))
        if w <= 0:
            for k in ("mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid",
                      "mic_last_pop", "mic_suffix_pop_std", "mic_pop_delta_vs_mid"):
                feat[k] = None
            for k in ("mic_cat_switch_now", "mic_last_cat_mismatch_r",
                      "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r",
                      "mic_is_recons", "mic_suffix_recons_r",
                      "mic_suffix_uniq_i", "mic_suffix_max_run_i", "mic_suffix_pop_ent"):
                feat[k] = 0.0
        else:
            feat["mic_last_gap"] = max(0.0, (ts[i] - win_ts[-1]) / timestamp_divisor)
            loc_ts  = win_ts + [ts[i]]
            loc_gaps = [max(0.0, (loc_ts[j] - loc_ts[j-1]) / timestamp_divisor) for j in range(1, len(loc_ts))]
            gm = safe_mean(loc_gaps)
            feat["mic_gap_mean"] = gm
            feat["mic_gap_delta_vs_mid"] = log_delta(gm, float(feat["mid_int_mean"])) if (gm is not None and feat["mid_int_mean"] is not None) else None
            cur_cat = cats[i]
            feat["mic_cat_switch_now"]       = 1.0 if cur_cat != win_cats[-1] else 0.0
            feat["mic_last_cat_mismatch_r"]  = sum(1 for c in win_cats if c != cur_cat) / float(w)
            feat["mic_suffix_cat_ent"]       = entropy_ratio(win_cats)
            feat["mic_suffix_cat_uniq_r"]    = uniq_ratio(win_cats)
            feat["mic_is_recons"]            = 1.0 if items[i] in set(win) else 0.0
            feat["mic_suffix_recons_r"]      = 1.0 - uniq_ratio(win)
            feat["mic_suffix_uniq_i"]        = uniq_ratio(win)
            feat["mic_suffix_max_run_i"]     = max_run_ratio(win)
            feat["mic_last_pop"]             = pops[i]
            feat["mic_suffix_pop_std"]       = safe_std(win_pops)
            feat["mic_suffix_pop_ent"]       = entropy_ratio([str(b) for b in win_bins])
            feat["mic_pop_delta_vs_mid"]     = log_delta(pops[i], float(feat["mid_pop_mean"])) if feat["mid_pop_mean"] is not None else None

        rows.append(feat)

    return rows


# ---------------------------------------------------------------------------
# Linear-time per-row implementation
# ---------------------------------------------------------------------------

def compute_session_rows(
    *,
    events:        List[Event],
    macro:         Dict[str, float | None],
    item_cat:      Dict[str, str],
    pop_score:     Dict[str, float],
    pop_bin:       Dict[str, int],
    micro_window:  int,
    mid_valid_cap: int,
    n_pop_bins:    int,
    timestamp_divisor: float,
) -> List[Dict[str, float | None]]:
    """Create event-causal rows in O(session_length * micro_window).

    Event-local micro fields may use event ``i``.  During sequential
    conversion, the row attached to event ``i`` becomes history only when
    predicting a later event.  Mid fields use events strictly before ``i``;
    no field reads an event after ``i``.
    """
    del n_pop_bins  # Kept in the API because it is part of the feature contract.
    rows: List[Dict[str, float | None]] = []
    items = [event.item for event in events]
    cats = [item_cat.get(item, "Unknown") for item in items]
    pops = [pop_score.get(item, 0.0) for item in items]
    bins = [pop_bin.get(item, 0) for item in items]
    timestamps = [event.ts_ms for event in events]

    item_counts: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    bin_counts: Counter[str] = Counter()
    cat_switches = 0
    item_max_run = 0
    item_current_run = 0
    previous_item: str | None = None
    previous_cat: str | None = None
    pop_sum = 0.0
    pop_mean = 0.0
    pop_m2 = 0.0
    gap_count = 0
    gap_mean = 0.0
    gap_m2 = 0.0

    for index in range(len(events)):
        feature = dict(macro)
        feature["mid_valid_r"] = min(index, mid_valid_cap) / float(mid_valid_cap)
        feature["mid_int_mean"] = gap_mean if gap_count else None
        feature["mid_int_std"] = (
            math.sqrt(max(0.0, gap_m2 / gap_count)) if gap_count > 1 else None
        )
        feature["mid_sess_age"] = (
            max(0.0, (timestamps[index - 1] - timestamps[0]) / timestamp_divisor)
            if index else None
        )
        feature["mid_cat_ent"] = entropy_ratio_from_counts(cat_counts, index)
        feature["mid_cat_top1"] = max(cat_counts.values()) / index if index else 0.0
        feature["mid_cat_switch_r"] = cat_switches / (index - 1) if index > 1 else 0.0
        feature["mid_cat_uniq_r"] = len(cat_counts) / index if index else 0.0
        feature["mid_item_uniq_r"] = len(item_counts) / index if index else 0.0
        feature["mid_repeat_r"] = 1.0 - feature["mid_item_uniq_r"] if index else None
        feature["mid_novel_r"] = feature["mid_item_uniq_r"] if index else None
        feature["mid_max_run_i"] = item_max_run / index if index else 0.0
        feature["mid_pop_mean"] = pop_mean if index else None
        feature["mid_pop_std"] = (
            math.sqrt(max(0.0, pop_m2 / index)) if index > 1 else None
        )
        feature["mid_pop_ent"] = entropy_ratio_from_counts(bin_counts, index)
        feature["mid_pop_trend"] = (
            log_delta(pops[index - 1], (pop_sum - pops[index - 1]) / (index - 1))
            if index > 1 else 0.0
        )

        start = max(0, index - micro_window)
        window_items = items[start:index]
        window_cats = cats[start:index]
        window_pops = pops[start:index]
        window_bins = bins[start:index]
        window_timestamps = timestamps[start:index]
        width = len(window_items)
        feature["mic_valid_r"] = width / float(max(1, micro_window))
        if width == 0:
            for name in (
                "mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid",
                "mic_last_pop", "mic_suffix_pop_std", "mic_pop_delta_vs_mid",
            ):
                feature[name] = None
            for name in (
                "mic_cat_switch_now", "mic_last_cat_mismatch_r",
                "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r",
                "mic_is_recons", "mic_suffix_recons_r", "mic_suffix_uniq_i",
                "mic_suffix_max_run_i", "mic_suffix_pop_ent",
            ):
                feature[name] = 0.0
        else:
            feature["mic_last_gap"] = max(
                0.0, (timestamps[index] - window_timestamps[-1]) / timestamp_divisor
            )
            local_timestamps = window_timestamps + [timestamps[index]]
            local_gaps = [
                max(0.0, (local_timestamps[pos] - local_timestamps[pos - 1]) / timestamp_divisor)
                for pos in range(1, len(local_timestamps))
            ]
            micro_gap_mean = safe_mean(local_gaps)
            feature["mic_gap_mean"] = micro_gap_mean
            feature["mic_gap_delta_vs_mid"] = (
                log_delta(micro_gap_mean, float(feature["mid_int_mean"]))
                if micro_gap_mean is not None and feature["mid_int_mean"] is not None
                else None
            )
            current_cat = cats[index]
            feature["mic_cat_switch_now"] = 1.0 if current_cat != window_cats[-1] else 0.0
            feature["mic_last_cat_mismatch_r"] = (
                sum(1 for category in window_cats if category != current_cat) / width
            )
            feature["mic_suffix_cat_ent"] = entropy_ratio(window_cats)
            feature["mic_suffix_cat_uniq_r"] = uniq_ratio(window_cats)
            feature["mic_is_recons"] = 1.0 if items[index] in set(window_items) else 0.0
            feature["mic_suffix_recons_r"] = 1.0 - uniq_ratio(window_items)
            feature["mic_suffix_uniq_i"] = uniq_ratio(window_items)
            feature["mic_suffix_max_run_i"] = max_run_ratio(window_items)
            feature["mic_last_pop"] = pops[index]
            feature["mic_suffix_pop_std"] = safe_std(window_pops)
            feature["mic_suffix_pop_ent"] = entropy_ratio([str(value) for value in window_bins])
            feature["mic_pop_delta_vs_mid"] = (
                log_delta(pops[index], float(feature["mid_pop_mean"]))
                if feature["mid_pop_mean"] is not None else None
            )

        rows.append(feature)

        current_item = items[index]
        current_cat = cats[index]
        item_counts[current_item] += 1
        cat_counts[current_cat] += 1
        bin_counts[str(bins[index])] += 1
        if previous_cat is not None and current_cat != previous_cat:
            cat_switches += 1
        item_current_run = item_current_run + 1 if previous_item == current_item else 1
        item_max_run = max(item_max_run, item_current_run)
        previous_item = current_item
        previous_cat = current_cat

        pop_value = float(pops[index])
        pop_sum += pop_value
        pop_delta = pop_value - pop_mean
        pop_mean += pop_delta / (index + 1)
        pop_m2 += pop_delta * (pop_value - pop_mean)

        if index > 0:
            gap_value = max(
                0.0, (timestamps[index] - timestamps[index - 1]) / timestamp_divisor
            )
            gap_count += 1
            gap_delta = gap_value - gap_mean
            gap_mean += gap_delta / gap_count
            gap_m2 += gap_delta * (gap_value - gap_mean)

    return rows


# ---------------------------------------------------------------------------
# Normalization stat fitting
# ---------------------------------------------------------------------------

def fit_norm_stats(
    cont_values: Dict[str, List[float]],
) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for name in CONTINUOUS_FEATURES:
        vals = sorted(cont_values.get(name, []))
        if not vals:
            stats[name] = {
                "type": "continuous", "transform": "log1p_winsorize_z_phi",
                "mean": 0.0, "std": 1.0, "q01": 0.0, "q99": 0.0, "missing_default": 0.5,
            }
            continue
        q01 = quantile_sorted(vals, 0.01)
        q99 = quantile_sorted(vals, 0.99)
        mu  = safe_mean(vals) or 0.0
        sd  = safe_std(vals)  or 1.0
        sd  = max(sd, 1e-12)
        stats[name] = {
            "type": "continuous", "transform": "log1p_winsorize_z_phi",
            "mean": float(mu), "std": float(sd),
            "q01": float(q01), "q99": float(q99), "missing_default": 0.5,
        }
    return stats


def normalize(name: str, value: float | None, stats: Dict[str, dict]) -> float:
    default = DEFAULT_VALUE.get(name, 0.5)
    if name in CONTINUOUS_FEATURES:
        if value is None:
            return default
        cfg = stats[name]
        x = math.log1p(max(0.0, float(value)))
        x = min(max(x, cfg["q01"]), cfg["q99"])
        z = (x - cfg["mean"]) / max(1e-12, cfg["std"])
        return clip01(phi(z))
    if name in CONTINUOUS_PAIR_BASE:
        if value is None:
            return default
        base = CONTINUOUS_PAIR_BASE[name]
        cfg = stats[base]
        z = float(value) / max(1e-12, float(cfg["std"]))
        return clip01(phi(z))
    if name in BOUNDED_PAIR_FEATURES:
        if value is None:
            return default
        return clip01(0.5 + 0.5 * float(value))
    if value is None:
        return default
    return clip01(float(value))


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------

def _load_module(filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _timestamp_bound(value: float) -> float | None:
    return None if value in (math.inf, -math.inf) else float(value)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    source_dataset = str(args.source_dataset)
    dataset = str(args.target_dataset)
    source_dir = Path(args.basic_root) / source_dataset
    out_dir    = Path(args.output_root) / dataset
    inter_in   = source_dir / f"{source_dataset}.inter"
    item_in    = source_dir / f"{source_dataset}.item"
    inter_out  = out_dir / f"{dataset}.inter"
    item_out   = out_dir / f"{dataset}.item"

    if not inter_in.exists() or not item_in.exists():
        raise SystemExit(f"Missing source files in: {source_dir}")
    planned_outputs = [
        inter_out, item_out,
        *[out_dir / f"{dataset}.{split}.inter" for split in ("train", "valid", "test")],
        out_dir / "feature_meta_v3.json",
        out_dir / f"{dataset}.session_split_summary.json",
        out_dir / f"{dataset}.build_summary.json",
    ]
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise SystemExit(
            "Output exists (use --overwrite): " + ", ".join(str(path) for path in existing_outputs)
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    source_hashes = {
        "inter_sha256": sha256_file(inter_in),
        "item_sha256": sha256_file(item_in),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
    }

    print(f"[1/7] loading basic dataset: {source_dir}")
    by_sid, sessions_by_user, session_order, item_cat = load_basic_data(inter_in, item_in)
    total_sessions = len(session_order)
    total_rows     = sum(len(by_sid[sid]) for sid in session_order)
    print(f"  sessions={total_sessions:,}  rows={total_rows:,}  items={len(item_cat):,}")

    timestamp_divisor = timestamp_scale_to_seconds(str(args.timestamp_unit))
    ratios = parse_split_ratios(str(args.split_ratios))
    frozen_membership: dict[str, str] | None = None
    frozen_source_rows: dict[str, int] | None = None
    frozen_session_order: dict[str, list[str]] | None = None
    if args.frozen_basic_splits:
        if args.tail_unseen_policy != "precleaned_frozen":
            raise SystemExit(
                "--frozen-basic-splits requires --tail-unseen-policy=precleaned_frozen"
            )
        frozen_membership, frozen_source_rows, frozen_session_order = load_frozen_split_membership(
            source_dir, source_dataset, session_order
        )
        train_n = sum(frozen_membership[sid] == "train" for sid in session_order)
        valid_n = sum(frozen_membership[sid] == "valid" for sid in session_order)
        test_n = sum(frozen_membership[sid] == "test" for sid in session_order)
        if min(train_n, valid_n, test_n) <= 0:
            raise SystemExit("frozen source split must contain non-empty train/valid/test")
        ratios = (
            train_n / total_sessions,
            valid_n / total_sessions,
            test_n / total_sessions,
        )
        session_order = [
            sid
            for split in ("train", "valid", "test")
            for sid in frozen_session_order[split]
        ]
    else:
        train_n, valid_n, test_n = split_counts(total_sessions, ratios)
    if not args.frozen_basic_splits and not math.isclose(float(args.fit_session_ratio), ratios[0], rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            "fit-session-ratio must equal the train split ratio so fitted statistics "
            f"use exactly training sessions: {args.fit_session_ratio} != {ratios[0]}"
        )
    fit_n = train_n
    effective_fit_ratio = fit_n / total_sessions
    unseen_filter_summary = {
        "policy": str(args.tail_unseen_policy),
        "minimum_session_length_policy": "retain_latest_context_plus_target_when_source_length_ge_2",
        "train_item_count": None,
        "dropped_non_target_unseen_rows": 0,
        "cold_target_sessions_retained": 0,
        "fallback_latest_cold_context_rows_retained": 0,
    }
    if args.tail_unseen_policy == "drop_non_target":
        unseen_filter_summary.update(filter_tail_unseen_context(by_sid, session_order, fit_n))
        total_rows = sum(len(by_sid[sid]) for sid in session_order)
        print(
            "  pre-feature unseen filter: "
            f"dropped={unseen_filter_summary['dropped_non_target_unseen_rows']:,}, "
            f"cold_targets={unseen_filter_summary['cold_target_sessions_retained']:,}"
        )
    elif args.tail_unseen_policy == "precleaned_frozen":
        if not args.frozen_basic_splits:
            raise SystemExit("precleaned_frozen policy requires --frozen-basic-splits")
        unseen_filter_summary.update(
            {
                "policy": "precleaned_frozen",
                "minimum_session_length_policy": "precleaned by source basic contract",
                "train_item_count": len(
                    {
                        event.item
                        for sid in session_order[:fit_n]
                        for event in by_sid[sid]
                    }
                ),
                "source_split_rows": frozen_source_rows,
            }
        )
    fit_sids = set(session_order[:fit_n])
    fit_last_start = by_sid[session_order[fit_n - 1]][0].ts_ms if fit_n > 0 else None
    print(f"  fit_sessions={fit_n}  (ratio={effective_fit_ratio})")

    print("[2/7] building popularity scores")
    pop_score, pop_bin, pop_bin_edges = build_popularity(by_sid, fit_sids, int(args.n_pop_bins))
    print(f"  pop items in fit={len(pop_score):,}")

    print("[3/7] computing session summaries + macro context")
    summaries      = summarize_sessions(by_sid, item_cat, pop_score, pop_bin, int(args.n_pop_bins), timestamp_divisor)
    macro_by_sid   = build_macro_by_session(sessions_by_user, summaries, timestamp_divisor)

    print("[4/7] fitting normalization stats (fit sessions only)")
    # float64 arrays avoid retaining millions of boxed Python float objects.
    cont_values: Dict[str, array] = {
        name: array("d") for name in CONTINUOUS_FEATURES
    }
    for sid in fit_sids:
        raw_rows = compute_session_rows(
            events=by_sid[sid], macro=macro_by_sid[sid],
            item_cat=item_cat, pop_score=pop_score, pop_bin=pop_bin,
            micro_window=int(args.micro_window), mid_valid_cap=int(args.mid_valid_cap),
            n_pop_bins=int(args.n_pop_bins),
            timestamp_divisor=timestamp_divisor,
        )
        for r in raw_rows:
            for name in CONTINUOUS_FEATURES:
                v = r.get(name)
                if v is not None:
                    cont_values[name].append(math.log1p(max(0.0, float(v))))
    norm_stats = fit_norm_stats(cont_values)

    print("[5/7] writing featured inter and frozen split files")
    split_paths = {
        "train": out_dir / f"{dataset}.train.inter",
        "valid": out_dir / f"{dataset}.valid.inter",
        "test": out_dir / f"{dataset}.test.inter",
    }
    inter_tmp = NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False,
        dir=str(out_dir), suffix=".inter.tmp"
    )
    split_tmps = {
        split: NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False,
            dir=str(out_dir), suffix=f".{split}.inter.tmp",
        )
        for split in split_paths
    }
    header = (
        ["session_id:token", "item_id:token", "timestamp:float", "user_id:token"]
        + [f"{name}:float" for name in ALL_FEATURES]
    )
    split_rows = {"train": 0, "valid": 0, "test": 0}
    split_timestamp_bounds = {
        split: {"min": math.inf, "max": -math.inf} for split in split_paths
    }
    try:
        with inter_tmp as fh, split_tmps["train"] as train_fh, split_tmps["valid"] as valid_fh, split_tmps["test"] as test_fh:
            writer = csv.writer(fh, delimiter="\t")
            split_writers = {
                "train": csv.writer(train_fh, delimiter="\t"),
                "valid": csv.writer(valid_fh, delimiter="\t"),
                "test": csv.writer(test_fh, delimiter="\t"),
            }
            writer.writerow(header)
            for split_writer in split_writers.values():
                split_writer.writerow(header)
            for session_index, sid in enumerate(session_order):
                split = (
                    frozen_membership[sid]
                    if frozen_membership is not None
                    else (
                        "train" if session_index < train_n else
                        "valid" if session_index < train_n + valid_n else "test"
                    )
                )
                raw_rows = compute_session_rows(
                    events=by_sid[sid], macro=macro_by_sid[sid],
                    item_cat=item_cat, pop_score=pop_score, pop_bin=pop_bin,
                    micro_window=int(args.micro_window), mid_valid_cap=int(args.mid_valid_cap),
                    n_pop_bins=int(args.n_pop_bins),
                    timestamp_divisor=timestamp_divisor,
                )
                for event, raw in zip(by_sid[sid], raw_rows):
                    values = [normalize(name, raw.get(name), norm_stats) for name in ALL_FEATURES]
                    output_row = (
                        [
                            sid,
                            event.item,
                            event.timestamp_raw
                            if event.timestamp_raw is not None
                            else format_timestamp(event.ts_ms),
                            event.user,
                        ]
                        + [format(value, ".10g") for value in values]
                    )
                    writer.writerow(output_row)
                    split_writers[split].writerow(output_row)
                    split_rows[split] += 1
                    split_timestamp_bounds[split]["min"] = min(
                        split_timestamp_bounds[split]["min"], event.ts_ms
                    )
                    split_timestamp_bounds[split]["max"] = max(
                        split_timestamp_bounds[split]["max"], event.ts_ms
                    )

            for output_handle in (fh, train_fh, valid_fh, test_fh):
                output_handle.flush()
                os.fsync(output_handle.fileno())
        os.replace(inter_tmp.name, inter_out)
        for split, temp_handle in split_tmps.items():
            os.replace(temp_handle.name, split_paths[split])
    except Exception:
        for temp_name in [inter_tmp.name, *[handle.name for handle in split_tmps.values()]]:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        raise

    shutil.copy2(item_in, item_out)

    print("[6/7] writing feature_meta_v3.json")
    norm_meta: Dict[str, dict] = {}
    for name in ALL_FEATURES:
        if name in CONTINUOUS_FEATURES:
            norm_meta[name] = dict(norm_stats[name])
        elif name in CONTINUOUS_PAIR_BASE:
            base = CONTINUOUS_PAIR_BASE[name]
            norm_meta[name] = {
                "type": "continuous_pair",
                "transform": "phi((log1p(lhs)-log1p(rhs))/train_std_log)",
                "base_feature": base,
                "mean": norm_stats[base]["mean"], "std": norm_stats[base]["std"],
                "q01": norm_stats[base]["q01"], "q99": norm_stats[base]["q99"],
                "missing_default": DEFAULT_VALUE.get(name, 0.5),
            }
        elif name in BOUNDED_PAIR_FEATURES:
            norm_meta[name] = {
                "type": "bounded_pair", "transform": "0.5_plus_half_diff",
                "missing_default": DEFAULT_VALUE.get(name, 0.5),
            }
        else:
            norm_meta[name] = {
                "type": "bounded", "transform": "clip01",
                "missing_default": DEFAULT_VALUE.get(name, 0.5),
            }

    meta = {
        "dataset": dataset,
        "source_dataset": source_dataset,
        "all_features": ALL_FEATURES,
        "families": FAMILIES,
        "macro_windows": [5, 10],
        "micro_window": int(args.micro_window),
        "mid_valid_cap": int(args.mid_valid_cap),
        "mid_scope_original": "strict_prefix",
        "mid_scope": "strict_prefix",
        "mid_constant_source": None,
        "mid_constant_postprocess": {
            "applied": False, "dataset": dataset,
            "mid_columns": len(MID_FEATURES),
            "rows": total_rows, "sessions": total_sessions,
        },
        "fit_sessions": {
            "fit_session_ratio": float(effective_fit_ratio),
            "fit_session_count": fit_n,
            "total_session_count": total_sessions,
            "fit_last_session_start_timestamp": float(fit_last_start) if fit_last_start is not None else None,
        },
        "n_pop_bins": int(args.n_pop_bins),
        "pop_bin_edges_log1p": pop_bin_edges,
        "macro_missing_policy": {
            "ctx_valid_r": "min(n_ctx, K) / K",
            "gap_last": "neutral 0.5 when no prior session",
            "average_features": "computed from available prior sessions",
            "pairwise_and_trend": "neutral 0.5 when insufficient context",
        },
        "normalization": {
            "bounded": "clip01",
            "continuous": "log1p -> winsorize(p01,p99) -> zscore -> phi",
            "pair_bounded": "0.5 + 0.5 * (lhs - rhs)",
            "pair_continuous": "phi((log1p(lhs)-log1p(rhs))/train_std_log)",
            "continuous_missing_default": 0.5,
        },
        "normalization_stats": norm_meta,
        "timestamp_unit": str(args.timestamp_unit),
        "row_information_contract": (
            "event_causal(timestamp_order): mid(t) uses events < t; explicitly "
            "event-local micro(t) fields may use event t; no field uses events > t; "
            "sequential conversion must use row t only to predict a later item"
        ),
        "tail_unseen_filter": unseen_filter_summary,
        "reconstruction_contract": "full-v5-leakage-correct-20260812",
        "source_preprocessing_contract": (
            "sessionized-core5-strict-split-v1" if args.frozen_basic_splits else None
        ),
        "compatibility_note": "schema-compatible with v3/v4; values are not identity-compatible with sampled release because known leakage and lost-builder defects are corrected",
        "source_hashes": source_hashes,
        "metadata_canonicalization": {
            "schema_version": 1,
            "contract": "full-v5-location-independent-metadata-v1",
            "method": "native_builder_v1",
            "removed_fields": ["output_root", "source_root"],
            "source_dataset": source_dataset,
            "target_dataset": dataset,
            "builder_sha256": source_hashes["builder_sha256"],
        },
    }
    (out_dir / "feature_meta_v3.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    split_summary = {
        "dataset": dataset,
        "ratios": {"train": ratios[0], "valid": ratios[1], "test": ratios[2]},
        "split_strategy": (
            "frozen_source_basic_membership" if args.frozen_basic_splits
            else "contiguous_by_session_start"
        ),
        "session_allocation": {
            "total_sessions": total_sessions,
            "train_sessions": train_n,
            "valid_sessions": valid_n,
            "test_sessions": test_n,
        },
        "write_stats": {
            "rows": split_rows,
            "sessions": {"train": train_n, "valid": valid_n, "test": test_n},
            "session_overlap": {"train_valid": 0, "train_test": 0, "valid_test": 0},
            "timestamp_bounds": {
                split: {
                    "min": _timestamp_bound(bounds["min"]),
                    "max": _timestamp_bound(bounds["max"]),
                }
                for split, bounds in split_timestamp_bounds.items()
            },
        },
        "temporal_check": {
            "train_before_valid": True,
            "valid_before_test": True,
            "basis": (
                "source basic frozen split membership, already contiguous by natural-session start"
                if args.frozen_basic_splits
                else "session_order sorted by session_start, first_source_row, session_id"
            ),
        },
    }
    (out_dir / f"{dataset}.session_split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Release full feature state before the final manifest serialization.
    del cont_values, norm_stats, summaries, macro_by_sid, pop_score, pop_bin
    del sessions_by_user, item_cat, fit_sids
    gc.collect()

    print("[7/7] finalizing build manifest")

    build_summary = {
        "dataset": dataset,
        "source_dataset": source_dataset,
        "source": str(source_dir),
        "target": str(out_dir),
        "rows": total_rows, "sessions": total_sessions, "features": len(ALL_FEATURES),
        "timestamp_unit": str(args.timestamp_unit),
        "feature_contract_version": "full-v5-leakage-correct-20260812",
        "source_hashes": source_hashes,
        "command": list(sys.argv),
        "split_summary": split_summary,
        "tail_unseen_filter": unseen_filter_summary,
        "frozen_basic_splits": bool(args.frozen_basic_splits),
        "split_session_counts": {"train": train_n, "valid": valid_n, "test": test_n},
    }
    (out_dir / f"{dataset}.build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"done → {out_dir}")
    print(f"  sessions={total_sessions:,}  rows={total_rows:,}  features={len(ALL_FEATURES)}")


if __name__ == "__main__":
    main()
