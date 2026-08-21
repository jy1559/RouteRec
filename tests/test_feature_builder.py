from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_feature_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_feature_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def event(index: int, item: str, timestamp: float) -> object:
    return MOD.Event(row_idx=index, item=item, ts_ms=timestamp, user="u")


def test_timestamp_units_produce_identical_seconds_features() -> None:
    events_s = [event(0, "a", 0), event(1, "b", 10), event(2, "c", 30)]
    events_ms = [event(0, "a", 0), event(1, "b", 10_000), event(2, "c", 30_000)]
    kwargs = dict(
        macro={}, item_cat={"a": "x", "b": "y", "c": "z"},
        pop_score={"a": 3.0, "b": 2.0, "c": 1.0},
        pop_bin={"a": 2, "b": 1, "c": 0}, micro_window=5,
        mid_valid_cap=10, n_pop_bins=10,
    )
    rows_s = MOD.compute_session_rows(events=events_s, timestamp_divisor=1.0, **kwargs)
    rows_ms = MOD.compute_session_rows(events=events_ms, timestamp_divisor=1000.0, **kwargs)
    for left, right in zip(rows_s, rows_ms):
        assert left == right


def test_event_causal_rows_do_not_use_future_events() -> None:
    first = [event(0, "a", 0), event(1, "b", 10), event(2, "c", 30)]
    changed = [event(0, "a", 0), event(1, "b", 10), event(2, "z", 99_999)]
    kwargs = dict(
        macro={}, item_cat={"a": "x", "b": "y", "c": "z", "z": "q"},
        pop_score={"a": 3.0, "b": 2.0, "c": 1.0, "z": 99.0},
        pop_bin={"a": 2, "b": 1, "c": 0, "z": 9}, micro_window=5,
        mid_valid_cap=10, n_pop_bins=10, timestamp_divisor=1.0,
    )
    rows_first = MOD.compute_session_rows(events=first, **kwargs)
    rows_changed = MOD.compute_session_rows(events=changed, **kwargs)
    assert rows_first[0] == rows_changed[0]
    assert rows_first[1] == rows_changed[1]


def test_linear_implementation_matches_reference() -> None:
    events = [
        event(0, "a", 0), event(1, "b", 10), event(2, "a", 30),
        event(3, "c", 55), event(4, "c", 90), event(5, "d", 140),
    ]
    kwargs = dict(
        events=events, macro={}, item_cat={"a": "x", "b": "y", "c": "y", "d": "z"},
        pop_score={"a": 8.0, "b": 3.0, "c": 2.0, "d": 1.0},
        pop_bin={"a": 3, "b": 2, "c": 1, "d": 0}, micro_window=5,
        mid_valid_cap=10, n_pop_bins=10, timestamp_divisor=1.0,
    )
    reference = MOD._compute_session_rows_reference(**kwargs)
    optimized = MOD.compute_session_rows(**kwargs)
    assert len(reference) == len(optimized)
    for expected, actual in zip(reference, optimized):
        assert expected.keys() == actual.keys()
        for name in expected:
            if expected[name] is None:
                assert actual[name] is None
            else:
                assert math.isclose(float(expected[name]), float(actual[name]), abs_tol=1e-12)


def test_unseen_context_is_filtered_before_features() -> None:
    by_sid = {
        "train": [event(0, "seen", 0), event(1, "target", 1)],
        "tail": [event(2, "cold-context", 2), event(3, "cold-target", 3)],
    }
    result = MOD.filter_tail_unseen_context(by_sid, ["train", "tail"], 1)
    assert [value.item for value in by_sid["tail"]] == ["cold-context", "cold-target"]
    assert result["dropped_non_target_unseen_rows"] == 0
    assert result["cold_target_sessions_retained"] == 1


def test_unseen_filter_keeps_latest_context_when_all_tail_context_is_cold() -> None:
    by_sid = {
        "train": [event(0, "seen", 0), event(1, "target", 1)],
        "tail": [
            event(2, "old-cold", 2), event(3, "latest-cold", 3),
            event(4, "cold-target", 4),
        ],
    }
    result = MOD.filter_tail_unseen_context(by_sid, ["train", "tail"], 1)
    assert [value.item for value in by_sid["tail"]] == ["latest-cold", "cold-target"]
    assert result["dropped_non_target_unseen_rows"] == 1


def test_popularity_is_train_count_with_log_quantile_bins() -> None:
    by_sid = {
        "s1": [event(0, "a", 0), event(1, "a", 1), event(2, "b", 2)],
        "s2": [event(3, "z", 3)],
    }
    scores, bins, edges = MOD.build_popularity(by_sid, {"s1"}, 10)
    assert scores == {"a": 2.0, "b": 1.0}
    assert "z" not in scores
    assert len(edges) == 11
    assert bins["a"] >= bins["b"]


def test_pair_transforms_match_versioned_contract() -> None:
    stats = {
        "mac5_pace_mean": {"mean": 0.0, "std": 2.0, "q01": 0.0, "q99": 1.0},
    }
    continuous = MOD.normalize("mac5_pace_trend", MOD.log_delta(4.0, 1.0), stats)
    assert math.isclose(continuous, MOD.phi(MOD.log_delta(4.0, 1.0) / 2.0))
    bounded = MOD.normalize("mac5_repeat_trend", 0.4, stats)
    assert math.isclose(bounded, 0.7)


def test_feature_surface_is_exactly_64_unique_names() -> None:
    assert len(MOD.ALL_FEATURES) == 64
    assert len(set(MOD.ALL_FEATURES)) == 64
    assert MOD.timestamp_scale_to_seconds("s") == 1.0
    assert MOD.timestamp_scale_to_seconds("ms") == 1000.0


def test_future_builder_emits_location_independent_metadata_natively() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"contract": "core5-location-independent-metadata-v1"' in source
    assert '"method": "native_builder_v1"' in source
    assert '"source_root": str(source_dir)' not in source
    assert '"output_root": str(out_dir)' not in source


def test_frozen_basic_membership_is_exact_and_contiguous(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    header = ["session_id:token", "item_id:token", "timestamp:float", "user_id:token"]
    rows = {
        "train": [["s0", "i0", "0", "u0"], ["s0", "i1", "1", "u0"]],
        "valid": [["s1", "i0", "10", "u1"], ["s1", "i1", "11", "u1"]],
        "test": [["s2", "i0", "20", "u2"], ["s2", "i1", "21", "u2"]],
    }
    for split, values in rows.items():
        with (source / f"source.{split}.inter").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            writer.writerows(values)
    membership, counts, order = MOD.load_frozen_split_membership(
        source, "source", ["s0", "s1", "s2"]
    )
    assert membership == {"s0": "train", "s1": "valid", "s2": "test"}
    assert counts == {"train": 2, "valid": 2, "test": 2}
    assert order == {"train": ["s0"], "valid": ["s1"], "test": ["s2"]}


def test_frozen_basic_membership_preserves_explicit_split_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    header = ["session_id:token", "item_id:token", "timestamp:float", "user_id:token"]
    for split, sid in (("train", "s0"), ("valid", "s2"), ("test", "s1")):
        with (source / f"source.{split}.inter").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            writer.writerow([sid, "i0", "0", "u0"])
    membership, _, order = MOD.load_frozen_split_membership(
        source, "source", ["s0", "s1", "s2"]
    )
    assert membership["s2"] == "valid"
    assert order == {"train": ["s0"], "valid": ["s2"], "test": ["s1"]}
