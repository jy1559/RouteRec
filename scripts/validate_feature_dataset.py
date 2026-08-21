#!/usr/bin/env python3
"""Independently validate a feature release built from frozen core5 splits.

The producing builder is intentionally not imported.  The basic release is
the row, session, split, and candidate-vocabulary authority.  This validator
streams every basic/feature interaction pair, verifies exact ordered identity,
and derives its own invariants and hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Sequence


SPLITS = ("train", "valid", "test")
BASIC_CONTRACT = "sessionized-core5-strict-split-v1"
FEATURE_CONTRACT = "core5-features-leakage-safe-v1"
FEATURE_CONTRACTS = {
    FEATURE_CONTRACT,
    "full-v5-leakage-correct-20260812",
}
METADATA_CONTRACT = "core5-location-independent-metadata-v1"
METADATA_CONTRACTS = {
    METADATA_CONTRACT,
    "full-v5-location-independent-metadata-v1",
}
BASE_HEADER = (
    "session_id:token",
    "item_id:token",
    "timestamp:float",
    "user_id:token",
)
FEATURES = (
    "mac5_ctx_valid_r", "mac5_gap_last", "mac5_pace_mean", "mac5_pace_trend",
    "mac5_theme_ent_mean", "mac5_theme_top1_mean", "mac5_theme_repeat_r", "mac5_theme_shift_r",
    "mac5_repeat_mean", "mac5_adj_cat_overlap_mean", "mac5_adj_item_overlap_mean", "mac5_repeat_trend",
    "mac5_pop_mean", "mac5_pop_std_mean", "mac5_pop_ent_mean", "mac5_pop_trend",
    "mac10_ctx_valid_r", "mac10_gap_last", "mac10_pace_mean", "mac10_pace_trend",
    "mac10_theme_ent_mean", "mac10_theme_top1_mean", "mac10_theme_repeat_r", "mac10_theme_shift_r",
    "mac10_repeat_mean", "mac10_adj_cat_overlap_mean", "mac10_adj_item_overlap_mean", "mac10_repeat_trend",
    "mac10_pop_mean", "mac10_pop_std_mean", "mac10_pop_ent_mean", "mac10_pop_trend",
    "mid_valid_r", "mid_int_mean", "mid_int_std", "mid_sess_age",
    "mid_cat_ent", "mid_cat_top1", "mid_cat_switch_r", "mid_cat_uniq_r",
    "mid_item_uniq_r", "mid_repeat_r", "mid_novel_r", "mid_max_run_i",
    "mid_pop_mean", "mid_pop_std", "mid_pop_ent", "mid_pop_trend",
    "mic_valid_r", "mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid",
    "mic_cat_switch_now", "mic_last_cat_mismatch_r", "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r",
    "mic_is_recons", "mic_suffix_recons_r", "mic_suffix_uniq_i", "mic_suffix_max_run_i",
    "mic_last_pop", "mic_suffix_pop_std", "mic_suffix_pop_ent", "mic_pop_delta_vs_mid",
)
FEATURE_HEADER = BASE_HEADER + tuple(f"{name}:float" for name in FEATURES)
HEX = frozenset("0123456789abcdef")


def plain(column: str) -> str:
    return column.lstrip("\ufeff").split(":", 1)[0]


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in HEX for character in text)


def update_row_hash(digest: object, row: Iterable[str]) -> None:
    values = tuple(row)
    digest.update(len(values).to_bytes(4, "big"))  # type: ignore[attr-defined]
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))  # type: ignore[attr-defined]
        digest.update(encoded)  # type: ignore[attr-defined]


class ErrorSink:
    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self.errors: list[str] = []
        self.suppressed = 0

    def add(self, message: str) -> None:
        if len(self.errors) < self.limit:
            self.errors.append(message)
        else:
            self.suppressed += 1

    def public(self) -> list[str]:
        if not self.suppressed:
            return list(self.errors)
        return [*self.errors, f"... {self.suppressed} additional errors suppressed"]


@dataclass
class PairScan:
    basic_path: Path
    feature_path: Path
    basic_rows: int = 0
    feature_rows: int = 0
    feature_values_checked: int = 0
    sessions: dict[str, int] = field(default_factory=dict)
    session_order: list[str] = field(default_factory=list)
    items: set[str] = field(default_factory=set)
    users: set[str] = field(default_factory=set)
    basic_sha256: str = ""
    feature_sha256: str = ""
    basic_identity_sha256: str = ""
    feature_base_identity_sha256: str = ""
    feature_content_identity_sha256: str = ""

    def public(self) -> dict[str, object]:
        lengths = list(self.sessions.values())
        return {
            "basic_rows": self.basic_rows,
            "feature_rows": self.feature_rows,
            "sessions": len(self.sessions),
            "users": len(self.users),
            "items": len(self.items),
            "session_length_min": min(lengths) if lengths else None,
            "session_length_max": max(lengths) if lengths else None,
            "feature_values_checked": self.feature_values_checked,
            "basic_sha256": self.basic_sha256,
            "feature_sha256": self.feature_sha256,
            "basic_identity_sha256": self.basic_identity_sha256,
            "feature_base_identity_sha256": self.feature_base_identity_sha256,
            "feature_content_identity_sha256": self.feature_content_identity_sha256,
        }


def parse_tsv_line(serialized: str) -> list[str]:
    return next(csv.reader([serialized], delimiter="\t"), [])


def scan_pair(
    basic_path: Path,
    feature_path: Path,
    *,
    errors: ErrorSink,
    aggregate_basic: object | None = None,
    aggregate_feature_base: object | None = None,
    aggregate_feature_content: object | None = None,
) -> PairScan:
    result = PairScan(basic_path=basic_path, feature_path=feature_path)
    raw_basic = hashlib.sha256()
    raw_feature = hashlib.sha256()
    basic_identity = hashlib.sha256()
    feature_base_identity = hashlib.sha256()
    feature_content_identity = hashlib.sha256()
    current_sid: str | None = None
    current_user: str | None = None
    current_timestamp = -math.inf
    closed_sessions: set[str] = set()

    with basic_path.open("r", encoding="utf-8-sig", newline="") as basic_handle, feature_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as feature_handle:
        basic_header_line = basic_handle.readline()
        feature_header_line = feature_handle.readline()
        raw_basic.update(basic_header_line.encode("utf-8"))
        raw_feature.update(feature_header_line.encode("utf-8"))
        basic_header = tuple(parse_tsv_line(basic_header_line))
        feature_header = tuple(parse_tsv_line(feature_header_line))
        if basic_header != BASE_HEADER:
            errors.add(f"{basic_path}: expected exact four-column basic header")
        if feature_header != FEATURE_HEADER:
            errors.add(f"{feature_path}: expected exact 4+64 feature header/order")

        for row_number, pair in enumerate(
            zip_longest(basic_handle, feature_handle), start=2
        ):
            basic_line, feature_line = pair
            basic_row: list[str] | None = None
            feature_row: list[str] | None = None
            if basic_line is None:
                errors.add(f"{basic_path}: ended before feature row {row_number}")
            else:
                raw_basic.update(basic_line.encode("utf-8"))
                basic_row = parse_tsv_line(basic_line)
                result.basic_rows += 1
                if len(basic_row) != len(BASE_HEADER):
                    errors.add(
                        f"{basic_path}:{row_number}: width {len(basic_row)} != {len(BASE_HEADER)}"
                    )
                    basic_row = None
            if feature_line is None:
                errors.add(f"{feature_path}: ended before basic row {row_number}")
            else:
                raw_feature.update(feature_line.encode("utf-8"))
                feature_row = parse_tsv_line(feature_line)
                result.feature_rows += 1
                if len(feature_row) != len(FEATURE_HEADER):
                    errors.add(
                        f"{feature_path}:{row_number}: width {len(feature_row)} != {len(FEATURE_HEADER)}"
                    )
                    feature_row = None

            if basic_row is not None:
                update_row_hash(basic_identity, basic_row)
                if aggregate_basic is not None:
                    update_row_hash(aggregate_basic, basic_row)
                sid, item, timestamp_raw, user = basic_row
                result.items.add(item)
                result.users.add(user)
                if sid != current_sid:
                    if current_sid is not None:
                        closed_sessions.add(current_sid)
                    if sid in closed_sessions:
                        errors.add(f"{basic_path}:{row_number}: non-contiguous session {sid!r}")
                    current_sid = sid
                    current_user = user
                    current_timestamp = -math.inf
                    result.session_order.append(sid)
                    result.sessions.setdefault(sid, 0)
                elif user != current_user:
                    errors.add(f"{basic_path}:{row_number}: session {sid!r} maps to multiple users")
                result.sessions[sid] = result.sessions.get(sid, 0) + 1
                try:
                    timestamp = float(timestamp_raw)
                except ValueError:
                    timestamp = math.nan
                if not math.isfinite(timestamp):
                    errors.add(f"{basic_path}:{row_number}: non-finite timestamp {timestamp_raw!r}")
                elif timestamp < current_timestamp:
                    errors.add(f"{basic_path}:{row_number}: timestamp regression in {sid!r}")
                else:
                    current_timestamp = timestamp

            if feature_row is not None:
                feature_base = feature_row[: len(BASE_HEADER)]
                update_row_hash(feature_base_identity, feature_base)
                update_row_hash(feature_content_identity, feature_row)
                if aggregate_feature_base is not None:
                    update_row_hash(aggregate_feature_base, feature_base)
                if aggregate_feature_content is not None:
                    update_row_hash(aggregate_feature_content, feature_row)
                if basic_row is not None and feature_base != basic_row:
                    errors.add(
                        f"{feature_path}:{row_number}: base identity differs from frozen basic row"
                    )
                for offset, raw_value in enumerate(feature_row[len(BASE_HEADER) :]):
                    result.feature_values_checked += 1
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = math.nan
                    if not math.isfinite(value) or value < 0.0 or value > 1.0:
                        errors.add(
                            f"{feature_path}:{row_number}: {FEATURES[offset]}={raw_value!r} "
                            "is not finite in [0,1]"
                        )

    result.basic_sha256 = raw_basic.hexdigest()
    result.feature_sha256 = raw_feature.hexdigest()
    result.basic_identity_sha256 = basic_identity.hexdigest()
    result.feature_base_identity_sha256 = feature_base_identity.hexdigest()
    result.feature_content_identity_sha256 = feature_content_identity.hexdigest()
    if result.basic_rows != result.feature_rows:
        errors.add(
            f"row count mismatch: {basic_path.name}={result.basic_rows}, "
            f"{feature_path.name}={result.feature_rows}"
        )
    if result.basic_identity_sha256 != result.feature_base_identity_sha256:
        errors.add(f"base identity digest mismatch: {basic_path.name} vs {feature_path.name}")
    return result


def load_json(path: Path, errors: ErrorSink) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.add(f"cannot read JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.add(f"JSON root is not an object: {path}")
        return {}
    return value


def item_ids(path: Path, errors: ErrorSink) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = {plain(name): name for name in (reader.fieldnames or [])}
        if "item_id" not in names:
            errors.add(f"{path}: missing item_id column")
            return values
        for row_number, row in enumerate(reader, start=2):
            value = str(row[names["item_id"]])
            if value in values:
                errors.add(f"{path}:{row_number}: duplicate item_id {value!r}")
            values.add(value)
    return values


def require_equal(errors: ErrorSink, actual: object, expected: object, label: str) -> None:
    if actual != expected:
        errors.add(f"{label}: expected {expected!r}, got {actual!r}")


def verify_recorded_hashes(
    summary: dict[str, object],
    actual: dict[str, dict[str, object]],
    errors: ErrorSink,
) -> None:
    recorded = summary.get("files")
    if not isinstance(recorded, dict):
        errors.add("basic summary is missing its files hash manifest")
        return
    for name, evidence in actual.items():
        entry = recorded.get(name)
        if not isinstance(entry, dict):
            errors.add(f"basic summary files manifest is missing {name}")
            continue
        require_equal(errors, entry.get("sha256"), evidence["sha256"], f"recorded hash {name}")
        require_equal(errors, entry.get("bytes"), evidence["bytes"], f"recorded size {name}")


def validate_metadata(
    *,
    basic_summary: dict[str, object],
    feature_meta: dict[str, object],
    build_summary: dict[str, object],
    split_summary: dict[str, object],
    source_dataset: str,
    target_dataset: str,
    basic_inter_sha: str,
    basic_item_sha: str,
    combined: PairScan,
    splits: dict[str, PairScan],
    errors: ErrorSink,
) -> None:
    require_equal(errors, basic_summary.get("contract"), BASIC_CONTRACT, "basic contract")
    is_core5_parent = (
        basic_summary.get("core5_contract", basic_summary.get("camera_ready_contract"))
        in {"core5-parent-strict-v1", "camera-ready-core5-parent-strict-v1"}
    )
    if is_core5_parent:
        if basic_summary.get("status") not in (
            "complete",
            "complete_staged_pending_independent_validation",
        ):
            errors.add(
                "core5 basic status is neither complete nor staged-complete"
            )
    else:
        require_equal(errors, basic_summary.get("status"), "complete", "basic status")
    require_equal(errors, basic_summary.get("target_dataset"), source_dataset, "basic target dataset")
    basic_output = basic_summary.get("output")
    if not isinstance(basic_output, dict):
        errors.add("basic summary is missing output counts")
    else:
        require_equal(
            errors, basic_output.get("rows"),
            {split: scan.basic_rows for split, scan in splits.items()},
            "basic output split rows",
        )
        require_equal(
            errors, basic_output.get("sessions"),
            {split: len(scan.sessions) for split, scan in splits.items()},
            "basic output split sessions",
        )
        require_equal(errors, basic_output.get("total_rows"), combined.basic_rows, "basic total rows")
        require_equal(
            errors, basic_output.get("total_sessions"), len(combined.sessions),
            "basic total sessions",
        )

    require_equal(errors, feature_meta.get("dataset"), target_dataset, "feature metadata dataset")
    require_equal(errors, feature_meta.get("source_dataset"), source_dataset, "feature metadata source")
    require_equal(errors, feature_meta.get("all_features"), list(FEATURES), "feature metadata surface")
    if feature_meta.get("reconstruction_contract") not in FEATURE_CONTRACTS:
        errors.add("feature reconstruction contract mismatch")
    require_equal(
        errors, feature_meta.get("source_preprocessing_contract"), BASIC_CONTRACT,
        "feature source preprocessing contract",
    )
    tail = feature_meta.get("tail_unseen_filter")
    if not isinstance(tail, dict) or tail.get("policy") != "precleaned_frozen":
        errors.add("feature metadata does not declare precleaned_frozen tail policy")
        tail = {}
    require_equal(
        errors, tail.get("source_split_rows"),
        {split: scan.basic_rows for split, scan in splits.items()},
        "metadata frozen source split rows",
    )
    require_equal(errors, tail.get("train_item_count"), len(splits["train"].items), "metadata train item count")
    fit = feature_meta.get("fit_sessions")
    if not isinstance(fit, dict):
        errors.add("feature metadata is missing fit_sessions")
    else:
        require_equal(errors, fit.get("fit_session_count"), len(splits["train"].sessions), "fit session count")
        require_equal(errors, fit.get("total_session_count"), len(combined.sessions), "fit total sessions")
        expected_ratio = len(splits["train"].sessions) / max(1, len(combined.sessions))
        try:
            observed_ratio = float(fit.get("fit_session_ratio"))
        except (TypeError, ValueError):
            observed_ratio = math.nan
        if not math.isclose(observed_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            errors.add(
                f"fit session ratio: expected {expected_ratio!r}, got {fit.get('fit_session_ratio')!r}"
            )
    require_equal(errors, feature_meta.get("mid_scope"), "strict_prefix", "mid feature scope")
    mid_postprocess = feature_meta.get("mid_constant_postprocess")
    if not isinstance(mid_postprocess, dict) or mid_postprocess.get("applied") is not False:
        errors.add("feature metadata does not prove the session-final mid postprocess is disabled")
    normalization_stats = feature_meta.get("normalization_stats")
    if not isinstance(normalization_stats, dict) or set(normalization_stats) != set(FEATURES):
        errors.add("normalization metadata does not cover exactly the canonical 64 features")
    basic_parameters = basic_summary.get("parameters")
    basic_timestamp_unit = (
        basic_parameters.get("timestamp_unit") if isinstance(basic_parameters, dict) else None
    )
    require_equal(
        errors, feature_meta.get("timestamp_unit"), basic_timestamp_unit,
        "basic/feature timestamp unit",
    )
    source_hashes = feature_meta.get("source_hashes")
    if not isinstance(source_hashes, dict):
        errors.add("feature metadata is missing source_hashes")
        source_hashes = {}
    require_equal(errors, source_hashes.get("inter_sha256"), basic_inter_sha, "source inter hash")
    require_equal(errors, source_hashes.get("item_sha256"), basic_item_sha, "source item hash")
    if not valid_sha256(source_hashes.get("builder_sha256")):
        errors.add("feature metadata builder_sha256 is invalid")

    canonical = feature_meta.get("metadata_canonicalization")
    if not isinstance(canonical, dict):
        errors.add("feature metadata is missing canonicalization contract")
    else:
        require_equal(errors, canonical.get("schema_version"), 1, "metadata schema version")
        if canonical.get("contract") not in METADATA_CONTRACTS:
            errors.add("metadata contract mismatch")
        require_equal(errors, canonical.get("method"), "native_builder_v1", "metadata method")
        require_equal(
            errors, canonical.get("removed_fields"), ["output_root", "source_root"],
            "metadata removed fields",
        )
        require_equal(errors, canonical.get("source_dataset"), source_dataset, "canonical source")
        require_equal(errors, canonical.get("target_dataset"), target_dataset, "canonical target")
        require_equal(
            errors, canonical.get("builder_sha256"), source_hashes.get("builder_sha256"),
            "canonical builder hash",
        )
    for forbidden in ("source_root", "output_root"):
        if forbidden in feature_meta:
            errors.add(f"location-dependent field remains in feature metadata: {forbidden}")

    require_equal(errors, build_summary.get("dataset"), target_dataset, "build dataset")
    require_equal(errors, build_summary.get("source_dataset"), source_dataset, "build source")
    if build_summary.get("feature_contract_version") not in FEATURE_CONTRACTS:
        errors.add("build contract mismatch")
    require_equal(errors, build_summary.get("frozen_basic_splits"), True, "frozen basic split flag")
    require_equal(errors, build_summary.get("rows"), combined.basic_rows, "build row count")
    require_equal(errors, build_summary.get("sessions"), len(combined.sessions), "build session count")
    require_equal(errors, build_summary.get("features"), len(FEATURES), "build feature count")
    require_equal(errors, build_summary.get("source_hashes"), source_hashes, "build/meta source hashes")
    require_equal(errors, build_summary.get("tail_unseen_filter"), tail, "build/meta tail policy")
    require_equal(
        errors, build_summary.get("split_session_counts"),
        {split: len(scans.sessions) for split, scans in splits.items()},
        "build split session counts",
    )

    require_equal(errors, split_summary.get("dataset"), target_dataset, "split summary dataset")
    require_equal(errors, build_summary.get("split_summary"), split_summary, "build/external split summary")
    require_equal(
        errors, split_summary.get("split_strategy"), "frozen_source_basic_membership",
        "split strategy",
    )
    write_stats = split_summary.get("write_stats")
    if not isinstance(write_stats, dict):
        errors.add("split summary is missing write_stats")
    else:
        require_equal(
            errors, write_stats.get("rows"),
            {split: scan.basic_rows for split, scan in splits.items()},
            "split write rows",
        )
        require_equal(
            errors, write_stats.get("sessions"),
            {split: len(scan.sessions) for split, scan in splits.items()},
            "split write sessions",
        )
        require_equal(
            errors, write_stats.get("session_overlap"),
            {"train_valid": 0, "train_test": 0, "valid_test": 0},
            "recorded split overlap",
        )


def validate_release(
    *,
    basic_root: Path,
    feature_root: Path,
    source_dataset: str,
    target_dataset: str,
    max_errors: int = 100,
) -> dict[str, object]:
    errors = ErrorSink(max_errors)
    basic_dir = basic_root / source_dataset
    feature_dir = feature_root / target_dataset
    basic_paths = {
        "combined": basic_dir / f"{source_dataset}.inter",
        **{split: basic_dir / f"{source_dataset}.{split}.inter" for split in SPLITS},
        "item": basic_dir / f"{source_dataset}.item",
        "summary": basic_dir / f"{source_dataset}.basic_summary.json",
    }
    feature_paths = {
        "combined": feature_dir / f"{target_dataset}.inter",
        **{split: feature_dir / f"{target_dataset}.{split}.inter" for split in SPLITS},
        "item": feature_dir / f"{target_dataset}.item",
        "meta": (
            feature_dir / "feature_metadata.json"
            if (feature_dir / "feature_metadata.json").is_file()
            else feature_dir / "feature_meta_v3.json"
        ),
        "build": feature_dir / f"{target_dataset}.build_summary.json",
        "split_summary": feature_dir / f"{target_dataset}.session_split_summary.json",
    }
    for path in [*basic_paths.values(), *feature_paths.values()]:
        if not path.is_file():
            errors.add(f"missing required file: {path}")
    if errors.errors:
        return {
            "schema_version": 1,
            "validator_contract": "frozen-core5-feature-validator-v1",
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "ok": False,
            "errors": errors.public(),
        }

    combined = scan_pair(
        basic_paths["combined"], feature_paths["combined"], errors=errors
    )
    aggregate_basic = hashlib.sha256()
    aggregate_feature_base = hashlib.sha256()
    aggregate_feature_content = hashlib.sha256()
    splits = {
        split: scan_pair(
            basic_paths[split], feature_paths[split], errors=errors,
            aggregate_basic=aggregate_basic,
            aggregate_feature_base=aggregate_feature_base,
            aggregate_feature_content=aggregate_feature_content,
        )
        for split in SPLITS
    }

    if combined.basic_identity_sha256 != aggregate_basic.hexdigest():
        errors.add("basic combined rows are not the exact ordered train+valid+test union")
    if combined.feature_base_identity_sha256 != aggregate_feature_base.hexdigest():
        errors.add("feature combined base rows are not the exact ordered split union")
    if combined.feature_content_identity_sha256 != aggregate_feature_content.hexdigest():
        errors.add("feature combined 68-column rows are not the exact ordered split union")

    split_session_sets = {split: set(scan.sessions) for split, scan in splits.items()}
    overlaps = {
        "train_valid": len(split_session_sets["train"] & split_session_sets["valid"]),
        "train_test": len(split_session_sets["train"] & split_session_sets["test"]),
        "valid_test": len(split_session_sets["valid"] & split_session_sets["test"]),
    }
    if any(overlaps.values()):
        errors.add(f"split sessions are not disjoint: {overlaps}")
    merged_lengths: dict[str, int] = {}
    for split in SPLITS:
        merged_lengths.update(splits[split].sessions)
    if combined.sessions != merged_lengths:
        errors.add("combined session membership/lengths differ from split union")

    train_items = splits["train"].items
    tail_items = splits["valid"].items | splits["test"].items
    unseen_tail = tail_items - train_items
    if unseen_tail:
        errors.add(f"valid/test contain {len(unseen_tail)} train-unseen items")
    basic_items = item_ids(basic_paths["item"], errors)
    feature_items = item_ids(feature_paths["item"], errors)
    if basic_items != train_items:
        errors.add("basic item table is not the exact training vocabulary")
    if feature_items != train_items:
        errors.add("feature item table is not the exact training vocabulary")
    basic_item_sha = sha256_file(basic_paths["item"])
    feature_item_sha = sha256_file(feature_paths["item"])
    if feature_item_sha != basic_item_sha:
        errors.add("feature item table is not an exact byte copy of the frozen basic item table")

    actual_min = min(merged_lengths.values(), default=0)
    actual_max = max(merged_lengths.values(), default=0)
    basic_summary = load_json(basic_paths["summary"], errors)
    is_core5_parent = (
        basic_summary.get("core5_contract", basic_summary.get("camera_ready_contract"))
        in {"core5-parent-strict-v1", "camera-ready-core5-parent-strict-v1"}
    )
    parameters = basic_summary.get("parameters")
    if not isinstance(parameters, dict):
        errors.add("basic summary is missing parameters")
        parameters = {}
    declared_min = parameters.get("minimum_session_length")
    declared_max = parameters.get("maximum_session_length")
    if declared_max is None and is_core5_parent:
        declared_max = parameters.get("maximum_chunk_length")
    if not isinstance(declared_min, int) or actual_min < declared_min:
        errors.add(f"session minimum {actual_min} violates declared minimum {declared_min!r}")
    if not isinstance(declared_max, int) or actual_max > declared_max:
        errors.add(f"session maximum {actual_max} violates declared maximum {declared_max!r}")
    invariants = basic_summary.get("invariants")
    if not isinstance(invariants, dict):
        errors.add("basic summary is missing invariants")
        invariants = {}
    if is_core5_parent and "minimum_session_length" not in invariants:
        require_equal(errors, invariants.get("session_length_5_to_50"), True, "core5 session length flag")
        require_equal(errors, invariants.get("valid_test_items_subset_train"), True, "core5 tail seen flag")
    else:
        require_equal(errors, invariants.get("minimum_session_length"), actual_min, "recorded minimum session length")
        require_equal(errors, invariants.get("maximum_session_length"), actual_max, "recorded maximum session length")
        require_equal(errors, invariants.get("minimum_session_length_pass"), True, "minimum session flag")
        require_equal(errors, invariants.get("maximum_session_length_pass"), True, "maximum session flag")
        require_equal(errors, invariants.get("valid_test_items_subset_of_train"), True, "tail seen flag")
    require_equal(errors, invariants.get("valid_test_unseen_item_count"), 0, "tail unseen count")

    basic_file_evidence = {
        f"{source_dataset}.inter": {
            "bytes": basic_paths["combined"].stat().st_size,
            "sha256": combined.basic_sha256,
        },
        **{
            f"{source_dataset}.{split}.inter": {
                "bytes": basic_paths[split].stat().st_size,
                "sha256": scans.basic_sha256,
            }
            for split, scans in splits.items()
        },
        f"{source_dataset}.item": {
            "bytes": basic_paths["item"].stat().st_size,
            "sha256": basic_item_sha,
        },
    }
    verify_recorded_hashes(basic_summary, basic_file_evidence, errors)

    feature_meta = load_json(feature_paths["meta"], errors)
    build_summary = load_json(feature_paths["build"], errors)
    split_summary = load_json(feature_paths["split_summary"], errors)
    validate_metadata(
        basic_summary=basic_summary,
        feature_meta=feature_meta,
        build_summary=build_summary,
        split_summary=split_summary,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        basic_inter_sha=combined.basic_sha256,
        basic_item_sha=basic_item_sha,
        combined=combined,
        splits=splits,
        errors=errors,
    )

    feature_file_evidence = {
        f"{target_dataset}.inter": {
            "bytes": feature_paths["combined"].stat().st_size,
            "sha256": combined.feature_sha256,
        },
        **{
            f"{target_dataset}.{split}.inter": {
                "bytes": feature_paths[split].stat().st_size,
                "sha256": scans.feature_sha256,
            }
            for split, scans in splits.items()
        },
        f"{target_dataset}.item": {
            "bytes": feature_paths["item"].stat().st_size,
            "sha256": feature_item_sha,
        },
        feature_paths["meta"].name: {
            "bytes": feature_paths["meta"].stat().st_size,
            "sha256": sha256_file(feature_paths["meta"]),
        },
        f"{target_dataset}.build_summary.json": {
            "bytes": feature_paths["build"].stat().st_size,
            "sha256": sha256_file(feature_paths["build"]),
        },
        f"{target_dataset}.session_split_summary.json": {
            "bytes": feature_paths["split_summary"].stat().st_size,
            "sha256": sha256_file(feature_paths["split_summary"]),
        },
    }
    report = {
        "schema_version": 1,
        "validator_contract": "frozen-core5-feature-validator-v1",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "ok": not errors.errors and errors.suppressed == 0,
        "errors": errors.public(),
        "combined": combined.public(),
        "splits": {split: scan.public() for split, scan in splits.items()},
        "invariants": {
            "exact_basic_feature_row_identity": (
                combined.basic_identity_sha256 == combined.feature_base_identity_sha256
                and all(
                    scan.basic_identity_sha256 == scan.feature_base_identity_sha256
                    for scan in splits.values()
                )
            ),
            "combined_is_exact_ordered_split_union": (
                combined.basic_identity_sha256 == aggregate_basic.hexdigest()
                and combined.feature_content_identity_sha256
                == aggregate_feature_content.hexdigest()
            ),
            "split_session_overlap": overlaps,
            "session_length_min": actual_min,
            "session_length_max": actual_max,
            "valid_test_unseen_items": len(unseen_tail),
            "item_table_equals_train_vocabulary": (
                basic_items == feature_items == train_items
            ),
            "feature_values_checked": sum(
                scan.feature_values_checked for scan in splits.values()
            ),
            "feature_schema_columns": len(FEATURE_HEADER),
        },
        "files": {"basic": basic_file_evidence, "feature": feature_file_evidence},
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-errors", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_release(
        basic_root=args.basic_root,
        feature_root=args.feature_root,
        source_dataset=str(args.source_dataset),
        target_dataset=str(args.target_dataset),
        max_errors=int(args.max_errors),
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
