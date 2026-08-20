#!/usr/bin/env python3
"""Independently validate a staged camera-ready core5 basic dataset tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


SPLITS = ("train", "valid", "test")
BASE_NAMES = ("session_id", "item_id", "timestamp", "user_id")
BASIC_CONTRACT = "sessionized-core5-strict-split-v1"
CAMERA_READY_CONTRACT = "camera-ready-core5-parent-strict-v1"


def plain(value: str) -> str:
    return str(value).lstrip("\ufeff").split(":", 1)[0]


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def parse_utc(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


@dataclass
class Scan:
    rows: int
    session_lengths: dict[str, int]
    session_users: dict[str, str]
    session_targets: dict[str, tuple[str, str]]
    item_counts: Counter[str]
    users: set[str]
    errors: list[str]

    @property
    def sessions(self) -> set[str]:
        return set(self.session_lengths)


def scan_inter(path: Path) -> Scan:
    errors: list[str] = []
    lengths: dict[str, int] = {}
    session_users: dict[str, str] = {}
    targets: dict[str, tuple[str, str]] = {}
    item_counts: Counter[str] = Counter()
    users: set[str] = set()
    closed: set[str] = set()
    previous_sid: str | None = None
    previous_timestamp: float | None = None
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = list(reader.fieldnames or [])
        names = {plain(name): name for name in header}
        if tuple(plain(name) for name in header) != BASE_NAMES:
            errors.append(f"{path.name}: interaction header is not exact base four")
        if not set(BASE_NAMES).issubset(names):
            return Scan(0, {}, {}, {}, Counter(), set(), errors + ["missing columns"])
        for row_number, row in enumerate(reader, start=2):
            sid = str(row[names["session_id"]])
            item = str(row[names["item_id"]])
            user = str(row[names["user_id"]])
            timestamp_text = str(row[names["timestamp"]])
            try:
                timestamp = float(timestamp_text)
            except ValueError:
                errors.append(f"{path.name}:{row_number}: invalid timestamp")
                continue
            if not math.isfinite(timestamp):
                errors.append(f"{path.name}:{row_number}: non-finite timestamp")
            if sid != previous_sid:
                if previous_sid is not None:
                    closed.add(previous_sid)
                if sid in closed:
                    errors.append(f"{path.name}:{row_number}: non-contiguous session {sid!r}")
                previous_sid = sid
                previous_timestamp = None
            if previous_timestamp is not None and timestamp < previous_timestamp:
                errors.append(f"{path.name}:{row_number}: timestamp regression in {sid!r}")
            previous_timestamp = timestamp
            if sid in session_users and session_users[sid] != user:
                errors.append(f"{path.name}:{row_number}: session has multiple users")
            session_users[sid] = user
            lengths[sid] = lengths.get(sid, 0) + 1
            targets[sid] = (item, timestamp_text)
            item_counts[item] += 1
            users.add(user)
            rows += 1
    return Scan(rows, lengths, session_users, targets, item_counts, users, errors)


def iter_data_lines(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        header = handle.readline()
        if not header:
            return
        yield from handle


def exact_union_error(combined: Path, split_paths: Mapping[str, Path]) -> str | None:
    expected = itertools.chain.from_iterable(iter_data_lines(split_paths[s]) for s in SPLITS)
    actual = iter_data_lines(combined)
    sentinel = object()
    for index, (left, right) in enumerate(
        itertools.zip_longest(actual, expected, fillvalue=sentinel), start=1
    ):
        if left != right:
            return f"combined row {index} differs from ordered train+valid+test union"
    return None


def load_item_ids(path: Path) -> tuple[set[str], int, list[str]]:
    errors: list[str] = []
    ids: set[str] = set()
    duplicates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = {plain(name): name for name in (reader.fieldnames or [])}
        if "item_id" not in names or "category" not in names:
            errors.append("item table must contain item_id and category")
            return ids, duplicates, errors
        for row_number, row in enumerate(reader, start=2):
            item = str(row[names["item_id"]])
            if item in ids:
                duplicates += 1
                errors.append(f"item row {row_number}: duplicate item {item!r}")
            ids.add(item)
    return ids, duplicates, errors


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def validate_lineage(
    path: Path,
    *,
    scans: Mapping[str, Scan],
    summary: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    output_sids = {split: scans[split].sessions for split in SPLITS}
    seen_chunks: set[str] = set()
    parent_split: dict[str, str] = {}
    parent_order: dict[str, tuple[float, int, str]] = {}
    parent_chunks: dict[str, set[int]] = defaultdict(set)
    parent_chunk_count: dict[str, int] = {}
    lineage_kept: dict[str, set[str]] = {split: set() for split in SPLITS}
    input_rows = Counter()
    final_rows = Counter()
    removed_rows = Counter()
    statuses = Counter()
    records = 0
    required = {
        "chunk_session_id", "parent_session_id", "split", "user_id",
        "parent_start_timestamp", "parent_end_timestamp", "parent_first_source_row",
        "chunk_index", "chunk_count", "original_rows", "final_rows",
        "rows_removed", "status", "original_target_item",
        "original_target_timestamp", "final_target_item", "final_target_timestamp",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            return {}, [f"lineage missing columns: {sorted(missing)}"]
        for row_number, row in enumerate(reader, start=2):
            records += 1
            sid = row["chunk_session_id"]
            parent = row["parent_session_id"]
            split = row["split"]
            status = row["status"]
            if split not in SPLITS:
                errors.append(f"lineage row {row_number}: invalid split {split!r}")
                continue
            if sid in seen_chunks:
                errors.append(f"lineage row {row_number}: duplicate chunk {sid!r}")
            seen_chunks.add(sid)
            if parent in parent_split and parent_split[parent] != split:
                errors.append(f"parent {parent!r} crosses splits")
            parent_split[parent] = split
            try:
                parent_start = float(row["parent_start_timestamp"])
                parent_first = int(row["parent_first_source_row"])
                chunk_index = int(row["chunk_index"])
                chunk_count = int(row["chunk_count"])
                original = int(row["original_rows"])
                final = int(row["final_rows"])
                removed = int(row["rows_removed"])
            except ValueError:
                errors.append(f"lineage row {row_number}: invalid numeric field")
                continue
            key = (parent_start, parent_first, parent)
            if parent in parent_order and parent_order[parent] != key:
                errors.append(f"parent {parent!r} has inconsistent ordering metadata")
            parent_order[parent] = key
            parent_chunks[parent].add(chunk_index)
            if parent in parent_chunk_count and parent_chunk_count[parent] != chunk_count:
                errors.append(f"parent {parent!r} has inconsistent chunk_count")
            parent_chunk_count[parent] = chunk_count
            if original - final != removed or min(original, final, removed) < 0:
                errors.append(f"lineage row {row_number}: row ledger does not balance")
            input_rows[split] += original
            final_rows[split] += final
            removed_rows[split] += removed
            statuses[status] += 1
            is_output = sid in output_sids[split]
            if status == "kept":
                lineage_kept[split].add(sid)
                if not is_output:
                    errors.append(f"lineage kept chunk {sid!r} absent from {split}")
                elif scans[split].session_lengths[sid] != final:
                    errors.append(f"lineage final length mismatch for {sid!r}")
                elif scans[split].session_users[sid] != row["user_id"]:
                    errors.append(f"lineage user mismatch for {sid!r}")
                target = scans[split].session_targets.get(sid)
                if target and (
                    target[0] != row["final_target_item"]
                    or float(target[1]) != float(row["final_target_timestamp"])
                ):
                    errors.append(f"lineage final target mismatch for {sid!r}")
                if split in ("valid", "test") and (
                    row["original_target_item"] != row["final_target_item"]
                    or float(row["original_target_timestamp"])
                    != float(row["final_target_timestamp"])
                ):
                    errors.append(f"tail chunk {sid!r} was retargeted")
            elif is_output:
                errors.append(f"dropped lineage chunk {sid!r} appears in {split}")

    for split in SPLITS:
        if lineage_kept[split] != output_sids[split]:
            errors.append(f"lineage kept set differs from {split} output session set")
        if final_rows[split] != scans[split].rows:
            errors.append(f"lineage final rows differ from {split} output rows")
    for parent, count in parent_chunk_count.items():
        if parent_chunks[parent] != set(range(count)):
            errors.append(f"parent {parent!r} chunk indices are incomplete")
            if len(errors) >= 100:
                break

    ordered = sorted(parent_order, key=lambda parent: parent_order[parent])
    split_sequence = [parent_split[parent] for parent in ordered]
    rank = {"train": 0, "valid": 1, "test": 2}
    if any(rank[right] < rank[left] for left, right in zip(split_sequence, split_sequence[1:])):
        errors.append("parent split is not a contiguous chronological 70/15/15 partition")
    expected_total = len(parent_order)
    expected_train = math.floor(expected_total * 0.7)
    expected_valid = (expected_total - expected_train) // 2
    expected_parent_counts = {
        "train": expected_train,
        "valid": expected_valid,
        "test": expected_total - expected_train - expected_valid,
    }
    actual_parent_counts = Counter(parent_split.values())
    if dict(actual_parent_counts) != expected_parent_counts:
        errors.append(
            f"parent split counts {dict(actual_parent_counts)} != {expected_parent_counts}"
        )
    declared = _as_dict(_as_dict(summary.get("parent_split")).get("parents"))
    if declared != expected_parent_counts:
        errors.append(f"summary parent counts {declared} != {expected_parent_counts}")
    return {
        "records": records,
        "parents": expected_total,
        "parent_counts": dict(actual_parent_counts),
        "input_rows": dict(input_rows),
        "final_rows": dict(final_rows),
        "removed_rows": dict(removed_rows),
        "statuses": dict(statuses),
        "parent_cross_split_count": sum(1 for error in errors if "crosses splits" in error),
        "chronological_parent_partition": not any(
            "contiguous chronological" in error for error in errors
        ),
    }, errors


def validate(args: argparse.Namespace) -> dict[str, object]:
    directory = args.root / args.dataset
    errors: list[str] = []
    required_paths = {
        "combined": directory / f"{args.dataset}.inter",
        **{split: directory / f"{args.dataset}.{split}.inter" for split in SPLITS},
        "item": directory / f"{args.dataset}.item",
        "summary": directory / f"{args.dataset}.basic_summary.json",
        "lineage": directory / f"{args.dataset}.parent_lineage.tsv",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        return {"ok": False, "dataset": args.dataset, "errors": [f"missing files: {missing}"]}

    scans = {name: scan_inter(required_paths[name]) for name in ("combined", *SPLITS)}
    for scan in scans.values():
        errors.extend(scan.errors)
    overlap = {
        "train_valid": len(scans["train"].sessions & scans["valid"].sessions),
        "train_test": len(scans["train"].sessions & scans["test"].sessions),
        "valid_test": len(scans["valid"].sessions & scans["test"].sessions),
    }
    if any(overlap.values()):
        errors.append(f"split session overlap: {overlap}")
    union_error = exact_union_error(
        required_paths["combined"], {split: required_paths[split] for split in SPLITS}
    )
    if union_error:
        errors.append(union_error)
    if scans["combined"].rows != sum(scans[split].rows for split in SPLITS):
        errors.append("combined row count differs from split sum")
    if any(
        length < 5 or length > 50
        for split in SPLITS
        for length in scans[split].session_lengths.values()
    ):
        errors.append("final session length is outside [5,50]")
    train_min_frequency = min(scans["train"].item_counts.values(), default=0)
    if train_min_frequency < 3:
        errors.append(f"train minimum item occurrence frequency is {train_min_frequency}")
    tail_items = scans["valid"].item_counts.keys() | scans["test"].item_counts.keys()
    unseen_items = set(tail_items) - set(scans["train"].item_counts)
    if unseen_items:
        errors.append(f"valid/test contain {len(unseen_items)} train-unseen items")
    unseen_targets = {
        target[0]
        for split in ("valid", "test")
        for target in scans[split].session_targets.values()
        if target[0] not in scans["train"].item_counts
    }
    if unseen_targets:
        errors.append(f"valid/test contain {len(unseen_targets)} train-unseen targets")

    item_ids, item_duplicates, item_errors = load_item_ids(required_paths["item"])
    errors.extend(item_errors)
    train_items = set(scans["train"].item_counts)
    if item_ids != train_items:
        errors.append(
            f"item table differs from train vocabulary: missing={len(train_items-item_ids)}, "
            f"extra={len(item_ids-train_items)}"
        )

    try:
        summary = json.loads(required_paths["summary"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        errors.append(f"invalid basic summary: {exc}")
    if summary.get("contract") != BASIC_CONTRACT:
        errors.append("basic contract mismatch")
    if summary.get("camera_ready_contract") != CAMERA_READY_CONTRACT:
        errors.append("camera-ready contract mismatch")
    if summary.get("target_dataset") != args.dataset:
        errors.append("summary target dataset mismatch")
    membership = _as_dict(summary.get("membership"))
    pre_split = _as_dict(membership.get("pre_split"))
    actual_pre = {
        "rows": pre_split.get("rows"),
        "sessions": pre_split.get("sessions"),
        "items": pre_split.get("items"),
    }
    expected_pre = {
        "rows": args.expected_rows,
        "sessions": args.expected_sessions,
        "items": args.expected_items,
    }
    if actual_pre != expected_pre:
        errors.append(f"pre-split gate mismatch: {actual_pre} != {expected_pre}")
    if args.expected_mean_length is not None and not math.isclose(
        float(pre_split.get("mean_session_length", math.nan)),
        args.expected_mean_length,
        rel_tol=0.0,
        abs_tol=0.001,
    ):
        errors.append("pre-split mean session length mismatch")

    parameters = _as_dict(summary.get("parameters"))
    if parameters.get("timestamp_unit") != args.timestamp_unit:
        errors.append("timestamp unit mismatch")
    for flag in ("retarget", "backfill", "resplit"):
        if parameters.get(flag) is not False:
            errors.append(f"{flag} must be false")
    strict = _as_dict(summary.get("strict_cleanup"))
    if strict.get("retarget_count") != 0 or strict.get("backfill_count") != 0:
        errors.append("strict cleanup reports retarget/backfill")
    ledger = _as_dict(summary.get("row_ledger"))
    try:
        source_rows = int(ledger["source_rows"])
        pre_rows = int(ledger["pre_split_rows"])
        pre_drop = int(ledger["pre_split_rows_dropped"])
        chunk_rows = int(ledger["chunk_rows_before_cleanup"])
        post_drop = int(ledger["post_split_rows_dropped"])
        output_rows = int(ledger["output_rows"])
        if source_rows - pre_rows != pre_drop:
            errors.append("pre-split row ledger arithmetic mismatch")
        if chunk_rows != pre_rows:
            errors.append("chunking changed row membership")
        if chunk_rows - output_rows != post_drop:
            errors.append("post-split row ledger arithmetic mismatch")
        if output_rows != scans["combined"].rows:
            errors.append("summary output row count mismatch")
        if ledger.get("balanced") is not True:
            errors.append("summary row ledger is not declared balanced")
    except (KeyError, TypeError, ValueError):
        errors.append("summary row ledger is incomplete")

    builder = _as_dict(summary.get("builder"))
    if not valid_sha256(builder.get("sha256")):
        errors.append("builder SHA-256 missing or invalid")
    if not isinstance(builder.get("command"), list) or not builder.get("command"):
        errors.append("builder command missing")
    if not parse_utc(builder.get("started_at_utc")) or not parse_utc(builder.get("finished_at_utc")):
        errors.append("builder UTC timestamps missing or invalid")
    source_files = _as_dict(_as_dict(summary.get("source")).get("files"))
    for role in ("interaction", "item_metadata"):
        record = _as_dict(source_files.get(role))
        if not valid_sha256(record.get("sha256")) or int(record.get("bytes", -1)) < 0:
            errors.append(f"source {role} hash/size record invalid")
            continue
        source_path = Path(str(record.get("path", "")))
        if not source_path.is_file():
            errors.append(f"source {role} path does not exist")
        elif (
            source_path.stat().st_size != record.get("bytes")
            or sha256_file(source_path) != record.get("sha256")
        ):
            errors.append(f"source {role} current hash/size differs from manifest")
    recorded_files = _as_dict(summary.get("files"))
    for path in required_paths.values():
        if path == required_paths["summary"]:
            continue
        record = _as_dict(recorded_files.get(path.name))
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            errors.append(f"recorded file hash mismatch: {path.name}")

    lineage, lineage_errors = validate_lineage(
        required_paths["lineage"], scans={split: scans[split] for split in SPLITS}, summary=summary
    )
    errors.extend(lineage_errors)
    if summary.get("source_kind") == "kuairec_adaptive":
        projection = _as_dict(membership.get("historical_v4_projection"))
        projection_actual = {
            "rows": projection.get("rows"),
            "sessions": projection.get("sessions"),
            "observed_items": projection.get("observed_items"),
        }
        projection_expected = {
            "rows": 3_862_479,
            "sessions": 209_312,
            "observed_items": 8_572,
        }
        if projection_actual != projection_expected:
            errors.append(
                f"historical-v4 evidence mismatch: {projection_actual} != {projection_expected}"
            )
        if projection.get("evidence_only_not_runtime") is not True:
            errors.append("historical-v4 projection is not marked evidence-only")
    output = _as_dict(summary.get("output"))
    if _as_dict(output.get("rows")) != {split: scans[split].rows for split in SPLITS}:
        errors.append("summary split row counts mismatch")
    if _as_dict(output.get("sessions")) != {
        split: len(scans[split].sessions) for split in SPLITS
    }:
        errors.append("summary split session counts mismatch")

    report = {
        "schema_version": 1,
        "validator_contract": "camera-ready-core5-basic-independent-validator-v1",
        "dataset": args.dataset,
        "ok": not errors,
        "errors": errors[:200],
        "error_count": len(errors),
        "expected_pre_split": expected_pre,
        "actual_pre_split": actual_pre,
        "scans": {
            name: {
                "rows": scan.rows,
                "sessions": len(scan.sessions),
                "items": len(scan.item_counts),
                "users": len(scan.users),
                "minimum_session_length": min(scan.session_lengths.values(), default=0),
                "maximum_session_length": max(scan.session_lengths.values(), default=0),
            }
            for name, scan in scans.items()
        },
        "train_minimum_item_occurrence_frequency": train_min_frequency,
        "valid_test_unseen_item_count": len(unseen_items),
        "valid_test_unseen_target_count": len(unseen_targets),
        "item_rows": len(item_ids),
        "item_duplicates": item_duplicates,
        "split_session_overlap": overlap,
        "combined_exact_ordered_split_union": union_error is None,
        "lineage": lineage,
        "summary_sha256": sha256_file(required_paths["summary"]),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--timestamp-unit", choices=("s", "ms"), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-sessions", type=int, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--expected-mean-length", type=float)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
