#!/usr/bin/env python3
"""Build non-overwriting six-dataset core5 split releases.

This production builder implements three explicit membership modes:

* ``preserve_session`` for Beauty, Foursquare, MovieLens-1M, and Retail Rocket;
* ``kuairec_adaptive`` for raw KuaiRec with the stabilized-user-mean watch rule;
* ``recovered_resessionize`` for the explicitly approximate LastFM source.

Natural parent sessions are split chronologically before balanced max-50
chunking.  Training is re-cored, the training vocabulary is frozen, and
validation/test sessions are cleaned without retargeting, backfill, resplit,
or parent migration.  A build is written below a hidden staging directory,
independently validated, and only then atomically published.  Existing final
directories are never replaced.
"""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


BASIC_CONTRACT = "sessionized-core5-strict-split-v1"
CORE5_PARENT_CONTRACT = "core5-parent-strict-v1"
SPLITS = ("train", "valid", "test")
BASE_HEADER = (
    "session_id:token",
    "item_id:token",
    "timestamp:float",
    "user_id:token",
)


def path_lexists(path: Path) -> bool:
    """Return true for every directory entry, including broken links."""

    return os.path.lexists(os.fspath(path))


def atomic_publish_directory(source: Path, target: Path) -> None:
    """Atomically publish a directory without ever replacing a target entry."""

    source = source.resolve(strict=True)
    target_parent = target.parent.resolve(strict=True)
    target = target_parent / target.name
    if source.stat().st_dev != target_parent.stat().st_dev:
        raise RuntimeError("publication source and target are not on the same filesystem")
    if path_lexists(target):
        raise FileExistsError(f"refusing to replace publication target: {target}")
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable; failing closed")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(f"refusing to replace publication target: {target}")
            raise OSError(error, os.strerror(error), os.fspath(target))
    else:
        os.rename(source, target)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    source_kind: str
    timestamp_unit: str
    expected_rows: int
    expected_sessions: int
    expected_items: int
    expected_mean_length: float | None = None
    release_source: str | None = None
    raw_interactions: str | None = None
    raw_item_metadata: str | None = None
    source_inter: str | None = None
    source_item: str | None = None


SPECS: dict[str, DatasetSpec] = {
    "beauty_core5_v1": DatasetSpec(
        "beauty_core5_v1", "preserve_session", "ms", 33_488, 4_243, 3_625,
        release_source="Datasets/release/beauty",
    ),
    "foursquare_core5_v1": DatasetSpec(
        "foursquare_core5_v1", "preserve_session", "s", 58_182, 5_994, 5_097,
        release_source="Datasets/release/foursquare",
    ),
    "movielens1m_core5_v1": DatasetSpec(
        "movielens1m_core5_v1", "preserve_session", "s", 574_018, 14_187, 3_280,
        release_source="Datasets/release/movielens1m",
    ),
    "retail_rocket_core5_v1": DatasetSpec(
        "retail_rocket_core5_v1", "preserve_session", "ms", 431_107, 48_800, 34_501,
        release_source="Datasets/release/retail_rocket",
    ),
    "kuairec_adaptive_core5_v1": DatasetSpec(
        "kuairec_adaptive_core5_v1", "kuairec_adaptive", "s",
        4_312_463, 209_312, 8_966,
        raw_interactions=(
            "Datasets/recovery/full_sources_20260812/raw/KuaiRec/data/big_matrix.csv"
        ),
        raw_item_metadata=(
            "Datasets/recovery/full_sources_20260812/raw/KuaiRec/data/item_categories.csv"
        ),
    ),
    "lastfm_recovered_core5_v1": DatasetSpec(
        "lastfm_recovered_core5_v1", "recovered_resessionize", "ms",
        15_756_683, 603_048, 546_008, 26.128,
        source_inter=(
            "Datasets/recovery/full_sources_20260812/processed/basic/lastfm/lastfm.inter"
        ),
        source_item=(
            "Datasets/recovery/full_sources_20260812/processed/basic/lastfm/lastfm.item"
        ),
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def plain(column: str) -> str:
    return str(column).lstrip("\ufeff").split(":", 1)[0]


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def canonical_timestamp(value: str) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite timestamp: {value!r}")
    integer = int(number)
    return integer if float(integer) == number else number


def format_timestamp(value: int | float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".17g")


@dataclass(frozen=True, slots=True)
class Event:
    item: str
    timestamp: int | float
    watch_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class Session:
    sid: str
    parent_id: str
    user: str
    events: tuple[Event, ...]
    first_source_row: int
    chunk_index: int = 0
    chunk_count: int = 1
    original_target_item: str | None = None
    original_target_timestamp: int | float | None = None

    @property
    def start(self) -> int | float:
        return self.events[0].timestamp

    @property
    def end(self) -> int | float:
        return self.events[-1].timestamp


def session_sort_key(session: Session) -> tuple[float, int, str]:
    return (float(session.start), int(session.first_source_row), str(session.parent_id))


def core_stats(sessions: Sequence[Session]) -> dict[str, int | float]:
    item_counts: Counter[str] = Counter()
    rows = 0
    for session in sessions:
        rows += len(session.events)
        item_counts.update(event.item for event in session.events)
    lengths = [len(session.events) for session in sessions]
    return {
        "rows": rows,
        "sessions": len(sessions),
        "users": len({session.user for session in sessions}),
        "items": len(item_counts),
        "mean_session_length": rows / len(sessions) if sessions else 0.0,
        "minimum_session_length": min(lengths, default=0),
        "maximum_session_length": max(lengths, default=0),
        "minimum_item_interaction_frequency": min(item_counts.values(), default=0),
    }


def iterative_core(
    sessions: Sequence[Session],
    *,
    minimum_session_length: int = 5,
    minimum_item_frequency: int = 3,
) -> tuple[list[Session], list[dict[str, int]]]:
    """Close an occurrence-frequency session/item core with a complete ledger."""

    work = list(sessions)
    history: list[dict[str, int]] = []
    iteration = 0
    while True:
        iteration += 1
        before_rows = sum(len(session.events) for session in work)
        before_sessions = len(work)

        length_kept: list[Session] = []
        short_pre_rows = 0
        short_pre_sessions = 0
        for session in work:
            if len(session.events) < minimum_session_length:
                short_pre_sessions += 1
                short_pre_rows += len(session.events)
            else:
                length_kept.append(session)

        item_counts: Counter[str] = Counter(
            event.item for session in length_kept for event in session.events
        )
        rare = {
            item for item, count in item_counts.items() if count < minimum_item_frequency
        }
        rare_rows = 0
        after_rare: list[Session] = []
        for session in length_kept:
            if rare:
                kept = tuple(event for event in session.events if event.item not in rare)
                rare_rows += len(session.events) - len(kept)
                after_rare.append(replace(session, events=kept))
            else:
                after_rare.append(session)

        next_work: list[Session] = []
        short_post_rows = 0
        short_post_sessions = 0
        for session in after_rare:
            if len(session.events) < minimum_session_length:
                short_post_sessions += 1
                short_post_rows += len(session.events)
            else:
                next_work.append(session)

        after_rows = sum(len(session.events) for session in next_work)
        record = {
            "iteration": iteration,
            "rows_before": before_rows,
            "rows_after": after_rows,
            "sessions_before": before_sessions,
            "sessions_after": len(next_work),
            "short_pre_sessions_removed": short_pre_sessions,
            "short_pre_rows_removed": short_pre_rows,
            "rare_items_removed": len(rare),
            "rare_item_rows_removed": rare_rows,
            "short_post_sessions_removed": short_post_sessions,
            "short_post_rows_removed": short_post_rows,
        }
        if (
            before_rows - after_rows
            != short_pre_rows + rare_rows + short_post_rows
        ):
            raise AssertionError(f"core ledger does not balance: {record}")
        history.append(record)
        work = next_work
        if after_rows == before_rows and len(work) == before_sessions:
            break
        if not work:
            raise ValueError("core filtering removed every session")
    return work, history


def _typed_names(reader: csv.DictReader) -> dict[str, str]:
    return {plain(name): name for name in (reader.fieldnames or [])}


def load_preserved_sessions(path: Path) -> tuple[list[Session], dict[str, object]]:
    """Load only the base four columns and preserve source session membership."""

    events_by_sid: dict[str, list[Event]] = {}
    user_by_sid: dict[str, str] = {}
    first_row_by_sid: dict[str, int] = {}
    last_ts_by_sid: dict[str, float] = {}
    source_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = _typed_names(reader)
        required = {"session_id", "item_id", "timestamp", "user_id"}
        if not required.issubset(names):
            raise ValueError(f"missing base columns in {path}: {sorted(required - set(names))}")
        for row_index, row in enumerate(reader):
            sid = sys.intern(str(row[names["session_id"]]))
            user = sys.intern(str(row[names["user_id"]]))
            item = sys.intern(str(row[names["item_id"]]))
            timestamp = canonical_timestamp(str(row[names["timestamp"]]))
            if sid in user_by_sid and user_by_sid[sid] != user:
                raise ValueError(f"source session {sid!r} contains multiple users")
            previous = last_ts_by_sid.get(sid)
            if previous is not None and float(timestamp) < previous:
                raise ValueError(f"timestamp regression inside preserved source session {sid!r}")
            if sid not in events_by_sid:
                events_by_sid[sid] = []
                user_by_sid[sid] = user
                first_row_by_sid[sid] = row_index
            events_by_sid[sid].append(Event(item=item, timestamp=timestamp))
            last_ts_by_sid[sid] = float(timestamp)
            source_rows += 1
    sessions = [
        Session(
            sid=sid,
            parent_id=sid,
            user=user_by_sid[sid],
            events=tuple(events),
            first_source_row=first_row_by_sid[sid],
        )
        for sid, events in events_by_sid.items()
    ]
    return sessions, {
        "rows": source_rows,
        "sessions": len(sessions),
        "users": len(set(user_by_sid.values())),
        "source_session_ids_preserved": True,
        "old_feature_columns_ignored": True,
    }


def _sessionize_sorted_user_events(
    by_user: Mapping[str, list[tuple[int | float, str, float | None, int]]],
    *,
    inactivity_gap_seconds: float,
    timestamp_divisor: float,
) -> tuple[list[Session], int]:
    sessions: list[Session] = []
    boundaries = 0
    for user, raw_events in by_user.items():
        raw_events.sort(key=lambda event: float(event[0]))  # stable for timestamp ties
        session_index = 0
        current: list[Event] = []
        first_source_row = -1
        previous: int | float | None = None

        def finish() -> None:
            nonlocal current, session_index, first_source_row
            if not current:
                return
            parent = f"{user}_s{session_index}"
            sessions.append(
                Session(
                    sid=parent,
                    parent_id=parent,
                    user=user,
                    events=tuple(current),
                    first_source_row=first_source_row,
                )
            )
            session_index += 1
            current = []
            first_source_row = -1

        for timestamp, item, watch_ratio, source_row in raw_events:
            if previous is not None:
                gap = (float(timestamp) - float(previous)) / timestamp_divisor
                if gap < 0:
                    raise AssertionError("per-user sort failed")
                if gap > inactivity_gap_seconds:
                    finish()
                    boundaries += 1
            if not current:
                first_source_row = source_row
            current.append(Event(item=item, timestamp=timestamp, watch_ratio=watch_ratio))
            previous = timestamp
        finish()
    return sessions, boundaries


def load_recovered_and_resessionize(
    path: Path,
    *,
    timestamp_divisor: float,
    inactivity_gap_seconds: float,
) -> tuple[list[Session], dict[str, object]]:
    by_user: dict[str, list[tuple[int | float, str, float | None, int]]] = defaultdict(list)
    source_sessions: set[str] = set()
    source_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = _typed_names(reader)
        required = {"session_id", "item_id", "timestamp", "user_id"}
        if not required.issubset(names):
            raise ValueError(f"missing base columns in {path}: {sorted(required - set(names))}")
        for row_index, row in enumerate(reader):
            user = sys.intern(str(row[names["user_id"]]))
            item = sys.intern(str(row[names["item_id"]]))
            timestamp = canonical_timestamp(str(row[names["timestamp"]]))
            by_user[user].append((timestamp, item, None, row_index))
            source_sessions.add(str(row[names["session_id"]]))
            source_rows += 1
    sessions, boundaries = _sessionize_sorted_user_events(
        by_user,
        inactivity_gap_seconds=inactivity_gap_seconds,
        timestamp_divisor=timestamp_divisor,
    )
    return sessions, {
        "rows": source_rows,
        "users": len(by_user),
        "source_session_ids_ignored": len(source_sessions),
        "resessionized_sessions": len(sessions),
        "gap_split_boundaries": boundaries,
        "source_is_raw_exact": False,
    }


def load_kuairec_raw(
    path: Path, *, inactivity_gap_seconds: float
) -> tuple[list[Session], dict[str, object]]:
    by_user: dict[str, list[tuple[int | float, str, float | None, int]]] = defaultdict(list)
    source_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "video_id", "timestamp", "watch_ratio"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"missing KuaiRec raw columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_index, row in enumerate(reader):
            user = sys.intern(str(int(row["user_id"])))
            item = sys.intern(str(int(row["video_id"])))
            timestamp = canonical_timestamp(row["timestamp"])
            watch_ratio = float(row["watch_ratio"])
            if not math.isfinite(watch_ratio):
                raise ValueError(f"non-finite watch_ratio at raw row {row_index + 2}")
            by_user[user].append((timestamp, item, watch_ratio, row_index))
            source_rows += 1
    sessions, boundaries = _sessionize_sorted_user_events(
        by_user,
        inactivity_gap_seconds=inactivity_gap_seconds,
        timestamp_divisor=1.0,
    )
    return sessions, {
        "rows": source_rows,
        "users": len(by_user),
        "natural_sessions": len(sessions),
        "gap_split_boundaries": boundaries,
    }


def adaptive_watch_filter(
    sessions: Sequence[Session],
) -> tuple[list[Session], dict[str, int | float]]:
    sums: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for session in sessions:
        for event in session.events:
            if event.watch_ratio is None:
                raise ValueError("adaptive watch filter received an event without watch_ratio")
            sums[session.user] += event.watch_ratio
            counts[session.user] += 1
    means = {user: sums[user] / counts[user] for user in sums}
    filtered: list[Session] = []
    kept_rows = 0
    for session in sessions:
        mean = means[session.user]
        kept = tuple(
            event
            for event in session.events
            if event.watch_ratio is not None
            and (event.watch_ratio > 1.0 or event.watch_ratio > mean)
        )
        kept_rows += len(kept)
        if kept:
            filtered.append(replace(session, events=kept))
    before_rows = sum(len(session.events) for session in sessions)
    return filtered, {
        "users": len(means),
        "rows_before": before_rows,
        "rows_after": kept_rows,
        "rows_removed": before_rows - kept_rows,
        "sessions_before": len(sessions),
        "sessions_after_nonempty": len(filtered),
        "empty_sessions_removed": len(sessions) - len(filtered),
        "rule_watch_ratio_gt_1_or_gt_stabilized_user_mean": True,
    }


def temporal_parent_split(
    sessions: Sequence[Session], train_ratio: float = 0.7
) -> dict[str, list[Session]]:
    ordered = sorted(sessions, key=session_sort_key)
    if len(ordered) < 3:
        raise ValueError("at least three parent sessions are required")
    train_count = int(math.floor(len(ordered) * train_ratio))
    train_count = min(max(1, train_count), len(ordered) - 2)
    tail = len(ordered) - train_count
    valid_count = tail // 2
    return {
        "train": ordered[:train_count],
        "valid": ordered[train_count : train_count + valid_count],
        "test": ordered[train_count + valid_count :],
    }


def historical_v4_projection(sessions: Sequence[Session]) -> dict[str, object]:
    """Reproduce the historical KuaiRec v4 tail projection as evidence only.

    The earliest 70% of parents are train.  For every tail parent, the final
    target is retained even when unseen, while train-unseen non-target rows are
    removed.  This intentionally differs from the strict runtime tail policy.
    """

    parents = temporal_parent_split(sessions, train_ratio=0.7)
    train_items = {
        event.item for session in parents["train"] for event in session.events
    }
    observed_items: set[str] = set()
    rows = 0
    tail_non_target_unseen_removed = 0
    for session in parents["train"]:
        rows += len(session.events)
        observed_items.update(event.item for event in session.events)
    for split in ("valid", "test"):
        for session in parents[split]:
            target_index = len(session.events) - 1
            for index, event in enumerate(session.events):
                if index != target_index and event.item not in train_items:
                    tail_non_target_unseen_removed += 1
                    continue
                rows += 1
                observed_items.add(event.item)
    sessions_count = sum(len(parents[split]) for split in SPLITS)
    return {
        "evidence_only_not_runtime": True,
        "policy": "train unchanged; tail original target retained; other train-unseen rows removed",
        "rows": rows,
        "sessions": sessions_count,
        "observed_items": len(observed_items),
        "mean_session_length": rows / sessions_count,
        "train_rows": sum(len(session.events) for session in parents["train"]),
        "train_items": len(train_items),
        "tail_non_target_unseen_rows_removed": tail_non_target_unseen_removed,
    }


def balanced_chunks(
    parents: Sequence[Session],
    *,
    maximum_session_length: int,
    split: str,
) -> list[Session]:
    result: list[Session] = []
    for parent in parents:
        length = len(parent.events)
        chunk_count = max(1, math.ceil(length / maximum_session_length))
        base, remainder = divmod(length, chunk_count)
        offset = 0
        for chunk_index in range(chunk_count):
            size = base + (1 if chunk_index < remainder else 0)
            events = parent.events[offset : offset + size]
            offset += size
            target = events[-1]
            result.append(
                Session(
                    sid=f"cr5_{split[:2]}_{len(result):012d}",
                    parent_id=parent.parent_id,
                    user=parent.user,
                    events=events,
                    first_source_row=parent.first_source_row + chunk_index,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    original_target_item=target.item,
                    original_target_timestamp=target.timestamp,
                )
            )
    return result


def strict_train_seen_tail(
    train: Sequence[Session],
    tail: Sequence[Session],
    *,
    minimum_session_length: int = 5,
) -> tuple[list[Session], dict[str, int], dict[str, dict[str, object]]]:
    train_items = {event.item for session in train for event in session.events}
    kept_sessions: list[Session] = []
    audit: dict[str, dict[str, object]] = {}
    totals: Counter[str] = Counter()
    for session in tail:
        original_rows = len(session.events)
        original_target = session.events[-1]
        if session.original_target_item != original_target.item:
            raise AssertionError("chunk target metadata differs before tail cleanup")
        if original_target.item not in train_items:
            totals["cold_target_sessions_dropped"] += 1
            totals["cold_target_rows_dropped"] += original_rows
            audit[session.sid] = {
                "status": "dropped_cold_target",
                "final_rows": 0,
                "rows_removed": original_rows,
            }
            continue
        kept = tuple(event for event in session.events if event.item in train_items)
        cold_rows = original_rows - len(kept)
        totals["cold_context_rows_dropped"] += cold_rows
        if len(kept) < minimum_session_length:
            totals["post_cold_sessions_below_minimum_dropped"] += 1
            totals["post_cold_retained_rows_in_dropped_sessions"] += len(kept)
            audit[session.sid] = {
                "status": "dropped_below_minimum_after_unseen_context",
                "final_rows": 0,
                "rows_removed": original_rows,
            }
            continue
        if kept[-1] is not original_target:
            raise AssertionError("tail cleanup retargeted a surviving session")
        final = session if kept is session.events else replace(session, events=kept)
        kept_sessions.append(final)
        audit[session.sid] = {
            "status": "kept",
            "final_rows": len(kept),
            "rows_removed": cold_rows,
        }
    totals["train_item_count"] = len(train_items)
    totals["input_sessions"] = len(tail)
    totals["input_rows"] = sum(len(session.events) for session in tail)
    totals["output_sessions"] = len(kept_sessions)
    totals["output_rows"] = sum(len(session.events) for session in kept_sessions)
    if totals["input_rows"] - totals["output_rows"] != sum(
        int(record["rows_removed"]) for record in audit.values()
    ):
        raise AssertionError("tail cleanup ledger does not balance")
    return kept_sessions, dict(totals), audit


def _load_item_table(path: Path) -> tuple[list[str], dict[str, list[str]], dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = list(reader.fieldnames or [])
        names = {plain(name): name for name in header}
        if "item_id" not in names:
            raise ValueError(f"item metadata lacks item_id: {path}")
        rows: dict[str, list[str]] = {}
        for row in reader:
            item = str(row[names["item_id"]])
            if item in rows:
                raise ValueError(f"duplicate item metadata: {item}")
            rows[item] = [str(row.get(column, "")) for column in header]
    return header, rows, {"source_metadata_rows": len(rows), "fallback_rows": 0}


def _load_kuairec_categories(
    path: Path,
) -> tuple[list[str], dict[str, list[str]], dict[str, object]]:
    rows: dict[str, list[str]] = {}
    malformed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                item = str(int(row["video_id"]))
                feat = ast.literal_eval(row["feat"])
                category = str(feat[0])
            except (KeyError, ValueError, TypeError, SyntaxError, IndexError):
                malformed += 1
                continue
            if item in rows:
                raise ValueError(f"duplicate KuaiRec item metadata: {item}")
            rows[item] = [item, category]
    return ["item_id:token", "category:token"], rows, {
        "source_metadata_rows": len(rows),
        "malformed_rows": malformed,
        "caption_category_source_present": False,
        "declared_fallback": "first item_categories.csv feat entry",
        "fallback_rows": 0,
    }


def _iter_sessions(splits: Mapping[str, Sequence[Session]]) -> Iterator[Session]:
    for split in SPLITS:
        yield from splits[split]


def _write_release(
    *,
    directory: Path,
    dataset: str,
    splits: Mapping[str, Sequence[Session]],
    all_chunks: Mapping[str, Sequence[Session]],
    tail_audits: Mapping[str, Mapping[str, Mapping[str, object]]],
    item_header: Sequence[str],
    item_rows: Mapping[str, Sequence[str]],
    item_metadata_stats: dict[str, object],
    summary: dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    handles: dict[str, object] = {}
    writers: dict[str, csv.writer] = {}
    try:
        handles["combined"] = (directory / f"{dataset}.inter").open(
            "x", encoding="utf-8", newline=""
        )
        writers["combined"] = csv.writer(handles["combined"], delimiter="\t")
        writers["combined"].writerow(BASE_HEADER)
        for split in SPLITS:
            handles[split] = (directory / f"{dataset}.{split}.inter").open(
                "x", encoding="utf-8", newline=""
            )
            writers[split] = csv.writer(handles[split], delimiter="\t")
            writers[split].writerow(BASE_HEADER)
        for split in SPLITS:
            for session in splits[split]:
                for event in session.events:
                    row = (
                        session.sid,
                        event.item,
                        format_timestamp(event.timestamp),
                        session.user,
                    )
                    writers["combined"].writerow(row)
                    writers[split].writerow(row)
        for handle in handles.values():
            handle.flush()  # type: ignore[attr-defined]
            os.fsync(handle.fileno())  # type: ignore[attr-defined]
    finally:
        for handle in handles.values():
            handle.close()  # type: ignore[attr-defined]

    train_items = sorted(
        {event.item for session in splits["train"] for event in session.events}
    )
    missing = [item for item in train_items if item not in item_rows]
    fallback_rows = 0
    item_path = directory / f"{dataset}.item"
    with item_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(item_header)
        for item in train_items:
            if item in item_rows:
                writer.writerow(item_rows[item])
            elif list(item_header) == ["item_id:token", "category:token"]:
                writer.writerow([item, "0"])
                fallback_rows += 1
            else:
                raise ValueError(f"{len(missing)} train items lack source metadata")
        handle.flush()
        os.fsync(handle.fileno())
    item_metadata_stats["fallback_rows"] = fallback_rows
    item_metadata_stats["train_vocabulary_rows"] = len(train_items)

    final_by_sid = {
        session.sid: session for session in _iter_sessions(splits)
    }
    parent_meta: dict[str, dict[str, int | float]] = {}
    for split in SPLITS:
        for chunk in all_chunks[split]:
            record = parent_meta.setdefault(
                chunk.parent_id,
                {
                    "start": chunk.events[0].timestamp,
                    "end": chunk.events[-1].timestamp,
                    "first_source_row": chunk.first_source_row - chunk.chunk_index,
                },
            )
            record["start"] = min(float(record["start"]), float(chunk.events[0].timestamp))
            record["end"] = max(float(record["end"]), float(chunk.events[-1].timestamp))
            record["first_source_row"] = min(
                int(record["first_source_row"]),
                chunk.first_source_row - chunk.chunk_index,
            )
    lineage_path = directory / f"{dataset}.parent_lineage.tsv"
    with lineage_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "chunk_session_id", "parent_session_id", "split", "user_id",
                "parent_start_timestamp", "parent_end_timestamp", "parent_first_source_row",
                "chunk_index", "chunk_count", "original_rows", "final_rows",
                "rows_removed", "status", "original_target_item",
                "original_target_timestamp", "final_target_item",
                "final_target_timestamp",
            ]
        )
        for split in SPLITS:
            for chunk in all_chunks[split]:
                final = final_by_sid.get(chunk.sid)
                if final is not None:
                    status = "kept"
                    final_rows = len(final.events)
                    rows_removed = len(chunk.events) - final_rows
                    final_target = final.events[-1]
                elif split == "train":
                    status = "dropped_train_core"
                    final_rows = 0
                    rows_removed = len(chunk.events)
                    final_target = None
                else:
                    record = tail_audits[split][chunk.sid]
                    status = str(record["status"])
                    final_rows = int(record["final_rows"])
                    rows_removed = int(record["rows_removed"])
                    final_target = None
                writer.writerow(
                    [
                        chunk.sid, chunk.parent_id, split, chunk.user,
                        format_timestamp(parent_meta[chunk.parent_id]["start"]),
                        format_timestamp(parent_meta[chunk.parent_id]["end"]),
                        int(parent_meta[chunk.parent_id]["first_source_row"]),
                        chunk.chunk_index, chunk.chunk_count, len(chunk.events),
                        final_rows, rows_removed, status,
                        chunk.original_target_item,
                        format_timestamp(chunk.original_target_timestamp)  # type: ignore[arg-type]
                        if chunk.original_target_timestamp is not None else "",
                        final_target.item if final_target else "",
                        format_timestamp(final_target.timestamp) if final_target else "",
                    ]
                )
        handle.flush()
        os.fsync(handle.fileno())

    summary["item_metadata"] = item_metadata_stats
    file_records = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    summary["files"] = file_records
    summary_path = directory / f"{dataset}.basic_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _core_drop_rows(history: Sequence[Mapping[str, int]]) -> int:
    return sum(
        int(record["short_pre_rows_removed"])
        + int(record["rare_item_rows_removed"])
        + int(record["short_post_rows_removed"])
        for record in history
    )


def build_staged_dataset(
    *,
    spec: DatasetSpec,
    repo_root: Path,
    staging_root: Path,
    started_at: str,
) -> Path:
    minimum_session_length = 5
    minimum_item_frequency = 3
    maximum_session_length = 50
    gap_seconds = 1800.0
    divisor = 1.0 if spec.timestamp_unit == "s" else 1000.0

    source_files: dict[str, Path] = {}
    if spec.source_kind == "preserve_session":
        assert spec.release_source
        source_dir = repo_root / spec.release_source
        source_name = source_dir.name
        source_inter = source_dir / f"{source_name}.inter"
        source_item = source_dir / f"{source_name}.item"
        source_files = {"interaction": source_inter, "item_metadata": source_item}
        sessions, source_stats = load_preserved_sessions(source_inter)
        item_header, item_rows, item_stats = _load_item_table(source_item)
    elif spec.source_kind == "recovered_resessionize":
        assert spec.source_inter and spec.source_item
        source_inter = repo_root / spec.source_inter
        source_item = repo_root / spec.source_item
        source_files = {"interaction": source_inter, "item_metadata": source_item}
        sessions, source_stats = load_recovered_and_resessionize(
            source_inter,
            timestamp_divisor=divisor,
            inactivity_gap_seconds=gap_seconds,
        )
        item_header, item_rows, item_stats = _load_item_table(source_item)
    elif spec.source_kind == "kuairec_adaptive":
        assert spec.raw_interactions and spec.raw_item_metadata
        source_inter = repo_root / spec.raw_interactions
        source_item = repo_root / spec.raw_item_metadata
        source_files = {"interaction": source_inter, "item_metadata": source_item}
        sessions, source_stats = load_kuairec_raw(
            source_inter, inactivity_gap_seconds=gap_seconds
        )
        item_header, item_rows, item_stats = _load_kuairec_categories(source_item)
    else:
        raise ValueError(f"unsupported source kind: {spec.source_kind}")
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    initial_core, initial_history = iterative_core(
        sessions,
        minimum_session_length=minimum_session_length,
        minimum_item_frequency=minimum_item_frequency,
    )
    after_initial = core_stats(initial_core)
    watch_summary: dict[str, object] | None = None
    final_history: list[dict[str, int]] = []
    if spec.source_kind == "kuairec_adaptive":
        watch_filtered, watch_summary = adaptive_watch_filter(initial_core)
        pre_split, final_history = iterative_core(
            watch_filtered,
            minimum_session_length=minimum_session_length,
            minimum_item_frequency=minimum_item_frequency,
        )
    else:
        pre_split = initial_core
    pre_split_stats = core_stats(pre_split)
    expected = {
        "rows": spec.expected_rows,
        "sessions": spec.expected_sessions,
        "items": spec.expected_items,
    }
    actual = {key: int(pre_split_stats[key]) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"{spec.name} pre-split identity gate failed: actual={actual}, expected={expected}"
        )
    if spec.expected_mean_length is not None and not math.isclose(
        float(pre_split_stats["mean_session_length"]),
        spec.expected_mean_length,
        rel_tol=0.0,
        abs_tol=0.001,
    ):
        raise RuntimeError(
            f"{spec.name} pre-split mean gate failed: "
            f"{pre_split_stats['mean_session_length']} != {spec.expected_mean_length}"
        )

    projection: dict[str, object] | None = None
    if spec.source_kind == "kuairec_adaptive":
        projection = historical_v4_projection(pre_split)
        projection_expected = {
            "rows": 3_862_479,
            "sessions": 209_312,
            "observed_items": 8_572,
        }
        projection_actual = {
            key: int(projection[key]) for key in projection_expected
        }
        if projection_actual != projection_expected:
            raise RuntimeError(
                "KuaiRec historical-v4 projection witness failed: "
                f"actual={projection_actual}, expected={projection_expected}"
            )
        projection["expected_counts"] = projection_expected
        projection["expected_counts_pass"] = True
        projection["handoff_mean_18_452_is_approximate"] = True

    parents = temporal_parent_split(pre_split, train_ratio=0.7)
    all_chunks = {
        split: balanced_chunks(
            parents[split], maximum_session_length=maximum_session_length, split=split
        )
        for split in SPLITS
    }
    train, train_history = iterative_core(
        all_chunks["train"],
        minimum_session_length=minimum_session_length,
        minimum_item_frequency=minimum_item_frequency,
    )
    valid, valid_cleanup, valid_audit = strict_train_seen_tail(train, all_chunks["valid"])
    test, test_cleanup, test_audit = strict_train_seen_tail(train, all_chunks["test"])
    splits = {"train": train, "valid": valid, "test": test}
    tail_audits = {"valid": valid_audit, "test": test_audit}

    output_stats = {
        "rows": {
            split: sum(len(session.events) for session in splits[split]) for split in SPLITS
        },
        "sessions": {split: len(splits[split]) for split in SPLITS},
    }
    output_stats["total_rows"] = sum(output_stats["rows"].values())  # type: ignore[union-attr]
    output_stats["total_sessions"] = sum(output_stats["sessions"].values())  # type: ignore[union-attr]

    source_rows = int(source_stats["rows"])
    pre_split_rows = int(pre_split_stats["rows"])
    chunk_rows = sum(
        len(session.events) for split in SPLITS for session in all_chunks[split]
    )
    output_rows = int(output_stats["total_rows"])
    pre_split_drops = source_rows - pre_split_rows
    post_split_drops = chunk_rows - output_rows
    expected_pre_drops = _core_drop_rows(initial_history) + _core_drop_rows(final_history)
    if watch_summary is not None:
        expected_pre_drops += int(watch_summary["rows_removed"])
    if pre_split_drops != expected_pre_drops:
        raise AssertionError("pre-split row ledger does not balance")
    expected_post_drops = _core_drop_rows(train_history)
    expected_post_drops += int(valid_cleanup["input_rows"]) - int(valid_cleanup["output_rows"])
    expected_post_drops += int(test_cleanup["input_rows"]) - int(test_cleanup["output_rows"])
    if post_split_drops != expected_post_drops:
        raise AssertionError("post-split row ledger does not balance")

    train_item_counts = Counter(
        event.item for session in train for event in session.events
    )
    train_items = set(train_item_counts)
    tail_items = {
        event.item
        for split in ("valid", "test")
        for session in splits[split]
        for event in session.events
    }
    if min(train_item_counts.values(), default=0) < minimum_item_frequency:
        raise AssertionError("final train core violates item occurrence >=3")
    if not tail_items.issubset(train_items):
        raise AssertionError("tail contains train-unseen items")
    if any(
        len(session.events) < minimum_session_length
        or len(session.events) > maximum_session_length
        for session in _iter_sessions(splits)
    ):
        raise AssertionError("final session length is outside [5,50]")
    final_lengths = [len(session.events) for session in _iter_sessions(splits)]
    final_minimum_session_length = min(final_lengths, default=0)
    final_maximum_session_length = max(final_lengths, default=0)

    source_manifest = {
        role: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in source_files.items()
    }
    builder_path = Path(__file__).resolve()
    summary: dict[str, object] = {
        "schema_version": 2,
        "contract": BASIC_CONTRACT,
        "core5_contract": CORE5_PARENT_CONTRACT,
        "status": "complete",
        "target_dataset": spec.name,
        "source_kind": spec.source_kind,
        "source_identity_note": (
            "recovered full basic approximation; source split/session ignored; not raw exact"
            if spec.source_kind == "recovered_resessionize"
            else "existing session artifact membership; old feature columns ignored"
            if spec.source_kind == "preserve_session"
            else "raw KuaiRec adaptive-watch membership"
        ),
        "source": {"files": source_manifest, "stats": source_stats},
        "parameters": {
            "timestamp_unit": spec.timestamp_unit,
            "inactivity_gap_seconds": gap_seconds,
            "minimum_session_length": minimum_session_length,
            "maximum_session_length": maximum_session_length,
            "minimum_item_interaction_occurrence_frequency": minimum_item_frequency,
            "maximum_chunk_length": maximum_session_length,
            "parent_split": "chronological 70/15/15 before within-parent chunking",
            "train_recore": True,
            "tail_policy": (
                "drop train-unseen target session; remove train-unseen context; drop below 5"
            ),
            "retarget": False,
            "backfill": False,
            "resplit": False,
        },
        "membership": {
            "initial_core_history": initial_history,
            "after_initial_core": after_initial,
            "adaptive_watch_filter": watch_summary,
            "final_core_history": final_history,
            "expected_pre_split": expected,
            "pre_split": pre_split_stats,
            "expected_gate_pass": True,
            "historical_v4_projection": projection,
        },
        "parent_split": {
            "parents": {split: len(parents[split]) for split in SPLITS},
            "chunks_before_cleanup": {split: len(all_chunks[split]) for split in SPLITS},
            "rows_before_cleanup": {
                split: sum(len(session.events) for session in all_chunks[split])
                for split in SPLITS
            },
            "parent_cross_split_count": 0,
        },
        "strict_cleanup": {
            "train_core_history": train_history,
            "valid": valid_cleanup,
            "test": test_cleanup,
            "tail_original_target_preserved_for_survivors": True,
            "retarget_count": 0,
            "backfill_count": 0,
        },
        "row_ledger": {
            "source_rows": source_rows,
            "pre_split_rows": pre_split_rows,
            "pre_split_rows_dropped": pre_split_drops,
            "chunk_rows_before_cleanup": chunk_rows,
            "post_split_rows_dropped": post_split_drops,
            "output_rows": output_rows,
            "source_minus_output": source_rows - output_rows,
            "balanced": source_rows - output_rows == pre_split_drops + post_split_drops,
        },
        "output": output_stats,
        "invariants": {
            "minimum_session_length": final_minimum_session_length,
            "maximum_session_length": final_maximum_session_length,
            "minimum_session_length_pass": final_minimum_session_length >= minimum_session_length,
            "maximum_session_length_pass": final_maximum_session_length <= maximum_session_length,
            "session_length_5_to_50": True,
            "train_item_occurrence_frequency_ge_3": True,
            "valid_test_items_subset_train": True,
            "valid_test_items_subset_of_train": True,
            "valid_test_unseen_item_count": 0,
            "split_session_ids_disjoint": True,
            "combined_exact_ordered_split_union": True,
            "item_exact_train_vocabulary": True,
            "valid_test_original_target_preserved": True,
            "zero_retarget": True,
            "zero_backfill": True,
        },
        "builder": {
            "path": str(builder_path),
            "sha256": sha256_file(builder_path),
            "command": list(sys.argv),
            "host": socket.gethostname(),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
        },
    }
    target = staging_root / spec.name
    _write_release(
        directory=target,
        dataset=spec.name,
        splits=splits,
        all_chunks=all_chunks,
        tail_audits=tail_audits,
        item_header=item_header,
        item_rows=item_rows,
        item_metadata_stats=item_stats,
        summary=summary,
    )
    return target


def publish_one(
    *,
    spec: DatasetSpec,
    repo_root: Path,
    basic_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    final_dir = basic_root / spec.name
    evidence_dir = evidence_root / spec.name
    if path_lexists(final_dir):
        raise FileExistsError(f"refusing to overwrite existing dataset: {final_dir}")
    if path_lexists(evidence_dir):
        raise FileExistsError(f"refusing to overwrite existing evidence: {evidence_dir}")
    basic_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{spec.name}.staging.", dir=str(basic_root))
    )
    evidence_stage = Path(
        tempfile.mkdtemp(prefix=f".{spec.name}.staging.", dir=str(evidence_root))
    )
    started_at = utc_now()
    try:
        staged_dir = build_staged_dataset(
            spec=spec,
            repo_root=repo_root,
            staging_root=stage_root,
            started_at=started_at,
        )
        validator = Path(__file__).resolve().with_name(
            "validate_core5_splits.py"
        )
        report = evidence_stage / "basic_validation_report.json"
        command = [
            sys.executable,
            str(validator),
            "--root", str(stage_root),
            "--dataset", spec.name,
            "--timestamp-unit", spec.timestamp_unit,
            "--expected-rows", str(spec.expected_rows),
            "--expected-sessions", str(spec.expected_sessions),
            "--expected-items", str(spec.expected_items),
            "--report", str(report),
        ]
        if spec.expected_mean_length is not None:
            command.extend(["--expected-mean-length", str(spec.expected_mean_length)])
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (evidence_stage / "validator.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (evidence_stage / "validator.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not report.is_file():
            raise RuntimeError(
                f"independent validator failed rc={completed.returncode}: {completed.stderr[-2000:]}"
            )
        validation = json.loads(report.read_text(encoding="utf-8"))
        if validation.get("ok") is not True:
            raise RuntimeError(f"independent validator returned ok=false: {validation}")
        publication = {
            "schema_version": 1,
            "dataset": spec.name,
            "status": "validated_before_atomic_publish",
            "started_at_utc": started_at,
            "validated_at_utc": utc_now(),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "validator_sha256": sha256_file(validator),
            "validator_command": command,
            "validation_report_sha256": sha256_file(report),
            "staged_tree": str(staged_dir),
            "final_tree": str(final_dir),
            "overwrite_allowed": False,
            "publication_primitive": "renameat2(RENAME_NOREPLACE)",
        }
        (evidence_stage / "publication.json").write_text(
            json.dumps(publication, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if path_lexists(final_dir) or path_lexists(evidence_dir):
            raise FileExistsError(
                "publication target appeared during build; refusing atomic rename: "
                f"data={path_lexists(final_dir)} evidence={path_lexists(evidence_dir)}"
            )
        if staged_dir.stat().st_dev != basic_root.stat().st_dev:
            raise RuntimeError("basic staging and final root are not on the same filesystem")
        if evidence_stage.stat().st_dev != evidence_root.stat().st_dev:
            raise RuntimeError("evidence staging and final root are not on the same filesystem")
        atomic_publish_directory(staged_dir, final_dir)
        atomic_publish_directory(evidence_stage, evidence_dir)
        stage_root.rmdir()
        return {
            "dataset": spec.name,
            "ok": True,
            "final_dir": str(final_dir),
            "evidence_dir": str(evidence_dir),
        }
    except Exception:
        failure = {
            "dataset": spec.name,
            "ok": False,
            "failed_at_utc": utc_now(),
            "staging_root_preserved": str(stage_root),
            "evidence_staging_preserved": str(evidence_stage),
        }
        try:
            (evidence_stage / "build_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(SPECS) + ("all",), required=True
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--basic-root", type=Path,
        default=Path("Datasets/core5_basic"),
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=Path("outputs/data_preparation/core5"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    basic_root = (repo_root / args.basic_root).resolve() if not args.basic_root.is_absolute() else args.basic_root.resolve()
    evidence_root = (repo_root / args.evidence_root).resolve() if not args.evidence_root.is_absolute() else args.evidence_root.resolve()
    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    results = []
    for name in names:
        print(json.dumps({"event": "build_start", "dataset": name, "utc": utc_now()}), flush=True)
        result = publish_one(
            spec=SPECS[name],
            repo_root=repo_root,
            basic_root=basic_root,
            evidence_root=evidence_root,
        )
        results.append(result)
        print(json.dumps({"event": "build_complete", **result, "utc": utc_now()}), flush=True)
    print(json.dumps({"ok": True, "results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
