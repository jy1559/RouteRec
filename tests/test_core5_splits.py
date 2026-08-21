from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_core5_splits.py"
VALIDATOR = ROOT / "scripts" / "validate_core5_splits.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("core5_split_builder", BUILDER)
validator = _load("core5_split_validator", VALIDATOR)


def _session(sid: str, user: str, items: list[str], start: int = 0):
    events = tuple(
        builder.Event(item=item, timestamp=start + index)
        for index, item in enumerate(items)
    )
    return builder.Session(
        sid=sid,
        parent_id=sid,
        user=user,
        events=events,
        first_source_row=start,
    )


def test_occurrence_core_and_adaptive_watch_rule_preserve_parent() -> None:
    sessions = [
        builder.Session(
            sid="u_s0",
            parent_id="u_s0",
            user="u",
            events=tuple(
                builder.Event(item=f"i{index}", timestamp=index, watch_ratio=ratio)
                for index, ratio in enumerate([0.5, 1.2, 0.9, 0.7, 1.1])
            ),
            first_source_row=0,
        )
    ]
    filtered, report = builder.adaptive_watch_filter(sessions)
    assert report["rule_watch_ratio_gt_1_or_gt_stabilized_user_mean"] is True
    assert filtered[0].parent_id == "u_s0"
    # mean=.88, so 1.2, .9, and 1.1 survive; no re-sessionization occurs.
    assert [event.watch_ratio for event in filtered[0].events] == [1.2, 0.9, 1.1]

    repeated = [_session("s", "u", ["z", "z", "z", "a", "a"])]
    result, _ = builder.iterative_core(
        repeated, minimum_session_length=3, minimum_item_frequency=3
    )
    assert [event.item for event in result[0].events] == ["z", "z", "z"]


def test_recovered_loader_sorts_and_uses_strict_gap(tmp_path: Path) -> None:
    path = tmp_path / "source.inter"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(builder.BASE_HEADER)
        writer.writerows(
            [
                ["ignored", "late", 3601, "u"],
                ["ignored", "zero", 0, "u"],
                ["ignored", "tie", 1800, "u"],
            ]
        )
    sessions, stats = builder.load_recovered_and_resessionize(
        path, timestamp_divisor=1.0, inactivity_gap_seconds=1800.0
    )
    assert stats["source_is_raw_exact"] is False
    assert stats["gap_split_boundaries"] == 1
    assert [[event.item for event in session.events] for session in sessions] == [
        ["zero", "tie"], ["late"]
    ]


def test_parent_split_chunk_and_tail_cleanup_never_retargets() -> None:
    parents = [
        _session(f"p{index}", f"u{index}", ["a", "b", "c", "d", "e", "f"], index * 10)
        for index in range(10)
    ]
    assignment = builder.temporal_parent_split(parents)
    assert {split: len(rows) for split, rows in assignment.items()} == {
        "train": 7, "valid": 1, "test": 2
    }
    chunks = builder.balanced_chunks(
        assignment["test"], maximum_session_length=5, split="test"
    )
    assert [len(chunk.events) for chunk in chunks] == [3, 3, 3, 3]
    long_parent = _session("long", "u", ["x"] * 101)
    assert [
        len(chunk.events)
        for chunk in builder.balanced_chunks([long_parent], maximum_session_length=50, split="train")
    ] == [34, 34, 33]

    train = [_session("tr", "u", ["a", "b", "c", "d", "e"])]
    raw = builder.balanced_chunks(
        [_session("tail", "v", ["a", "cold", "b", "c", "d", "e"], 100)],
        maximum_session_length=50,
        split="valid",
    )
    cleaned, totals, audit = builder.strict_train_seen_tail(train, raw)
    assert [event.item for event in cleaned[0].events] == ["a", "b", "c", "d", "e"]
    assert cleaned[0].events[-1].item == raw[0].original_target_item == "e"
    assert totals["cold_context_rows_dropped"] == 1
    assert audit[raw[0].sid]["status"] == "kept"


def _write_preserved_source(repo: Path, session_count: int = 30) -> None:
    source = repo / "Datasets" / "release" / "source"
    source.mkdir(parents=True)
    with (source / "source.item").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["item_id:token", "category:token"])
        for index in range(6):
            writer.writerow([f"i{index}", f"c{index % 2}"])
    with (source / "source.inter").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([*builder.BASE_HEADER, "old_feature:float"])
        for session_index in range(session_count):
            for offset in range(6):
                writer.writerow(
                    [
                        f"legacy_{session_index}", f"i{offset}",
                        session_index * 100 + offset, f"u{session_index}", 0.5,
                    ]
                )


def _synthetic_spec() -> object:
    return builder.DatasetSpec(
        name="synthetic_core5_v1",
        source_kind="preserve_session",
        timestamp_unit="s",
        expected_rows=180,
        expected_sessions=30,
        expected_items=6,
        release_source="Datasets/release/source",
    )


def test_end_to_end_stage_validate_atomic_publish_and_refuse_overwrite(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_preserved_source(repo)
    basic_root = tmp_path / "basic"
    evidence_root = tmp_path / "evidence"
    result = builder.publish_one(
        spec=_synthetic_spec(),
        repo_root=repo,
        basic_root=basic_root,
        evidence_root=evidence_root,
    )
    assert result["ok"] is True
    final = basic_root / "synthetic_core5_v1"
    report = json.loads(
        (evidence_root / "synthetic_core5_v1" / "basic_validation_report.json").read_text()
    )
    assert report["ok"] is True
    assert report["combined_exact_ordered_split_union"] is True
    assert report["valid_test_unseen_target_count"] == 0
    summary = json.loads((final / "synthetic_core5_v1.basic_summary.json").read_text())
    assert summary["source"]["stats"]["old_feature_columns_ignored"] is True
    assert summary["row_ledger"]["balanced"] is True
    with pytest.raises(FileExistsError):
        builder.publish_one(
            spec=_synthetic_spec(),
            repo_root=repo,
            basic_root=basic_root,
            evidence_root=evidence_root,
        )


def test_validator_rejects_retargeted_lineage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_preserved_source(repo)
    stage = tmp_path / "stage"
    stage.mkdir()
    builder.build_staged_dataset(
        spec=_synthetic_spec(), repo_root=repo, staging_root=stage,
        started_at=builder.utc_now(),
    )
    lineage = stage / "synthetic_core5_v1" / "synthetic_core5_v1.parent_lineage.tsv"
    text = lineage.read_text(encoding="utf-8")
    # Change one kept tail original target while leaving the runtime row untouched.
    lines = text.splitlines()
    header = lines[0].split("\t")
    split_index = header.index("split")
    status_index = header.index("status")
    target_index = header.index("original_target_item")
    for index in range(1, len(lines)):
        fields = lines[index].split("\t")
        if fields[split_index] == "valid" and fields[status_index] == "kept":
            fields[target_index] = "tampered"
            lines[index] = "\t".join(fields)
            break
    lineage.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        root=stage,
        dataset="synthetic_core5_v1",
        timestamp_unit="s",
        expected_rows=180,
        expected_sessions=30,
        expected_items=6,
        expected_mean_length=None,
    )
    report = validator.validate(args)
    assert report["ok"] is False
    assert any("retargeted" in error or "hash mismatch" in error for error in report["errors"])


def test_atomic_publication_never_replaces_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    (target / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        builder.atomic_publish_directory(source, target)
    assert (source / "new.txt").read_text(encoding="utf-8") == "new"
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
