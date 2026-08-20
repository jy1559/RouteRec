from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


basic_builder = _load(
    "camera_ready_basic_for_feature_test",
    ROOT / "scripts" / "rebuild_camera_ready_core5_basic.py",
)
feature_driver = _load(
    "camera_ready_feature_driver",
    ROOT / "scripts" / "build_camera_ready_core5_features.py",
)


def _write_source(repo: Path) -> object:
    source = repo / "Datasets" / "release" / "source"
    source.mkdir(parents=True)
    with (source / "source.item").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["item_id:token", "category:token"])
        for index in range(6):
            writer.writerow([f"i{index}", f"c{index % 2}"])
    with (source / "source.inter").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(basic_builder.BASE_HEADER)
        for session_index in range(30):
            for offset in range(6):
                # ``.17g`` preserves these frozen source tokens while the old
                # feature writer's ``.15g`` round trip shortened them.  This
                # reproduces the production KuaiRec gate failure.
                timestamp = 1593051862.7030001 + session_index * 10_000 + offset
                writer.writerow(
                    [f"s{session_index}", f"i{offset}", timestamp, f"u{session_index}"]
                )
    return basic_builder.DatasetSpec(
        name="synthetic_core5_v1",
        source_kind="preserve_session",
        timestamp_unit="s",
        expected_rows=180,
        expected_sessions=30,
        expected_items=6,
        release_source="Datasets/release/source",
    )


def test_feature_stage_validate_and_atomic_publish(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    # The driver invokes these production scripts from its repo root.
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("build_full_feature_dataset.py", "validate_full_feature_dataset.py"):
        (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    spec = _write_source(repo)
    basic_root = tmp_path / "basic"
    evidence_root = tmp_path / "evidence"
    basic_builder.publish_one(
        spec=spec, repo_root=repo, basic_root=basic_root, evidence_root=evidence_root
    )
    feature_root = tmp_path / "features"
    result = feature_driver.build_one(
        dataset="synthetic_core5_v1",
        timestamp_unit="s",
        repo_root=repo,
        basic_root=basic_root,
        feature_root=feature_root,
        evidence_root=evidence_root,
    )
    assert result["ok"] is True
    final = feature_root / "synthetic_core5_v1"
    assert final.is_dir()
    header = (final / "synthetic_core5_v1.inter").read_text().splitlines()[0]
    assert len(header.split("\t")) == 68
    basic_rows = (basic_root / "synthetic_core5_v1" / "synthetic_core5_v1.inter").read_text().splitlines()
    feature_rows = (final / "synthetic_core5_v1.inter").read_text().splitlines()
    assert [row.split("\t")[:4] for row in feature_rows[1:]] == [
        row.split("\t") for row in basic_rows[1:]
    ]
    report = json.loads(
        (evidence_root / "synthetic_core5_v1" / "feature_build" / "feature_validation_report.json").read_text()
    )
    assert report["ok"] is True
    publication = json.loads(
        (evidence_root / "synthetic_core5_v1" / "feature_build" / "publication.json").read_text()
    )
    assert publication["feature_contract"]["mid_scope"] == "strict_prefix"
    assert "--overwrite" not in publication["builder"]["command"]
    with pytest.raises(FileExistsError):
        feature_driver.build_one(
            dataset="synthetic_core5_v1",
            timestamp_unit="s",
            repo_root=repo,
            basic_root=basic_root,
            feature_root=feature_root,
            evidence_root=evidence_root,
        )


def test_feature_publication_never_replaces_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    (target / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        feature_driver.atomic_publish_directory(source, target)
    assert (source / "new.txt").read_text(encoding="utf-8") == "new"
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
