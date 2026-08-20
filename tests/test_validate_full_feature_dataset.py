from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_full_feature_dataset.py"
SPEC = importlib.util.spec_from_file_location("core5_feature_validator", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _make_release(tmp_path: Path) -> tuple[Path, Path]:
    basic_root = tmp_path / "basic"
    feature_root = tmp_path / "feature"
    basic_dir = basic_root / "source"
    feature_dir = feature_root / "target"
    basic_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)

    split_rows: dict[str, list[list[str]]] = {split: [] for split in module.SPLITS}
    allocation = {"train": 3, "valid": 1, "test": 1}
    prefix = {"train": "tr", "valid": "va", "test": "te"}
    clock = 0
    for split, count in allocation.items():
        for session_index in range(count):
            sid = f"c5_{prefix[split]}_{session_index}"
            for offset, item in enumerate(("a", "b", "c", "a", "b")):
                split_rows[split].append(
                    [sid, item, str(clock + offset), f"u_{split}_{session_index}"]
                )
            clock += 100
    combined_rows = [row for split in module.SPLITS for row in split_rows[split]]
    base_header = list(module.BASE_HEADER)
    _write_tsv(basic_dir / "source.inter", base_header, combined_rows)
    for split in module.SPLITS:
        _write_tsv(basic_dir / f"source.{split}.inter", base_header, split_rows[split])
    _write_tsv(
        basic_dir / "source.item",
        ["item_id:token", "category:token"],
        [["a", "x"], ["b", "y"], ["c", "z"]],
    )

    basic_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(basic_dir.iterdir())
        if path.is_file()
    }
    basic_summary = {
        "schema_version": 1,
        "contract": module.BASIC_CONTRACT,
        "status": "complete",
        "target_dataset": "source",
        "parameters": {
            "timestamp_unit": "s",
            "minimum_session_length": 5,
            "maximum_session_length": 50,
            "minimum_item_interaction_frequency": 3,
        },
        "output": {
            "rows": {split: len(split_rows[split]) for split in module.SPLITS},
            "sessions": allocation,
            "total_rows": len(combined_rows),
            "total_sessions": sum(allocation.values()),
        },
        "invariants": {
            "minimum_session_length": 5,
            "maximum_session_length": 5,
            "minimum_session_length_pass": True,
            "maximum_session_length_pass": True,
            "valid_test_items_subset_of_train": True,
            "valid_test_unseen_item_count": 0,
            "split_session_overlap": {
                "train_valid": 0,
                "train_test": 0,
                "valid_test": 0,
            },
        },
        "files": basic_files,
    }
    (basic_dir / "source.basic_summary.json").write_text(
        json.dumps(basic_summary, indent=2) + "\n", encoding="utf-8"
    )

    feature_header = list(module.FEATURE_HEADER)
    feature_rows = {
        split: [row + ["0"] * len(module.FEATURES) for row in split_rows[split]]
        for split in module.SPLITS
    }
    _write_tsv(
        feature_dir / "target.inter",
        feature_header,
        [row for split in module.SPLITS for row in feature_rows[split]],
    )
    for split in module.SPLITS:
        _write_tsv(feature_dir / f"target.{split}.inter", feature_header, feature_rows[split])
    (feature_dir / "target.item").write_bytes((basic_dir / "source.item").read_bytes())

    source_hashes = {
        "inter_sha256": _sha(basic_dir / "source.inter"),
        "item_sha256": _sha(basic_dir / "source.item"),
        "builder_sha256": "a" * 64,
    }
    feature_meta = {
        "dataset": "target",
        "source_dataset": "source",
        "all_features": list(module.FEATURES),
        "reconstruction_contract": module.FEATURE_CONTRACT,
        "source_preprocessing_contract": module.BASIC_CONTRACT,
        "timestamp_unit": "s",
        "fit_sessions": {
            "fit_session_ratio": 3 / 5,
            "fit_session_count": 3,
            "total_session_count": 5,
        },
        "mid_scope": "strict_prefix",
        "mid_constant_postprocess": {"applied": False},
        "normalization_stats": {
            name: {"type": "bounded"} for name in module.FEATURES
        },
        "tail_unseen_filter": {
            "policy": "precleaned_frozen",
            "source_split_rows": {
                split: len(split_rows[split]) for split in module.SPLITS
            },
            "train_item_count": 3,
        },
        "source_hashes": source_hashes,
        "metadata_canonicalization": {
            "schema_version": 1,
            "contract": module.METADATA_CONTRACT,
            "method": "native_builder_v1",
            "removed_fields": ["output_root", "source_root"],
            "source_dataset": "source",
            "target_dataset": "target",
            "builder_sha256": "a" * 64,
        },
    }
    (feature_dir / "feature_meta_v3.json").write_text(
        json.dumps(feature_meta, indent=2) + "\n", encoding="utf-8"
    )
    split_summary = {
        "dataset": "target",
        "split_strategy": "frozen_source_basic_membership",
        "write_stats": {
            "rows": {split: len(split_rows[split]) for split in module.SPLITS},
            "sessions": allocation,
            "session_overlap": {"train_valid": 0, "train_test": 0, "valid_test": 0},
        },
    }
    (feature_dir / "target.session_split_summary.json").write_text(
        json.dumps(split_summary, indent=2) + "\n", encoding="utf-8"
    )
    build_summary = {
        "dataset": "target",
        "source_dataset": "source",
        "rows": len(combined_rows),
        "sessions": sum(allocation.values()),
        "features": len(module.FEATURES),
        "feature_contract_version": module.FEATURE_CONTRACT,
        "source_hashes": source_hashes,
        "tail_unseen_filter": feature_meta["tail_unseen_filter"],
        "frozen_basic_splits": True,
        "split_session_counts": allocation,
        "split_summary": split_summary,
    }
    (feature_dir / "target.build_summary.json").write_text(
        json.dumps(build_summary, indent=2) + "\n", encoding="utf-8"
    )
    return basic_root, feature_root


def _validate(basic_root: Path, feature_root: Path) -> dict[str, object]:
    return module.validate_release(
        basic_root=basic_root,
        feature_root=feature_root,
        source_dataset="source",
        target_dataset="target",
    )


def test_valid_frozen_core5_feature_release_passes(tmp_path: Path) -> None:
    basic_root, feature_root = _make_release(tmp_path)
    report = _validate(basic_root, feature_root)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["invariants"]["feature_schema_columns"] == 68
    assert report["invariants"]["feature_values_checked"] == 25 * 64
    assert report["invariants"]["item_table_equals_train_vocabulary"] is True


def test_rejects_non_finite_or_out_of_range_feature(tmp_path: Path) -> None:
    basic_root, feature_root = _make_release(tmp_path)
    path = feature_root / "target" / "target.valid.inter"
    rows = list(csv.reader(path.open("r", encoding="utf-8", newline=""), delimiter="\t"))
    rows[1][4] = "nan"
    rows[2][5] = "1.01"
    _write_tsv(path, rows[0], rows[1:])
    report = _validate(basic_root, feature_root)
    assert report["ok"] is False
    assert any("is not finite in [0,1]" in error for error in report["errors"])


def test_rejects_feature_base_identity_change(tmp_path: Path) -> None:
    basic_root, feature_root = _make_release(tmp_path)
    path = feature_root / "target" / "target.test.inter"
    rows = list(csv.reader(path.open("r", encoding="utf-8", newline=""), delimiter="\t"))
    rows[1][1] = "different-item"
    _write_tsv(path, rows[0], rows[1:])
    report = _validate(basic_root, feature_root)
    assert report["ok"] is False
    assert any("base identity differs" in error for error in report["errors"])


def test_rejects_non_frozen_metadata_or_stale_source_hash(tmp_path: Path) -> None:
    basic_root, feature_root = _make_release(tmp_path)
    path = feature_root / "target" / "feature_meta_v3.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["source_preprocessing_contract"] = "other"
    meta["tail_unseen_filter"]["policy"] = "drop_non_target"
    meta["source_hashes"]["inter_sha256"] = "0" * 64
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    report = _validate(basic_root, feature_root)
    assert report["ok"] is False
    joined = "\n".join(report["errors"])
    assert "source preprocessing contract" in joined
    assert "precleaned_frozen" in joined
    assert "source inter hash" in joined


def test_cli_writes_machine_readable_failure_report(tmp_path: Path) -> None:
    basic_root, feature_root = _make_release(tmp_path)
    (feature_root / "target" / "target.item").unlink()
    report_path = tmp_path / "report.json"
    code = module.main(
        [
            "--basic-root", str(basic_root),
            "--feature-root", str(feature_root),
            "--source-dataset", "source",
            "--target-dataset", "target",
            "--report", str(report_path),
        ]
    )
    assert code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert any("missing required file" in error for error in report["errors"])
