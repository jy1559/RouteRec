#!/usr/bin/env python3
"""Build, independently validate, and atomically publish core5 features.

Each dataset is built into a hidden same-filesystem directory.  The existing
feature builder consumes the frozen basic split, recomputes every one of the
64 engineered features from scratch with train-only fitting and strict-prefix
mid scope, and never receives ``--overwrite``.  The exhaustive validator must
return ``ok=true`` before the complete feature tree is renamed into its final
non-existing destination.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DATASETS = {
    "beauty_core5_v1": "ms",
    "foursquare_core5_v1": "s",
    "movielens1m_core5_v1": "s",
    "retail_rocket_core5_v1": "ms",
    "kuairec_adaptive_core5_v1": "s",
    "lastfm_recovered_core5_v1": "ms",
}


def path_lexists(path: Path) -> bool:
    """Return true for every directory entry, including a broken symlink."""

    return os.path.lexists(os.fspath(path))


def atomic_publish_directory(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing entry."""

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(directory: Path) -> dict[str, object]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "contract": "core5-feature-tree-manifest-v1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(record["bytes"]) for record in files),
    }


def run_logged(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    started_at = utc_now()
    started = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    return {
        "command": command,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": completed.returncode,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }


def build_one(
    *,
    dataset: str,
    timestamp_unit: str,
    repo_root: Path,
    basic_root: Path,
    feature_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    basic_dir = basic_root / dataset
    final_dir = feature_root / dataset
    evidence_dir = evidence_root / dataset
    evidence_final = evidence_dir / "feature_build"
    if not basic_dir.is_dir():
        raise FileNotFoundError(f"validated basic dataset does not exist: {basic_dir}")
    if path_lexists(final_dir):
        raise FileExistsError(f"refusing to overwrite feature dataset: {final_dir}")
    if path_lexists(evidence_final):
        raise FileExistsError(f"refusing to overwrite feature evidence: {evidence_final}")
    feature_root.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{dataset}.feature-staging.", dir=str(feature_root))
    )
    evidence_stage = Path(
        tempfile.mkdtemp(prefix=f".{dataset}.feature-staging.", dir=str(evidence_dir))
    )
    builder = repo_root / "scripts" / "build_feature_dataset.py"
    validator = repo_root / "scripts" / "validate_feature_dataset.py"
    if not builder.is_file() or not validator.is_file():
        raise FileNotFoundError("feature builder or validator is missing")
    try:
        build_command = [
            sys.executable,
            "-u",
            str(builder),
            "--basic-root", str(basic_root),
            "--output-root", str(stage_root),
            "--source-dataset", dataset,
            "--target-dataset", dataset,
            "--timestamp-unit", timestamp_unit,
            "--tail-unseen-policy", "precleaned_frozen",
            "--frozen-basic-splits",
        ]
        if "--overwrite" in build_command:
            raise AssertionError("core5 feature build must never pass --overwrite")
        build_run = run_logged(
            build_command,
            stdout_path=evidence_stage / "builder.stdout.log",
            stderr_path=evidence_stage / "builder.stderr.log",
        )
        if build_run["returncode"] != 0:
            raise RuntimeError(f"feature builder failed: {build_run}")
        staged_dir = stage_root / dataset
        if not staged_dir.is_dir():
            raise RuntimeError("feature builder succeeded without producing the staged dataset")

        validation_report = evidence_stage / "feature_validation_report.json"
        validate_command = [
            sys.executable,
            "-u",
            str(validator),
            "--basic-root", str(basic_root),
            "--feature-root", str(stage_root),
            "--source-dataset", dataset,
            "--target-dataset", dataset,
            "--report", str(validation_report),
        ]
        validation_run = run_logged(
            validate_command,
            stdout_path=evidence_stage / "validator.stdout.log",
            stderr_path=evidence_stage / "validator.stderr.log",
        )
        if validation_run["returncode"] != 0 or not validation_report.is_file():
            raise RuntimeError(f"feature validator failed: {validation_run}")
        validation = json.loads(validation_report.read_text(encoding="utf-8"))
        if validation.get("ok") is not True:
            raise RuntimeError("feature validator returned ok=false")

        manifest = tree_manifest(staged_dir)
        manifest_path = staged_dir / f"{dataset}.core5_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Re-read every recorded file after manifest creation to detect a late mutation.
        for record in manifest["files"]:  # type: ignore[index]
            path = staged_dir / str(record["path"])
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"staged feature artifact mutated after validation: {path}")

        publication = {
            "schema_version": 1,
            "contract": "core5-feature-publication-v1",
            "dataset": dataset,
            "timestamp_unit": timestamp_unit,
            "status": "validated_before_atomic_publish",
            "host": socket.gethostname(),
            "builder": {
                "path": str(builder),
                "sha256": sha256_file(builder),
                **build_run,
            },
            "validator": {
                "path": str(validator),
                "sha256": sha256_file(validator),
                **validation_run,
                "report_sha256": sha256_file(validation_report),
            },
            "tree_manifest": {
                "path": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "file_count_excluding_manifest": manifest["file_count"],
                "bytes_excluding_manifest": manifest["total_bytes"],
            },
            "feature_contract": {
                "columns": 68,
                "base_columns": 4,
                "engineered_features": 64,
                "mid_scope": "strict_prefix",
                "fit_scope": "train_only",
                "frozen_basic_splits": True,
                "finite_unit_interval_required": True,
                "base_four_byte_order_identity_required": True,
            },
            "overwrite_allowed": False,
            "publication_primitive": "renameat2(RENAME_NOREPLACE)",
            "final_dir": str(final_dir),
            "validated_at_utc": utc_now(),
        }
        (evidence_stage / "publication.json").write_text(
            json.dumps(publication, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if path_lexists(final_dir) or path_lexists(evidence_final):
            raise FileExistsError("feature publication target appeared during build")
        if stage_root.stat().st_dev != feature_root.stat().st_dev:
            raise RuntimeError("feature staging root is not on final filesystem")
        if evidence_stage.stat().st_dev != evidence_dir.stat().st_dev:
            raise RuntimeError("feature evidence staging root is not on final filesystem")
        atomic_publish_directory(evidence_stage, evidence_final)
        atomic_publish_directory(staged_dir, final_dir)
        stage_root.rmdir()
        return {
            "dataset": dataset,
            "ok": True,
            "final_dir": str(final_dir),
            "evidence_dir": str(evidence_final),
            "builder_seconds": build_run["elapsed_seconds"],
            "validator_seconds": validation_run["elapsed_seconds"],
        }
    except Exception as exc:
        try:
            (evidence_stage / "feature_build_failure.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at_utc": utc_now(),
                        "staging_root_preserved": str(stage_root),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS) + ("all",), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--basic-root", type=Path,
        default=Path("Datasets/core5_basic"),
    )
    parser.add_argument(
        "--feature-root", type=Path,
        default=Path("Datasets/core5"),
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
    feature_root = (repo_root / args.feature_root).resolve() if not args.feature_root.is_absolute() else args.feature_root.resolve()
    evidence_root = (repo_root / args.evidence_root).resolve() if not args.evidence_root.is_absolute() else args.evidence_root.resolve()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    for name in names:
        print(json.dumps({"event": "feature_build_start", "dataset": name, "utc": utc_now()}), flush=True)
        result = build_one(
            dataset=name,
            timestamp_unit=DATASETS[name],
            repo_root=repo_root,
            basic_root=basic_root,
            feature_root=feature_root,
            evidence_root=evidence_root,
        )
        results.append(result)
        print(json.dumps({"event": "feature_build_complete", **result, "utc": utc_now()}), flush=True)
    print(json.dumps({"ok": True, "results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
