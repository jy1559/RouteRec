#!/usr/bin/env python3
"""Strictly re-evaluate a trusted RouteRec attempt checkpoint on test data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.datasets import infer_recbole_dataset_config, normalize_dataset_name
from routerec.runner import deep_merge, run_test_only, save_json_result


def _yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(loaded)


def _checkpoint_from_attempt(attempt_dir: Path) -> Path:
    candidates = sorted(
        path for path in (attempt_dir / "checkpoints").glob("*.pth") if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one best checkpoint under {attempt_dir / 'checkpoints'}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _saved_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    import torch

    # RecBole 1.2.1 pickles its Config object.  Only evaluate checkpoints that
    # were created by this trusted local experiment workspace.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint is not a RecBole state mapping")
    config = payload.get("config")
    final = getattr(config, "final_config_dict", None)
    if not isinstance(final, Mapping):
        raise ValueError("checkpoint does not contain a frozen RecBole Config")
    return dict(final)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-dir",
        required=True,
        help="Successful attempt directory containing resolved_config.yaml and checkpoints/.",
    )
    parser.add_argument("--checkpoint", help="Optional explicit checkpoint inside the attempt directory.")
    parser.add_argument("--gpu", default="0", help="One physical GPU id to expose as logical cuda:0.")
    parser.add_argument("--data-path", help="Host-specific replacement for the frozen dataset root.")
    parser.add_argument("--output", help="Output JSON; defaults to a timestamped file inside the attempt.")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    if not str(args.gpu).strip() or "," in str(args.gpu):
        raise ValueError("--gpu must identify exactly one physical GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    os.environ["ROUTEREC_PHYSICAL_GPU_ID"] = str(args.gpu).strip()
    os.environ["ROUTEREC_LOGICAL_GPU_ID"] = "0"

    attempt_dir = Path(args.attempt_dir).expanduser().resolve()
    if not attempt_dir.is_dir():
        raise FileNotFoundError(f"attempt directory does not exist: {attempt_dir}")
    resolved_path = attempt_dir / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"attempt is missing resolved_config.yaml: {attempt_dir}")

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else _checkpoint_from_attempt(attempt_dir)
    )
    try:
        checkpoint.relative_to((attempt_dir / "checkpoints").resolve())
    except ValueError as exc:
        raise ValueError("checkpoint must be contained in the selected attempt") from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    recorded_config = _yaml_mapping(resolved_path)
    checkpoint_config = _saved_checkpoint_config(checkpoint)
    recorded_model = str(recorded_config.get("model", ""))
    recorded_dataset = normalize_dataset_name(str(recorded_config.get("dataset", "")))
    if str(checkpoint_config.get("model", "")) != recorded_model:
        raise ValueError("checkpoint model does not match resolved_config.yaml")
    if normalize_dataset_name(str(checkpoint_config.get("dataset", ""))) != recorded_dataset:
        raise ValueError("checkpoint dataset does not match resolved_config.yaml")

    # Use the checkpoint-embedded frozen config as the reconstruction source.
    # A host path may change when artifacts move, so that one field is
    # intentionally replaceable and the dataset schema is re-inferred.
    evaluation_config = dict(checkpoint_config)
    if args.data_path:
        evaluation_config["data_path"] = str(Path(args.data_path).expanduser().resolve())
        evaluation_config = deep_merge(
            evaluation_config,
            infer_recbole_dataset_config(
                dataset=recorded_dataset,
                data_path=evaluation_config["data_path"],
                repo_root=ROOT,
            ),
        )
    evaluation_config["gpu_id"] = 0
    evaluation_config["use_gpu"] = True
    evaluation_config["seed"] = int(checkpoint_config.get("seed", recorded_config.get("seed", 42)))
    evaluation_config["reproducibility"] = True

    result = run_test_only(
        dataset=recorded_dataset,
        model=recorded_model,
        config_dict=evaluation_config,
        checkpoint_path=str(checkpoint),
        strict_checkpoint=True,
        show_progress=bool(args.show_progress),
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else attempt_dir / "reevaluations" / f"test-{timestamp}.json"
    )
    save_json_result(output, result)
    print(json.dumps({"output": str(output), "test_result": result["test_result"]}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    raise SystemExit(main())
