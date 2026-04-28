#!/usr/bin/env python3
"""Single-run training entrypoint for RouteRec and bundled baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.datasets import resolve_dataset_runtime
from routerec.runner import build_config_dict, run_training, save_json_result


def _parse_overrides(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"Override key is empty: {item}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RouteRec with one dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset id or paper alias (for example: ml-1m, lastfm, kuairec)")
    parser.add_argument("--model", default="RouteRec", help="Model name (default: RouteRec)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional data root override. By default RouteRec auto-discovers Datasets/processed/feature_added_v4.",
    )
    parser.add_argument("--no-dataset-preset", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--output", default="outputs/train/latest.json")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra config override as key=value. JSON values are supported.",
    )
    args = parser.parse_args()

    resolved_dataset = resolve_dataset_runtime(
        dataset=args.dataset,
        data_path=args.data_path,
        repo_root=ROOT,
        require_existing=True,
    )
    overrides = _parse_overrides(args.override)
    config_dict = build_config_dict(
        dataset=resolved_dataset.dataset_name,
        model=args.model,
        base_config_path=ROOT / "configs/models/routerec_default.yaml",
        dataset_presets_path=ROOT / "configs/models/routerec_dataset_presets.yaml",
        use_dataset_preset=not args.no_dataset_preset,
        overrides=overrides,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        data_path=resolved_dataset.data_path,
    )

    result = run_training(
        dataset=resolved_dataset.dataset_name,
        model=args.model,
        config_dict=config_dict,
        save_model=not args.no_save_model,
    )
    save_json_result(ROOT / args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
