#!/usr/bin/env python3
"""Single-run training entrypoint for RouteRec and bundled baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional data root override. By default RouteRec searches Datasets/core5 and Datasets/release.",
    )
    parser.add_argument(
        "--use-dataset-preset",
        action="store_true",
        help="Apply the documented dataset-specific initialization preset.",
    )
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly evaluate test after loading the best validation checkpoint.",
    )
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--run-dir", default=None, help="Unique artifact directory for this attempt.")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra config override as key=value. JSON values are supported.",
    )
    args = parser.parse_args()
    if args.no_save_model and args.evaluate_test:
        parser.error("test evaluation requires a saved best-validation checkpoint")

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
        use_dataset_preset=bool(args.use_dataset_preset),
        overrides=overrides,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        data_path=resolved_dataset.data_path,
    )

    if args.run_dir:
        run_dir = (ROOT / args.run_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "outputs" / "runs" / f"manual_{resolved_dataset.dataset_name}_{args.model}_s{args.seed}_{stamp}_{os.getpid()}"

    result = run_training(
        dataset=resolved_dataset.dataset_name,
        model=args.model,
        config_dict=config_dict,
        save_model=not args.no_save_model,
        run_dir=run_dir,
        evaluate_test=bool(args.evaluate_test),
        show_progress=args.show_progress,
    )
    output_path = (ROOT / args.output).resolve() if args.output else run_dir / "result.json"
    save_json_result(output_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
