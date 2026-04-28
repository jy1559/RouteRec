#!/usr/bin/env python3
"""Evaluate a trained RouteRec checkpoint on test split."""

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
from routerec.runner import build_config_dict, run_test_only, save_json_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RouteRec checkpoint on test split.")
    parser.add_argument("--dataset", required=True, help="Dataset id or paper alias (for example: ml-1m, lastfm, kuairec)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="RouteRec")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional data root override. By default RouteRec auto-discovers Datasets/processed/feature_added_v4.",
    )
    parser.add_argument("--output", default="outputs/test/latest.json")
    args = parser.parse_args()

    resolved_dataset = resolve_dataset_runtime(
        dataset=args.dataset,
        data_path=args.data_path,
        repo_root=ROOT,
        require_existing=True,
    )

    config_dict = build_config_dict(
        dataset=resolved_dataset.dataset_name,
        model=args.model,
        base_config_path=ROOT / "configs/models/routerec_default.yaml",
        dataset_presets_path=ROOT / "configs/models/routerec_dataset_presets.yaml",
        use_dataset_preset=True,
        overrides={},
        epochs=1,
        train_batch_size=4096,
        eval_batch_size=8192,
        seed=args.seed,
        data_path=resolved_dataset.data_path,
    )

    result = run_test_only(
        dataset=resolved_dataset.dataset_name,
        model=args.model,
        config_dict=config_dict,
        checkpoint_path=args.checkpoint,
    )
    save_json_result(ROOT / args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
