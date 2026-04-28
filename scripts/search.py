#!/usr/bin/env python3
"""Bounded hyperparameter search for RouteRec using appendix-aligned ranges."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.datasets import resolve_dataset_runtime
from routerec.runner import build_config_dict, run_training, save_json_result


def _sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _pick(rng: random.Random, values: list[object]) -> object:
    return values[rng.randrange(0, len(values))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded random search for RouteRec.")
    parser.add_argument("--dataset", required=True, help="Dataset id or paper alias (for example: ml-1m, lastfm, kuairec)")
    parser.add_argument("--model", default="RouteRec")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional data root override. By default RouteRec auto-discovers Datasets/processed/feature_added_v4.",
    )
    parser.add_argument("--output", default="outputs/search/latest.json")
    args = parser.parse_args()

    resolved_dataset = resolve_dataset_runtime(
        dataset=args.dataset,
        data_path=args.data_path,
        repo_root=ROOT,
        require_existing=True,
    )

    grid_path = ROOT / "configs/search_spaces/paper_bounded_grid.yaml"
    with grid_path.open("r", encoding="utf-8") as f:
        search_space = yaml.safe_load(f) or {}

    shared = search_space.get("shared", {})
    routerec = search_space.get("routerec_additions", {})
    lr_intervals = search_space.get("dataset_learning_rate_intervals", {})
    lr_low, lr_high = lr_intervals.get(resolved_dataset.dataset_name, [1.5e-4, 2.0e-3])

    rng = random.Random(args.seed)
    all_results: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for trial in range(1, args.trials + 1):
        embed = int(_pick(rng, shared.get("width_embedding", [128])))
        overrides = {
            "learning_rate": _sample_log_uniform(rng, float(lr_low), float(lr_high)),
            "weight_decay": _pick(rng, routerec.get("weight_decay", shared.get("weight_decay", [1e-6]))),
            "MAX_ITEM_LIST_LENGTH": int(_pick(rng, shared.get("max_history_length", [20]))),
            "embedding_size": embed,
            "hidden_size": embed,
            "d_ff": int(_pick(rng, shared.get("inner_width", [256]))),
            "n_layers": int(_pick(rng, shared.get("layers", [2]))),
            "num_heads": int(_pick(rng, shared.get("heads", [2]))),
            "hidden_dropout_prob": float(_pick(rng, routerec.get("dropout", shared.get("hidden_dropout", [0.15])))),
            "attn_dropout_prob": float(_pick(rng, routerec.get("attention_dropout", shared.get("attention_dropout", [0.1])))),
            "d_router_hidden": int(_pick(rng, routerec.get("router_width", [64]))),
            "expert_scale": int(_pick(rng, routerec.get("expert_scale", [3]))),
            "stage_feature_dropout_prob": float(_pick(rng, routerec.get("feature_dropout", [0.03]))),
            "route_consistency_lambda": float(_pick(rng, routerec.get("route_consistency_lambda", [2.5e-4]))),
            "z_loss_lambda": float(_pick(rng, routerec.get("z_loss_lambda", [1e-4]))),
        }

        config_dict = build_config_dict(
            dataset=resolved_dataset.dataset_name,
            model=args.model,
            base_config_path=ROOT / "configs/models/routerec_default.yaml",
            dataset_presets_path=ROOT / "configs/models/routerec_dataset_presets.yaml",
            use_dataset_preset=True,
            overrides=overrides,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed + trial,
            data_path=resolved_dataset.data_path,
        )

        try:
            result = run_training(
                dataset=resolved_dataset.dataset_name,
                model=args.model,
                config_dict=config_dict,
                save_model=False,
            )
            score = float(result.get("best_valid_score", float("-inf")))
            trial_result = {"trial": trial, "score": score, "overrides": overrides, "result": result}
            all_results.append(trial_result)
            if best is None or score > float(best["score"]):
                best = trial_result
        except Exception as exc:
            all_results.append({"trial": trial, "error": str(exc), "overrides": overrides})

    payload = {
        "dataset": resolved_dataset.dataset_name,
        "model": args.model,
        "trials": args.trials,
        "best": best,
        "results": all_results,
    }
    save_json_result(ROOT / args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
