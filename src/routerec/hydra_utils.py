"""
Hydra-based config utilities for RecBole experiments.
Provides robust hierarchical config composition and CLI overrides.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def validate_feature_mode(cfg: DictConfig | Dict[str, Any]) -> None:
    """Validate the stable public feature-mode surface."""
    feature_mode = str(cfg.get("feature_mode", "full")).strip().lower()
    if feature_mode not in {"full", "none"}:
        raise RuntimeError(
            f"unsupported feature_mode={feature_mode!r}; expected 'full' or 'none'"
        )


def _detect_item_file(cfg: DictConfig) -> Optional[Path]:
    """Locate the dataset's item or interaction file for counting items."""
    data_path = Path(cfg.get("data_path", "."))
    dataset = str(cfg.get("dataset", "")).strip()
    candidates = [
        data_path / dataset / f"{dataset}.item",
        data_path / f"{dataset}.item",
        data_path / dataset / f"{dataset}.train.inter",
        data_path / f"{dataset}.train.inter",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _count_items(cfg: DictConfig, threshold: int) -> Optional[int]:
    """Count (or lower-bound) item cardinality with early stop for large sets."""
    path = _detect_item_file(cfg)
    if path is None:
        return None

    sep = cfg.get("field_separator", "\t")
    item_field = cfg.get("ITEM_ID_FIELD", "item_id")

    if path.suffix == ".item":
        with path.open("r", encoding="utf-8") as f:
            count = sum(1 for _ in f) - 1  # subtract header
        return max(count, 0)

    # Fall back to counting unique item ids from interactions (early stopping)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(sep)
        try:
            idx = header.index(item_field)
        except ValueError:
            return None

        seen = set()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(sep)
            if len(parts) <= idx:
                continue
            seen.add(parts[idx])
            if len(seen) > threshold:
                return len(seen)
    return len(seen)


def configure_eval_sampling(cfg: DictConfig) -> DictConfig:
    """
    Configure evaluation sampling policy.
    
    IMPORTANT: Always use 'full' mode for FullSortEvalDataLoader.
    For large datasets, precomputed negatives + score masking is applied 
    in recbole_train.py (NOT RecBole's NegSampleEvalDataLoader which is slow).
    
    RecBole's NegSampleEvalDataLoader has step=1 issue (1 user per batch) 
    when sample_num is large, making it extremely slow.
    """
    # Ensure eval_args exists
    if "eval_args" not in cfg:
        OmegaConf.update(cfg, "eval_args", {})
    
    # Always use 'full' mode for FullSortEvalDataLoader (batch by session)
    # Large datasets use precomputed negatives + score masking in Trainer
    OmegaConf.update(cfg.eval_args, "mode", {'valid': 'full', 'test': 'full'})
    
    return cfg


def load_hydra_config(
    config_dir: Path,
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load config using Hydra with composition and overrides.
    
    Args:
        config_dir: Directory containing config files
        config_name: Name of base config file (without .yaml)
        overrides: List of override strings (e.g., ["model=sasrec", "epochs=10"])
    
    Returns:
        Resolved config dict
    """
    # Hydra defaults handle model group loading
    # Just pass overrides as-is
    overrides_list = overrides or []
    
    # Clean up any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.3")
    
    # Compose config with overrides
    cfg = compose(config_name=config_name, overrides=overrides_list)
    validate_feature_mode(cfg)
    
    # Clean up
    GlobalHydra.instance().clear()
    
    return OmegaConf.to_container(cfg, resolve=True)


def expand_search_space(cfg: Dict[str, Any], max_configs: int = 200) -> List[Dict[str, Any]]:
    """Expand grid search configurations with optional random sampling.
    
    Args:
        cfg: Config dict with optional 'search' block
        max_configs: Maximum number of configs to return. If total combinations
                     exceed this, randomly sample max_configs combinations.
    """
    import random
    
    if "search" not in cfg:
        return [cfg]

    search = cfg["search"]
    keys = []
    values = []
    for k, v in search.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)

    if not keys:
        return [cfg]

    # Calculate total combinations
    all_combos = list(itertools.product(*values))
    total = len(all_combos)
    
    # Random sample if exceeds max_configs
    if total > max_configs:
        print(f"Random search: {total} combinations -> sampling {max_configs}")
        all_combos = random.sample(all_combos, max_configs)
    
    configs = []
    for combo in all_combos:
        new_cfg = copy.deepcopy(cfg)
        new_cfg.pop("search", None)
        for k, v in zip(keys, combo):
            new_cfg[k] = v
        configs.append(new_cfg)
    return configs


def normalize_search_stages(cfg: Dict[str, Any], max_configs: int) -> List[Dict[str, Any]]:
    """Normalize staged search settings for successive halving.

    Each stage can override:
      - max_configs: cap on sampled configs for the stage
      - epochs: short budget for early stages
      - stopping_step: early-stopping patience for the stage
    - top_ratio or top_k: how many configs to keep for next stage
    - min_keep: minimum remaining options per parameter during narrowing
    - drop_min/drop_max: range of options to drop per parameter
    - drop_gap: minimum score gap to justify dropping more than drop_min
    """
    stages = cfg.get("search_stages") or []
    if not isinstance(stages, list) or not stages:
        return []

    normalized = []
    for stage in stages:
        if stage is None:
            continue
        stage = dict(stage)
        stage_max = stage.get("max_configs", max_configs)
        stage_epochs = stage.get("epochs", cfg.get("epochs"))
        stage_stopping = stage.get("stopping_step", None)
        top_ratio = stage.get("top_ratio", None)
        top_k = stage.get("top_k", None)
        window = stage.get("window", None)
        cat_keep = stage.get("cat_keep", None)
        min_keep = stage.get("min_keep", None)
        drop_min = stage.get("drop_min", None)
        drop_max = stage.get("drop_max", None)
        drop_gap = stage.get("drop_gap", None)
        normalized.append(
            {
                "max_configs": int(stage_max) if stage_max is not None else max_configs,
                "epochs": int(stage_epochs) if stage_epochs is not None else cfg.get("epochs"),
                "stopping_step": stage_stopping,
                "top_ratio": top_ratio,
                "top_k": top_k,
                "window": window,
                "cat_keep": cat_keep,
                "min_keep": min_keep,
                "drop_min": drop_min,
                "drop_max": drop_max,
                "drop_gap": drop_gap,
            }
        )
    return normalized


def narrow_search_space(
    search: Dict[str, Any],
    stage_results: List[tuple],
    window: Optional[int] = None,
    cat_keep: Optional[int] = None,
    min_keep: int = 2,
    drop_min: int = 1,
    drop_max: int = 2,
    drop_gap: float = 0.002,
) -> Dict[str, Any]:
    """Narrow search space based on stage performance.

    Default behavior: drop weak options per parameter with safeguards.
    - Keep at least min_keep options per parameter.
    - Drop between drop_min and drop_max options per stage.
    - If the best gap at a cutoff is smaller than drop_gap, drop_min is used.

    Legacy behavior (if drop_min/drop_max are None):
    - Numeric lists: keep a window around the median of top configs.
    - Categorical lists: keep most frequent values from top configs.
    """
    if not search or not stage_results:
        return search

    use_drop = drop_min is not None and drop_max is not None

    if not use_drop:
        top_configs = [cfg for _, cfg in stage_results]
        narrowed = {}
        for key, values in search.items():
            if not isinstance(values, list) or len(values) == 0:
                narrowed[key] = values
                continue

            top_vals = [cfg.get(key, None) for cfg in top_configs if key in cfg]
            top_vals = [v for v in top_vals if v is not None]
            if not top_vals:
                narrowed[key] = values
                continue

            if all(isinstance(v, (int, float)) for v in values) and all(
                isinstance(v, (int, float)) for v in top_vals
            ):
                sorted_vals = sorted(values)
                top_vals_sorted = sorted(top_vals)
                median = top_vals_sorted[len(top_vals_sorted) // 2]
                closest_idx = min(range(len(sorted_vals)), key=lambda i: abs(sorted_vals[i] - median))
                half = max(1, (window or 3) // 2)
                start = max(0, closest_idx - half)
                end = min(len(sorted_vals), closest_idx + half + 1)
                narrowed[key] = sorted_vals[start:end]
            else:
                freq = {}
                for v in top_vals:
                    freq[v] = freq.get(v, 0) + 1
                ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                keep = [v for v, _ in ranked[: max(1, (cat_keep or 2))]]
                narrowed[key] = [v for v in values if v in keep]
                if not narrowed[key]:
                    narrowed[key] = values
        return narrowed

    # Drop-based narrowing
    narrowed = {}
    score_map = []
    for score, cfg in stage_results:
        score_map.append((float(score) if score is not None else float("-inf"), cfg))

    for key, values in search.items():
        if not isinstance(values, list) or len(values) == 0:
            narrowed[key] = values
            continue

        if len(values) <= min_keep:
            narrowed[key] = values
            continue

        # Mean score per option
        value_scores: Dict[Any, List[float]] = {v: [] for v in values}
        for score, cfg in score_map:
            if key in cfg:
                v = cfg.get(key)
                if v in value_scores:
                    value_scores[v].append(score)

        mean_scores = {}
        for v in values:
            scores = value_scores.get(v, [])
            if scores:
                mean_scores[v] = sum(scores) / len(scores)
            else:
                mean_scores[v] = float("-inf")

        # Sort by score desc; stabilize by original order
        ranked = sorted(values, key=lambda v: (mean_scores[v], -values.index(v)), reverse=True)

        max_drop_allowed = min(drop_max, len(values) - min_keep)
        min_drop_allowed = min(drop_min, max_drop_allowed)
        if max_drop_allowed <= 0:
            narrowed[key] = values
            continue

        best_drop = min_drop_allowed
        best_gap = float("-inf")
        for drop_n in range(min_drop_allowed, max_drop_allowed + 1):
            keep_n = len(values) - drop_n
            if keep_n < min_keep or keep_n >= len(ranked):
                continue
            gap = mean_scores[ranked[keep_n - 1]] - mean_scores[ranked[keep_n]]
            if gap > best_gap:
                best_gap = gap
                best_drop = drop_n

        if best_gap < drop_gap:
            best_drop = min_drop_allowed

        keep_n = len(values) - best_drop
        keep_set = set(ranked[:keep_n])
        narrowed[key] = [v for v in values if v in keep_set]

    return narrowed
