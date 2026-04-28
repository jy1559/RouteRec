"""Shared training and evaluation entry helpers for RouteRec scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .datasets import infer_recbole_dataset_config, normalize_dataset_name
from .runtime import enable_custom_model_resolver


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_config_dict(
    *,
    dataset: str,
    model: str,
    base_config_path: Path,
    dataset_presets_path: Path | None,
    use_dataset_preset: bool,
    overrides: dict[str, Any],
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
    data_path: str | None,
) -> dict[str, Any]:
    dataset_name = normalize_dataset_name(dataset)
    config = _read_yaml(base_config_path)
    config["model"] = model

    if use_dataset_preset and dataset_presets_path and dataset_presets_path.exists():
        preset_root = _read_yaml(dataset_presets_path)
        dataset_preset = preset_root.get(dataset_name)
        if isinstance(dataset_preset, dict):
            config.update(dataset_preset)

    config.update(overrides)
    config.update(infer_recbole_dataset_config(dataset=dataset_name, data_path=data_path))
    config.setdefault("train_neg_sample_args", None)
    config["epochs"] = int(epochs)
    config["train_batch_size"] = int(train_batch_size)
    config["eval_batch_size"] = int(eval_batch_size)
    config["seed"] = int(seed)
    config["reproducibility"] = True
    config["state"] = "INFO"
    if data_path:
        config["data_path"] = data_path
    return config


def run_training(
    *,
    dataset: str,
    model: str,
    config_dict: dict[str, Any],
    save_model: bool,
) -> dict[str, Any]:
    enable_custom_model_resolver()

    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    recbole_cfg = Config(model=model, dataset=dataset, config_dict=config_dict)
    init_seed(recbole_cfg["seed"], recbole_cfg["reproducibility"])
    init_logger(recbole_cfg)

    dataset_obj = create_dataset(recbole_cfg)
    train_data, valid_data, test_data = data_preparation(recbole_cfg, dataset_obj)

    model_cls = get_model(recbole_cfg["model"])
    model_obj = model_cls(recbole_cfg, dataset_obj).to(recbole_cfg["device"])
    trainer_cls = get_trainer(recbole_cfg["MODEL_TYPE"], recbole_cfg["model"])
    trainer = trainer_cls(recbole_cfg, model_obj)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        verbose=True,
        saved=bool(save_model),
        show_progress=True,
    )
    test_result = trainer.evaluate(test_data, load_best_model=bool(save_model), show_progress=True)

    return {
        "model": model,
        "dataset": dataset,
        "best_valid_score": float(best_valid_score),
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "checkpoint_path": str(getattr(trainer, "saved_model_file", "") or ""),
        "device": str(recbole_cfg["device"]),
    }


def run_test_only(
    *,
    dataset: str,
    model: str,
    config_dict: dict[str, Any],
    checkpoint_path: str,
) -> dict[str, Any]:
    enable_custom_model_resolver()

    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    recbole_cfg = Config(model=model, dataset=dataset, config_dict=config_dict)
    init_seed(recbole_cfg["seed"], recbole_cfg["reproducibility"])
    init_logger(recbole_cfg)

    dataset_obj = create_dataset(recbole_cfg)
    _, _, test_data = data_preparation(recbole_cfg, dataset_obj)

    model_cls = get_model(recbole_cfg["model"])
    model_obj = model_cls(recbole_cfg, dataset_obj).to(recbole_cfg["device"])

    ckpt = torch.load(checkpoint_path, map_location=recbole_cfg["device"])
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        state_dict = ckpt
    model_obj.load_state_dict(state_dict, strict=False)

    trainer_cls = get_trainer(recbole_cfg["MODEL_TYPE"], recbole_cfg["model"])
    trainer = trainer_cls(recbole_cfg, model_obj)
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

    return {
        "model": model,
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "test_result": test_result,
        "device": str(recbole_cfg["device"]),
    }


def save_json_result(output_path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
