"""Shared training and evaluation entry helpers for RouteRec scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import yaml

from .continuation import (
    FIXED_BUDGET_POLICY,
    ContinuationError,
    load_continuation_state,
    save_continuation_state,
)
from .datasets import infer_recbole_dataset_config, normalize_dataset_name
from .lr_scheduler import build_lr_scheduler, current_learning_rates
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


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML artifact without exposing a partially written file."""

    _ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=True, allow_unicode=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _metric_value(result: dict[str, Any], key: str) -> float:
    normalized = {str(name).lower(): value for name, value in result.items()}
    lookup = str(key).lower()
    if lookup not in normalized:
        raise KeyError(f"Validation result is missing required metric {key!r}; got {sorted(normalized)}")
    value = float(normalized[lookup])
    if not math.isfinite(value):
        raise ValueError(f"Validation metric {key!r} is non-finite: {value}")
    return value


def _config_value(config: Any, key: str, default: Any) -> Any:
    """Read RecBole config keys without treating its missing-key None as a value."""

    try:
        value = config[key]
    except Exception:
        value = default
    return default if value is None else value


def _dataset_from_loader(data_loader: Any) -> Any:
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None:
        dataset = getattr(data_loader, "_dataset", None)
    if dataset is None:
        raise RuntimeError("Could not locate RecBole dataset on data loader")
    return dataset


def _train_item_mask(train_data: Any, *, n_items: int, iid_field: str, list_suffix: str) -> Any:
    import torch

    dataset = _dataset_from_loader(train_data)
    interaction = dataset.inter_feat
    mask = torch.zeros(int(n_items), dtype=torch.bool)
    for field in (iid_field, iid_field + list_suffix):
        if field not in interaction.columns:
            continue
        values = interaction[field]
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        values = values.detach().long().cpu().reshape(-1)
        values = values[(values > 0) & (values < int(n_items))]
        if values.numel() > 0:
            mask[values.unique()] = True
    if not bool(mask.any()):
        raise RuntimeError("Train-seen candidate mask is empty")
    return mask


def _hash_tensor(hasher: Any, tensor: Any, *, chunk_rows: int = 8192) -> None:
    import torch

    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().cpu()
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(json.dumps(list(tensor.shape)).encode("ascii"))
    if tensor.ndim == 0:
        hasher.update(tensor.contiguous().numpy().tobytes())
        return
    for start in range(0, int(tensor.shape[0]), chunk_rows):
        hasher.update(tensor[start : start + chunk_rows].contiguous().numpy().tobytes())


def split_fingerprint(data_loader: Any, *, iid_field: str, uid_field: str, time_field: str) -> dict[str, Any]:
    """Fingerprint evaluation sample identity separately from sequence content."""
    dataset = _dataset_from_loader(data_loader)
    interaction = dataset.inter_feat
    key_hasher = hashlib.sha256()
    content_hasher = hashlib.sha256()
    key_fields = [field for field in (uid_field, iid_field, time_field) if field in interaction.columns]
    content_fields = list(key_fields)
    item_list_field = getattr(dataset, "item_id_list_field", iid_field + "_list")
    length_field = getattr(dataset, "item_list_length_field", "item_length")
    for field in (length_field, item_list_field):
        if field in interaction.columns and field not in content_fields:
            content_fields.append(field)
    for field in key_fields:
        key_hasher.update(field.encode("utf-8"))
        _hash_tensor(key_hasher, interaction[field])
    for field in content_fields:
        content_hasher.update(field.encode("utf-8"))
        _hash_tensor(content_hasher, interaction[field])
    return {
        "rows": int(len(interaction)),
        "sample_key_fields": key_fields,
        "sample_key_sha256": key_hasher.hexdigest(),
        "sequence_fields": content_fields,
        "sequence_sha256": content_hasher.hexdigest(),
    }


def _parameter_counts(model: Any) -> dict[str, int | str]:
    named = list(model.named_parameters())
    total = int(sum(parameter.numel() for _, parameter in named))
    item_embedding = int(
        sum(
            parameter.numel()
            for name, parameter in named
            if name == "item_embedding.weight" or ".item_embedding.weight" in name
        )
    )
    non_item = total - item_embedding
    logical_active_non_item = non_item
    executed_non_item = non_item
    active_rule = "all parameters execute"

    # RouteRec's factorized top-3 x top-2 router gives six experts non-zero
    # mathematical weight per sample.  The current dense PyTorch realization
    # still executes every expert MLP; record both facts instead of conflating
    # sparse influence with wall-clock sparse dispatch.
    stage_executor = getattr(model, "stage_executor", None)
    stage_blocks = getattr(stage_executor, "stage_blocks", None)
    if stage_blocks is not None:
        logical_active_non_item = non_item
        logical_expert_total = 0
        logical_expert_active = 0
        for block in stage_blocks.values():
            experts = getattr(block, "experts", None)
            if experts is None:
                continue
            per_expert = [int(sum(p.numel() for p in expert.parameters())) for expert in experts]
            if not per_expert:
                continue
            logical_expert_total += sum(per_expert)
            primitive_specs = getattr(block, "primitive_specs", {}) or {}
            e_spec = primitive_specs.get("e_scalar")
            d_spec = primitive_specs.get("d_cond")
            e_top_k = int(getattr(e_spec, "top_k", 0) or 0)
            d_top_k = int(getattr(d_spec, "top_k", 0) or 0)
            active_experts = e_top_k * d_top_k if e_top_k and d_top_k else len(per_expert)
            active_experts = min(len(per_expert), max(active_experts, 1))
            logical_expert_active += sum(per_expert[:active_experts])
        if logical_expert_total:
            logical_active_non_item = non_item - logical_expert_total + logical_expert_active
            active_rule = "factorized top-3 x top-2 logical support; dense all-expert execution"

    return {
        "total": total,
        "item_embedding": item_embedding,
        "non_item": non_item,
        "logical_active_non_item": int(logical_active_non_item),
        "executed_non_item": int(executed_non_item),
        "active_rule": active_rule,
    }


def _measure_efficiency_probe(trainer: Any, valid_data: Any, config: Any) -> dict[str, Any]:
    import torch

    enabled = bool(_config_value(config, "routerec_efficiency_enabled", False))
    if not enabled:
        return {"enabled": False}
    warmup = int(_config_value(config, "routerec_efficiency_warmup_batches", 2))
    repetitions = int(_config_value(config, "routerec_efficiency_measure_batches", 5))
    profile_flops = bool(_config_value(config, "routerec_efficiency_profile_flops", True))
    if repetitions < 1 or warmup < 0:
        raise ValueError("invalid efficiency probe repetitions")

    batch = next(iter(valid_data))
    previous_stats = getattr(trainer, "_routerec_current_filter_stats", None)
    trainer._routerec_current_filter_stats = {
        "total_targets": 0,
        "seen_targets": 0,
        "unseen_targets": 0,
        "dropped_rows": 0,
    }
    trainer.model.eval()
    try:
        with torch.no_grad():
            for _ in range(warmup):
                trainer._full_sort_batch_eval(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            last_scores = None
            for _ in range(repetitions):
                _, last_scores, _, _ = trainer._full_sort_batch_eval(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            inference_peak = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )

        if last_scores is None:
            raise RuntimeError("efficiency probe did not produce scores")
        queries_per_batch = int(last_scores.size(0))
        candidate_items = int(last_scores.size(1))
        profiled_flops = 0
        if profile_flops:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.no_grad(), torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
            ) as prof:
                trainer._full_sort_batch_eval(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profiled_flops = int(
                sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
            )
    finally:
        trainer._routerec_current_filter_stats = previous_stats
    return {
        "enabled": True,
        "scope": "first validation full-sort batch including host-to-device and candidate masking",
        "warmup_batches": warmup,
        "measure_batches": repetitions,
        "queries_per_batch": queries_per_batch,
        "candidate_items": candidate_items,
        "mean_batch_seconds": elapsed / repetitions,
        "query_throughput_per_second": (
            queries_per_batch * repetitions / elapsed if elapsed > 0 else 0.0
        ),
        "mean_query_latency_ms_amortized": (
            elapsed * 1000.0 / (queries_per_batch * repetitions)
            if queries_per_batch > 0
            else 0.0
        ),
        "inference_peak_cuda_memory_bytes": inference_peak,
        "profiled_forward_flops_supported_ops": profiled_flops,
        "flop_note": "torch.profiler supported-operator count; not a complete algorithmic FLOP proof",
    }


def _build_paper_trainer(base_cls: type, metric_keys: tuple[str, ...]) -> type:
    """Add composite selection, train-seen masking, and RouteRec schedules."""

    class PaperTrainer(base_cls):  # type: ignore[misc, valid-type]
        _routerec_metric_keys = metric_keys

        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            self._routerec_epoch = int(epoch_idx)
            optimizer = getattr(self, "optimizer", None)
            self._routerec_last_learning_rates = (
                current_learning_rates(optimizer) if optimizer is not None else []
            )
            if hasattr(self.model, "set_schedule_epoch"):
                schedule_total = int(
                    _config_value(
                        self.config,
                        "routerec_schedule_total_epochs",
                        self.config["epochs"],
                    )
                )
                self.model.set_schedule_epoch(int(epoch_idx), schedule_total)
            started = time.perf_counter()
            loss = super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=loss_func,
                show_progress=show_progress,
            )
            self._routerec_last_train_seconds = time.perf_counter() - started
            self._routerec_last_train_loss = _jsonable(loss)
            return loss

        def _valid_epoch(self, valid_data, show_progress=False):
            self._routerec_eval_split = "valid"
            self._routerec_current_filter_stats = {
                "total_targets": 0,
                "seen_targets": 0,
                "unseen_targets": 0,
                "dropped_rows": 0,
            }
            started = time.perf_counter()
            valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
            values = [_metric_value(valid_result, key) for key in self._routerec_metric_keys]
            objective = float(sum(values) / len(values))
            epoch = int(getattr(self, "_routerec_epoch", -1))
            record = {
                "epoch": epoch,
                "train_loss": getattr(self, "_routerec_last_train_loss", None),
                "train_seconds": getattr(self, "_routerec_last_train_seconds", None),
                "learning_rates": list(
                    getattr(self, "_routerec_last_learning_rates", [])
                ),
                "valid_seconds": time.perf_counter() - started,
                "validation_objective": objective,
                "validation_metrics": _jsonable(valid_result),
                "candidate_filter": dict(self._routerec_current_filter_stats),
            }
            best = float(getattr(self, "_routerec_observed_best", float("-inf")))
            if objective > best:
                self._routerec_observed_best = objective
                self._routerec_best_epoch = epoch
            lr_scheduler = getattr(self, "_routerec_lr_scheduler", None)
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer = getattr(self, "optimizer", None)
            record["next_learning_rates"] = (
                current_learning_rates(optimizer) if optimizer is not None else []
            )
            history = getattr(self, "_routerec_history", None)
            if not isinstance(history, list):
                history = []
                self._routerec_history = history
            history.append(record)
            history_path = getattr(self, "_routerec_history_path", None)
            if history_path:
                with Path(history_path).open("a", encoding="utf-8", buffering=1) as stream:
                    stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            return objective, valid_result

        def _full_sort_batch_eval(self, batched_data):
            import torch

            interaction, scores, positive_u, positive_i = super()._full_sort_batch_eval(batched_data)
            allowed_cpu = getattr(self, "_routerec_train_item_mask", None)
            if allowed_cpu is None:
                raise RuntimeError("Train-seen candidate mask was not initialized")
            if int(allowed_cpu.numel()) != int(scores.size(1)):
                raise RuntimeError(
                    f"Candidate mask size {allowed_cpu.numel()} != score width {scores.size(1)}"
                )
            allowed = allowed_cpu.to(device=scores.device)
            scores[:, ~allowed] = float("-inf")
            # RecBole keeps ``positive_u``/``positive_i`` on CPU even when
            # full-sort scores live on CUDA.  Candidate masking belongs on the
            # score device, while row/positive bookkeeping stays on the index
            # tensors' device.  Moving the positive ids into ``allowed`` caused
            # a real cuda:0/cpu mismatch in the four-GPU preflight.
            positive_i_cpu = positive_i.detach().cpu().long()
            seen_cpu = allowed_cpu.index_select(0, positive_i_cpu)
            stats = getattr(self, "_routerec_current_filter_stats", None)
            if isinstance(stats, dict):
                total = int(positive_i.numel())
                seen_count = int(seen_cpu.sum().item())
                stats["total_targets"] += total
                stats["seen_targets"] += seen_count
                stats["unseen_targets"] += total - seen_count
            if bool(seen_cpu.all()):
                return interaction, scores, positive_u, positive_i

            keep_positive = seen_cpu.nonzero(as_tuple=False).reshape(-1)
            if keep_positive.numel() == 0:
                empty_scores = scores.new_empty((0, scores.size(1)))
                empty_index = positive_i.new_empty((0,), dtype=positive_i.dtype)
                if isinstance(stats, dict):
                    stats["dropped_rows"] += int(scores.size(0))
                return interaction[[]], empty_scores, empty_index, empty_index

            kept_positive_u = positive_u.index_select(
                0, keep_positive.to(device=positive_u.device)
            ).long()
            kept_positive_i = positive_i.index_select(
                0, keep_positive.to(device=positive_i.device)
            ).long()
            row_mask = torch.zeros(
                scores.size(0), dtype=torch.bool, device=positive_u.device
            )
            row_mask[kept_positive_u] = True
            kept_rows = row_mask.nonzero(as_tuple=False).reshape(-1)
            if isinstance(stats, dict):
                stats["dropped_rows"] += int(scores.size(0) - kept_rows.numel())
            remap = positive_u.new_full((scores.size(0),), -1, dtype=positive_u.dtype)
            remap[kept_rows] = torch.arange(
                kept_rows.numel(), device=positive_u.device, dtype=positive_u.dtype
            )
            return (
                interaction[kept_rows.detach().cpu()],
                scores.index_select(0, kept_rows.to(device=scores.device)),
                remap.index_select(0, kept_positive_u),
                kept_positive_i,
            )

    PaperTrainer.__name__ = f"Paper{base_cls.__name__}"
    PaperTrainer.__qualname__ = PaperTrainer.__name__
    return PaperTrainer


def _bind_isolated_cuda_config(config: dict[str, Any]) -> None:
    """Keep RecBole from replacing the scheduler's physical GPU binding."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return
    if "," in visible:
        raise RuntimeError(
            "A RouteRec worker must receive exactly one CUDA_VISIBLE_DEVICES token"
        )
    # RecBole Config._init_device writes gpu_id back to CUDA_VISIBLE_DEVICES.
    # Passing logical zero here would therefore redirect every isolated worker
    # to physical GPU 0. Preserve the scheduler-provided physical token; it
    # still becomes logical cuda:0 after CUDA applies the visibility mask.
    config["gpu_id"] = visible
    config["use_gpu"] = True


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
            config = deep_merge(config, dataset_preset)

    config = deep_merge(config, overrides)
    config = deep_merge(
        config,
        infer_recbole_dataset_config(dataset=dataset_name, data_path=data_path),
    )
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
    run_dir: Path | None = None,
    evaluate_test: bool = True,
    validation_metric_keys: tuple[str, ...] = ("Hit@10", "NDCG@10", "MRR@10"),
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train one isolated run under the paper validation protocol.

    Ordinary HPO callers must set ``evaluate_test=False``.  Explicit
    test-aware exploratory search may retain test metrics, but its proposer
    and promoter must still consume validation only.  A unique ``run_dir`` is
    strongly recommended; it owns the checkpoint and epoch history and avoids
    RecBole's second-resolution checkpoint-name collision.
    """
    # RecBole 1.2.1 saves its Config object in checkpoints and calls
    # ``torch.load`` without an explicit mode.  PyTorch 2.6 changed that
    # default to weights-only, which cannot reconstruct RecBole's Config.
    # These checkpoints are created locally by this worker and are trusted.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    enable_custom_model_resolver()

    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    if not validation_metric_keys:
        raise ValueError("validation_metric_keys must not be empty")
    if evaluate_test and not save_model:
        raise ValueError("Test evaluation requires the best validation checkpoint (save_model=True)")

    effective_config = dict(config_dict)
    if save_model and run_dir is None:
        raise ValueError("save_model=True requires an isolated run_dir")
    if not bool(effective_config.get("routerec_frozen_split")):
        raise ValueError(
            "Paper training requires routerec_frozen_split=true; refusing to re-split the combined .inter file"
        )
    if str(effective_config.get("USER_ID_FIELD", "")) != "session_id":
        raise ValueError("Paper training requires USER_ID_FIELD=session_id")
    if list(effective_config.get("benchmark_filename") or []) != ["train", "valid", "test"]:
        raise ValueError("Paper training requires frozen benchmark_filename=[train, valid, test]")
    if effective_config.get("routerec_mask_non_train_items") is False:
        raise ValueError("Paper training cannot disable train-seen candidate masking")
    if effective_config.get("routerec_exclude_unseen_targets") is False:
        raise ValueError("Paper training cannot include unseen positive targets in the main metric denominator")
    if run_dir is not None:
        run_dir = Path(run_dir).resolve()
        _ensure_dir(run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        _ensure_dir(checkpoint_dir)
        effective_config["checkpoint_dir"] = str(checkpoint_dir)
        cache_parent = run_dir.parents[3] if len(run_dir.parents) > 3 else run_dir.parent
        effective_config.setdefault("routerec_split_cache_dir", str(cache_parent / "cache" / "splits"))
        effective_config["save_dataset"] = False
        effective_config["save_dataloaders"] = False
    effective_config["metrics"] = ["Hit", "NDCG", "MRR"]
    effective_config["topk"] = sorted({10, *[int(str(key).split("@", 1)[1]) for key in validation_metric_keys]})
    # RecBole still requires a built-in valid metric during Config parsing;
    # PaperTrainer replaces its checkpoint objective with the composite.
    effective_config["valid_metric"] = "MRR@10"
    effective_config["valid_metric_bigger"] = True
    effective_config["eval_args"] = deep_merge(
        dict(effective_config.get("eval_args") or {}),
        {"mode": {"valid": "full", "test": "full"}},
    )
    _bind_isolated_cuda_config(effective_config)

    recbole_cfg = Config(model=model, dataset=dataset, config_dict=effective_config)
    init_seed(recbole_cfg["seed"], recbole_cfg["reproducibility"])
    init_logger(recbole_cfg)

    if not bool(recbole_cfg["use_gpu"]) or not torch.cuda.is_available():
        raise RuntimeError("Paper training worker unexpectedly resolved use_gpu=false")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "An isolated worker must see exactly one logical CUDA device; "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}, count={torch.cuda.device_count()}"
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset_started = time.perf_counter()
    dataset_obj = create_dataset(recbole_cfg)
    dataset_seconds = time.perf_counter() - dataset_started
    preparation_started = time.perf_counter()
    train_data, valid_data, test_data = data_preparation(recbole_cfg, dataset_obj)
    data_preparation_seconds = time.perf_counter() - preparation_started
    if run_dir is not None:
        _atomic_write_yaml(
            run_dir / "dataset_prepared.yaml",
            {
                "schema_version": 1,
                "dataset": dataset,
                "dataset_creation_seconds": float(dataset_seconds),
                "data_preparation_seconds": float(data_preparation_seconds),
                "train_rows": int(len(_dataset_from_loader(train_data).inter_feat)),
                "valid_rows": int(len(_dataset_from_loader(valid_data).inter_feat)),
                "test_rows": int(len(_dataset_from_loader(test_data).inter_feat)),
            },
        )

    model_started = time.perf_counter()
    model_cls = get_model(recbole_cfg["model"])
    # RecBole's normal quick-start path constructs sequential recommenders
    # against train_data.dataset.  Custom contrastive baselines inspect the
    # generated ITEM_SEQ fields during __init__, which do not exist on the
    # pre-split dataset object.
    model_obj = model_cls(recbole_cfg, _dataset_from_loader(train_data)).to(
        recbole_cfg["device"]
    )
    model_initialization_seconds = time.perf_counter() - model_started
    trainer_started = time.perf_counter()
    base_trainer_cls = get_trainer(recbole_cfg["MODEL_TYPE"], recbole_cfg["model"])
    trainer_cls = _build_paper_trainer(base_trainer_cls, tuple(validation_metric_keys))
    trainer = trainer_cls(recbole_cfg, model_obj)
    lr_scheduler, lr_scheduler_spec = build_lr_scheduler(
        trainer.optimizer,
        effective_config,
    )
    trainer._routerec_lr_scheduler = lr_scheduler
    trainer._routerec_lr_scheduler_spec = lr_scheduler_spec
    trainer._routerec_train_item_mask = _train_item_mask(
        train_data,
        n_items=int(dataset_obj.item_num),
        iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
        list_suffix=str(recbole_cfg["LIST_SUFFIX"]),
    )
    if run_dir is not None:
        trainer._routerec_history_path = str(run_dir / "history.jsonl")
    continuation_spec = effective_config.get("routerec_continuation")
    continuation_runtime = effective_config.get("routerec_continuation_runtime")
    continuation_resume = None
    if continuation_spec is not None:
        if not isinstance(continuation_runtime, dict):
            raise ContinuationError(
                "continuation scientific config requires a validated runtime binding"
            )
        if str(effective_config.get("routerec_early_stopping_policy")) != FIXED_BUDGET_POLICY:
            raise ContinuationError("exact continuation requires fixed-budget early stopping policy")
        schedule_total = int(effective_config["routerec_schedule_total_epochs"])
        if int(effective_config["stopping_step"]) < schedule_total:
            raise ContinuationError("stopping_step is too small for fixed-budget continuation")
        if continuation_runtime.get("mode") == "resume":
            if run_dir is None:
                raise ContinuationError("resume requires an isolated run directory")
            continuation_resume = load_continuation_state(
                trainer=trainer,
                loaders={"train": train_data, "valid": valid_data, "test": test_data},
                state_path=Path(str(continuation_runtime["state_path"])),
                state_sha256=str(continuation_runtime["state_sha256"]),
                parent_best_checkpoint=Path(
                    str(continuation_runtime["best_checkpoint_path"])
                ),
                parent_best_sha256=str(
                    continuation_runtime["best_checkpoint_sha256"]
                ),
                inherited_best_output=run_dir / "checkpoints" / "inherited_best.pth",
                expected_contract=dict(continuation_runtime["contract"]),
            )
            history_path = Path(trainer._routerec_history_path)
            if history_path.exists():
                raise ContinuationError("child history path already exists before resume")
            temporary = history_path.with_name(
                f".{history_path.name}.{os.getpid()}.continuation.tmp"
            )
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                for row in trainer._routerec_history:
                    stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, history_path)
            temporary.unlink()
        elif continuation_runtime.get("mode") not in {"fresh", "reference"}:
            raise ContinuationError("validated continuation runtime mode is invalid")
    trainer_initialization_seconds = time.perf_counter() - trainer_started

    segment_start_epoch = int(getattr(trainer, "start_epoch", 0))
    started = time.perf_counter()
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        verbose=bool(show_progress),
        saved=bool(save_model),
        show_progress=bool(show_progress),
    )
    train_elapsed = time.perf_counter() - started
    continuation_result = None
    if continuation_spec is not None:
        completed_epoch = int(getattr(trainer, "_routerec_epoch", -1))
        expected_completed_epoch = int(recbole_cfg["epochs"]) - 1
        if completed_epoch != expected_completed_epoch:
            raise ContinuationError(
                "fixed-budget segment ended early: "
                f"completed epoch {completed_epoch}, expected {expected_completed_epoch}"
            )
        if run_dir is None:
            raise ContinuationError("continuation output requires an isolated run directory")
        continuation_result = save_continuation_state(
            trainer=trainer,
            loaders={"train": train_data, "valid": valid_data, "test": test_data},
            output=(
                run_dir
                / "continuations"
                / f"epoch_{completed_epoch:06d}.pth"
            ),
            contract=dict(continuation_runtime["contract"]),
        )
        continuation_result["mode"] = str(continuation_runtime["mode"])
        continuation_result["segment_start_epoch"] = segment_start_epoch
        continuation_result["segment_end_epoch"] = completed_epoch
        continuation_result["segment_epochs"] = completed_epoch - segment_start_epoch + 1
        continuation_result["resume"] = continuation_resume
    training_peak_cuda_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    efficiency = _measure_efficiency_probe(trainer, valid_data, recbole_cfg)
    test_result = None
    test_filter_stats = None
    if evaluate_test:
        trainer._routerec_eval_split = "test"
        trainer._routerec_current_filter_stats = {
            "total_targets": 0,
            "seen_targets": 0,
            "unseen_targets": 0,
            "dropped_rows": 0,
        }
        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            show_progress=bool(show_progress),
        )
        test_filter_stats = dict(trainer._routerec_current_filter_stats)

    split_fingerprints = {
        "train": split_fingerprint(
            train_data,
            iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
            uid_field=str(recbole_cfg["USER_ID_FIELD"]),
            time_field=str(recbole_cfg["TIME_FIELD"]),
        ),
        "valid": split_fingerprint(
            valid_data,
            iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
            uid_field=str(recbole_cfg["USER_ID_FIELD"]),
            time_field=str(recbole_cfg["TIME_FIELD"]),
        ),
        "test": split_fingerprint(
            test_data,
            iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
            uid_field=str(recbole_cfg["USER_ID_FIELD"]),
            time_field=str(recbole_cfg["TIME_FIELD"]),
        ),
    }

    checkpoint_path = str(getattr(trainer, "saved_model_file", "") or "")
    if save_model and not Path(checkpoint_path).is_file():
        raise RuntimeError(f"Best checkpoint was not created: {checkpoint_path}")
    if save_model and run_dir is not None:
        resolved_checkpoint = Path(checkpoint_path).resolve()
        try:
            resolved_checkpoint.relative_to((run_dir / "checkpoints").resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Checkpoint escaped the isolated attempt directory: {resolved_checkpoint}"
            ) from exc
    peak_cuda_bytes = training_peak_cuda_bytes

    parameter_counts = _parameter_counts(model_obj)
    history = _jsonable(getattr(trainer, "_routerec_history", []))
    train_examples = int(len(_dataset_from_loader(train_data).inter_feat))
    epoch_train_seconds = [
        float(row["train_seconds"])
        for row in history
        if isinstance(row, dict) and isinstance(row.get("train_seconds"), (int, float))
    ]
    measured_epoch_seconds = float(sum(epoch_train_seconds))
    result = {
        "model": model,
        "dataset": dataset,
        "seed": int(recbole_cfg["seed"]),
        "selection": {
            "name": "mean_at_10",
            "metric_keys": list(validation_metric_keys),
            "objective": float(best_valid_score),
        },
        "best_valid_score": float(best_valid_score),
        "best_valid_result": _jsonable(best_valid_result),
        "best_epoch": int(getattr(trainer, "_routerec_best_epoch", -1)),
        "test_result": _jsonable(test_result),
        "test_evaluated": bool(evaluate_test),
        "test_candidate_filter": test_filter_stats,
        "checkpoint_path": checkpoint_path,
        "continuation": continuation_result,
        "model_initialization_report": _jsonable(
            getattr(model_obj, "fame_pretrain_report", None)
        ),
        "device": str(recbole_cfg["device"]),
        "physical_gpu_id": os.environ.get("ROUTEREC_PHYSICAL_GPU_ID"),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameter_count": int(parameter_counts["total"]),
        "parameter_counts": parameter_counts,
        "train_seconds": train_elapsed,
        "peak_cuda_memory_bytes": peak_cuda_bytes,
        "runtime_breakdown": {
            "dataset_creation_seconds": dataset_seconds,
            "data_preparation_seconds": data_preparation_seconds,
            "model_initialization_seconds": model_initialization_seconds,
            "trainer_initialization_seconds": trainer_initialization_seconds,
            "fit_seconds": train_elapsed,
            "train_examples_per_epoch": train_examples,
            "measured_epoch_train_seconds": measured_epoch_seconds,
            "train_examples_per_second": (
                train_examples * len(epoch_train_seconds) / measured_epoch_seconds
                if measured_epoch_seconds > 0
                else 0.0
            ),
        },
        "efficiency": efficiency,
        "split_fingerprints": split_fingerprints,
        "resolved_config": _jsonable(dict(recbole_cfg.final_config_dict)),
        "history": history,
    }
    if run_dir is not None:
        _atomic_write_yaml(run_dir / "resolved_config.yaml", result["resolved_config"])
    return result


def run_test_only(
    *,
    dataset: str,
    model: str,
    config_dict: dict[str, Any],
    checkpoint_path: str,
    strict_checkpoint: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    enable_custom_model_resolver()

    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    effective_config = dict(config_dict)
    # A checkpoint stores RecBole's *final* Config, including values derived
    # during the first Config construction.  Those fields are not valid input
    # to a second construction: merely retaining `local_rank` switches RecBole
    # into distributed initialization, and its normalized CE negative-sampling
    # dictionary is rejected when fed back as user config.
    for derived_key in (
        "MODEL_TYPE",
        "MODEL_INPUT_TYPE",
        "device",
        "single_spec",
        "local_rank",
        "nproc",
        "world_size",
        "ip",
        "port",
        "offset",
    ):
        effective_config.pop(derived_key, None)
    if str(effective_config.get("loss_type", "")).upper() == "CE":
        effective_config["train_neg_sample_args"] = None
    effective_config["eval_neg_sample_args"] = None

    # Final Config also contains the already-appended dataset directory, while
    # Config.__init__ expects its parent and appends the dataset name itself.
    configured_data_path = Path(str(effective_config.get("data_path", ""))).expanduser()
    if configured_data_path.name == normalize_dataset_name(dataset):
        effective_config["data_path"] = str(configured_data_path.parent)
    effective_config["metrics"] = ["Hit", "NDCG", "MRR"]
    effective_config["topk"] = [10]
    effective_config["valid_metric"] = "MRR@10"
    effective_config["valid_metric_bigger"] = True
    effective_config["eval_args"] = deep_merge(
        dict(effective_config.get("eval_args") or {}),
        {"mode": {"valid": "full", "test": "full"}},
    )
    if not bool(effective_config.get("routerec_frozen_split")):
        raise ValueError("Test evaluation requires the frozen paper split")
    if str(effective_config.get("USER_ID_FIELD", "")) != "session_id":
        raise ValueError("Test evaluation requires USER_ID_FIELD=session_id")
    _bind_isolated_cuda_config(effective_config)

    recbole_cfg = Config(model=model, dataset=dataset, config_dict=effective_config)
    init_seed(recbole_cfg["seed"], recbole_cfg["reproducibility"])
    init_logger(recbole_cfg)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not torch.cuda.is_available():
        raise RuntimeError("Test evaluation requires CUDA")
    if visible_devices and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "An isolated test worker must see exactly one logical CUDA device; "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}, count={torch.cuda.device_count()}"
        )
    torch.cuda.reset_peak_memory_stats()

    total_started = time.perf_counter()
    dataset_started = time.perf_counter()
    dataset_obj = create_dataset(recbole_cfg)
    dataset_seconds = time.perf_counter() - dataset_started
    preparation_started = time.perf_counter()
    train_data, _, test_data = data_preparation(recbole_cfg, dataset_obj)
    data_preparation_seconds = time.perf_counter() - preparation_started

    model_started = time.perf_counter()
    model_cls = get_model(recbole_cfg["model"])
    # Match RecBole's training construction path.  Sequential/contrastive
    # models such as DuoRec and FEARec inspect ITEM_SEQ fields generated during
    # data_preparation(), which are present on train_data.dataset but not on the
    # original pre-split dataset object.
    model_obj = model_cls(recbole_cfg, _dataset_from_loader(train_data)).to(
        recbole_cfg["device"]
    )
    model_initialization_seconds = time.perf_counter() - model_started

    checkpoint_started = time.perf_counter()
    ckpt = torch.load(
        checkpoint_path,
        map_location=recbole_cfg["device"],
        weights_only=False,
    )
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        state_dict = ckpt
    model_obj.load_state_dict(state_dict, strict=bool(strict_checkpoint))
    checkpoint_load_seconds = time.perf_counter() - checkpoint_started

    trainer_started = time.perf_counter()
    base_trainer_cls = get_trainer(recbole_cfg["MODEL_TYPE"], recbole_cfg["model"])
    trainer_cls = _build_paper_trainer(base_trainer_cls, ("Hit@10", "NDCG@10", "MRR@10"))
    trainer = trainer_cls(recbole_cfg, model_obj)
    trainer._routerec_train_item_mask = _train_item_mask(
        train_data,
        n_items=int(dataset_obj.item_num),
        iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
        list_suffix=str(recbole_cfg["LIST_SUFFIX"]),
    )
    trainer._routerec_eval_split = "test"
    trainer._routerec_current_filter_stats = {
        "total_targets": 0,
        "seen_targets": 0,
        "unseen_targets": 0,
        "dropped_rows": 0,
    }
    trainer_initialization_seconds = time.perf_counter() - trainer_started
    evaluation_started = time.perf_counter()
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=bool(show_progress))
    evaluation_seconds = time.perf_counter() - evaluation_started
    total_seconds = time.perf_counter() - total_started
    test_rows = int(len(_dataset_from_loader(test_data).inter_feat))

    return {
        "model": model,
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "strict_checkpoint": bool(strict_checkpoint),
        "test_result": _jsonable(test_result),
        "test_candidate_filter": dict(trainer._routerec_current_filter_stats),
        "device": str(recbole_cfg["device"]),
        "physical_gpu_id": os.environ.get("ROUTEREC_PHYSICAL_GPU_ID"),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameter_counts": _parameter_counts(model_obj),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_breakdown": {
            "dataset_creation_seconds": dataset_seconds,
            "data_preparation_seconds": data_preparation_seconds,
            "model_initialization_seconds": model_initialization_seconds,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "trainer_initialization_seconds": trainer_initialization_seconds,
            "test_evaluation_seconds": evaluation_seconds,
            "total_seconds": total_seconds,
            "test_rows": test_rows,
            "test_rows_per_second": test_rows / evaluation_seconds if evaluation_seconds > 0 else 0.0,
        },
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "split_fingerprints": {
            "train": split_fingerprint(
                train_data,
                iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
                uid_field=str(recbole_cfg["USER_ID_FIELD"]),
                time_field=str(recbole_cfg["TIME_FIELD"]),
            ),
            "test": split_fingerprint(
                test_data,
                iid_field=str(recbole_cfg["ITEM_ID_FIELD"]),
                uid_field=str(recbole_cfg["USER_ID_FIELD"]),
                time_field=str(recbole_cfg["TIME_FIELD"]),
            ),
        },
    }


def save_json_result(output_path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(output_path.parent)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
