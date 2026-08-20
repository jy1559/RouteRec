"""Exact, provenance-bound continuation for staged RouteRec training.

Validation-best checkpoints and continuation checkpoints serve different
purposes.  The former is the frozen input to post-hoc evaluation.  The latter
captures the *last completed epoch* together with optimizer, early-stopping,
data-loader, and RNG state so a promoted HPO trial can continue without
repeating earlier epochs.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import yaml


CONTINUATION_SCHEMA_VERSION = 2
SUPPORTED_CONTINUATION_SCHEMA_VERSIONS = frozenset({1, 2})
CONTINUATION_KIND = "routerec_exact_continuation"
FIXED_BUDGET_POLICY = "fixed_budget_no_early_stop"


class ContinuationError(ValueError):
    """Raised when a continuation lineage or state is incomplete or unsafe."""


def _scheduler_checkpoint(trainer: Any) -> dict[str, Any]:
    spec = _mapping(
        getattr(trainer, "_routerec_lr_scheduler_spec", None),
        "trainer LR scheduler spec",
    )
    schedule_type = str(spec.get("type", ""))
    scheduler = getattr(trainer, "_routerec_lr_scheduler", None)
    if schedule_type == "constant":
        if scheduler is not None:
            raise ContinuationError("constant LR contract unexpectedly owns a scheduler")
        state = None
    elif schedule_type == "warmup_cosine":
        if scheduler is None or not hasattr(scheduler, "state_dict"):
            raise ContinuationError("warmup_cosine scheduler is absent at continuation boundary")
        state = scheduler.state_dict()
    else:
        raise ContinuationError(f"unsupported continuation LR scheduler {schedule_type!r}")
    return {**spec, "state": state}


def _restore_scheduler_checkpoint(
    *,
    trainer: Any,
    payload: Mapping[str, Any],
    schema_version: int,
    completed_epoch: int,
    restore_state: bool,
) -> None:
    saved = _mapping(payload, "continuation scheduler")
    current = _mapping(
        getattr(trainer, "_routerec_lr_scheduler_spec", None),
        "trainer LR scheduler spec",
    )
    scheduler = getattr(trainer, "_routerec_lr_scheduler", None)
    if schema_version == 1:
        if saved != {"type": "constant", "state": None}:
            raise ContinuationError("legacy continuation has an invalid scheduler payload")
        if current.get("type") != "constant" or scheduler is not None:
            raise ContinuationError("legacy constant continuation cannot resume another scheduler")
        return

    saved_contract = {key: value for key, value in saved.items() if key != "state"}
    if saved_contract != current:
        raise ContinuationError("continuation LR scheduler contract does not match current config")
    if saved_contract.get("type") == "constant":
        if saved.get("state") is not None or scheduler is not None:
            raise ContinuationError("constant continuation must be state-free")
        return
    if saved_contract.get("type") != "warmup_cosine":
        raise ContinuationError("continuation LR scheduler type is unsupported")
    if scheduler is None or not hasattr(scheduler, "load_state_dict"):
        raise ContinuationError("current warmup_cosine scheduler is absent")
    state = _mapping(saved.get("state"), "continuation scheduler state")
    last_epoch = int(state.get("last_epoch", -1))
    expected_last_epoch = int(completed_epoch) + 1
    if last_epoch != expected_last_epoch:
        raise ContinuationError(
            "continuation scheduler epoch differs from completed training boundary: "
            f"last_epoch={last_epoch}, expected={expected_last_epoch}"
        )
    if restore_state:
        scheduler.load_state_dict(state)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationError(f"{label} must be a mapping")
    return dict(value)


def _validate_parent_test_contract(
    *,
    metrics: Mapping[str, Any],
    scientific: Mapping[str, Any],
) -> None:
    """Allow test-bearing HPO parents only under the explicit no-selection contract."""

    test_evaluated = metrics.get("test_evaluated")
    if test_evaluated is False:
        return
    run = _mapping(scientific.get("run"), "parent run config")
    evaluation = _mapping(scientific.get("evaluation"), "parent evaluation config")
    if (
        test_evaluated is not True
        or str(run.get("mode")) != "testaware_search"
        or evaluation.get("allow_test") is not True
        or metrics.get("test_used_for_selection") is not False
    ):
        raise ContinuationError(
            "test-evaluated HPO parent lacks the explicit testaware_search "
            "test_used_for_selection=false contract"
        )


def continuation_compatibility_view(scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return the semantic surface that must not drift across HPO rungs."""

    payload = _mapping(scientific, "scientific config")
    run = _mapping(payload.get("run"), "run")
    trainer = _mapping(payload.get("trainer"), "trainer")
    provenance = _mapping(payload.get("provenance"), "provenance")
    continuation = _mapping(trainer.get("continuation"), "trainer.continuation")
    return {
        "run": {
            key: run.get(key)
            for key in ("mode", "hpo_seed", "seed", "protocol_id")
        },
        "data": payload.get("data"),
        "model": payload.get("model"),
        "optimization": payload.get("optimization"),
        "trainer": {
            key: trainer.get(key)
            for key in (
                "eval_step",
                "train_batch_size",
                "eval_batch_size",
                "reproducibility",
                "grad_clip_norm",
            )
        },
        "evaluation": payload.get("evaluation"),
        "provenance": {
            key: provenance.get(key)
            for key in ("paper_contract_id", "source_id", "identity_status")
        },
        "continuation": {
            key: continuation.get(key)
            for key in (
                "schema_version",
                "budget_schedule",
                "schedule_total_epochs",
                "early_stopping_policy",
                "lineage_id",
            )
        },
    }


def continuation_compatibility_digest(scientific: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        continuation_compatibility_view(scientific),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state(loaders: Mapping[str, Any]) -> dict[str, Any]:
    """Capture all RNG surfaces that can affect the next training epoch."""

    import torch

    loader_states: dict[str, Any] = {}
    for name, loader in sorted(loaders.items()):
        generator = getattr(loader, "generator", None)
        if generator is None or not hasattr(generator, "get_state"):
            raise ContinuationError(f"{name} loader lacks a restorable torch generator")
        loader_states[str(name)] = generator.get_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": loader_states,
    }


def restore_rng_state(state: Mapping[str, Any], loaders: Mapping[str, Any]) -> None:
    """Restore RNG only after dataset/model/trainer reconstruction is complete."""

    import torch

    payload = _mapping(state, "rng state")
    expected = {"python", "numpy", "torch_cpu", "torch_cuda_all", "loaders"}
    if set(payload) != expected:
        raise ContinuationError(
            f"rng state keys differ: expected {sorted(expected)}, got {sorted(payload)}"
        )
    loader_states = _mapping(payload["loaders"], "rng state loaders")
    if set(loader_states) != {str(name) for name in loaders}:
        raise ContinuationError("continuation loader-generator set does not match current loaders")

    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    cuda_states = list(payload["torch_cuda_all"])
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ContinuationError(
                "continuation CUDA RNG device count does not match the isolated worker"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise ContinuationError("CUDA RNG state cannot be restored without CUDA")

    for name, loader in loaders.items():
        generator = getattr(loader, "generator", None)
        if generator is None or not hasattr(generator, "set_state"):
            raise ContinuationError(f"{name} loader lacks a restorable torch generator")
        generator.set_state(loader_states[str(name)])


def _atomic_torch_save_noreplace(payload: Mapping[str, Any], output: Path) -> None:
    import torch

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"continuation output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"continuation temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream, pickle_protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        parent_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy_noreplace(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ContinuationError(f"inherited best checkpoint is missing or empty: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"inherited checkpoint output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"checkpoint temporary path already exists: {temporary}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, output)
        parent_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_continuation_state(
    *,
    trainer: Any,
    loaders: Mapping[str, Any],
    output: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Save the boundary state after validation of the last completed epoch."""

    completed_epoch = int(getattr(trainer, "_routerec_epoch", -1))
    if completed_epoch < 0:
        raise ContinuationError("cannot save continuation before a completed epoch")
    best_checkpoint = Path(str(getattr(trainer, "saved_model_file", ""))).resolve()
    if not best_checkpoint.is_file():
        raise ContinuationError("validation-best checkpoint is absent at continuation boundary")
    payload = {
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "kind": CONTINUATION_KIND,
        "contract": dict(contract),
        "completed_epoch": completed_epoch,
        "start_epoch": int(getattr(trainer, "start_epoch", 0)),
        "cur_step": int(getattr(trainer, "cur_step", 0)),
        "best_valid_score": float(getattr(trainer, "best_valid_score")),
        "best_valid_result": getattr(trainer, "best_valid_result", None),
        "routerec_best_epoch": int(getattr(trainer, "_routerec_best_epoch", -1)),
        "routerec_observed_best": float(
            getattr(trainer, "_routerec_observed_best", getattr(trainer, "best_valid_score"))
        ),
        "state_dict": trainer.model.state_dict(),
        "other_parameter": trainer.model.other_parameter(),
        "optimizer": trainer.optimizer.state_dict(),
        "train_loss_dict": dict(getattr(trainer, "train_loss_dict", {})),
        "history": list(getattr(trainer, "_routerec_history", [])),
        "rng_state": capture_rng_state(loaders),
        "scheduler": _scheduler_checkpoint(trainer),
        "best_checkpoint": {
            "sha256": sha256_file(best_checkpoint),
            "size": int(best_checkpoint.stat().st_size),
            "epoch": int(getattr(trainer, "_routerec_best_epoch", -1)),
        },
    }
    _atomic_torch_save_noreplace(payload, output)
    return {
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "kind": CONTINUATION_KIND,
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "size": int(output.stat().st_size),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "best_checkpoint_sha256": payload["best_checkpoint"]["sha256"],
        "best_checkpoint_epoch": payload["best_checkpoint"]["epoch"],
        "contract": dict(contract),
    }


def load_continuation_state(
    *,
    trainer: Any,
    loaders: Mapping[str, Any],
    state_path: Path,
    state_sha256: str,
    parent_best_checkpoint: Path,
    parent_best_sha256: str,
    inherited_best_output: Path,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore a verified parent boundary into a fresh worker."""

    import torch

    state_path = state_path.resolve()
    parent_best_checkpoint = parent_best_checkpoint.resolve()
    if sha256_file(state_path) != str(state_sha256):
        raise ContinuationError("continuation state SHA-256 mismatch")
    if sha256_file(parent_best_checkpoint) != str(parent_best_sha256):
        raise ContinuationError("parent validation-best checkpoint SHA-256 mismatch")
    payload = torch.load(
        state_path,
        # The payload contains CPU RNG ByteTensors in addition to model and
        # optimizer tensors. Loading the whole pickle onto CUDA corrupts the
        # RNG-state type contract; model/optimizer loaders move their own
        # tensors to the parameter device after CPU deserialization.
        map_location="cpu",
        weights_only=False,
    )
    payload = _mapping(payload, "continuation payload")
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in SUPPORTED_CONTINUATION_SCHEMA_VERSIONS:
        raise ContinuationError("unsupported continuation schema version")
    if payload.get("kind") != CONTINUATION_KIND:
        raise ContinuationError("checkpoint is not an exact RouteRec continuation state")
    if _mapping(payload.get("contract"), "continuation contract") != dict(expected_contract):
        raise ContinuationError("continuation contract does not match the current job")
    completed_epoch = int(payload.get("completed_epoch", -1))
    if completed_epoch < 0:
        raise ContinuationError("continuation completed_epoch is invalid")
    _restore_scheduler_checkpoint(
        trainer=trainer,
        payload=_mapping(payload.get("scheduler"), "continuation scheduler"),
        schema_version=schema_version,
        completed_epoch=completed_epoch,
        restore_state=False,
    )
    best = _mapping(payload.get("best_checkpoint"), "continuation best checkpoint")
    if best.get("sha256") != str(parent_best_sha256):
        raise ContinuationError("continuation state and inherited best checkpoint disagree")

    trainer.model.load_state_dict(payload["state_dict"], strict=True)
    trainer.model.load_other_parameter(payload.get("other_parameter"))
    trainer.optimizer.load_state_dict(payload["optimizer"])
    _restore_scheduler_checkpoint(
        trainer=trainer,
        payload=_mapping(payload.get("scheduler"), "continuation scheduler"),
        schema_version=schema_version,
        completed_epoch=completed_epoch,
        restore_state=True,
    )
    trainer.start_epoch = completed_epoch + 1
    trainer.cur_step = int(payload["cur_step"])
    trainer.best_valid_score = float(payload["best_valid_score"])
    trainer.best_valid_result = payload.get("best_valid_result")
    trainer.train_loss_dict = dict(payload.get("train_loss_dict") or {})
    trainer._routerec_best_epoch = int(payload.get("routerec_best_epoch", -1))
    trainer._routerec_observed_best = float(payload.get("routerec_observed_best"))
    trainer._routerec_history = list(payload.get("history") or [])
    trainer._routerec_epoch = completed_epoch

    _atomic_copy_noreplace(parent_best_checkpoint, inherited_best_output)
    if sha256_file(inherited_best_output) != str(parent_best_sha256):
        raise ContinuationError("inherited best checkpoint copy changed bytes")
    trainer.saved_model_file = str(inherited_best_output.resolve())

    # This must be last: construction, loading, and copying are allowed to use
    # RNG, but the next training batch must see the exact parent boundary.
    restore_rng_state(_mapping(payload.get("rng_state"), "rng state"), loaders)
    return {
        "schema_version": schema_version,
        "kind": CONTINUATION_KIND,
        "parent_state_path": str(state_path),
        "parent_state_sha256": str(state_sha256),
        "parent_best_checkpoint": str(parent_best_checkpoint),
        "parent_best_checkpoint_sha256": str(parent_best_sha256),
        "completed_epoch": completed_epoch,
        "resumed_at_epoch": completed_epoch + 1,
    }


def verify_artifact_manifest(root: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify exact parent-attempt coverage before trusting a pickle artifact."""

    root = root.resolve()
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != str(expected_sha256):
        raise ContinuationError("parent artifact manifest is missing or has the wrong SHA-256")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "artifact manifest")
    records_raw = manifest.get("files")
    if not isinstance(records_raw, list):
        raise ContinuationError("parent artifact manifest files must be a list")
    records: dict[str, dict[str, Any]] = {}
    for raw in records_raw:
        record = _mapping(raw, "artifact record")
        relative = str(record.get("path", ""))
        if not relative or relative in records:
            raise ContinuationError("parent artifact manifest has an invalid or duplicate path")
        records[relative] = record
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and not path.name.endswith(".tmp")
    }
    if set(actual) != set(records):
        raise ContinuationError("parent artifact manifest does not exactly cover the attempt")
    for relative, path in actual.items():
        record = records[relative]
        if int(record.get("size", -1)) != path.stat().st_size:
            raise ContinuationError(f"parent artifact size mismatch: {relative}")
        if str(record.get("sha256", "")) != sha256_file(path):
            raise ContinuationError(f"parent artifact SHA-256 mismatch: {relative}")
    return manifest


def build_parent_binding(*, repo_root: Path, attempt: Path) -> dict[str, Any]:
    """Build the exact immutable parent block consumed by a promoted rung."""

    repo_root = repo_root.resolve()
    attempt = attempt.resolve()
    studies_root = (repo_root / "outputs" / "studies").resolve()
    try:
        relative_attempt = attempt.relative_to(repo_root).as_posix()
        attempt.relative_to(studies_root)
    except ValueError as exc:
        raise ContinuationError("parent attempt must be contained under outputs/studies") from exc
    if not (attempt / "SUCCESS").is_file():
        raise ContinuationError("parent attempt is not terminally successful")
    artifact_manifest_path = attempt / "artifact_manifest.json"
    artifact_manifest_sha256 = sha256_file(artifact_manifest_path)
    manifest = verify_artifact_manifest(attempt, artifact_manifest_sha256)
    records = {str(row["path"]): row for row in manifest["files"]}
    metrics = _mapping(
        json.loads((attempt / "metrics.json").read_text(encoding="utf-8")),
        "parent metrics",
    )
    provenance = _mapping(
        json.loads((attempt / "provenance.json").read_text(encoding="utf-8")),
        "parent provenance",
    )
    identity = _mapping(provenance.get("actual_identity"), "parent actual identity")
    continuation = _mapping(metrics.get("continuation"), "parent continuation metrics")
    parent_scientific = _mapping(
        yaml.safe_load((attempt / "scientific_config.yaml").read_text(encoding="utf-8")),
        "parent scientific config",
    )
    _validate_parent_test_contract(metrics=metrics, scientific=parent_scientific)
    continuation_path = str(continuation.get("path", ""))
    continuation_sha256 = str(continuation.get("sha256", ""))
    best_checkpoint_path = str(metrics.get("checkpoint", ""))
    best_checkpoint_sha256 = str(continuation.get("best_checkpoint_sha256", ""))
    for relative, expected_sha, label in (
        (continuation_path, continuation_sha256, "continuation"),
        (best_checkpoint_path, best_checkpoint_sha256, "best checkpoint"),
    ):
        record = records.get(relative)
        if record is None or str(record.get("sha256")) != expected_sha:
            raise ContinuationError(f"sealed parent {label} identity is incomplete")
    required_identity = {
        key: str(identity.get(key, ""))
        for key in ("config_id", "config_hash", "source_id", "data_id", "protocol_id")
    }
    if not required_identity["config_id"].startswith("cfg-"):
        raise ContinuationError("parent provenance lacks an exact config_id")
    if len(required_identity["config_hash"]) != 64:
        raise ContinuationError("parent provenance lacks an exact config_hash")
    if not required_identity["source_id"].startswith("src-"):
        raise ContinuationError("parent provenance lacks an exact source_id")
    if not required_identity["data_id"].startswith("data-"):
        raise ContinuationError("parent provenance lacks an exact data_id")
    return {
        "attempt_path": relative_attempt,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "continuation_path": continuation_path,
        "continuation_sha256": continuation_sha256,
        "best_checkpoint_path": best_checkpoint_path,
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "completed_epoch": int(continuation.get("completed_epoch", -1)),
        "job_id": str(metrics.get("job_id", "")),
        **required_identity,
    }


def resolve_parent_runtime(
    *,
    repo_root: Path,
    current_scientific: Mapping[str, Any],
    current_job: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate parent attempt, identities, hashes, and rung compatibility."""

    trainer = _mapping(current_scientific.get("trainer"), "trainer")
    continuation = trainer.get("continuation")
    if continuation is None:
        return None
    continuation = _mapping(continuation, "trainer.continuation")
    current_provenance = _mapping(current_scientific.get("provenance"), "provenance")
    current_source_id = str(
        current_job.get("source_id") or current_provenance.get("source_id") or ""
    )
    current_data_id = str(current_job.get("data_id") or "")
    current_protocol_id = str(current_job.get("protocol_id") or "")
    mode = str(continuation.get("mode"))
    if mode in {"fresh", "reference"}:
        return {
            "mode": mode,
            "contract": {
                "compatibility_digest": continuation_compatibility_digest(current_scientific),
                "lineage_id": str(continuation["lineage_id"]),
                "budget_schedule": list(continuation["budget_schedule"]),
                "schedule_total_epochs": int(continuation["schedule_total_epochs"]),
                "early_stopping_policy": str(continuation["early_stopping_policy"]),
                "model": str(current_job["model"]),
                "dataset": str(current_job["dataset"]),
                "seed": int(current_job["seed"]),
                "source_id": current_source_id,
                "data_id": current_data_id,
                "protocol_id": current_protocol_id,
            },
        }

    parent = _mapping(continuation.get("parent"), "trainer.continuation.parent")
    repo_root = repo_root.resolve()
    attempt = (repo_root / str(parent["attempt_path"])).resolve()
    studies_root = (repo_root / "outputs" / "studies").resolve()
    try:
        attempt.relative_to(studies_root)
    except ValueError as exc:
        raise ContinuationError("parent attempt must be contained under outputs/studies") from exc
    current_attempt_raw = os.environ.get("ROUTEREC_ATTEMPT_DIR")
    if current_attempt_raw and attempt == Path(current_attempt_raw).resolve():
        raise ContinuationError("a continuation job cannot parent itself")
    if not (attempt / "SUCCESS").is_file():
        raise ContinuationError("parent attempt is not terminally successful")

    manifest = verify_artifact_manifest(attempt, str(parent["artifact_manifest_sha256"]))
    records = {str(row["path"]): row for row in manifest["files"]}
    state_rel = str(parent["continuation_path"])
    best_rel = str(parent["best_checkpoint_path"])
    for relative, expected_sha, label in (
        (state_rel, str(parent["continuation_sha256"]), "continuation"),
        (best_rel, str(parent["best_checkpoint_sha256"]), "best checkpoint"),
    ):
        path = (attempt / relative).resolve()
        try:
            path.relative_to(attempt)
        except ValueError as exc:
            raise ContinuationError(f"parent {label} path escapes its attempt") from exc
        record = records.get(relative)
        if record is None or str(record.get("sha256")) != expected_sha:
            raise ContinuationError(f"parent {label} is absent from the sealed artifact manifest")

    parent_scientific = _mapping(
        yaml.safe_load((attempt / "scientific_config.yaml").read_text(encoding="utf-8")),
        "parent scientific config",
    )
    parent_metrics = _mapping(
        json.loads((attempt / "metrics.json").read_text(encoding="utf-8")),
        "parent metrics",
    )
    parent_provenance = _mapping(
        json.loads((attempt / "provenance.json").read_text(encoding="utf-8")),
        "parent provenance",
    )
    identity = _mapping(parent_provenance.get("actual_identity"), "parent actual identity")
    expected_identity = {
        "job_id": str(parent["job_id"]),
        "config_id": str(parent["config_id"]),
        "config_hash": str(parent["config_hash"]),
        "source_id": str(parent["source_id"]),
        "data_id": str(parent["data_id"]),
        "protocol_id": str(parent["protocol_id"]),
    }
    if str(parent_metrics.get("job_id")) != expected_identity["job_id"]:
        raise ContinuationError("parent job identity mismatch")
    for key in ("config_id", "config_hash", "source_id", "data_id", "protocol_id"):
        if str(identity.get(key)) != expected_identity[key]:
            raise ContinuationError(f"parent {key} identity mismatch")
    current_identity = {
        "source_id": current_source_id,
        "data_id": current_data_id,
        "protocol_id": current_protocol_id,
    }
    for key in ("source_id", "data_id", "protocol_id"):
        if current_identity[key] != expected_identity[key]:
            raise ContinuationError(f"current and parent {key} differ")
    _validate_parent_test_contract(metrics=parent_metrics, scientific=parent_scientific)
    if continuation_compatibility_digest(parent_scientific) != continuation_compatibility_digest(
        current_scientific
    ):
        raise ContinuationError("current scientific settings drifted from the parent rung")
    if int(parent_metrics.get("seed", -1)) != int(current_job["seed"]):
        raise ContinuationError("current and parent training seed differ")
    if str(parent_metrics.get("model")) != str(current_job["model"]):
        raise ContinuationError("current and parent model differ")
    if str(parent_metrics.get("dataset")) != str(current_job["dataset"]):
        raise ContinuationError("current and parent dataset differ")
    parent_continuation = _mapping(parent_metrics.get("continuation"), "parent continuation metrics")
    if int(parent_continuation.get("completed_epoch", -1)) != int(parent["completed_epoch"]):
        raise ContinuationError("parent continuation epoch does not match its sealed metrics")

    contract = {
        "compatibility_digest": continuation_compatibility_digest(current_scientific),
        "lineage_id": str(continuation["lineage_id"]),
        "budget_schedule": list(continuation["budget_schedule"]),
        "schedule_total_epochs": int(continuation["schedule_total_epochs"]),
        "early_stopping_policy": str(continuation["early_stopping_policy"]),
        "model": str(current_job["model"]),
        "dataset": str(current_job["dataset"]),
        "seed": int(current_job["seed"]),
        "source_id": current_source_id,
        "data_id": current_data_id,
        "protocol_id": current_protocol_id,
    }
    return {
        "mode": "resume",
        "contract": contract,
        "state_path": str((attempt / state_rel).resolve()),
        "state_sha256": str(parent["continuation_sha256"]),
        "best_checkpoint_path": str((attempt / best_rel).resolve()),
        "best_checkpoint_sha256": str(parent["best_checkpoint_sha256"]),
        "parent_attempt": str(attempt),
        "parent_job_id": str(parent["job_id"]),
        "parent_completed_epoch": int(parent["completed_epoch"]),
    }


__all__ = [
    "CONTINUATION_KIND",
    "CONTINUATION_SCHEMA_VERSION",
    "SUPPORTED_CONTINUATION_SCHEMA_VERSIONS",
    "FIXED_BUDGET_POLICY",
    "ContinuationError",
    "build_parent_binding",
    "capture_rng_state",
    "continuation_compatibility_digest",
    "continuation_compatibility_view",
    "load_continuation_state",
    "resolve_parent_runtime",
    "restore_rng_state",
    "save_continuation_state",
    "sha256_file",
    "verify_artifact_manifest",
]
