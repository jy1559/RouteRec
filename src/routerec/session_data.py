"""Frozen-session split support for RecBole 1.2.1.

The prepared RouteRec data stores raw interaction rows in three immutable
``*.train.inter``, ``*.valid.inter`` and ``*.test.inter`` files.  RecBole's
``SequentialDataset`` benchmark path normally expects those files to contain
already-augmented ``*_list`` columns.  This module installs a deliberately
small compatibility patch: load all three files with RecBole's shared token
mapping, then convert each split to next-item samples without re-splitting the
combined interaction file.

Training uses every prefix in a training session.  Validation and test use the
last item of each prepared session as the target.  This is the protocol used by
the historical RouteRec experiment runner and, unlike a normal RecBole random
split, preserves the frozen session boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .datasets import find_feature_metadata

try:
    import fcntl
except ImportError:  # pragma: no cover - RouteRec production workers are Linux.
    fcntl = None


_PATCHED = False
_PATCH_VERSION = 6
_TARGET_MID_BROADCAST_FLAG = "routerec_causal_mid_target_broadcast"
_TARGET_MID_CONTRACTS = {
    "core5-features-leakage-safe-v1",
    "full-v5-leakage-correct-20260812",
}

# These are the models whose interaction access has been audited in the public
# source. Keep unknown models on the all-history path:
# silently projecting a future model to item-only data could change behavior.
_ITEM_ONLY_HISTORY_MODELS = {
    "bsarec",
    "difsr",
    "duorec",
    "fame",
    "fdsa",
    "fearec",
    "gru4rec",
    "sasrec",
    "sigma",
}
_ROUTEREC_HISTORY_MODELS = {"routerec"}


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        value = config.get(key, default)
        return default if value is None else value
    try:
        value = config[key]
    except Exception:
        value = getattr(config, key, default)
    return default if value is None else value


def _is_routerec_model(config: Any) -> bool:
    return _model_key(config) in _ROUTEREC_HISTORY_MODELS


def _model_key(config: Any) -> str:
    return str(_config_get(config, "model", "")).lower().replace("_", "")


def _is_route_feature(name: str) -> bool:
    return name.startswith(("mac_", "mac5_", "mac10_", "mid_", "mic_"))


def _target_mid_broadcast_enabled(config: Any) -> bool:
    """Whether a RouteRec sample uses its target-row strict-prefix mid cue.

    The target interaction itself is never copied into model history.  Only its
    ``mid_*`` row is eligible, and the feature builder guarantees that those
    fields were computed from events strictly before the target row.
    """

    return _is_routerec_model(config) and bool(
        _config_get(config, _TARGET_MID_BROADCAST_FLAG, False)
    )


def _history_fields(dataset: Any, fields: list[str]) -> list[str]:
    fields = [field for field in fields if field != dataset.uid_field]
    if _is_routerec_model(dataset.config):
        return [field for field in fields if field == dataset.iid_field or _is_route_feature(field)]
    # Baselines consume item histories. TiSASRec additionally consumes the
    # timestamp sequence; FDSA/DIFSR obtain item attributes from item_feat.
    # Materializing every 64-feature history for all baselines is behaviorally
    # inert and makes full-data caches tens of GiB larger.
    allowed = {dataset.iid_field}
    if _model_key(dataset.config) == "tisasrec":
        allowed.add(dataset.time_field)
        return [field for field in fields if field in allowed]
    if _model_key(dataset.config) in _ITEM_ONLY_HISTORY_MODELS:
        return [field for field in fields if field in allowed]
    return fields


def _history_profile(config: Any) -> str:
    if _is_routerec_model(config):
        if _target_mid_broadcast_enabled(config):
            return "routerec_route_features_target_mid_broadcast_v1"
        return "routerec_route_features"
    if _model_key(config) == "tisasrec":
        return "tisasrec_item_timestamp_elapsed_seconds_v1"
    if _model_key(config) in _ITEM_ONLY_HISTORY_MODELS:
        return "baseline_item_only"
    return "all_fields"


def _broadcast_target_mid_history(
    *,
    history_values: Any,
    target_idx: Any,
    seq_len: Any,
    max_len: int,
    chunk_size: int = 16384,
) -> Any:
    """Broadcast each target-row mid cue over valid history slots.

    ``history_values[target_idx]`` is safe only under the guarded full-v5
    strict-prefix contract.  Padding remains exactly zero.  The helper works
    for scalar fields and preserves possible trailing value dimensions.
    """

    import torch

    sample_count = int(target_idx.numel())
    out_shape = (sample_count, max_len) + tuple(history_values.shape[1:])
    history = torch.zeros(out_shape, dtype=history_values.dtype, device=history_values.device)
    offsets = torch.arange(max_len, dtype=torch.long, device=seq_len.device)
    chunk_size = max(int(chunk_size), 1)
    for begin in range(0, sample_count, chunk_size):
        finish = min(begin + chunk_size, sample_count)
        target_values = history_values[target_idx[begin:finish]]
        broadcast = target_values.unsqueeze(1).expand(
            (finish - begin, max_len) + tuple(history_values.shape[1:])
        )
        valid = offsets.unsqueeze(0) < seq_len[begin:finish].unsqueeze(1)
        while valid.dim() < broadcast.dim():
            valid = valid.unsqueeze(-1)
        history[begin:finish] = broadcast.masked_fill(~valid, 0)
    return history


def _validate_target_mid_broadcast_contract(dataset: Any, split_names: list[str]) -> None:
    """Fail closed unless the dataset proves target-row mid cues are strict-prefix."""

    requested = bool(_config_get(dataset.config, _TARGET_MID_BROADCAST_FLAG, False))
    if not requested:
        return
    if not _is_routerec_model(dataset.config):
        raise ValueError(f"{_TARGET_MID_BROADCAST_FLAG}=true is RouteRec-only")
    data_root = _benchmark_data_root(dataset, split_names)
    metadata_path = find_feature_metadata(data_root)
    if metadata_path is None:
        raise ValueError(
            f"{_TARGET_MID_BROADCAST_FLAG}=true requires feature metadata"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read target-mid contract: {metadata_path}") from exc
    if metadata.get("reconstruction_contract") not in _TARGET_MID_CONTRACTS:
        raise ValueError(
            "target-mid broadcast requires a supported leakage-safe "
            "reconstruction_contract"
        )
    if metadata.get("mid_scope") != "strict_prefix":
        raise ValueError("target-mid broadcast requires feature metadata mid_scope='strict_prefix'")
    mid_features = {
        str(name) for name in metadata.get("all_features", []) if str(name).startswith("mid_")
    }
    if len(mid_features) != 16:
        raise ValueError(
            f"target-mid broadcast requires exactly 16 declared mid fields, got {len(mid_features)}"
        )


def _tisas_timestamp_unit(dataset: Any, split_names: list[str]) -> str:
    data_root = _benchmark_data_root(dataset, split_names)
    metadata_path = find_feature_metadata(data_root)
    if metadata_path is None:
        raise ValueError("TiSASRec exact-seconds loading requires feature metadata")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read TiSASRec timestamp unit: {metadata_path}") from exc
    unit = metadata.get("timestamp_unit")
    if unit not in {"s", "ms"}:
        raise ValueError(f"TiSASRec requires declared timestamp_unit s or ms, got {unit!r}")
    return str(unit)


def _load_tisas_elapsed_seconds(
    dataset: Any,
    split_names: list[str],
    split_name: str,
    expected_rows: int,
) -> Any:
    """Load exact session-relative seconds before RecBole float32 conversion.

    Only the first and third TSV fields are inspected.  This preserves decimal
    KuaiRec seconds and avoids the 65-second-scale quantization produced by
    casting millisecond Unix timestamps near 1e12 to float32.
    """

    import torch

    data_root = _benchmark_data_root(dataset, split_names)
    dataset_name = str(getattr(dataset, "dataset_name", _config_get(dataset.config, "dataset")))
    path = data_root / f"{dataset_name}.{split_name}.inter"
    unit = _tisas_timestamp_unit(dataset, split_names)
    divisor = Decimal(1000 if unit == "ms" else 1)
    values: list[float] = []
    seen: set[str] = set()
    current_sid: str | None = None
    origin = Decimal(0)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        plain = [column.split(":", 1)[0] for column in header[:4]]
        if plain != ["session_id", "item_id", "timestamp", "user_id"]:
            raise ValueError(f"unexpected TiSASRec interaction header: {path}: {plain}")
        for line_number, line in enumerate(handle, 2):
            first = line.find("\t")
            second = line.find("\t", first + 1)
            third = line.find("\t", second + 1)
            if min(first, second, third) < 0:
                raise ValueError(f"{path}:{line_number}: malformed interaction row")
            sid = line[:first]
            raw = line[second + 1:third]
            try:
                timestamp = Decimal(raw)
            except InvalidOperation as exc:
                raise ValueError(f"{path}:{line_number}: invalid timestamp {raw!r}") from exc
            if not timestamp.is_finite():
                raise ValueError(f"{path}:{line_number}: non-finite timestamp")
            if sid != current_sid:
                if sid in seen:
                    raise ValueError(f"{path}:{line_number}: non-contiguous session {sid!r}")
                seen.add(sid)
                current_sid = sid
                origin = timestamp
            elapsed = (timestamp - origin) / divisor
            if elapsed < 0:
                raise ValueError(f"{path}:{line_number}: timestamp regression in {sid!r}")
            values.append(float(elapsed))
    if len(values) != int(expected_rows):
        raise ValueError(
            f"{path}: exact timestamp rows {len(values)} != RecBole rows {expected_rows}"
        )
    return torch.tensor(values, dtype=torch.float64)


def _convert_interactions(
    dataset: Any,
    inter_feat: Any,
    *,
    training: bool,
    exact_time_values: Any | None = None,
) -> Any | None:
    """Convert one raw interaction split to sequential samples."""
    import numpy as np
    import torch
    from recbole.data.interaction import Interaction

    if not hasattr(inter_feat, "interaction"):
        raise TypeError("RouteRec frozen split conversion requires a RecBole Interaction")

    fields = list(inter_feat.columns)
    if not fields:
        return None

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    time_field = dataset.time_field
    for required in (uid_field, iid_field, time_field):
        if required not in fields:
            raise ValueError(f"Frozen split is missing required field: {required}")

    max_len = int(dataset.config["MAX_ITEM_LIST_LENGTH"])
    if max_len < 1:
        raise ValueError("MAX_ITEM_LIST_LENGTH must be positive")
    list_suffix = str(_config_get(dataset.config, "LIST_SUFFIX", "_list"))
    length_field = str(_config_get(dataset.config, "ITEM_LIST_LENGTH_FIELD", "item_length"))
    chunk_size = max(int(_config_get(dataset.config, "sequence_convert_chunk_size", 16384)), 1024)
    feature_fp16 = bool(_config_get(dataset.config, "routerec_feature_fp16", True))

    uid_values = inter_feat[uid_field]
    time_values = exact_time_values if exact_time_values is not None else inter_feat[time_field]
    if not torch.is_tensor(time_values):
        time_values = torch.as_tensor(time_values)
    if int(time_values.shape[0]) != len(inter_feat):
        raise ValueError("exact timestamp row count does not match interaction split")
    uid_np = uid_values.detach().cpu().numpy()
    time_np = time_values.detach().cpu().numpy()
    original_idx = np.arange(len(uid_np), dtype=np.int64)
    order_np = np.lexsort((original_idx, time_np, uid_np))
    order = torch.from_numpy(order_np).long()
    uid_sorted = uid_np[order_np]

    if uid_sorted.size < 2:
        return None
    changes = np.nonzero(uid_sorted[1:] != uid_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(uid_sorted)]))

    target_parts: list[Any] = []
    seq_start_parts: list[Any] = []
    seq_len_parts: list[Any] = []
    for start, end in zip(starts, ends):
        count = int(end - start)
        if count < 2:
            continue
        if training:
            relative_target = np.arange(1, count, dtype=np.int64)
        else:
            relative_target = np.asarray([count - 1], dtype=np.int64)
        target = start + relative_target
        seq_len = np.minimum(relative_target, max_len)
        target_parts.append(target)
        seq_start_parts.append(target - seq_len)
        seq_len_parts.append(seq_len)

    if not target_parts:
        return None

    target_idx = torch.from_numpy(np.concatenate(target_parts)).long()
    seq_start = torch.from_numpy(np.concatenate(seq_start_parts)).long()
    seq_len = torch.from_numpy(np.concatenate(seq_len_parts)).long()
    sample_count = int(target_idx.numel())
    offsets = torch.arange(max_len, dtype=torch.long)
    selected_history_fields = set(_history_fields(dataset, fields))

    converted: dict[str, Any] = {length_field: seq_len}
    for field in fields:
        values = time_values if field == time_field and exact_time_values is not None else inter_feat[field]
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        values = values[order]
        converted[field] = values[target_idx]
        if field not in selected_history_fields:
            continue

        history_values = values
        if feature_fp16 and _is_route_feature(field) and history_values.is_floating_point():
            history_values = history_values.to(torch.float16)
        if _target_mid_broadcast_enabled(dataset.config) and field.startswith("mid_"):
            converted[field + list_suffix] = _broadcast_target_mid_history(
                history_values=history_values,
                target_idx=target_idx,
                seq_len=seq_len,
                max_len=max_len,
                chunk_size=chunk_size,
            )
            continue
        out_shape = (sample_count, max_len) + tuple(history_values.shape[1:])
        history = torch.zeros(out_shape, dtype=history_values.dtype)

        for begin in range(0, sample_count, chunk_size):
            finish = min(begin + chunk_size, sample_count)
            starts_chunk = seq_start[begin:finish]
            lengths_chunk = seq_len[begin:finish]
            index = starts_chunk.unsqueeze(1) + offsets.unsqueeze(0)
            valid = offsets.unsqueeze(0) < lengths_chunk.unsqueeze(1)
            index = index.clamp(min=0, max=max(int(history_values.shape[0]) - 1, 0))
            gathered = history_values[index]
            mask = ~valid
            while mask.dim() < gathered.dim():
                mask = mask.unsqueeze(-1)
            history[begin:finish] = gathered.masked_fill(mask, 0)
        converted[field + list_suffix] = history

    return Interaction(converted)


def _make_sequence_dataset(base_dataset: Any, converted: Any) -> Any:
    from recbole.utils.enum_type import FeatureType

    list_suffix = str(_config_get(base_dataset.config, "LIST_SUFFIX", "_list"))
    length_field = str(_config_get(base_dataset.config, "ITEM_LIST_LENGTH_FIELD", "item_length"))
    max_len = int(base_dataset.config["MAX_ITEM_LIST_LENGTH"])
    new_dataset = base_dataset.copy(converted)
    new_dataset.inter_feat = converted
    new_dataset.item_id_list_field = base_dataset.iid_field + list_suffix
    new_dataset.item_list_length_field = length_field

    for field in converted.columns:
        if not field.endswith(list_suffix):
            continue
        base_field = field[: -len(list_suffix)]
        setattr(new_dataset, f"{base_field}_list_field", field)
        if field in new_dataset.field2type:
            continue
        base_type = new_dataset.field2type.get(base_field)
        if base_type in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            new_dataset.field2type[field] = FeatureType.TOKEN_SEQ
        else:
            new_dataset.field2type[field] = FeatureType.FLOAT_SEQ
        new_dataset.field2seqlen[field] = max_len

    if length_field not in new_dataset.field2type:
        new_dataset.field2type[length_field] = FeatureType.TOKEN
        new_dataset.field2seqlen[length_field] = 1
    return new_dataset


def _benchmark_data_root(dataset: Any, split_names: list[str]) -> Path:
    """Resolve both RecBole forms of ``data_path`` without duplicating the dataset name."""

    dataset_name = str(getattr(dataset, "dataset_name", _config_get(dataset.config, "dataset", "dataset")))
    configured = Path(str(_config_get(dataset.config, "data_path", ""))).expanduser()
    candidates = (configured / dataset_name, configured)
    for candidate in candidates:
        if all((candidate / f"{dataset_name}.{split}.inter").is_file() for split in split_names):
            return candidate.resolve()
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve frozen benchmark directory for {dataset_name}; checked {expected}"
    )


def _cache_key(dataset: Any, split_names: list[str]) -> str:
    dataset_name = str(getattr(dataset, "dataset_name", _config_get(dataset.config, "dataset", "dataset")))
    data_root = _benchmark_data_root(dataset, split_names)
    files = []
    for split in split_names:
        path = data_root / f"{dataset_name}.{split}.inter"
        stat = path.stat()
        content_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content_hash.update(chunk)
        files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "sha256": content_hash.hexdigest(),
            }
        )
    payload = {
        "patch_version": _PATCH_VERSION,
        "dataset": dataset_name,
        "files": files,
        "max_len": int(dataset.config["MAX_ITEM_LIST_LENGTH"]),
        "uid": dataset.uid_field,
        "iid": dataset.iid_field,
        "time": dataset.time_field,
        "load_col": _config_get(dataset.config, "load_col", {}),
        "feature_fp16": bool(_config_get(dataset.config, "routerec_feature_fp16", True)),
        "history_profile": _history_profile(dataset.config),
        "target_mid_broadcast": _target_mid_broadcast_enabled(dataset.config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _ExclusiveCacheBuildLock:
    """Serialize only the first build of a content-addressed split cache."""

    def __init__(self, cache_file: Path) -> None:
        self.lock_file = cache_file.with_name(f".{cache_file.name}.build.lock")
        self._handle: Any = None

    def __enter__(self) -> "_ExclusiveCacheBuildLock":
        if fcntl is None:
            raise RuntimeError("RouteRec split-cache build locking requires POSIX fcntl")
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.lock_file.open("a+b")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _load_cached_splits(dataset: Any, cache_file: Path) -> list[Any] | None:
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("rb") as stream:
            splits = pickle.load(stream)
        if not isinstance(splits, list) or len(splits) != 3:
            return None
        for split in splits:
            split.config = dataset.config
        dataset.logger.info("RouteRec: loaded frozen sequence cache %s", cache_file)
        return splits
    except Exception as exc:
        dataset.logger.warning("RouteRec: ignored unusable frozen sequence cache %s (%s)", cache_file, exc)
        return None


def _save_cached_splits(dataset: Any, cache_file: Path, splits: list[Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(splits, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_file)
        dataset.logger.info("RouteRec: saved frozen sequence cache %s", cache_file)
    except Exception as exc:
        dataset.logger.warning("RouteRec: could not save frozen sequence cache (%s)", exc)
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass


def install_frozen_session_split_patch() -> None:
    """Install the RecBole patch once in the current worker process."""
    global _PATCHED
    if _PATCHED:
        return

    import numpy as np
    from recbole.data.dataset.sequential_dataset import SequentialDataset

    original_benchmark_presets = SequentialDataset._benchmark_presets
    original_build = SequentialDataset.build

    def patched_benchmark_presets(self: Any) -> Any:
        list_suffix = str(_config_get(self.config, "LIST_SUFFIX", "_list"))
        item_list_field = self.iid_field + list_suffix
        self.item_id_list_field = item_list_field
        if item_list_field in self.inter_feat.columns:
            return original_benchmark_presets(self)
        self._routerec_raw_benchmark = True
        self.item_list_length_field = str(_config_get(self.config, "ITEM_LIST_LENGTH_FIELD", "item_length"))
        for field in self.inter_feat.columns:
            if field != self.uid_field:
                setattr(self, f"{field}_list_field", field + list_suffix)
        return None

    def patched_build(self: Any) -> list[Any]:
        is_raw_benchmark = bool(getattr(self, "_routerec_raw_benchmark", False))
        if self.benchmark_filename_list is None or not is_raw_benchmark:
            return original_build(self)
        if not bool(_config_get(self.config, "routerec_frozen_split", False)):
            raise ValueError(
                "Raw benchmark files require routerec_frozen_split=true; refusing an implicit protocol change"
            )
        if str(self.uid_field) != "session_id":
            raise ValueError(
                f"Frozen RouteRec protocol requires USER_ID_FIELD=session_id, got {self.uid_field!r}"
            )

        split_names = [str(name) for name in self.benchmark_filename_list]
        if split_names != ["train", "valid", "test"]:
            raise ValueError(f"Expected frozen benchmark splits [train, valid, test], got {split_names}")
        _validate_target_mid_broadcast_contract(self, split_names)

        cache_root_raw = str(_config_get(self.config, "routerec_split_cache_dir", "")).strip()
        cache_file: Path | None = None
        if cache_root_raw and bool(_config_get(self.config, "enable_session_split_cache", True)):
            cache_file = Path(cache_root_raw).expanduser().resolve() / f"{_cache_key(self, split_names)}.pkl"
            cached = _load_cached_splits(self, cache_file)
            if cached is not None:
                return cached

        def build_uncached() -> list[Any]:
            self._change_feat_format()
            boundaries = [0] + list(np.cumsum(self.file_size_list))
            raw_splits = [
                self.inter_feat[start:end]
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            if len(raw_splits) != 3:
                raise ValueError(f"Expected three raw split tensors, got {len(raw_splits)}")

            datasets = []
            for index, (name, raw_split) in enumerate(zip(split_names, raw_splits)):
                exact_time_values = None
                if _model_key(self.config) == "tisasrec":
                    exact_time_values = _load_tisas_elapsed_seconds(
                        self, split_names, name, len(raw_split)
                    )
                converted = _convert_interactions(
                    self,
                    raw_split,
                    training=index == 0,
                    exact_time_values=exact_time_values,
                )
                if converted is None:
                    raise ValueError(f"Frozen split {name} produced no sequential samples")
                datasets.append(_make_sequence_dataset(self, converted))
                self.logger.info(
                    "RouteRec frozen split %s: raw_rows=%d sequential_samples=%d",
                    name,
                    len(raw_split),
                    len(converted),
                )

            if cache_file is not None:
                _save_cached_splits(self, cache_file, datasets)
            return datasets

        if cache_file is None:
            return build_uncached()

        # The first worker owns the expensive conversion and atomic cache write.
        # Waiters do not hold the lock while unpickling the completed cache, so
        # training can fan back out across GPUs as soon as the one-time build ends.
        with _ExclusiveCacheBuildLock(cache_file):
            if not cache_file.is_file():
                return build_uncached()
        cached = _load_cached_splits(self, cache_file)
        if cached is None:
            raise RuntimeError(
                f"Content-addressed RouteRec split cache is unusable after locked build: {cache_file}"
            )
        return cached

    SequentialDataset._benchmark_presets = patched_benchmark_presets
    SequentialDataset.build = patched_build
    _PATCHED = True


__all__ = ["install_frozen_session_split_patch"]
