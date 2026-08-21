"""Dataset name normalization and repository-local path resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_RELATIVE_DATA_ROOTS = (
    Path("Datasets/core5"),
    Path("Datasets/release"),
)

# New releases use the stable name. The second entry keeps already prepared
# prepared data readable without exposing a versioned name elsewhere.
FEATURE_METADATA_FILENAMES = ("feature_metadata.json", "feature_meta_v3.json")

CANONICAL_DATASETS = (
    "beauty",
    "foursquare",
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "movielens1m",
    "retail_rocket",
    "beauty_core5_v1",
    "foursquare_core5_v1",
    "movielens1m_core5_v1",
    "retail_rocket_core5_v1",
    "kuairec_adaptive_core5_v1",
    "lastfm_recovered_core5_v1",
)

_ALIAS_TO_CANONICAL = {
    # The paper dataset is ``beauty`` (33,488 prepared interactions).  A
    # different, much smaller ``amazon_beauty`` directory is present in the
    # historical archive.  Treating both as canonical silently changed the
    # benchmark depending on the spelling used at the CLI.
    "amazon beauty": "beauty",
    "amazon-beauty": "beauty",
    "amazon_beauty": "beauty",
    "beauty": "beauty",
    "foursquare": "foursquare",
    "kuairec": "KuaiRecLargeStrictPosV2_0.2",
    "kuairec20": "KuaiRecLargeStrictPosV2_0.2",
    "KuaiRecLargeStrictPosV2_0.2": "KuaiRecLargeStrictPosV2_0.2",
    "lastfm": "lastfm0.03",
    "lastfm0.03": "lastfm0.03",
    "ml-1m": "movielens1m",
    "ml1m": "movielens1m",
    "movielens-1m": "movielens1m",
    "movielens1m": "movielens1m",
    "movie lens 1m": "movielens1m",
    "retail rocket": "retail_rocket",
    "retail-rocket": "retail_rocket",
    "retail_rocket": "retail_rocket",
    "retailrocket": "retail_rocket",
}


def _dataset_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


_CANONICAL_BY_KEY = {_dataset_key(name): name for name in CANONICAL_DATASETS}
_CANONICAL_BY_KEY.update(
    {_dataset_key(alias): canonical for alias, canonical in _ALIAS_TO_CANONICAL.items()}
)

_DATASET_FILE_SUFFIXES = (
    ".train.inter",
    ".valid.inter",
    ".test.inter",
    ".inter",
    ".item",
)


@dataclass(frozen=True)
class ResolvedDataset:
    requested_name: str
    dataset_name: str
    data_path: str | None
    dataset_dir: Path | None
    auto_discovered: bool
    searched_paths: tuple[Path, ...]


def normalize_dataset_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("dataset name must not be empty")
    return _CANONICAL_BY_KEY.get(_dataset_key(cleaned), cleaned)


def default_dataset_roots(repo_root: Path) -> list[Path]:
    root = Path(repo_root).resolve()
    return [(root / rel_path).resolve() for rel_path in DEFAULT_RELATIVE_DATA_ROOTS]


def find_feature_metadata(dataset_dir: Path) -> Path | None:
    """Return the first supported feature-metadata file in a dataset directory."""
    root = Path(dataset_dir)
    for filename in FEATURE_METADATA_FILENAMES:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _resolve_override_path(data_path: str | None, repo_root: Path | None) -> Path | None:
    if not data_path:
        return None
    path = Path(data_path).expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = Path(repo_root).resolve() / path
    return path.resolve()


def _looks_like_dataset_dir(path: Path, dataset_name: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / f"{dataset_name}{suffix}").exists() for suffix in _DATASET_FILE_SUFFIXES)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def resolve_dataset_dir(
    *,
    dataset: str,
    data_path: str | None,
    repo_root: Path | None = None,
) -> Path | None:
    resolved = resolve_dataset_runtime(
        dataset=dataset,
        data_path=data_path,
        repo_root=repo_root,
        require_existing=False,
    )
    return resolved.dataset_dir


def resolve_dataset_runtime(
    *,
    dataset: str,
    data_path: str | None,
    repo_root: Path | None = None,
    require_existing: bool = False,
) -> ResolvedDataset:
    dataset_name = normalize_dataset_name(dataset)
    repo_root_path = Path(repo_root).resolve() if repo_root is not None else None
    override_path = _resolve_override_path(data_path, repo_root_path)

    searched_paths: list[Path] = []
    candidate_dirs: list[Path] = []
    auto_discovered = False

    if override_path is not None:
        candidate_dirs.extend([override_path / dataset_name, override_path])
    elif repo_root_path is not None:
        auto_discovered = True
        for root in default_dataset_roots(repo_root_path):
            candidate_dirs.append(root / dataset_name)

    for candidate_dir in _unique_paths(candidate_dirs):
        searched_paths.append(candidate_dir)
        if not _looks_like_dataset_dir(candidate_dir, dataset_name):
            continue
        data_root = candidate_dir.parent if candidate_dir.name == dataset_name else candidate_dir
        return ResolvedDataset(
            requested_name=str(dataset),
            dataset_name=dataset_name,
            data_path=str(data_root),
            dataset_dir=candidate_dir,
            auto_discovered=auto_discovered,
            searched_paths=tuple(searched_paths),
        )

    if require_existing:
        pretty_paths = ", ".join(str(path) for path in searched_paths) or "<no search roots>"
        raise FileNotFoundError(
            f"Could not locate dataset '{dataset}' (normalized to '{dataset_name}'). "
            f"Checked: {pretty_paths}"
        )

    if override_path is not None:
        fallback_root = override_path.parent if override_path.name == dataset_name else override_path
        return ResolvedDataset(
            requested_name=str(dataset),
            dataset_name=dataset_name,
            data_path=str(fallback_root),
            dataset_dir=None,
            auto_discovered=False,
            searched_paths=tuple(searched_paths),
        )

    fallback_root = None
    if repo_root_path is not None:
        fallback_root = default_dataset_roots(repo_root_path)[0]

    return ResolvedDataset(
        requested_name=str(dataset),
        dataset_name=dataset_name,
        data_path=str(fallback_root) if fallback_root is not None else None,
        dataset_dir=None,
        auto_discovered=auto_discovered,
        searched_paths=tuple(searched_paths),
    )


def _interaction_header_path(dataset_dir: Path, dataset_name: str) -> Path | None:
    for suffix in ("train.inter", "inter"):
        candidate = dataset_dir / f"{dataset_name}.{suffix}"
        if candidate.exists():
            return candidate
    return None


def _parse_typed_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")
    if not first_line:
        return []
    return [column.split(":", 1)[0].strip() for column in first_line.split("\t") if column.strip()]


def _split_summary_path(dataset_dir: Path, dataset_name: str) -> Path | None:
    candidates = [
        dataset_dir / f"{dataset_name}.session_split_summary.json",
        dataset_dir / f"{dataset_name}.v4_split_summary.json",
        dataset_dir / f"{dataset_name}.split_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_recbole_dataset_config(
    *,
    dataset: str,
    data_path: str | None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    dataset_dir = resolve_dataset_dir(dataset=dataset, data_path=data_path, repo_root=repo_root)
    if dataset_dir is None:
        return {}

    dataset_name = normalize_dataset_name(dataset)
    header_path = _interaction_header_path(dataset_dir, dataset_name)
    if header_path is None:
        return {}

    columns = _parse_typed_header(header_path)
    if not columns:
        return {}

    config: dict[str, Any] = {
        "field_separator": "\t",
        "load_col": {"inter": columns},
    }
    # The paper prediction unit is a reconstructed session.  Using the raw
    # user id here joins multiple sessions into one RecBole sequence and is a
    # different task.  ``user_id`` remains loaded as an auxiliary field.
    if "session_id" in columns:
        config["USER_ID_FIELD"] = "session_id"
        config["SESSION_ID_FIELD"] = "session_id"
    elif "user_id" in columns:
        config["USER_ID_FIELD"] = "user_id"
    if "item_id" in columns:
        config["ITEM_ID_FIELD"] = "item_id"
    if "timestamp" in columns:
        config["TIME_FIELD"] = "timestamp"

    split_suffixes = ("train", "valid", "test")
    has_benchmark_splits = all(
        (dataset_dir / f"{dataset_name}.{suffix}.inter").exists() for suffix in split_suffixes
    )
    if has_benchmark_splits:
        # RouteRec installs a small RecBole compatibility patch that converts
        # these raw, frozen split files to sequential samples.  Do not fall
        # back to re-splitting the combined ``.inter`` file.
        config["benchmark_filename"] = list(split_suffixes)
        config["routerec_frozen_split"] = True

    summary_path = _split_summary_path(dataset_dir, dataset_name)
    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        ratios = summary.get("ratios") if isinstance(summary, dict) else None
        if isinstance(ratios, dict):
            train_ratio = ratios.get("train")
            valid_ratio = ratios.get("valid")
            test_ratio = ratios.get("test")
            if all(isinstance(value, (int, float)) for value in (train_ratio, valid_ratio, test_ratio)):
                config["eval_args"] = {
                    "split": {"RS": [float(train_ratio), float(valid_ratio), float(test_ratio)]},
                    "group_by": "user",
                    "order": "TO",
                    "mode": "full",
                }

    return config
