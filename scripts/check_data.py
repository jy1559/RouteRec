#!/usr/bin/env python3
"""Check that the six paper datasets can be resolved from this repository."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.datasets import resolve_dataset_runtime


PAPER_DATASETS = (
    "beauty",
    "foursquare",
    "kuairec",
    "lastfm",
    "ml-1m",
    "retail-rocket",
)


def _data_rows(path: Path) -> int:
    newline_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    line_count = newline_count + (1 if path.stat().st_size and last_byte != b"\n" else 0)
    return max(0, line_count - 1)


def main() -> int:
    missing: list[str] = []
    for dataset in PAPER_DATASETS:
        resolved = resolve_dataset_runtime(
            dataset=dataset,
            data_path=None,
            repo_root=ROOT,
            require_existing=False,
        )
        if resolved.dataset_dir is None:
            missing.append(f"{dataset} -> {resolved.dataset_name}")
            print(f"MISSING  {dataset:14} -> {resolved.dataset_name}")
            continue

        required = [
            resolved.dataset_dir / f"{resolved.dataset_name}.{split}.inter"
            for split in ("train", "valid", "test")
        ]
        missing_splits = [path.name for path in required if not path.is_file()]
        if missing_splits:
            missing.append(f"{dataset}: missing {', '.join(missing_splits)}")
            print(f"MISSING  {dataset:14} -> frozen split incomplete")
            continue

        files = list(resolved.dataset_dir.glob(f"{resolved.dataset_name}.*"))
        size = sum(path.stat().st_size for path in files if path.is_file())
        rows = {}
        for path in required:
            rows[path.name.split(".")[-2]] = _data_rows(path)
        relative = resolved.dataset_dir.relative_to(ROOT)
        print(
            f"OK       {dataset:14} -> {relative} "
            f"(raw split rows train/valid/test={rows['train']}/{rows['valid']}/{rows['test']}; "
            f"{size / 2**20:.1f} MiB)"
        )

    if missing:
        print("\nMissing datasets:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("\nAll six frozen release datasets are available. Run check_protocol.py before performance experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
