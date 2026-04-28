"""Repository layout metadata for RouteRec."""

from __future__ import annotations

REQUIRED_DIRECTORIES = (
    "src/routerec",
    "src/routerec/models",
    "configs",
    "configs/models",
    "configs/search_spaces",
    "scripts",
    "tests",
    "docs",
    "paper",
    "assets",
)

REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    "pyproject.toml",
    "docs/repo-layout.md",
    "docs/model-selection.md",
    "docs/reproducibility.md",
    "configs/models/routerec_default.yaml",
    "configs/models/routerec_dataset_presets.yaml",
    "configs/search_spaces/paper_bounded_grid.yaml",
    "scripts/train.py",
    "scripts/search.py",
    "scripts/test.py",
)
