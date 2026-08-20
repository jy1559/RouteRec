"""Repository layout metadata for RouteRec."""

from __future__ import annotations

REQUIRED_DIRECTORIES = (
    "src/routerec",
    "src/routerec/models",
    "configs",
    "configs/models",
    "configs/paper",
    "configs/search_spaces",
    "scripts",
    "tests",
    "docs",
    "Datasets",
    "paper",
    "assets",
)

REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    "pyproject.toml",
    "environment.yml",
    "Datasets/README.md",
    "docs/repo-layout.md",
    "docs/model-selection.md",
    "docs/reproducibility.md",
    "docs/data-contract.md",
    "configs/models/routerec_default.yaml",
    "configs/models/routerec_dataset_presets.yaml",
    "configs/paper/protocol.yaml",
    "configs/paper/models/routerec_a12.yaml",
    "configs/search_spaces/paper_bounded_grid.yaml",
    "scripts/train.py",
    "scripts/test.py",
    "scripts/rebuild_camera_ready_core5_basic.py",
    "scripts/build_camera_ready_core5_features.py",
    "scripts/validate_camera_ready_core5_basic.py",
)
