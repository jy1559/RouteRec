"""Repository layout metadata for RouteRec."""

from __future__ import annotations

REQUIRED_DIRECTORIES = (
    "src/routerec",
    "src/routerec/models",
    "src/routerec/models/routerec",
    ".github/workflows",
    "configs",
    "configs/models",
    "configs/paper",
    "configs/search_spaces",
    "scripts",
    "tests",
    "docs",
    "Datasets",
)

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".gitignore",
    "pyproject.toml",
    "environment.yml",
    "Datasets/README.md",
    "docs/repo-layout.md",
    "docs/model-selection.md",
    "docs/reproducibility.md",
    "docs/data-contract.md",
    "docs/release-checklist.md",
    "configs/models/routerec_default.yaml",
    "configs/models/routerec_dataset_presets.yaml",
    "configs/paper/protocol.yaml",
    "configs/paper/models/routerec_a12.yaml",
    "configs/search_spaces/paper_bounded_grid.yaml",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/build_core5_splits.py",
    "scripts/build_core5_features.py",
    "scripts/validate_core5_splits.py",
    "scripts/validate_feature_dataset.py",
    ".github/workflows/ci.yml",
)
