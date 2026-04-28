# Recommended Public Repository Layout

This repository should optimize for reproducibility, readability, and low noise.

## Keep In The Root

- `README.md`: project overview, install, quickstart, reproduction entrypoint
- `pyproject.toml`: package metadata and dependencies
- `.gitignore`: exclude logs, checkpoints, caches, and paper build artifacts

## Main Directories

- `src/routerec/`: core library code, model implementations, adapters, utilities
- `configs/`: concise experiment configs for released models and datasets
- `scripts/`: CLI wrappers for train, eval, export, and validation
- `tests/`: smoke tests and regression checks for the released surface
- `docs/`: repo layout notes, model selection notes, and reproducibility instructions
- `paper/`: final table and figure manifests, not every draft file
- `assets/`: a small number of figures or media files used in docs

## Command Surface

- `python scripts/train.py --dataset <name>`
- `python scripts/search.py --dataset <name>`
- `python scripts/test.py --dataset <name> --checkpoint <path>`
- `python scripts/check_repo.py`

## What The Final Repo Avoids

- giant log dumps
- temporary cleanup scripts for one server session
- dozens of abandoned model variants without documentation
- hidden dependencies on local dataset paths
- notebook-only pipelines for critical results

## Suggested End State

The final public repo should be able to answer these questions quickly:

1. How do I install the environment?
2. How do I train the main model on one dataset?
3. How do I evaluate and reproduce the paper metrics?
4. Which files correspond to the released RouteRec method?
5. Which outputs are expected after a successful run?
