# RouteRec

RouteRec is the paper-focused codebase for behavior-guided sparse routing in sequential recommendation. The public surface is intentionally narrow: a single released model name, paper-aligned dataset presets, bounded search ranges, and three CLI entrypoints for train, search, and test.

## Repository Layout

```text
RouteRec/
├── src/routerec/        # model code, utilities, dataset resolution
├── configs/             # RouteRec defaults, dataset presets, search spaces
├── scripts/             # train / search / test / repo checks
├── tests/               # smoke and regression tests for the released surface
├── docs/                # layout, model-selection, reproducibility notes
├── paper/               # paper-facing artifacts
└── assets/              # figures used in docs or paper assets
```

## Installation

RouteRec officially targets Python 3.10 or 3.11.

```bash
python3 -m pip install -e .
```

This installs the core runtime used by the released scripts, including RecBole, Torch, Hydra, and YAML support.

If you only have Python 3.12 available, upstream RecBole dependency resolution currently fails because of its pinned `ray` requirement. In that case, use a 3.10/3.11 environment for the clean one-command install path.

## Dataset Handling

The CLI resolves dataset storage automatically. When `--data-path` is omitted, RouteRec searches these repository-local roots in order:

1. `Datasets/release`
2. `Datasets/processed/feature_added_v4`
3. `Datasets/processed/feature_added_v3`
4. `Datasets/processed/basic`

This lets the released commands stay stable even if the backing data root changes later. The resolver also accepts paper-friendly aliases:

- `kuairec` -> `KuaiRecLargeStrictPosV2_0.2`
- `lastfm` -> `lastfm0.03`
- `ml-1m` -> `movielens1m`
- `retail-rocket` -> `retail_rocket`

If you are restoring the prepared processed archive, extract it from the repository root so that the archive's `Datasets/...` tree lands directly under this repo:

```bash
tar -xzf /path/to/FeaturedMoE_dataset_agent_backup_20260424.tar.gz -C /path/to/RouteRec
```

For the paper-facing command surface, RouteRec also supports a clean alias root at `Datasets/release`. In this workspace it can point to the restored `feature_added_v4` tree, so user-facing commands do not need to mention the engineering version baked into the original data pipeline.

More details are in `docs/reproducibility.md`.

At runtime, RouteRec infers the RecBole field schema directly from the dataset header and reads split ratios from the accompanying split-summary JSON when it is present. The raw `*.train.inter`, `*.valid.inter`, and `*.test.inter` files are retained as reproducibility artifacts; the released runner does not feed them through RecBole benchmark mode because these files are still raw interaction rows rather than pre-augmented sequence-list examples.

## Experimental Setup

The released experiment surface follows the paper's shared sessionized temporal protocol across six datasets: Beauty, Foursquare, KuaiRec, LastFM, ML-1M, and Retail Rocket.

- Main comparison metrics: HR@10, NDCG@10, and MRR@20
- Candidate set: seen-target evaluation
- Filtering: sessions shorter than 5 and items with fewer than 3 interactions are removed
- Sessionization: 30-minute threshold for Foursquare, KuaiRec, LastFM, ML-1M, and Retail Rocket; 14-day threshold for Beauty
- Split: chronological 70% / 15% / 15% train / validation / test
- Released sampled subsets: KuaiRec 20%, LastFM 3%

## Main Commands

Use the repository root as the working directory.

```bash
python3 scripts/check_repo.py
python3 -m unittest discover -s tests
```

```bash
python3 scripts/train.py --dataset ml-1m
```

```bash
python3 scripts/search.py --dataset kuairec --trials 12
```

```bash
python3 scripts/test.py --dataset lastfm --checkpoint /path/to/model.pth
```

For a quick smoke run, keep the model unsaved and reduce the budget:

```bash
python3 scripts/train.py --dataset ml-1m --epochs 1 --no-save-model
```

## Configuration Contract

- Base defaults: `configs/models/routerec_default.yaml`
- Dataset presets: `configs/models/routerec_dataset_presets.yaml`
- Bounded search ranges: `configs/search_spaces/paper_bounded_grid.yaml`

The search script samples inside the bounded ranges and applies dataset-specific learning-rate intervals from the search-space file after dataset alias normalization.

## Outputs

- `outputs/train/`: training summaries and best validation payloads
- `outputs/search/`: per-trial search records and selected best trial
- `outputs/test/`: checkpoint evaluation summaries

## Additional Notes

- The released model identity is always `RouteRec`.
- Dataset files are intentionally kept outside version control; see `.gitignore`.
- `docs/reproducibility.md` is the authoritative reference for dataset layout and paper-aligned settings.
