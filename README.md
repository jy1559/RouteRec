# RouteRec

RouteRec is the reference implementation of behavior-guided sparse routing for
sequential recommendation. This branch is an independently assembled
camera-ready release candidate; it is not a final tagged release yet.

The repository contains the RouteRec model, the paper baselines, reproducible
training and evaluation entry points, and source code for rebuilding the
sessionized core-filtered datasets. Prepared datasets, checkpoints, experiment
logs, and result tables are intentionally excluded.

## Evaluation contract

- Prediction unit: `session_id`.
- Split policy: consume the provided `*.train.inter`, `*.valid.inter`, and
  `*.test.inter` files as frozen chronological splits.
- Training examples: all valid prefixes; validation and test: the last target
  of each session.
- Candidate set: items observed in training. Rows with unseen positive targets
  are excluded from the validation/test denominator.
- Metrics: HR@10, NDCG@10, and MRR@10.
- Model selection: the mean of validation HR@10, NDCG@10, and MRR@10.
- Test use: after validation selection, evaluate each frozen checkpoint once.

The default RouteRec configuration uses the submitted A12 layout:

```text
[attn, macro_ffn, mid_ffn, attn, micro_ffn]
wrapper: w5_exd
group router: feature, top-3
intra-group router: hidden + feature, top-2
active experts: 3 x 2 = 6
global top-k: disabled
load-balancing loss: disabled
```

## Installation

The pinned environment targets Python 3.10, PyTorch 2.6, and RecBole 1.2.1.

```bash
bash scripts/create_env.sh
micromamba activate routerec
python -m pip install -e .
python scripts/check_repo.py
python -m unittest discover -s tests
```

GPU availability and dataset files are checked separately:

```bash
python scripts/check_gpu.py --min-devices 1
python scripts/check_data.py
```

## Data preparation

Dataset files are not distributed in this repository. Place acquired source
files under `Datasets/` as described in [`Datasets/README.md`](Datasets/README.md)
and [`docs/data-contract.md`](docs/data-contract.md).

The six-dataset core-filtering pipeline is implemented in:

```bash
python scripts/rebuild_camera_ready_core5_basic.py --help
python scripts/build_camera_ready_core5_features.py --help
python scripts/validate_camera_ready_core5_basic.py --help
```

All default output paths are repository-relative and may be overridden from the
command line. Builders publish into a new directory and refuse to overwrite an
existing release.

## Training and evaluation

Run one model on one dataset:

```bash
python scripts/train.py \
  --dataset beauty_core5_v1 \
  --model RouteRec \
  --epochs 100
```

Evaluate a trusted best-validation checkpoint:

```bash
python scripts/test.py --attempt-dir outputs/runs/<attempt> --gpu 0
```

RouteRec uses one process per GPU; multiple GPUs should run independent jobs,
not a distributed copy of one job.

## Baselines

The paper baseline surface includes SASRec, GRU4Rec, TiSASRec, DuoRec, BSARec,
FEARec, DIFSR, FAME, and FDSA. SASRec and GRU4Rec use RecBole implementations;
the remaining implementations are bundled under `src/routerec/models/`.

## Repository layout

```text
configs/       model, dataset, and protocol configuration
Datasets/      local data only; README is the sole tracked file
docs/          protocol and reproducibility documentation
scripts/       setup, validation, data preparation, train, and test entry points
src/routerec/  RouteRec and baseline implementations
tests/         unit and regression tests
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for artifact and
reporting requirements. Do not treat a successful smoke test as reproduction of
the paper's reported results.
