# RouteRec

Official implementation of **RouteRec: Behavior-Guided Sparse Routing for
Sequential Recommendation**.

This repository contains the RouteRec model, the comparison baselines used by
the project, reproducible training and evaluation entry points, and the source
code for rebuilding the sessionized core-filtered datasets. Prepared datasets,
checkpoints, logs, and generated result tables are intentionally excluded.

## Evaluation contract

- Prediction unit: `session_id`.
- Split policy: use the supplied `*.train.inter`, `*.valid.inter`, and
  `*.test.inter` files as frozen chronological splits.
- Training examples: all valid prefixes; validation and test: the last target
  of each session.
- Candidate set: items observed in training. Rows with unseen positive targets
  are excluded from the validation/test denominator.
- Metrics: HR@10, NDCG@10, and MRR@10.
- Model selection: mean validation HR@10, NDCG@10, and MRR@10.
- Test use: evaluate a frozen best-validation checkpoint once.

The default RouteRec configuration uses the A12 layout:

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

The reference environment uses Python 3.10, PyTorch 2.6, and RecBole 1.2.1.

```bash
bash scripts/create_env.sh
micromamba activate routerec
python -m pip install -e ".[test]"
python scripts/check_repo.py
python -m pytest -q
```

Check local data and accelerator access separately:

```bash
python scripts/check_data.py
python scripts/check_gpu.py --min-devices 1
```

## Data preparation

Dataset files are not distributed in this repository. Acquire them under
`Datasets/` according to [Datasets/README.md](Datasets/README.md) and the
[data contract](docs/data-contract.md).

The public core5 pipeline is:

```bash
python scripts/build_core5_splits.py --help
python scripts/validate_core5_splits.py --help
python scripts/build_core5_features.py --help
python scripts/validate_feature_dataset.py --help
```

Builders use repository-relative defaults, stage outputs before publication,
and refuse to overwrite an existing release.

## Training and evaluation

Run one model on one dataset:

```bash
python scripts/train.py \
  --dataset beauty_core5_v1 \
  --model RouteRec \
  --epochs 100 \
  --seed 42
```

Add `--use-dataset-preset` to apply the documented dataset initialization
preset. Training selects and saves the best validation checkpoint without
evaluating test by default.

Evaluate a trusted local attempt after its configuration is frozen:

```bash
python scripts/evaluate.py --attempt-dir outputs/runs/<attempt> --gpu 0
```

RouteRec uses one process per GPU. Run independent jobs on separate devices;
do not turn a single run into distributed training.

## Baselines

The comparison surface includes SASRec, GRU4Rec, TiSASRec, DuoRec, BSARec,
FEARec, DIFSR, FAME, and FDSA. SASRec and GRU4Rec use RecBole implementations;
the remaining implementations are bundled under `src/routerec/models/`.

## Repository layout

```text
configs/       model, dataset, protocol, and bounded-search configuration
Datasets/      local data only; README is the sole tracked file
docs/          data, selection, reproducibility, and release documentation
scripts/       setup, checks, data preparation, training, and evaluation
src/routerec/  RouteRec and baseline implementations
tests/         unit and regression tests
```

See [docs/reproducibility.md](docs/reproducibility.md) for the evidence required
to support a reported result. A successful smoke test is execution evidence,
not reproduction of a paper score.

## Citation

If you use RouteRec, please cite the CIKM '26 paper. Machine-readable metadata
is available in [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{song2026routerec,
  author    = {Junyeong Song and Jaemin Yoo},
  title     = {RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  doi       = {10.1145/3799682.3841104},
  isbn      = {979-8-4007-2539-5}
}
```

## License

RouteRec's original code, configuration, tests, and documentation are licensed
under the [Apache License 2.0](LICENSE). Bundled baseline files retain their
applicable upstream licenses and attribution; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The article's ACM eRights metadata specifies the Creative Commons Attribution
4.0 International (CC BY 4.0) license. That publication license applies to the
article and is separate from the software licenses in this repository.
