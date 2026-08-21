# Reproducibility

This document defines the minimum evidence required for a reproducible RouteRec
result. A smoke test demonstrates that code executes; it does not reproduce a
paper result.

## Environment

`environment.yml` records the reference Python, PyTorch, RecBole, NumPy, and
Hydra versions. Create and inspect the environment with:

```bash
bash scripts/create_env.sh
micromamba activate routerec
python -m pip install -e ".[test]"
python scripts/check_repo.py
python -m pytest -q
```

Record any deliberate dependency or accelerator change with the run. Do not
load an untrusted checkpoint with Python pickle compatibility enabled.

## Source and configuration

For every reported run, preserve:

- the Git commit;
- the complete resolved configuration;
- the model and dataset identifiers;
- the command, seed, start/end time, and software versions;
- the GPU model and visible-device mapping;
- the selected validation checkpoint and epoch.

`src/routerec/` is executable truth. `configs/paper/` records the paper-facing
protocol and model contract. Historical recovery material and current defaults
are not substitutes for a resolved run configuration.

## Data and evaluation

All paper-facing runs use:

- `session_id` as the sequential entity;
- frozen train/validation/test split membership;
- all-prefix training and last-target validation/test examples;
- a training-seen candidate set with unseen-positive rows excluded;
- HR@10, NDCG@10, and MRR@10;
- the mean of those three validation metrics for checkpoint selection;
- no test evaluation during search or model selection.

See [`data-contract.md`](data-contract.md). A dataset name is not a content
identity: preserve source-file hashes, row counts, headers, preprocessing
parameters, and the generated split summary.

## Running one configuration

```bash
python scripts/train.py \
  --dataset beauty_core5_v1 \
  --model RouteRec \
  --seed 42 \
  --epochs 100
```

Training is validation-only by default and saves the best validation
checkpoint. After the configuration has been frozen, evaluate a trusted local
attempt exactly once:

```bash
python scripts/evaluate.py --attempt-dir outputs/runs/<attempt> --gpu 0
```

Changing `--data-path` is only a host-path relocation; the data content must
remain identical.

## Evidence required per result

At minimum retain:

1. resolved configuration and environment versions;
2. source and dataset content identities;
3. raw split counts and generated sample counts;
4. epoch history and validation objective;
5. selected checkpoint and artifact hashes;
6. test metrics only for a predeclared final run;
7. parameter, timing, and memory evidence for efficiency claims;
8. stdout/stderr and failure records.

For paired statistical analysis, also retain per-session predictions or ranks
for identical evaluated sessions.

## Seeds and reporting

The paper protocol uses independent seeds 42, 43, and 44. Report every seed,
the arithmetic mean, and sample standard deviation (`ddof=1`). Do not infer
statistical significance from three unpaired means. For small margins,
predeclare a paired session-level bootstrap or randomization analysis, report
effect sizes and confidence intervals, and correct the declared comparison
family.

## Publication boundary

Datasets, checkpoints, outputs, private instructions, credentials, host details,
and local archives stay outside Git. Publish a release tag only after the code,
configs, data identities, and paper tables have been reviewed together.
