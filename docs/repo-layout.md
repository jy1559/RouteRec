# Repository layout

## Tracked public tree

```text
RouteRec/
  README.md
  environment.yml
  pyproject.toml
  configs/
    models/                model defaults and baseline settings
    paper/                 protocol and dataset configuration
    search_spaces/         disclosed bounded search space
  src/routerec/
    models/routerec/       stable RouteRec implementation package
    models/*.py            bundled baseline implementations
    datasets.py            dataset names and repository-relative discovery
    runner.py              training, validation selection, and test helpers
    session_data.py        frozen-split sequential conversion
  scripts/
    train.py               isolated one-model training
    evaluate.py            trusted-checkpoint test evaluation
    check_*.py             repository, data, and GPU checks
    *core5*.py             core-filtered dataset preparation and validation
  tests/                   unit and regression tests
  docs/                    public protocol and reproducibility documentation
  Datasets/README.md       acquisition and local layout guidance
  .github/workflows/       clean CPU validation workflow
```

`src/routerec/` is the executable source of truth.

## Untracked local tree

```text
Datasets/**      raw, prepared, and derived datasets
outputs/         runs, checkpoints, logs, and reports
artifacts/       exported experiment artifacts
local_archive/   historical recovery material
```

These paths must not be force-added. A public release may include a small
reviewed result manifest, but it must not include prepared datasets, private
paths, credentials, checkpoints, or operational logs.

## Public-release standard

A reader should be able to determine:

1. the exact data, split, candidate, and metric contract;
2. the complete model configuration used for a run;
3. how to rebuild or validate each dataset without bundled restricted data;
4. how validation selection is separated from test evaluation;
5. how per-seed values and aggregate statistics are regenerated;
6. which claims are limitations rather than demonstrated results.
