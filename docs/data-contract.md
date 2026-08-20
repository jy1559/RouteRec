# Data and evaluation contract

RouteRec treats a reconstructed session as the prediction entity. Dataset
aliases are conveniences only; reproducibility depends on exact content and
preprocessing identity.

## Required split files

Each prepared dataset directory contains:

```text
<dataset>.train.inter
<dataset>.valid.inter
<dataset>.test.inter
```

These files are frozen chronological splits. They must not be concatenated and
re-split at training time. Interaction tables contain at least
`session_id`, `item_id`, `timestamp`, and `user_id`; feature-enabled releases add
the paper's engineered cue columns.

## Sequence construction

- `session_id` is the sequential entity.
- Training uses every valid prefix-target pair.
- Validation and test use the last target of each session.
- Sequence order follows timestamp order within the frozen session.
- Any maximum-history truncation must be recorded in the resolved config.

## Candidate and metric policy

The candidate vocabulary is fitted on the training split. Items absent from
training are masked during validation and test. A row whose positive target is
absent from the training vocabulary is excluded from the metric denominator
and its exclusion count is reported.

The authoritative metrics are HR@10, NDCG@10, and MRR@10. Checkpoint selection
uses:

```text
(validation HR@10 + validation NDCG@10 + validation MRR@10) / 3
```

Search and checkpoint selection must not inspect test performance. Test is
evaluated only after a configuration and best-validation checkpoint are frozen.

## Core-filtered rebuild

The public data pipeline provides three stages:

1. `rebuild_camera_ready_core5_basic.py` builds and validates a four-column,
   sessionized core-filtered release.
2. `build_camera_ready_core5_features.py` fits feature statistics on training
   data and publishes the experiment-facing feature tree.
3. `validate_camera_ready_core5_basic.py` and
   `validate_full_feature_dataset.py` independently verify the outputs.

Default outputs are repository-relative:

```text
Datasets/core5_basic/
Datasets/core5/
outputs/data_preparation/core5/
```

Builders stage into new directories, validate before publication, and refuse to
overwrite an existing target.

## Content identity

For every prepared dataset preserve:

- source URL/version and acquisition date where permitted;
- raw and generated file SHA-256 hashes;
- typed headers, row counts, session counts, and item counts;
- timestamp unit and sessionization parameters;
- core thresholds and convergence report;
- split membership and train-vocabulary filtering counts;
- feature normalization statistics fitted on training only;
- the exact builder commit and command.

Do not claim that two paths or aliases are the same dataset without matching
content evidence. Do not commit raw or prepared dataset files.
