# Datasets

Dataset files are local-only and are ignored by Git. Only this README is
tracked.

The repository supports two explicit data surfaces:

```text
Datasets/
  release/       acquired/prepared source datasets
  core5_basic/   four-column core-filtered audit output
  core5/         experiment-facing feature output
```

The six paper dataset identities are Beauty, Foursquare, KuaiRec, LastFM,
MovieLens-1M, and Retail Rocket. Acquisition and redistribution terms differ by
dataset, so raw or prepared files are not bundled here.

To rebuild the core-filtered releases, inspect the required inputs and options:

```bash
python scripts/rebuild_camera_ready_core5_basic.py --help
python scripts/build_camera_ready_core5_features.py --help
python scripts/validate_camera_ready_core5_basic.py --help
```

The builders use frozen chronological splits, fit filtering/normalization state
without validation or test leakage, publish atomically, and refuse to replace an
existing target. Run `python scripts/check_data.py` after preparation.

See [`docs/data-contract.md`](../docs/data-contract.md) for the evaluation and
identity contract.
