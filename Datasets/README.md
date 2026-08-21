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

Place acquired source files in the following stable input layout before
running the builders:

```text
Datasets/release/
  beauty/beauty.inter + beauty.item
  foursquare/foursquare.inter + foursquare.item
  movielens1m/movielens1m.inter + movielens1m.item
  retail_rocket/retail_rocket.inter + retail_rocket.item
  kuairec/big_matrix.csv + item_categories.csv
  lastfm/lastfm.inter + lastfm.item
```

The first four interaction files must already contain the typed base columns
`session_id`, `item_id`, `timestamp`, and `user_id`. The KuaiRec CSV files use
the official `big_matrix` and item-category schemas. The LastFM interaction
table uses the same four typed base columns and is re-sessionized by the public
builder. Keep each dataset's acquisition record and license with the local
source; those files remain untracked.

To rebuild the core-filtered releases, inspect the required inputs and options:

```bash
python scripts/build_core5_splits.py --help
python scripts/validate_core5_splits.py --help
python scripts/build_core5_features.py --help
python scripts/validate_feature_dataset.py --help
```

The builders use frozen chronological splits, fit filtering/normalization state
without validation or test leakage, publish atomically, and refuse to replace an
existing target. Run `python scripts/check_data.py` after preparation.

See [`docs/data-contract.md`](../docs/data-contract.md) for the evaluation and
identity contract.
