# Reproducibility Notes

## Runtime Baseline

The clean installation path for this repository is a Python 3.10 or 3.11 environment.
RecBole currently pulls a `ray<=2.6.3` constraint that does not resolve cleanly on Python 3.12, so 3.12 should be treated as a manual-workaround environment rather than the default reproducibility target.

## Dataset Layout

RouteRec expects RecBole-style processed datasets under one of these roots, in this order:

1. `Datasets/release`
2. `Datasets/processed/feature_added_v4`
3. `Datasets/processed/feature_added_v3`
4. `Datasets/processed/basic`

Each dataset should live in its own subdirectory, for example:

```text
Datasets/release/movielens1m/
  movielens1m.train.inter
  movielens1m.valid.inter
  movielens1m.test.inter
  movielens1m.item
  feature_meta_v3.json
```

`Datasets/release` is the clean public-facing alias. In this workspace it can simply point at the restored `feature_added_v4` tree. The `feature_meta_v3.json` filename is still expected because the feature-bank schema version is retained from the earlier engineering pipeline.

The current runner infers `USER_ID_FIELD`, `ITEM_ID_FIELD`, `TIME_FIELD`, `load_col`, and split ratios from the dataset files themselves:

- field names are parsed from the typed header of `dataset.inter` or `dataset.train.inter`
- split ratios are read from `dataset.session_split_summary.json`, `dataset.v4_split_summary.json`, or `dataset.split_summary.json` when one of those files is present
- raw split files are kept for auditability, but they are not passed into RecBole benchmark mode because SequentialDataset benchmark mode expects pre-augmented `*_list` fields rather than raw interaction rows

If you have the prepared archive used during paper development, extract it from the repository root so that the bundled `Datasets/...` tree lands in place:

```bash
tar -xzf /path/to/FeaturedMoE_dataset_agent_backup_20260424.tar.gz -C /path/to/RouteRec
```

## Accepted Dataset Names

The CLI accepts both canonical RecBole ids and paper-friendly aliases:

- `beauty`
- `amazon_beauty`
- `foursquare`
- `kuairec` -> `KuaiRecLargeStrictPosV2_0.2`
- `lastfm` -> `lastfm0.03`
- `ml-1m` or `movielens1m` -> `movielens1m`
- `retail-rocket` or `retail_rocket` -> `retail_rocket`

## Paper Protocol Summary

The released experiments follow the sessionized temporal protocol described in the paper draft:

- Metrics: HR@$k$, NDCG@$k$, and MRR@$k$ with the main comparison reported on HR@10, NDCG@10, and MRR@20.
- Candidate space: seen-target evaluation, restricted to items observed in training.
- Sessionization: 30-minute inactivity threshold for Foursquare, KuaiRec, LastFM, ML-1M, and Retail Rocket.
- Beauty sessionization: 14-day inactivity threshold due to the sparse review stream.
- Filtering: remove sessions shorter than 5 and items with fewer than 3 interactions.
- Split: chronological 70% / 15% / 15% train / validation / test.
- Sampled subsets: KuaiRec uses the released 20% processed subset; LastFM uses the released 3% processed subset.

## Main Commands

Use the repository root as the working directory.

```bash
python3 scripts/check_repo.py
python3 -m unittest discover -s tests
```

```bash
python3 scripts/train.py --dataset ml-1m
python3 scripts/search.py --dataset kuairec --trials 12
python3 scripts/test.py --dataset lastfm --checkpoint /path/to/model.pth
```

You only need `--data-path` when you want to override the repository-local dataset roots.

## Outputs

- `outputs/train/*.json`: training summaries and best validation scores
- `outputs/search/*.json`: per-trial search records and the selected best trial
- `outputs/test/*.json`: evaluation summaries for a saved checkpoint

The CLI normalizes dataset aliases before applying presets and before locating the dataset directory, so the same command surface works even if the storage root changes later.
