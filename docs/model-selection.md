# Model Selection Notes

## Final Public Model Identity

- Single exposed model name: `RouteRec`
- Default config entry: `model: RouteRec`
- CLI default (`train/search/test`) also uses `RouteRec`

## Defaults and Presets

- Generic default: `configs/models/routerec_default.yaml`
- Dataset presets: `configs/models/routerec_dataset_presets.yaml`
- Presets are intended as practical starting points, not universal optima.

## Bounded Search Policy

`configs/search_spaces/paper_bounded_grid.yaml` defines the tuning envelope:

- shared optimization ranges
- RouteRec-specific ranges
- dataset-specific learning-rate intervals

`scripts/search.py` samples within these bounds and writes all trial records to `outputs/search`.
