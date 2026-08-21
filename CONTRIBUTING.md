# Contributing

Thank you for helping improve RouteRec. Keep changes focused, reproducible, and
separate from private experiment artifacts.

## Before opening a change

1. Explain the problem and the intended scientific or engineering effect.
2. Preserve the frozen split, candidate-set, and @10 metric contracts unless a
   protocol change is explicitly proposed and documented.
3. Add or update regression tests for behavior changes.
4. Do not commit datasets, checkpoints, logs, generated outputs, credentials,
   host paths, or unpublished result values.

## Local checks

```bash
python -m pip install -e ".[test]"
python scripts/check_repo.py
python -m compileall -q src scripts
python -m pytest -q
```

For data-pipeline changes, also run the relevant builder and independent
validator on a small disposable fixture. A passing test or smoke run must not
be described as reproducing a paper result.

## Pull requests

Include the exact commands run, the affected model/data contract, and any
backward-compatibility implications. Keep experimental scores out of a pull
request unless their complete provenance and publication status are clear.
