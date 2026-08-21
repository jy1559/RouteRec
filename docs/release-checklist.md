# Public release checklist

## Repository boundary

- [ ] Only source, configs, tests, and public documentation are tracked.
- [ ] No datasets, checkpoints, outputs, logs, credentials, private paths, or
      server details appear in the Git index.
- [ ] A clean clone passes `python scripts/check_repo.py`, `compileall`, and the
      complete unit-test suite.

## Reproducibility

- [ ] Dataset acquisition guidance and redistribution limits are documented.
- [ ] Every reported dataset has a content identity and validated split summary.
- [ ] Final model configs, seeds, selection rule, and test boundary match the
      paper.
- [ ] Reported tables and figures are generated from reviewed machine-readable
      artifacts.

## Project metadata

- [x] The author list and order are final, then `CITATION.cff` is added.
- [x] The repository license is chosen by the rights holders and added.
- [x] Paper DOI, venue metadata, and release version are added only after they
      are authoritative.

## Publication

- [ ] CI passes on the intended public branch.
- [ ] README commands have been rerun from a clean environment.
- [ ] The release diff and tag contents receive owner review before publication.
