# Model selection policy

## RouteRec identity

The paper model is exposed as `RouteRec`. A main-paper run must pass the semantic validator against `configs/paper/models/routerec_a12.yaml`; the model name alone is not sufficient evidence that the A12 routing contract was used.

## Validation objective

All selection stages maximize:

```text
(HR@10 + NDCG@10 + MRR@10) / 3
```

on the frozen validation split. The selected checkpoint is the best epoch by this objective, not the final epoch. Historical experiments at another cutoff are not an alternative selection criterion.

Test is disabled during screen, HPO, and promotion. A test result may be produced only after a configuration has been frozen in a final campaign.

## Staged tuning

Recovered settings are center hypotheses because the exact final configurations may have been lost or may belong to a different protocol variant.

The camera-ready search order is:

1. verify data/model/evaluation identity;
2. paired LR/weight-decay screen at seed 42 with fixed structure;
3. promote validation leaders to larger budgets;
4. if necessary, open one structural/regularization/router block at a time;
5. freeze one configuration;
6. run a new final campaign on seeds 42/43/44 and evaluate test once per seed.

Search breadth, budgets, seeds, and promotion rules must be declared before inspecting results. Failed/missing candidates must be resolved before ranking an incomplete leaderboard.

Every published configuration should record its validation-only search space,
budget, selected checkpoint, and seed before test evaluation.

## Baselines are gated

The submitted comparison set is SASRec, GRU4Rec, TiSASRec, FEARec, DuoRec, BSARec, FAME, DIF-SR, FDSA, and RouteRec. The current registry is useful for code recovery, but baseline execution is not yet certified for camera-ready results.

Before a baseline enters a paper campaign, verify:

- the intended implementation and model-specific config path;
- the identical frozen-session and train-seen evaluation protocol;
- the same @10 metrics and composite validation rule;
- a documented, reasonably matched HPO budget;
- exact source/config/data identities and attempt artifacts.

Do not run every registered model merely because it imports successfully, and do not give RouteRec an undocumented wider search.
