# MODEL_REGISTRY.md

## Population-18 — free-agent draft pool (trained 2026-06-09)

**Status: TRAINED** — 400/400 updates, 3.62M steps, exit 0. All 18 latest.pt present.
Key result: mean_episode_steps 223 → 148 (−34%), win rate ~0.5 throughout (healthy self-play).
Artifacts: `training_runs/population_18_v1/` (~1.1 GB, gitignored).
Registry with checkpoint paths: `training_runs/population_18_v1/registry.json`.

18 independent, faction-locked, uniquely-seeded nets. 6 per faction. No culling.
Architecture: `MaskedActorCritic` (hidden_size 128, ~738k params each),
`train_obs_v3.1` / `OBSERVATION_DIM 5188`. Spec source of truth:
`src/training/population.py::build_default_population`. The live, per-run registry
(checkpoint paths, updates trained, status) is generated at
`training_runs/<run>/registry.json` and `training_runs/<run>/MODEL_REGISTRY.md`.

genotype_hash = sha256 of {architecture, obs/action interface, faction,
hidden_size, learning_rate, entropy_coef, clip_coef, gae_lambda, recipe, seed}.

| net_id | faction | recipe | seed | lr | entropy | clip | genotype_hash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| humans_1 | humans | baseline | 20260601 | 0.0003 | 0.01 | 0.2 | `5f50eef25fc592cd` |
| humans_2 | humans | low_entropy | 20260602 | 0.0003 | 0.005 | 0.15 | `f0091163006d1aac` |
| humans_3 | humans | high_entropy | 20260603 | 0.0003 | 0.02 | 0.25 | `a7e56d4b506e4e15` |
| humans_4 | humans | fast_lr | 20260604 | 0.0005 | 0.01 | 0.2 | `e042a84c0268609b` |
| humans_5 | humans | slow_explore | 20260605 | 0.00015 | 0.015 | 0.2 | `ca03482d06585a55` |
| humans_6 | humans | alt_mix | 20260606 | 0.0004 | 0.007 | 0.18 | `759ad21d875829b7` |
| infected_1 | infected | baseline | 20260701 | 0.0003 | 0.01 | 0.2 | `0cab933c35c10206` |
| infected_2 | infected | low_entropy | 20260702 | 0.0003 | 0.005 | 0.15 | `4eb51290602309fe` |
| infected_3 | infected | high_entropy | 20260703 | 0.0003 | 0.02 | 0.25 | `3f4f1a895122d59e` |
| infected_4 | infected | fast_lr | 20260704 | 0.0005 | 0.01 | 0.2 | `01d1749c2f2eff01` |
| infected_5 | infected | slow_explore | 20260705 | 0.00015 | 0.015 | 0.2 | `ff61b4f4b26a44bb` |
| infected_6 | infected | alt_mix | 20260706 | 0.0004 | 0.007 | 0.18 | `dd0342458be2b583` |
| robots_1 | robots | baseline | 20260801 | 0.0003 | 0.01 | 0.2 | `a2ffbe970f3ac94f` |
| robots_2 | robots | low_entropy | 20260802 | 0.0003 | 0.005 | 0.15 | `cd28d4f41f203621` |
| robots_3 | robots | high_entropy | 20260803 | 0.0003 | 0.02 | 0.25 | `a5ecb8d48fbe9de5` |
| robots_4 | robots | fast_lr | 20260804 | 0.0005 | 0.01 | 0.2 | `e703ed57c82c4dcd` |
| robots_5 | robots | slow_explore | 20260805 | 0.00015 | 0.015 | 0.2 | `cfb835da4a6ee63d` |
| robots_6 | robots | alt_mix | 20260806 | 0.0004 | 0.007 | 0.18 | `14718fff2a4c628f` |

All 18 genotype hashes are distinct (verified by `assert_population_unique` and
`tests/test_population.py`). Generation 0, no parents (fresh population).

## Per-model schema (generated registry)

```text
net_id / faction / seed / recipe / genotype_hash / hidden_size /
learning_rate / entropy_coef / clip_coef / parameter_count /
checkpoint / policy_state_id / generation / parent_ids /
status (training|trained) / updates_trained
```
