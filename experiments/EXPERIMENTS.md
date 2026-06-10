# EXPERIMENTS.md

```text
run_id:
date:
goal:
hypothesis:
git_state:
config:
data_version:
models:
metrics:
status:
conclusion:
next_step:
```

## population_18_v1 (prepared, not yet trained)

```text
run_id:        population_18_v1
date:          2026-06-09 (prepared)
goal:          18 unique faction-locked nets (6/faction) able to play any
               position with any ally; frozen-snapshot self-play; no culling.
hypothesis:    mixed-faction self-play with position + teammate randomization
               yields a composable free-agent draft pool without selection.
git_state:     branch main (see config.json git_commit at launch)
config:        scripts/train_population.py (defaults: updates 400,
               episodes-per-net 3, snapshot-interval 5); spec in
               src/training/population.py::build_default_population
data_version:  n/a (RL self-play, env-generated experience)
models:        18 nets, MaskedActorCritic hidden 128 (~738k params each),
               train_obs_v3.1 / obs 5188; 18 distinct genotype hashes
metrics:       training_runs/population_18_v1/metrics.jsonl (per-update
               controlled_win_rate, loss_by_net, coverage)
status:        TRAINED — 400/400 updates, exit 0, no crashes
conclusion:    All 18 nets trained for full 400 updates × 3 ep/net = 3.62M total
               steps. mean_episode_steps dropped 223.9 → 148.3 (−34%, 75 steps
               shorter) — clear evidence nets learned to end games decisively
               rather than stalling to max_ticks. Win rate oscillated healthy
               ~0.5 throughout (frozen-snapshot self-play working correctly; not
               pinned at 1.0 or 0.0). No NaN/Inf/OOM. All 18 latest.pt present.
next_step:     teammate-swap / role-stress eval; ready for league draft (no culling)
```

See `docs/cloudcode/POPULATION_18_RUNBOOK.md` for launch + monitoring.
