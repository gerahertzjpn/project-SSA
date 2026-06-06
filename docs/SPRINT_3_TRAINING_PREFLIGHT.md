# Sprint 3 Training Preflight

Date: 2026-06-04

Status: preparation only. Do not start neural-network training until the user
explicitly approves the run.

Latest preparation result on 2026-06-04:

- no-learning preflight: passed;
- team random/random throughput, 16 workers, 10 sec: `6.08M` world
  steps/hour;
- team scripted/scripted throughput, 16 workers, 10 sec: `3.87M` world
  steps/hour;
- simulation crash gate, 100 matches: `issue_summary {}`;
- full pytest: `234 passed`;
- baseline-only CLI dry run with `updates=0`: completed with `total_steps=0`.
- existing `training_runs/sprint3_team_scripted_warm_v1/checkpoints/latest.pt`
  failed compatibility preflight because it does not carry current
  `train_obs_v1.0 / 1117 / 1356` metadata. Do not resume it.

## Current Interface

```text
OBSERVATION_VERSION = train_obs_v1.0
OBSERVATION_DIM = 1117
ACTION_DIM = 1356
controlled players = A1,A2,A3
default workers = 16
torch threads = 1
```

The current action interface is one categorical policy output with bounded
`PRODUCE_MOVE` actions. The policy can produce one unit and move one group in
the same world tick, but it still has no per-unit micro-control.

## Checkpoint Rule

Do not warm-start from Sprint 2 checkpoints for the next large run.

Reason: Sprint 2 checkpoints predate the current 2D combat, Lord-route
observation, `JOIN_LORD` / `HUNT_LORD`, and `PRODUCE_MOVE` interface. They are
proof evidence only. A valid warm-start must be explicitly compatible with
`train_obs_v1.0`, `OBSERVATION_DIM=1117`, and `ACTION_DIM=1356`.

## No-Learning Preflight

Run before any real training:

```powershell
python scripts/preflight_sprint3_training.py --seed 20260604 --opponent-bot scripted --ally-bot scripted
```

Expected:

- `ok: true`;
- observations are finite and shape `1117`;
- masks are shape `1356` and include `HOLD`;
- sampled legal actions advance the world by exactly one tick;
- replay/trace recording are disabled;
- policy forward pass is finite;
- temporary team checkpoint save/load roundtrip succeeds.

Optional compatibility check for a team checkpoint:

```powershell
python scripts/preflight_sprint3_training.py --checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/latest.pt
```

## Throughput Gate

Run a team-environment benchmark without learning:

```powershell
python scripts/benchmark_team_training_env_steps.py --duration-sec 30 --workers 16 --seed 20260604 --opponent-bot scripted --ally-bot scripted
```

Track both:

- `world_steps_per_hour`: simulator ticks/hour;
- `agent_decisions_per_hour`: three policy decisions per world tick.

The project speed gate is still at least `3M` world steps/hour unless we make an
explicit redesign decision.

## Crash Gate

Run the simulation invariant sweep:

```powershell
python scripts/crash_test_simulation.py --matches 100 --seed-base 2026060200
```

Expected:

```text
seeds_with_issues 0
issue_summary {}
```

This catches engine-breaking replay/state bugs such as:

- battles without hostile sides;
- live units owned by dead players;
- dead players still owning nodes or moving groups;
- resurrected Lord missing or duplicated;
- Lord targeting dead cores;
- Lord moving without `LORD_PUSH` intent.

## Test Gate

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

## Baseline-Only Eval Without Training

If we want fresh random-legal baseline metrics before the real run, use
`updates=0`. This creates a run directory and logs baseline data, but performs no
PPO update:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_preflight_baseline_v1 --overwrite --updates 0 --workers 16 --rollout-steps 512 --eval-episodes 0 --checkpoint-interval 1 --opponent-bot scripted --ally-bot scripted --seed 20260604 --torch-threads 1 --random-baseline-episodes 100
```

Latest 20-episode dry-run result was only a smoke check:

- random-legal team win-rate: `50%`;
- loss-rate: `45%`;
- draw-rate: `5%`;
- normal win-rate: `15%`;
- death-time win-rate: `35%`;
- invalid action rate: `0.0` for all three controlled players;
- total PPO training steps: `0`.

Use `100+` episodes for a real baseline table before judging quality.

## First Real Training Command

Only after explicit approval:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --overwrite --updates 300 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --opponent-bot scripted --ally-bot scripted --seed 20260604 --torch-threads 1 --random-baseline-episodes 100
```

## Metrics To Watch During Training

- eval argmax win-rate vs scripted;
- eval stochastic win-rate vs scripted;
- random-legal baseline gap;
- scripted-control baseline gap;
- draw-rate;
- normal wins vs death-time wins;
- reward, shaping, and terminal reward separately;
- entropy per agent;
- dominant action rate per agent;
- invalid action rate;
- duplicate-target and idle-with-units diagnostics when available;
- Lord kill/control/core-destroy stats.

Learning is not accepted by reward alone. A promising run must show both reward
movement and real eval win-rate movement.
