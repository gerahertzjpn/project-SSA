# Sprint 3 Self-Play Preparation

Date: 2026-06-04

Status: infrastructure prepared, no neural-network training started.

Latest no-learning smoke checks on 2026-06-04:

- policy-opponent env preflight with a synthetic compatible checkpoint: passed;
- `train_team_policy.py --opponent-bot policy --updates 0`: passed with
  `total_steps=0`;
- `evaluate_team_checkpoints.py --opponent-bot policy`: passed on a 1-episode
  smoke eval;
- `build_self_play_pool.py`: wrote a compatible pool manifest;
- policy-opponent throughput with synthetic `hidden_size=32` checkpoint:
  - stochastic opponent: about `2.60M` world steps/hour;
  - argmax opponent: about `2.29M` world steps/hour.
- policy-opponent throughput with synthetic `hidden_size=128` checkpoint:
  - stochastic opponent: about `2.61M` world steps/hour;
  - argmax opponent: about `2.30M` world steps/hour.
- scripted no-policy throughput on the same 16-worker benchmark:
  about `3.98M` world steps/hour.

The policy-opponent throughput is below the `3M` world-steps/hour training
target. For the planned fresh Sprint 3 run, the slower self-play budget is
explicitly accepted after the scripted-quality gates are closed. Keep this as a
known throughput debt and report the real-checkpoint speed before starting
self-play.

## Goal

After the fresh Sprint 3 scripted-baseline run produces a compatible
`train_obs_v1.0` team checkpoint, we need a safe path into self-play without
changing the RL interface.

The first self-play mode is intentionally simple:

```text
current learner team A1,A2,A3
vs
frozen policy team B1,B2,B3 loaded from a compatible team checkpoint
```

This is not full league matchmaking yet. It is the bridge from Sprint 3 into the
first Sprint 4-style pool.

## Current Support

The engine/training stack now supports:

- `TeamTrainingEnvConfig(opponent_bot="policy")`;
- `opponent_policy_checkpoint=<team checkpoint>`;
- `opponent_policy_deterministic=False` for stochastic frozen opponents;
- `opponent_policy_deterministic=True` for argmax frozen opponents;
- `PolicyBot` mirrored B-side slot loading:
  - `B1` can use `A1` policy state;
  - `B2` can use `A2` policy state;
  - `B3` can use `A3` policy state.

No action-space or observation-space change is introduced by this.

## Compatibility Rule

Only checkpoints with current metadata may enter self-play:

```text
OBSERVATION_VERSION = train_obs_v1.0
OBSERVATION_DIM = 1117
ACTION_DIM = 1356
```

Old Sprint 2 and old Sprint 3 warm checkpoints are rejected.

## Build A Pool Manifest

After a real fresh Sprint 3 run:

```powershell
python scripts/build_self_play_pool.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --updates 100,150,200,250,300 --out training_runs/self_play_pools/sprint3_team_scripted_fresh_v1_pool.json
```

This writes a JSON manifest with compatible checkpoint paths and metadata.

The current trainer does not yet sample directly from the manifest. For the
first self-play run, pick one frozen checkpoint from the manifest explicitly.
Manifest-driven opponent sampling is the next league/pool step.

## Evaluate Candidate Against Frozen Policy

Policy-vs-policy eval, deterministic learner against stochastic frozen opponent:

```powershell
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --updates 300 --modes argmax,stochastic --seed-bases 20260604,20270604,20280604,20290604 --episodes-per-seed 50 --workers 16 --torch-threads 1 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_200.pt --opponent-policy-mode stochastic --out-dir training_runs/evals/sprint3_team_u300_vs_frozen_u200
```

Run both opponent modes when deciding whether a checkpoint is robust:

```text
--opponent-policy-mode stochastic
--opponent-policy-mode argmax
```

## First Self-Play Training Command

Only after explicit approval, only after a compatible frozen checkpoint exists,
and only after the policy-opponent throughput gate is accepted:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_selfplay_vs_u200_v1 --overwrite --updates 300 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_200.pt --opponent-policy-mode stochastic --ally-bot scripted --seed 20260605 --torch-threads 1 --random-baseline-episodes 100
```

Notes:

- The learner is fresh unless `--resume` is explicitly used.
- The opponent is frozen. It does not learn during this run.
- Use stochastic frozen opponents first because the future league path expects
  stochastic policies.
- Use argmax frozen opponents as a harder exploit check, not as the only
  training opponent.

## What To Measure

Self-play is dangerous if we only watch reward. Track:

- win-rate vs scripted baseline;
- win-rate vs random-legal;
- win-rate vs frozen policy opponent;
- argmax and stochastic eval separately;
- draw-rate;
- normal wins vs death-time wins;
- entropy per agent;
- dominant action rate per agent;
- invalid action rate;
- action-sequence diversity;
- Lord kill/control/core impact;
- whether behavior still works in replay, not only in metrics.

## Gate Into League / Sprint 4

Do not start a true league until:

- fresh scripted-baseline checkpoint is stable;
- policy-vs-policy eval works with both argmax and stochastic modes;
- at least one self-play run can checkpoint/resume;
- frozen opponent checkpoints are compatible and reproducible;
- no single checkpoint dominates only by exploiting one frozen opponent;
- self-play does not destroy scripted-baseline performance.

The next implementation step after this preparation is manifest-driven opponent
sampling:

```text
sample opponent checkpoint from pool per episode or per worker
log opponent id in every episode summary
periodically add new learner snapshots to pool
keep scripted/random anchors in the opponent mix
```
