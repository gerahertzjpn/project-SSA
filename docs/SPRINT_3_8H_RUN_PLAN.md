# Sprint 3 Fresh 8-Hour Run Plan

Date: 2026-06-04

Status: preparation only. No neural-network training has been started by this
document.

## Current Fixed Point

- Observation/action interface is current:
  - `OBSERVATION_VERSION = train_obs_v1.0`
  - `OBSERVATION_DIM = 1117`
  - `ACTION_DIM = 1356`
- B-side policy perspective is mirrored into A-side slots:
  - `B1 -> A1`
  - `B2 -> A2`
  - `B3 -> A3`
- The action mask and action decode are tested against this mirrored contract.
- Team checkpoints with `policy_states` are supported for B-side policy bots by
  mirrored slot lookup.
- Self-play infrastructure exists, but self-play throughput is below the target:
  - scripted no-policy benchmark: about `3.98M` world steps/hour;
  - h128 policy-opponent stochastic benchmark: about `2.61M` world steps/hour;
  - h128 policy-opponent argmax benchmark: about `2.30M` world steps/hour.

Self-play speed is therefore a known risk. For this specific 8-hour run, the
lower speed is explicitly accepted: self-play may start below `3M` world
steps/hour if the quality gates are closed. The low self-play throughput remains
a required later optimization item and must be reported clearly.

## Preflight Before Starting The 8-Hour Run

Run these before the actual training command:

```powershell
pytest -q
python scripts/preflight_sprint3_training.py --seed 20260604 --opponent-bot scripted --ally-bot scripted --hidden-size 128
python scripts/crash_test_simulation.py --matches 100 --seed-base 20260604
python scripts/benchmark_team_training_env_steps.py --duration-sec 30 --workers 16 --seed 20260604 --opponent-bot scripted --ally-bot scripted
```

Gate:

- all tests pass;
- crash test has no engine-breaking invariant failures;
- scripted throughput stays near or above `3M` world steps/hour;
- replay/UI issues are not counted here unless they reveal simulation bugs.

## Phase 1 - Fresh Scripted Training From Zero

Primary run command:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --overwrite --updates 300 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --opponent-bot scripted --ally-bot scripted --seed 20260604 --torch-threads 1 --random-baseline-episodes 100
```

This is the main Sprint 3 proof run. It trains a fresh learner team against the
scripted baseline under the corrected current rules.

Watch separately:

- train reward;
- eval argmax win-rate vs scripted;
- eval stochastic win-rate vs scripted;
- win-rate vs random baseline;
- draw-rate;
- normal wins vs death-time wins;
- entropy per player;
- dominant action rate per player;
- invalid action rate;
- action distribution by player;
- Lord kill/control/core impact.

Do not call the run successful because shaped reward rises. It must also win.

## Phase 2 - Real Resume Gate

After the first useful checkpoint, verify restart behavior with a real resume.
Example:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --resume --updates 1 --workers 16 --rollout-steps 512 --eval-episodes 10 --eval-interval 1 --checkpoint-interval 1 --opponent-bot scripted --ally-bot scripted --seed 20260604 --torch-threads 1
```

Gate:

- resumed update loads the latest checkpoint;
- total steps continue instead of resetting;
- checkpoint metadata remains compatible;
- no observation/action dimension mismatch.

If resume fails, stop and fix before spending the remaining run budget.

## Phase 3 - Checkpoint Sweep

Evaluate checkpoints by scripted performance, not by random-only performance:

```powershell
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --updates 50,100,150,200,250,300 --modes argmax,stochastic --seed-bases 20260604,20270604,20280604,20290604 --episodes-per-seed 50 --workers 16 --torch-threads 1 --opponent-bot scripted --ally-bot scripted --out-dir training_runs/evals/sprint3_team_scripted_fresh_v1_scripted_sweep
```

Also keep the random sanity sweep:

```powershell
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --updates 50,100,150,200,250,300 --modes argmax,stochastic --seed-bases 20260604,20270604,20280604,20290604 --episodes-per-seed 50 --workers 16 --torch-threads 1 --opponent-bot random --ally-bot random --out-dir training_runs/evals/sprint3_team_scripted_fresh_v1_random_sweep
```

Gate into the next step:

- argmax win-rate vs scripted is stably above the scripted-control baseline;
- stochastic eval does not collapse compared with argmax;
- performance is stable across seed bases;
- draw-rate is not masking weak play;
- invalid action rate is zero or effectively zero;
- action distribution does not collapse to one degenerate action;
- at least one checkpoint is clearly selectable as the valid Sprint 3 checkpoint.

If the best checkpoint only beats random but fails scripted, keep training or
change opponent mix. Do not start self-play from a weak checkpoint.

## Phase 4 - Build Candidate Pool

Only after Phase 3 gates pass:

```powershell
python scripts/build_self_play_pool.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --updates 100,150,200,250,300 --out training_runs/self_play_pools/sprint3_team_scripted_fresh_v1_pool.json
```

Pick one frozen checkpoint for the first self-play opponent. Prefer the best
scripted checkpoint, not necessarily the best random checkpoint.

## Phase 5 - Real Checkpoint Self-Play Benchmark

Before any self-play training, benchmark with the actual selected checkpoint:

```powershell
python scripts/benchmark_team_training_env_steps.py --duration-sec 30 --workers 16 --seed 20260604 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_300.pt --opponent-policy-mode stochastic --ally-bot scripted
python scripts/benchmark_team_training_env_steps.py --duration-sec 30 --workers 16 --seed 20260604 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_300.pt --opponent-policy-mode argmax --ally-bot scripted
```

Gate:

- target: at least `3M` world steps/hour;
- accepted exception for this run: below `3M` does not block self-play;
- if below target, report it as a known throughput debt and prefer a shorter
  scoped self-play segment if the remaining wall-clock budget is tight.

## Phase 6 - First Self-Play Only If Gates Pass

If all gates pass and speed is accepted:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_selfplay_vs_selected_v1 --overwrite --updates 100 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_300.pt --opponent-policy-mode stochastic --ally-bot scripted --seed 20260605 --torch-threads 1 --random-baseline-episodes 100
```

This first self-play run is not a full league. It is learner vs one frozen
policy team. It must preserve scripted-baseline skill.

Self-play success gate:

- does not lose scripted eval performance;
- learns useful variation against the frozen policy;
- stochastic behavior remains playable;
- checkpoint/resume works;
- no single exploit dominates the only frozen opponent.

## End-Of-Run Report

The final report should include:

- exact commands launched;
- wall-clock timing for each phase;
- checkpoint chosen and why;
- train reward vs eval win-rate;
- argmax and stochastic eval tables;
- random, scripted, and policy-opponent results separately;
- draw and death-time win rates;
- invalid-rate/action-distribution summary;
- Lord impact summary;
- throughput numbers;
- whether self-play was started, skipped, or blocked;
- next blocking issue.
