# Sprint 3 Progress Log

Started: 2026-06-03T13:50:15+09:00

## Baseline Snapshot

- Start commit: `87a6b03411b190162040ae285c6ad409b6adf24e`
- Worktree at start: clean
- Active plan: `docs/SPRINT_3_PROJECT_PLAN.md`
- Accepted Sprint 2 checkpoint:
  `training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt`

## 0. Preflight

Commands:

```powershell
python -m pytest -q
python scripts/evaluate_checkpoints.py --run-dir training_runs/sprint2_scripted_clean_v1 --updates 100 --modes argmax,stochastic --seed-bases 20260603,20270603 --episodes-per-seed 25 --workers 16 --torch-threads 1 --ally-bot scripted --opponent-bot scripted --include-random-legal --out-dir training_runs/evals/sprint3_preflight_sprint2_checkpoint
```

Results:

- Tests: `124 passed in 16.43s`
- Scripted control over 50 episodes: `34%` win, `64%` loss, `2%` draw
- Random-legal A1 over 50 episodes: `26%` win, `38%` loss, `36%` draw
- Sprint 2 checkpoint update 100 argmax over 50 episodes:
  - `100%` win
  - `0%` loss
  - `0%` draw
  - `56%` normal wins
  - `44%` death-time wins
  - invalid action rate `0.0`
- Sprint 2 checkpoint update 100 stochastic over 50 episodes:
  - `78%` win
  - `10%` loss
  - `12%` draw
  - `18%` normal wins
  - `60%` death-time wins
  - invalid action rate `0.0`

Decision:

- Sprint 2 checkpoint and eval loop are healthy enough to use as Sprint 3 warm
  start/reference.
- Proceed to multi-agent synchronized 3v3 training infrastructure.

## 1. Team Training Environment

Implemented:

- `src/training/team_env.py`
- `TeamTrainingEnvConfig`
- `TeamTrainingEnv`
- synchronized `step({"A1": action, "A2": action, "A3": action})`
- one engine tick per multi-agent step
- per-player observations, masks, decoded actions, invalid-action accounting
- shared team terminal reward and team shaping by default
- dead controlled players are masked to `HOLD` only

Test command:

```powershell
python -m pytest tests/test_team_training_env.py -q
```

Result:

- `7 passed in 1.46s`

Decision:

- The first Sprint 3 infrastructure requirement is met: three controlled
  teammates can act on the same world tick without converting 3v3 into three
  sequential single-agent turns.

## 2. Team PPO, CLI, Resume, Eval, Diagnostics

Implemented:

- `src/training/team_ppo_trainer.py`
- `scripts/train_team_policy.py`
- `scripts/evaluate_team_checkpoints.py`
- `scripts/analyze_team_coordination.py`
- `tests/test_team_ppo_trainer.py`

Key design:

- Three separate policies/optimizers are trained for `A1`, `A2`, and `A3`.
- Rollouts are collected from one synchronized team environment.
- Checkpoints store separate `policy_states` and `optimizer_states` per player.
- Sprint 2 accepted checkpoint can warm-start all three team policies.
- Eval supports argmax and stochastic modes, plus random-legal and scripted
  controls.
- Coordination diagnostics track duplicate move targets, movement into current
  node battles, idle-with-units, action distributions, and basic Lord impact.

Test/smoke commands:

```powershell
python -m pytest tests/test_team_training_env.py tests/test_team_ppo_trainer.py -q
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_smoke --overwrite --updates 1 --workers 2 --rollout-steps 32 --eval-episodes 2 --eval-interval 1 --checkpoint-interval 1 --random-baseline-episodes 2 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1 --hidden-size 128
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_smoke --resume training_runs/sprint3_team_smoke/checkpoints/latest.pt --updates 1 --workers 2 --rollout-steps 32 --eval-episodes 2 --eval-interval 1 --checkpoint-interval 1 --random-baseline-episodes 0 --opponent-bot scripted --seed 20260603 --torch-threads 1 --hidden-size 128
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_smoke --updates latest --modes argmax,stochastic --seed-bases 20260603 --episodes-per-seed 2 --workers 2 --torch-threads 1 --opponent-bot scripted --include-random-legal --include-scripted-control --out-dir training_runs/evals/sprint3_team_smoke_eval
python scripts/analyze_team_coordination.py --checkpoint training_runs/sprint3_team_smoke/checkpoints/latest.pt --episodes 2 --seed 20260603 --mode argmax --opponent-bot scripted --out analysis/sprint3_team_smoke_coordination.json
```

Results:

- Team env/trainer tests: `12 passed`
- Team trainer smoke wrote `training_runs/sprint3_team_smoke/checkpoints/latest.pt`
- Resume smoke started from `start_update: 2`
- Team eval smoke wrote `training_runs/evals/sprint3_team_smoke_eval/summary.json`
- Coordination smoke wrote `analysis/sprint3_team_smoke_coordination.json`

Decision:

- The Sprint 3 training/eval process path is alive end-to-end on tiny settings.
- Next step is a full test suite run before starting the first real 16-worker
  team training run.

## 3. Full Test Gate Before Real Run

Command:

```powershell
python -m pytest -q
```

Result:

- `136 passed in 17.78s`

Decision:

- Safe to start the first real Sprint 3 team training run from a fixed commit.

## 4. Baseline Performance Fix

First real run attempt:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_warm_v1 --overwrite --updates 300 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --random-baseline-episodes 50 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1
```

Observation:

- The process started correctly from clean commit `6d4e8a2`, but the initial
  random-legal baseline produced no output for several minutes.
- Root cause: team random-legal baseline in `train_team_policy.py` was
  sequential, unlike rollout collection and checkpoint eval.

Fix:

- Added `evaluate_random_legal_team_parallel(...)`.
- `train_team_policy.py` now uses the parallel baseline path.

Verification:

```powershell
python -m py_compile src/training/team_ppo_trainer.py scripts/train_team_policy.py
python -m pytest tests/test_team_ppo_trainer.py -q
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_baseline_smoke --overwrite --updates 0 --workers 4 --rollout-steps 32 --eval-episodes 2 --random-baseline-episodes 4 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1
```

Results:

- `tests/test_team_ppo_trainer.py`: `5 passed`
- Baseline smoke completed and wrote
  `training_runs/sprint3_team_baseline_smoke/metrics.jsonl`

Decision:

- Restart the real run after committing this performance fix.

## 5. Eval Performance Fix

Observation during real run:

- Update 10 successfully saved a checkpoint and produced the first real argmax
  eval.
- Update 10 argmax eval over 50 episodes:
  - `86%` win
  - `12%` loss
  - `2%` draw
  - `44%` normal wins
  - `42%` death-time wins
  - invalid action rate `0.0`
- However the update took `344.577s` because `evaluate_team_policies(...)` in
  the train CLI was sequential.

Fix:

- Added `evaluate_team_policies_parallel(...)`.
- `train_team_policy.py` now uses the parallel eval path for training-time
  argmax eval.

Verification:

```powershell
python -m py_compile src/training/team_ppo_trainer.py scripts/train_team_policy.py
python -m pytest tests/test_team_ppo_trainer.py -q
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_parallel_eval_smoke --overwrite --updates 1 --workers 4 --rollout-steps 32 --eval-episodes 4 --eval-interval 1 --checkpoint-interval 1 --random-baseline-episodes 0 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1
```

Results:

- `tests/test_team_ppo_trainer.py`: `5 passed`
- Parallel eval smoke completed and wrote
  `training_runs/sprint3_team_parallel_eval_smoke/metrics.jsonl`

Decision:

- Resume the real run from
  `training_runs/sprint3_team_scripted_warm_v1/checkpoints/latest.pt`
  (checkpoint update 10).

## 6. Eval Pool Reuse Fix

Observation during resumed real run:

- Update 20 eval completed in `64.495s`, much faster than the old `344.577s`.
- Update 20 argmax eval over 50 episodes:
  - `84%` win
  - `14%` loss
  - `2%` draw
  - `48%` normal wins
  - `36%` death-time wins
  - invalid action rate `0.0`
- But Windows reported multiple `[WinError 1455]` pagefile errors while spawning
  a second eval process pool on top of the already-live rollout pool.

Fix:

- `evaluate_team_policies_parallel(...)` now accepts an existing multiprocessing
  pool.
- `train_team_policy.py` passes the live rollout pool into eval, avoiding a
  second wave of torch/CUDA process imports.

Verification:

```powershell
python -m py_compile src/training/team_ppo_trainer.py scripts/train_team_policy.py
python -m pytest tests/test_team_ppo_trainer.py -q
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_pool_reuse_smoke --overwrite --updates 1 --workers 4 --rollout-steps 32 --eval-episodes 4 --eval-interval 1 --checkpoint-interval 1 --random-baseline-episodes 0 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1
```

Results:

- `tests/test_team_ppo_trainer.py`: `5 passed`
- Pool reuse smoke completed without pagefile errors.

Decision:

- Resume the real run from checkpoint update 20.

## 7. Real Run Resume After Pool Reuse

Run:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_warm_v1 --resume training_runs/sprint3_team_scripted_warm_v1/checkpoints/latest.pt --updates 280 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --random-baseline-episodes 0 --opponent-bot scripted --seed 20260603 --torch-threads 1
```

State cleanup before resume:

- `metrics.jsonl` had uncheckpointed update `21-24` records from the interrupted
  attempt.
- A backup was saved as
  `training_runs/sprint3_team_scripted_warm_v1/metrics_before_resume_prune_20260603.jsonl`.
- The active metrics file was pruned back to checkpoint update `20`, so resumed
  updates start cleanly from `21`.

Early resumed eval checkpoints:

| Update | Win | Loss | Draw | Normal Wins | Death-Time Wins | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 30 | `94%` | `6%` | `0%` | `46%` | `48%` | Pool reuse worked; no pagefile errors observed. |
| 40 | `60%` | `20%` | `20%` | `6%` | `54%` | Bad/noisy checkpoint; keep watching trend. |
| 50 | `96%` | `0%` | `4%` | `50%` | `46%` | Recovered strongly after update 40 dip. |

Current interpretation:

- The run is alive and producing usable checkpoints.
- Invalid action rate is still `0.0`.
- Checkpoint selection must be based on eval sweeps, not simply the latest
  checkpoint; update `40` is already a good example of why.
