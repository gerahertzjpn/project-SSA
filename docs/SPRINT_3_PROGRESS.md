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
| 60 | `98%` | `0%` | `2%` | `54%` | `44%` | Strong, still mixed normal/death-time wins. |
| 70 | `100%` | `0%` | `0%` | `58%` | `42%` | First perfect 50-episode argmax eval. |
| 80 | `98%` | `2%` | `0%` | `68%` | `30%` | More normal wins; shorter games. |
| 100 | `100%` | `0%` | `0%` | `76%` | `24%` | Best visible finish profile so far. |
| 110 | `84%` | `12%` | `4%` | `54%` | `30%` | Dip; reinforces need for sweep-based selection. |
| 120 | `98%` | `2%` | `0%` | `74%` | `24%` | Strong, close to update 100. |
| 130 | `98%` | `2%` | `0%` | `68%` | `30%` | Strong but worse finish profile than 100/120. |
| 140 | `98%` | `2%` | `0%` | `82%` | `16%` | Best normal-win profile so far. |
| 150 | `96%` | `4%` | `0%` | `58%` | `38%` | Regression in finish profile. |
| 160 | `96%` | `4%` | `0%` | `70%` | `26%` | Working checkpoint, not a top candidate. |
| 170 | `92%` | `6%` | `2%` | `76%` | `16%` | Lower raw win-rate. |
| 180 | `90%` | `10%` | `0%` | `64%` | `26%` | Late-training dip. |
| 190 | `100%` | `0%` | `0%` | `52%` | `48%` | Perfect raw win-rate but too death-time heavy. |
| 200 | `94%` | `6%` | `0%` | `72%` | `22%` | Better finish profile than 190 but lower raw win. |
| 210 | `96%` | `4%` | `0%` | `76%` | `20%` | Solid, not top. |
| 220 | `100%` | `0%` | `0%` | `90%` | `10%` | Best candidate so far by win and finish profile. |
| 230 | `100%` | `0%` | `0%` | `78%` | `22%` | Perfect raw win-rate, worse finish profile than 220. |
| 240 | `100%` | `0%` | `0%` | `98%` | `2%` | New best visible candidate; much faster games. |

Current interpretation:

- The run is alive and producing usable checkpoints.
- Invalid action rate is still `0.0`.
- Checkpoint selection must be based on eval sweeps, not simply the latest
  checkpoint; update `40` is already a good example of why.
- Update `100` is the best current candidate by argmax eval because it combines
  perfect win-rate with fewer death-time wins and shorter mean episode length.
- Update `140` is also a serious candidate because its normal-win share is
  higher than update `100`, despite a slightly lower raw win-rate.
- Update `220` is now the strongest visible candidate: perfect argmax win-rate,
  high normal-win share, low death-time share, and shorter games.
- Update `240` supersedes update `220` in the single seed-block view: it keeps
  perfect argmax win-rate and almost eliminates death-time wins.
- A1 argmax has become heavily concentrated on `PRODUCE:SKIRMISHER`; this must
  be checked in stochastic eval and multi-seed sweeps before accepting the
  checkpoint for league use.

## 8. Real Run Completed To Update 300

Run result:

- The resumed real run reached update `300`.
- `checkpoint_update_300.pt` and `latest.pt` were written successfully.
- The process then failed during final CLI summary generation, not during
  training, because `latest_metrics(...)` hit a BOM-prefixed JSONL line.

Technical fix:

- `src/training/team_ppo_trainer.py::latest_metrics(...)` now strips `\ufeff`
  at the start of every JSONL line.
- The same fix was applied to `src/training/ppo_trainer.py` for single-agent
  runs.
- Added `tests/test_metrics_io.py` to lock this behavior.

Verification:

```powershell
python -m pytest tests/test_metrics_io.py tests/test_team_ppo_trainer.py -q
python -c "from pathlib import Path; from src.training.team_ppo_trainer import latest_metrics; m=latest_metrics(Path('training_runs/sprint3_team_scripted_warm_v1/metrics.jsonl')); print(len(m)); print(m[-1]['kind'], m[-1].get('update'))"
```

Results:

- `6 passed`
- Current real-run metrics file reads fully: `310` records; last record is
  `team_ppo_update 300`.

Final built-in argmax eval checkpoints:

| Update | Win | Loss | Draw | Normal Wins | Death-Time Wins | Mean Steps | A3 Dominant Argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 250 | `94%` | `6%` | `0%` | `92%` | `2%` | `153.96` | `PRODUCE:SKIRMISHER` at `26%` |
| 260 | `96%` | `4%` | `0%` | `90%` | `6%` | `155.80` | `PRODUCE:SKIRMISHER` at `30%` |
| 270 | `98%` | `2%` | `0%` | `88%` | `10%` | `158.84` | `HOLD` at `53%` |
| 280 | `94%` | `6%` | `0%` | `86%` | `8%` | `164.84` | `HOLD` at `55%` |
| 290 | `96%` | `4%` | `0%` | `82%` | `14%` | `190.10` | `HOLD` at `57%` |
| 300 | `86%` | `14%` | `0%` | `82%` | `4%` | `186.82` | `HOLD` at `70%` |

Interpretation:

- The team policy clearly learned under the corrected rules: it repeatedly
  exceeded the random-legal team baseline (`30%` win, `66%` loss, `4%` draw) and
  the scripted-control baseline (`34%` win, `64%` loss, `2%` draw from Sprint 2
  preflight).
- The latest checkpoint is not the best checkpoint.
- Update `240` remains the strongest visible candidate from the built-in eval:
  `100%` win, `98%` normal wins, `2%` death-time wins, shortest mean episode
  length.
- Update `220` remains a backup candidate: `100%` win, `90%` normal wins,
  `10%` death-time wins.
- Update `270` is worth checking as a late neighbor but has a clear A3 HOLD
  concentration risk.
- Updates `280-300` should not be accepted without surprising sweep results;
  A3 argmax collapses toward HOLD and raw win-rate drops by update `300`.

Next action:

- Run a separate multi-seed argmax/stochastic eval sweep on candidate
  checkpoints `220`, `240`, `250`, `260`, `270`, plus controls.
- Select a checkpoint by scripted performance and stability, not by training
  recency.

## 9. Candidate Eval Sweep

Run:

```powershell
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_scripted_warm_v1 --updates 220,240,250,260,270 --modes argmax,stochastic --seed-bases 20260603,20270603,20280603 --episodes-per-seed 50 --workers 16 --torch-threads 1 --opponent-bot scripted --ally-bot scripted --include-random-legal --include-scripted-control --out-dir training_runs/evals/sprint3_team_scripted_warm_v1_candidates
```

Scope:

- `150` episodes per checkpoint/mode (`3` seed blocks x `50` episodes).
- Both argmax eval and stochastic eval.
- Scripted-control and random-legal controls included.

Controls:

| Control | Win | Loss | Draw | Normal Wins | Death-Time Wins | Mean Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted team | `36.7%` | `57.3%` | `6.0%` | `0.0%` | `36.7%` | `300.00` |
| Random-legal team | `22.0%` | `71.3%` | `6.7%` | `0.0%` | `22.0%` | `299.85` |

Checkpoint sweep:

| Update | Mode | Win | Loss | Draw | Normal Wins | Death-Time Wins | Mean Steps | Sequence Diversity | Opening Diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `220` | argmax | `100.0%` | `0.0%` | `0.0%` | `94.0%` | `6.0%` | `174.93` | `100.0%` | `40.7%` |
| `220` | stochastic | `95.3%` | `4.7%` | `0.0%` | `78.7%` | `16.7%` | `200.35` | `100.0%` | `100.0%` |
| `240` | argmax | `100.0%` | `0.0%` | `0.0%` | `96.7%` | `3.3%` | `143.55` | `99.3%` | `43.3%` |
| `240` | stochastic | `95.3%` | `4.7%` | `0.0%` | `85.3%` | `10.0%` | `187.55` | `100.0%` | `100.0%` |
| `250` | argmax | `96.7%` | `3.3%` | `0.0%` | `91.3%` | `5.3%` | `159.65` | `100.0%` | `75.3%` |
| `250` | stochastic | `96.0%` | `4.0%` | `0.0%` | `84.7%` | `11.3%` | `195.59` | `100.0%` | `100.0%` |
| `260` | argmax | `98.7%` | `1.3%` | `0.0%` | `96.0%` | `2.7%` | `151.59` | `100.0%` | `74.0%` |
| `260` | stochastic | `95.3%` | `4.7%` | `0.0%` | `83.3%` | `12.0%` | `190.96` | `100.0%` | `100.0%` |
| `270` | argmax | `97.3%` | `2.7%` | `0.0%` | `90.0%` | `7.3%` | `155.99` | `99.3%` | `68.7%` |
| `270` | stochastic | `97.3%` | `2.7%` | `0.0%` | `88.0%` | `9.3%` | `183.08` | `100.0%` | `100.0%` |

Decision:

- Best checkpoint: `checkpoint_update_240.pt`.
- Reason:
  - Best argmax gate: `100%` win, `0%` loss, `96.7%` normal wins.
  - Fastest argmax games among top candidates: `143.55` mean ticks.
  - Stochastic mode does not collapse: `95.3%` win, `85.3%` normal wins,
    `100%` opening diversity.
  - It beats both controls by a very large margin.
- Backup checkpoint: `checkpoint_update_220.pt`.
- Late checkpoints `250-270` are viable but not preferred:
  - `270` has the best stochastic win-rate among candidates, but its argmax
    gate is weaker and A3 shows a clear HOLD concentration risk.
  - `260` is close, but still loses in argmax and has worse stochastic
    normal-win share than `240`.

## 10. Coordination Analysis For Update 240

Runs:

```powershell
python scripts/analyze_team_coordination.py --checkpoint training_runs/sprint3_team_scripted_warm_v1/checkpoints/checkpoint_update_240.pt --episodes 150 --seed 20260603 --mode argmax --opponent-bot scripted --ally-bot scripted --out training_runs/evals/sprint3_team_scripted_warm_v1_candidates/coordination_update_240_argmax.json
python scripts/analyze_team_coordination.py --checkpoint training_runs/sprint3_team_scripted_warm_v1/checkpoints/checkpoint_update_240.pt --episodes 150 --seed 20260603 --mode stochastic --opponent-bot scripted --ally-bot scripted --out training_runs/evals/sprint3_team_scripted_warm_v1_candidates/coordination_update_240_stochastic.json
```

Results:

| Mode | Win | Loss | Draw | Normal Wins | Death-Time Wins | Mean Ticks | Duplicate Target Rate | Mean Unique Move Destinations | Node Battle Move Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax | `99.3%` | `0.7%` | `0.0%` | `96.7%` | `3.3%` | `140.17` | `13.2%` | `1.71` | `25.7%` |
| stochastic | `96.0%` | `4.0%` | `0.0%` | `82.0%` | `18.0%` | `195.79` | `6.7%` | `1.56` | `18.9%` |

Lord notes:

- Argmax:
  - Neutral Lord kill rate: `38.7%`.
  - Spawn rate: `38.0%`.
  - Owner counts after neutral kill: `TEAM_B=57`, `NONE=1`.
  - Expended rate: `18.7%`.
  - Mean neutral kill tick: `70.98`.
- Stochastic:
  - Neutral Lord kill rate: `76.0%`.
  - Spawn rate: `76.0%`.
  - Owner counts after neutral kill: `TEAM_B=114`.
  - Expended rate: `59.3%`.
  - Mean neutral kill tick: `80.25`.

Interpretation:

- Update `240` wins mostly through normal victory, not reward-only shaping and
  not death-time farming.
- Stochastic league-style play remains strong, but it is visibly less clean than
  argmax: more death-time wins and longer games.
- A notable strategic smell remains: in the coordination sample, the resurrected
  Lord is mostly awarded to `TEAM_B`, yet the learned `TEAM_A` policy still wins
  heavily. This may mean the policy is strong enough to win without Lord control,
  or that scripted `TEAM_B` spends Lord advantage poorly.
- The current Sprint 3 gate for "synchronized 3-agent policy can learn and beat
  scripted reliably under corrected rules" is satisfied by update `240`.
- Before starting league/self-play work, keep `checkpoint_update_240.pt` as the
  accepted warm-start checkpoint and monitor Lord-control incentives in future
  mixed-policy training.
