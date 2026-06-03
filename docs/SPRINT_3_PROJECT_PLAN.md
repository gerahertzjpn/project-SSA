# Sprint 3 Project Plan: 3v3 Multi-Agent Training

Date: 2026-06-03

Sprint 2 status: closed for the single-agent proof. Accepted checkpoint:
`training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt`.
That checkpoint is proof evidence only after the `train_obs_v0.5` interface
change; it must not be resumed or warm-started directly.

Sprint 3 goal: move from one controlled player (`A1`) to three controlled
teammates (`A1`, `A2`, `A3`) that learn coordinated 3v3 play under the current
game rules.

Current pre-training balance pass is documented in `docs/BALANCE_AUDIT.md`.
The active ruleset for the next large run uses core `80hp/6 atk`, neutral guard
damage `2 * strength`, integer death-time core damage, and neutral income
`ceil(strength / 2)`.

## Non-Negotiables

- Do not treat reward alone as learning. Track real win-rate separately.
- Eval must include deterministic argmax and stochastic sampling.
- Checkpoint/resume must be verified by a real restarted process.
- Training generation must keep exact gameplay parity with replay generation.
- Keep 16 workers as the default CPU load unless a fresh benchmark shows a
  better stable point.
- Do not use old pre-fix exploit checkpoints as quality evidence.
- If game rules change during Sprint 3, current Sprint 3 checkpoints become
  stale until revalidated.
- Observation interface changes also invalidate checkpoints with mismatched
  metadata or input layers. Current interface is `train_obs_v0.5` /
  `OBSERVATION_DIM=1028`.
- Policy observations/actions are left-side normalized: B-side players are
  mirrored into matching A-side slots (`B1->A1`, `B2->A2`, `B3->A3`) before the
  policy sees the state, and MOVE source/destination ids are mirrored back
  before entering the simulator.

## Main Design Choice

Sprint 3 should start with three separate policies, not one shared policy:

- `A1_policy`
- `A2_policy`
- `A3_policy`

They may share the same network architecture and can optionally be initialized
from the Sprint 2 checkpoint, but their weights and optimizers must be saved
separately. This matches the roadmap requirement: three separate networks learn
team behavior.

Each environment step must be synchronized:

1. build observation and legal-action mask for each controlled player;
2. sample/choose all three actions;
3. set all three actions into external bots;
4. advance the engine by exactly one tick;
5. assign the same team terminal result to all three agents, with optional
   small per-agent diagnostics logged separately.

No agent should get to advance the world alone while the other two wait.

## Files To Add Or Extend

- `src/training/team_env.py`
  - multi-agent `reset`, `step`, `action_masks`;
  - controlled players: default `A1,A2,A3`;
  - baseline or policy-controlled opponents;
  - same reward contract as Sprint 2, but team-shared.

- `src/training/team_ppo_trainer.py`
  - three policies and three optimizers;
  - rollout collection for all controlled players;
  - per-agent action distribution, entropy, invalid-rate;
  - team-level episode summaries.

- `scripts/train_team_policy.py`
  - CLI for 3v3 training;
  - checkpoint/resume;
  - JSONL metrics;
  - optional Sprint 2 warm start.

- `scripts/evaluate_team_checkpoints.py`
  - argmax and stochastic eval;
  - 50+ episodes minimum, preferably 200 for gates;
  - seed blocks;
  - scripted/random/frozen-policy controls.

- `scripts/analyze_team_coordination.py`
  - duplicate-target rate;
  - split-front coverage;
  - idle-with-units rate;
  - assist/reinforcement rate;
  - Lord contest and Lord impact stats.

- tests:
  - `tests/test_team_training_env.py`
  - `tests/test_team_ppo_trainer.py`
  - `tests/test_team_evaluate_checkpoints.py`

## Eight-Hour Runbook

### 0:00-0:30 - Preflight

Commands:

```powershell
git status --short
python -m pytest -q
python scripts/evaluate_checkpoints.py --run-dir training_runs/sprint2_scripted_clean_v1 --updates 100 --modes argmax,stochastic --seed-bases 20260603,20270603 --episodes-per-seed 25 --workers 16 --torch-threads 1 --ally-bot scripted --opponent-bot scripted --include-random-legal --out-dir training_runs/evals/sprint3_preflight_sprint2_checkpoint
```

Expected:

- tests green;
- Sprint 2 checkpoint still loads;
- eval still shows strong scripted performance;
- git state recorded before new code work.

### 0:30-2:30 - Implement Multi-Agent Training Skeleton

Build the smallest correct version first:

- `TeamTrainingEnv` with three external controlled bots;
- synchronized tick stepping;
- dict observations: `{"A1": obs, "A2": obs, "A3": obs}`;
- dict masks and dict actions;
- shared terminal reward;
- no replay/trace recording in training mode;
- baseline opponents initially `scripted` or `random`.

Tests to write before long training:

- reset returns three observations of dimension `1028`;
- each controlled player has a legal mask with at least `HOLD`;
- one `step()` advances the engine by one tick total, not three ticks;
- all three actions are consumed on the same tick;
- terminal team win/loss/draw is reported consistently for all agents;
- illegal actions are counted and handled, not silently hidden;
- replay/training generation parity tests remain green.

### 2:30-3:15 - Trainer And Resume Smoke

Add or adapt a team trainer.

Smoke commands should look like this:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_smoke --overwrite --updates 2 --workers 4 --rollout-steps 128 --eval-episodes 20 --eval-interval 1 --checkpoint-interval 1 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1

python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_smoke --resume training_runs/sprint3_team_smoke/checkpoints/latest.pt --updates 1 --workers 4 --rollout-steps 128 --eval-episodes 20 --eval-interval 1 --checkpoint-interval 1 --opponent-bot scripted --seed 20260603 --torch-threads 1
```

Smoke pass criteria:

- no NaN observations/losses;
- invalid action rate near `0.0`;
- checkpoints include all three policies and optimizers;
- resumed run starts from the next update;
- metrics JSONL contains per-agent and team summaries.

### 3:15-4:00 - Eval Harness

Create team eval with these modes:

- argmax team policy vs scripted team;
- stochastic team policy vs scripted team;
- random-legal controlled team vs scripted team;
- scripted control team vs scripted team;
- optional `3x Sprint2` frozen policies vs scripted team.

Minimum command shape:

```powershell
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_smoke --updates latest --modes argmax,stochastic --seed-bases 20260603,20270603,20280603,20290603 --episodes-per-seed 25 --workers 16 --torch-threads 1 --opponent-bot scripted --include-random-legal --include-scripted-control --out-dir training_runs/evals/sprint3_team_smoke_eval
```

This eval does not need to pass quality gates yet. It only proves measurement.

### 4:00-6:30 - First Real Team Run

Start with the cleaner version: trained `A1,A2,A3` vs scripted `B1,B2,B3`,
all three initialized from the accepted Sprint 2 checkpoint.

Command target:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_warm_v1 --overwrite --updates 300 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --ally-init training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt --opponent-bot scripted --seed 20260603 --torch-threads 1
```

During the run, watch every 10 updates:

- eval argmax win-rate;
- eval stochastic win-rate;
- draw-rate;
- normal win-rate vs death-time win-rate;
- mean reward and mean shaping separately;
- entropy per agent;
- dominant action per agent;
- invalid action rate;
- duplicate-target rate;
- idle-with-units rate;
- Lord kill/control/impact stats.

Stop early only for hard failures:

- exceptions or broken checkpoint save;
- NaN loss/reward/observation;
- invalid action rate meaningfully above zero;
- all agents collapse to one dominant action in argmax and win-rate does not
  improve;
- reward rises while eval win-rate stays flat for several eval intervals.

### 6:30-7:15 - Checkpoint Sweep

Evaluate promising checkpoints. Do not pick by train reward.

Command target:

```powershell
python scripts/evaluate_team_checkpoints.py --run-dir training_runs/sprint3_team_scripted_warm_v1 --updates 50,100,150,200,250,300 --modes argmax,stochastic --seed-bases 20260603,20270603,20280603,20290603 --episodes-per-seed 50 --workers 16 --torch-threads 1 --opponent-bot scripted --include-random-legal --include-scripted-control --out-dir training_runs/evals/sprint3_team_scripted_warm_v1_sweep
```

Pick candidates only if they are stable by seed block. One lucky aggregate is
not enough.

### 7:15-7:45 - Decide Next Training Branch

If argmax and stochastic both beat scripted control comfortably:

- start first pool/self-play infrastructure;
- freeze the best checkpoint as `sprint3_team_scripted_warm_v1_best`;
- next run samples opponents from scripted, random, and previous policy pools.

If argmax wins but stochastic is weak:

- continue with entropy pressure or stochastic-focused eval;
- do not enter league-mode training yet.

If reward rises but win-rate does not:

- treat it as reward hacking;
- inspect action distribution and map objectives;
- adjust reward only after replay/stat evidence.

If all agents duplicate the same front:

- add coordination diagnostics first;
- only then consider reward shaping for split-front coverage or assist behavior.

### 7:45-8:00 - Report

Final report must include:

- commit hash;
- tests run and results;
- training commands;
- total steps and steps/sec;
- best checkpoint path;
- argmax and stochastic eval table;
- random-legal and scripted control table;
- action distribution per agent;
- coordination diagnostics;
- Lord statistics;
- whether Sprint 3 gates are open, partially open, or blocked.

## Sprint 3 Gate Draft

Sprint 3 is not closed by "it trains". It closes only when all are true:

- three controlled policies can be trained and resumed reliably;
- argmax team eval beats scripted/non-coordinated control by a stable margin over
  multiple seed blocks;
- stochastic eval remains useful and does not collapse;
- reward and win-rate move together;
- invalid action rate is effectively zero;
- replay samples are watchable enough to see team behavior;
- coordination diagnostics show less blind duplication than baselines.

## First Command After Approval

Do not start the eight-hour sequence blindly. The first execution step should be:

```powershell
git status --short
python -m pytest -q
```

If that is clean, start the preflight eval and then implement the multi-agent
training skeleton.
