# Sprint 2 Training Prep

This sprint prepares and smoke-tests the first RL training run. Large training
runs should start only after the trainer, eval, checkpoint, and resume loop pass.

## Current Scope

- Single controlled player environment: default controlled player is `A1`.
- Other players are baseline bots (`scripted` or `random`).
- The environment is Gym-like (`reset`, `step`, `action_mask`) but intentionally
  does not require Gymnasium yet.
- Replay recording is disabled by default for training rollouts, so long runs do
  not accumulate replay JSON in memory.
- Combat trace recording is disabled by default for training rollouts.
- A first masked PPO runner is available for smoke training against scripted or
  random baseline bots.

## Generation Modes

There are two simulation generation profiles:

- `replay`: classic generation for viewer playback and debugging.
  - `record_trace = true`
  - `record_replay = true`
  - replay JSON is saved
- `training`: headless generation for long neural-network rollouts.
  - `record_trace = false`
  - `record_replay = false`
  - replay JSON is not saved

Important contract:

- Both modes must use the same simulation rules, RNG seeds, bot actions, movement,
  combat, income, production, routing, and tick order.
- Training mode may disable only observer/debug artifacts that cannot affect the
  resulting world state.
- Do not disable, shortcut, batch, or approximate anything that changes what the
  policy experiences or what actions mean.

Current disabled artifacts in `training` mode:

- per-tick replay snapshots;
- per-combat trace/event summaries used by the viewer.

These are intentionally behavior-equivalent to replay mode. Tests compare final
state parity between `replay` and `training` generation for identical seeds.

Future league note:

- Neural policies in the league are expected to be stochastic.
- Generation mode is orthogonal to that. A stochastic policy should still sample
  from its action distribution in training/evaluation according to the league
  rules; `training` mode only controls recording overhead.

## Important Future Note: Neutral Aggro

Current implementation:

- `TRANSIT` through neutral control points does not aggro neutral guards.
- `CAPTURE` or `ATTACK` starts neutral battle and immediately adds the attacking
  team to `neutral_aggro_teams`.
- That means neutral guards currently react to action intent, not strictly to
  first received damage.

Future design candidate:

- Neutral guards should not know intent.
- Neutral guards should aggro only after actually receiving damage.

This is intentionally **not changed now**. If we change it later, it is a
simulation-rule change and old training checkpoints should be considered stale.

## Action Space

Fixed discrete action space:

- `0`: `HOLD`
- `1..3`: produce `VANGUARD`, `RANGER`, `SKIRMISHER`
- remaining actions: `MOVE` from a graph node to another graph node in two intent modes:
  `CAPTURE` and `TRANSIT`

Current size:

- `ACTION_DIM = 316`

Move source selection is explicit:

- the source node is part of the `MOVE` action;
- the source must contain controllable units for the controlled player;
- the source must be ready this tick;
- the source-to-destination route must be legal.

The current single-head action model also serializes economy and maneuver: one
tick can be `PRODUCE` or `MOVE`, not both. If the agent struggles to combine
production with map pressure, the first candidate fix is a factorized policy
interface with separate production and movement heads.

Legal action masks are provided by `legal_action_mask(obs, player_id)`.
Illegal decoded actions fall back to `HOLD`.

## Observation

Current vector:

- `OBSERVATION_VERSION = train_obs_v0.4`
- `OBSERVATION_DIM = 1028`
- dtype: `float32`

The vector includes:

- tick/time remaining/death-time timing/resources/team/alive/global battle counts;
- per-node type and owner relation;
- per-node self/ally/enemy/neutral unit counts;
- per-node reachability, route distance, and estimated travel time from the
  current controllable force;
- neutral HP/income/aggro;
- persistent core HP;
- ready lock signal;
- moving groups aggregated by node with type-wise composition for
  Vanguard/Ranger/Skirmisher by self/ally/enemy and inside/outgoing/incoming
  buckets;
- node battle HP summaries plus type-wise battle composition;
- edge battle summaries projected onto both endpoint nodes, including HP and
  Vanguard/Ranger/Skirmisher composition by self/ally/enemy.

This is a training interface, not a replay format. Replay schema remains separate.

## Reward Skeleton

Default reward:

- terminal win: `+1.0`
- terminal death-time win: `+0.5`
- terminal loss: `-1.0`
- draw: `0.0`
- small shaping delta: control/resource/alive-score delta times `0.02`

This is deliberately conservative. The first real training should check a
learning curve and baseline comparison before treating reward weights as valid.

## First-Run Failure Mode To Watch

Track win-rate against the scripted baseline separately from reward.

Main trap:

- shaped reward goes up;
- real wins do not improve.

That is reward-hacking, not useful learning. A run is considered promising only
when both are true:

- average reward improves;
- eval win-rate against the scripted baseline rises above random/legal policy
  behavior and stabilizes.

Current trainer rules:

- training rollouts are stochastic policy samples;
- eval rollouts use deterministic `argmax` actions;
- eval metrics are logged separately from training reward;
- `TrainingEnv` must use the same real `TICK_LIMIT` death-time finalization as
  `GameEngine.run()`; artificial `truncated` draws are allowed only for
  short smoke `max_ticks` below the real tick limit;
- train/eval summaries log action distributions, including dominant action rate,
  so an untrained argmax policy collapsing to `HOLD` is visible immediately;
- eval uses 50 episodes by default; smaller values are smoke-only and should not
  be used to judge win-rate;
- normal win-rate and death-time win-rate are logged separately;
- random-legal baseline eval is logged at the start of a fresh run;
- checkpoint/resume must be verified by launching a new process with
  `--resume`, not just by checking that a file exists.

## Smoke Commands

Training environment smoke, no learning:

```powershell
python scripts/training_smoke.py --episodes 3 --seed 20260602 --max-ticks 60
```

First trainer smoke:

```powershell
python scripts/train_policy.py --run-dir training_runs/first_smoke --overwrite --updates 3 --workers 4 --rollout-steps 300 --eval-episodes 50
```

Checkpoint/resume verification:

```powershell
python scripts/train_policy.py --run-dir training_runs/first_smoke --resume training_runs/first_smoke/checkpoints/latest.pt --updates 1 --workers 2 --rollout-steps 120 --eval-episodes 50
```

Expected properties:

- JSON output;
- `kind` is `random_rollout_smoke_no_training`;
- all observations have dimension `1028`;
- no replay JSON is stored in engine memory during the rollout.

Full verification:

```powershell
python -m pytest -q
```

## Next Training Step

Do not start by tuning a large multi-agent run.

Recommended next step after trainer smoke:

1. Train only one controlled player against scripted/random baselines.
2. Log episode return, win/draw/loss rate, entropy, action distribution, and
   invalid-action rate.
3. Save checkpoint and verify resume.
4. Run evaluation without learning against scripted baseline.

Only after this works should we move toward 3v3 multi-agent self-play.

## Sprint 2 Clean Closure

Clean single-agent training was rerun after the Lord/ranged-abuse fixes.

Run:

```powershell
python scripts/train_policy.py --run-dir training_runs/sprint2_scripted_clean_v1 --overwrite --updates 400 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --random-baseline-episodes 100 --ally-bot scripted --opponent-bot scripted --seed 20260603 --torch-threads 1
```

The run was intentionally stopped after checkpoint-backed update `100` because
that checkpoint already passed the scripted gate. Resume was verified in a real
process:

```powershell
python scripts/train_policy.py --run-dir training_runs/sprint2_scripted_clean_v1 --resume training_runs/sprint2_scripted_clean_v1/checkpoints/latest.pt --updates 390 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 50 --checkpoint-interval 10 --random-baseline-episodes 0 --ally-bot scripted --opponent-bot scripted --seed 20260603 --torch-threads 1
```

Final eval:

```powershell
python scripts/evaluate_checkpoints.py --run-dir training_runs/sprint2_scripted_clean_v1 --updates 50,100 --modes argmax,stochastic --seed-bases 20260603,20270603,20280603,20290603 --episodes-per-seed 50 --workers 16 --torch-threads 1 --ally-bot scripted --opponent-bot scripted --include-random-legal --out-dir training_runs/evals/sprint2_clean_scripted_eval_50_100
```

Gate result:

- `checkpoint_update_100.pt` is the accepted Sprint 2 checkpoint.
- Scripted control: `34.5%` win, `45.5%` loss, `20.0%` draw over 200 episodes.
- Random-legal A1: `37.0%` win, `37.5%` loss, `25.5%` draw over 200 episodes.
- Update 50 argmax: `66.0%` win, `0.0%` loss, `34.0%` draw; all wins were death-time wins.
- Update 100 argmax: `99.5%` win, `0.0%` loss, `0.5%` draw.
- Update 100 argmax split: `47.0%` normal wins, `52.5%` death-time wins.
- Update 100 argmax seed blocks: `100%`, `100%`, `98%`, `100%` win.
- Update 100 stochastic: `77.0%` win, `13.0%` loss, `10.0%` draw.
- Invalid action rate was `0.0%` in all reported evals.

Conclusion:

- The PPO loop learns under the corrected simulation rules.
- The accepted checkpoint beats scripted and random-legal controls by a wide,
  stable margin in deterministic argmax eval.
- Stochastic policy quality is useful but weaker than argmax; future league
  work should continue tracking stochastic eval separately.
- Sprint 2 is closed for the single-agent proof. The next step is Sprint 3
  league/self-play infrastructure, not treating the old pre-fix exploit
  checkpoints as valid policy-quality evidence.

## Invalidated Runs

- `training_runs/sprint2_first_train` was run on the latest simulation code, but
  before `TrainingEnv` used real death-time finalization at `TICK_LIMIT`.
  Its 300-tick games were reported as `truncated` draws, so that run is a
  diagnostic artifact only and must not be treated as a valid learning result.
- All checkpoints from before `train_obs_v0.4`, edge-battle/type-aware movement
  observations, and source-aware `MOVE` actions
  are incompatible with the current policy interface and must not be resumed.
