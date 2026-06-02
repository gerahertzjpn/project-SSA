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
- remaining actions: `MOVE` to each graph node in two intent modes:
  `CAPTURE` and `TRANSIT`

Current size:

- `ACTION_DIM = 30`

Move source selection is deterministic:

- choose a controllable source node with units;
- source must be ready this tick;
- route must be legal;
- prefer more units, then shorter route, then stable node order.

Legal action masks are provided by `legal_action_mask(obs, player_id)`.
Illegal decoded actions fall back to `HOLD`.

## Observation

Current vector:

- `OBSERVATION_VERSION = train_obs_v0.2`
- `OBSERVATION_DIM = 505`
- dtype: `float32`

The vector includes:

- tick/resources/team/alive/global battle counts;
- per-node type and owner relation;
- per-node self/ally/enemy/neutral unit counts;
- neutral HP/income/aggro;
- persistent core HP;
- ready lock signal;
- moving groups aggregated by node;
- node battle HP summaries.

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
python scripts/train_policy.py --run-dir training_runs/first_smoke --overwrite --updates 3 --workers 4 --rollout-steps 300 --eval-episodes 4
```

Checkpoint/resume verification:

```powershell
python scripts/train_policy.py --run-dir training_runs/first_smoke --resume training_runs/first_smoke/checkpoints/latest.pt --updates 1 --workers 2 --rollout-steps 120 --eval-episodes 2
```

Expected properties:

- JSON output;
- `kind` is `random_rollout_smoke_no_training`;
- all observations have dimension `505`;
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
