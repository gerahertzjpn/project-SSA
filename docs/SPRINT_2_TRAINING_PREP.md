# Sprint 2 Training Prep

This sprint prepares the first RL training run. It does **not** train a neural
network yet.

## Current Scope

- Single controlled player environment: default controlled player is `A1`.
- Other players are baseline bots (`scripted` or `random`).
- The environment is Gym-like (`reset`, `step`, `action_mask`) but intentionally
  does not require Gymnasium yet.
- Replay recording is disabled by default for training rollouts, so long runs do
  not accumulate replay JSON in memory.
- Combat trace recording is disabled by default for training rollouts.

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

- `OBSERVATION_VERSION = train_obs_v0.1`
- `OBSERVATION_DIM = 492`
- dtype: `float32`

The vector includes:

- tick/resources/team/alive/global battle counts;
- per-node type and owner relation;
- per-node self/ally/enemy/neutral unit counts;
- neutral HP/income/aggro;
- ready lock signal;
- moving groups aggregated by node;
- node battle HP summaries.

This is a training interface, not a replay format. Replay schema remains separate.

## Reward Skeleton

Default reward:

- terminal win: `+1.0`
- terminal loss: `-1.0`
- draw: `0.0`
- small shaping delta: control/resource/alive-score delta times `0.02`

This is deliberately conservative. The first real training should check a
learning curve and baseline comparison before treating reward weights as valid.

## Smoke Commands

Training environment smoke, no learning:

```powershell
python scripts/training_smoke.py --episodes 3 --seed 20260602 --max-ticks 60
```

Expected properties:

- JSON output;
- `kind` is `random_rollout_smoke_no_training`;
- all observations have dimension `492`;
- no replay JSON is stored in engine memory during the rollout.

Full verification:

```powershell
python -m pytest -q
```

## Next Training Step

Do not start by tuning a large multi-agent run.

Recommended next step:

1. Wrap `TrainingEnv` with the chosen trainer (CleanRL/SB3/Gymnasium adapter).
2. Train only one controlled player against scripted/random baselines.
3. Log episode return, win/draw/loss rate, entropy, action distribution, and
   invalid-action rate.
4. Save checkpoint and verify resume.
5. Run evaluation without learning against scripted baseline.

Only after this works should we move toward 3v3 multi-agent self-play.
