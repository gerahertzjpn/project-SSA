# Current 2D Status

Date: 2026-06-04

This is the current status after switching active battles to `combat_2d_map`
and before the next large training run.

## Current Balance Snapshot

Latest balance snapshot after `LORD_ATTACK_RANGE=4.6`,
`CORE_ATTACK_RANGE=5.0`, explicit `JOIN_LORD` / `HUNT_LORD` actions, and
`train_obs_v1.0`:

300 random-vs-random matches with seed `20260611`:

- winners: `TEAM_A 157`, `TEAM_B 143`, `DRAW 0`;
- win rates: `TEAM_A 52.33%`, `TEAM_B 47.67%`;
- winner types: `normal 139`, `death_time 161`;
- ticks: min `80`, median `275.5`, mean `246.663`, max `300`.

Lord trace from 300 random-vs-random matches:

- neutral Lord killed: `296 / 300`;
- spawned after kill: `184 / 300`;
- killed but no owner: `112 / 300`;
- first Lord kill tick: min `22`, median `72`, mean `77.622`, max `243`;
- Lord owner teams: `TEAM_A 91`, `TEAM_B 93`;
- owner-team win rates after Lord spawn: `TEAM_A 76.92%`, `TEAM_B 76.34%`;
- spawned Lord destroyed a core: `124 / 184` (`67.39%` per spawn);
- spawned Lord died before core: `44 / 184` (`23.91%` per spawn);
- spawned Lord alive at game end: `16 / 184` (`8.70%` per spawn).

Saved data:

```text
analysis/current_balance_suite_lord_range4_6_core_range5_300m_100lord.json
analysis/random_lord_stats_range4_6_core5_300trace.json
```

Previous historical snapshot before the latest Lord/core range pass:

300 random-vs-random matches with seed `20260604`:

- winners: `TEAM_A 163`, `TEAM_B 135`, `DRAW 2`;
- win rates: `TEAM_A 54.33%`, `TEAM_B 45.00%`, `DRAW 0.67%`;
- winner types: `normal 123`, `death_time 175`, `draw_no_core_damage 2`;
- winner type rates: `normal 41.00%`, `death_time 58.33%`;
- ticks: min `94`, median `282.5`, mean `253.393`, max `300`;
- core combat damage mean/median: `224.207 / 240.0`.

Lord stats from the same 300 matches:

- neutral Lord killed: `246 / 300` (`82.00%`);
- Lord spawned after kill: `196 / 300` (`65.33%`);
- killed but no owner: `50 / 300` (`16.67%`);
- first Lord kill tick mean/median: `124.057 / 118`;
- Lord owner teams: `TEAM_A 102`, `TEAM_B 94`;
- spawned Lord destroyed a core: `92 / 196` (`46.94%` per spawn);
- spawned Lord died before core: `69 / 196` (`35.20%` per spawn);
- spawned Lord alive at game end: `35 / 196` (`17.86%` per spawn).

Interpretation:

- Side balance is not obviously broken from 300 random games, but `TEAM_A`
  has a mild lead and should be watched with larger samples.
- Death-time is still very common. We are not changing it directly yet because
  the current plan is to rebalance first.
- Lord is contested often, but the spawned Lord is no longer a near-guaranteed
  core kill.
- The old `docs/BALANCE_AUDIT.md` micro-combat results are historical after
  `combat_2d_map`; they used the old 1D helper.

Saved data:

```text
analysis/current_2d_random_lord_300.json
```

## Training Metadata

Current interface:

```text
OBSERVATION_VERSION = train_obs_v1.0
OBSERVATION_DIM = 1117
ACTION_DIM = 1356
```

Action breakdown:

```text
HOLD: 1
PRODUCE: 3
MOVE source -> destination with intent: 312
MOVE source -> ALLY_LORD with JOIN_LORD intent: 13
MOVE source -> ENEMY_LORD with HUNT_LORD intent: 13
PRODUCE_MOVE bounded composite actions: 1014
```

The interface version was bumped because the policy now sees resurrected Lord
state explicitly and can combine one production command with one movement
command in a single tick:

- allied Lord alive flag;
- owner relation from the acting player's perspective;
- allied Lord HP;
- ready ETA and arrival ETA;
- waiting support-unit counts for explicit JOIN_LORD escorts;
- per-node flags for Lord route and Lord target.
- bounded `PRODUCE_MOVE` actions: one produced unit type plus one move action;
- movement resolves before production, so new units do not immediately join the
  same-tick move.

Old checkpoints should now fail compatibility checks instead of silently
running as current models.

Replay schema is now `0.7.2`. The replay/observation Baron payload includes:

```text
allied_lord_path
allied_lord_ready_tick
allied_lord_arrival_tick
allied_lord_support_units
allied_lord_ready_in
allied_lord_eta
```

The resurrected Lord waits `2` ticks before push. During those ticks it remains
visible at Baron as `lord_delay`, while both teams already know target, route,
ready tick, and ETA.

`JOIN_LORD` support only comes from explicit special actions. Ordinary survivors
that killed the neutral Lord do not automatically join the push. `HUNT_LORD`
lets a player send a group toward the current enemy resurrected Lord position.

## Current Known Issues

### 1. Balance Is Not Final

The current snapshot is enough to say random-vs-random is playable, but not
enough to accept the ruleset for training. We still need a proper 2D-specific
balance audit:

- equal-budget unit fights using `combat2d`;
- neutral-point ROI;
- Lord raid thresholds;
- core push thresholds;
- random and scripted baseline stats;
- policy-vs-baseline re-eval only after new training.

### 2. Scripted Baseline Is Slow To Analyze

Quick scripted-vs-scripted attempts with full and reduced trace ran too long for
this pass and were stopped. This needs a focused performance check. It may be
that scripted games are simply long and fight-heavy under the current rules, or
that the scripted bot creates inefficient high-contention states.

### 3. Node Battle Merging

The engine currently has at most one active battle per node key, such as
`node:CP1`.

That means two separate fights at two different entrances to the same point are
merged into one objective battle. This is intentional for now because:

- RL sees one strategic contest for the objective;
- resolution and ownership stay simple;
- performance stays predictable.

The limitation is that the engine cannot represent two physically separate
local skirmishes inside the same node at the same time. If approach-local
position becomes strategically important, we need a more detailed battle key,
for example `node:CP1:approach:A1->CP1`, plus merge rules when fights overlap.

### 4. Post-Battle Aggregation

During battle, units are individual `UnitInstance` objects with real `x/y`.
After battle, survivors are aggregated back into `{unit_type: count}` in nodes
or moving groups.

Gameplay impact:

- ordinary units do not currently carry individual identity, so aggregation
  does not change their gameplay;
- allied Lord HP is persisted separately and remains special-cased;
- future unit veterancy, per-unit buffs, or persistent HP for ordinary units
  would require a different representation.

Visual impact:

- this is one of the causes of identity jumps in the viewer;
- PixiJS can smooth handoffs, but exact continuity needs persistent replay ids
  across production, stationed state, movement, battle, survival, and regrouping.

Performance impact:

- aggregation is fast;
- emitting persistent visual ids in replay mode should have little or no
  training cost if disabled in headless training;
- carrying persistent individual units in training would be more expensive and
  should be benchmarked before adoption.

### 5. Realer 2D Physics Cost

There are several levels of "fixing" simplified 2D:

- visual-only PixiJS renderer: no training performance hit;
- persistent replay visual ids only: near-zero training hit if replay-only;
- lightweight spatial buckets for target queries/separation: likely modest
  overhead, roughly `10-30%` if implemented carefully;
- real steering/collision/pathfinding in the simulation loop: high risk,
  potentially `2x-10x` slower depending on unit counts and implementation.

Current benchmark after `combat2d`:

```text
1082.0 steps/sec with 16 workers
approximately 3.90M steps/hour
```

The next engine-side realism change must keep the `>=3M steps/hour` gate.

### 6. Factorized Produce + Move

Current action model is a single categorical action:

```text
either HOLD
or PRODUCE one unit type
or MOVE one source/destination/intent
```

The source node is already part of the action space, so the older concern that
the policy cannot choose the group/source has been addressed.

If we factorize into separate heads:

```text
produce_head: none / Vanguard / Ranger / Skirmisher
move_head: hold-move / source -> destination with intent
```

the raw output size stays similar (`4 + 313` logits instead of `316` logits),
but the environment applies two decisions per tick. Training complexity should
not explode, but code complexity rises:

- two masks instead of one;
- two log-probs/entropies in PPO;
- more invalid-action handling;
- reward attribution changes because production and movement happen together.

Expected effect:

- compute cost: small increase;
- implementation complexity: medium;
- learning behavior: likely cleaner economy+movement coordination, but it must
  be tested.

### 7. Viewer State

Canvas viewer remains useful for fallback and diagnostics, but it is visually
rough.

New experimental path:

```text
viewer_pixi/index.html
```

It uses PixiJS display objects and replay-provided battle `x/y` positions. It
does not simulate gameplay and does not affect training.

Current viewer fixes:

- destroyed cores render as white dead nodes instead of neutral/yellow nodes;
- Pixi no longer applies previous-tick handoff backfill on stable
  `visual_units`, removing the "moves forward, teleports back, moves again"
  artifact;
- edge-battle survivors resume movement with direction-aware progress;
- `node_enter` reinforcements join approach battles at the target boundary;
- `node_exit` reinforcements join approach battles at the source boundary.

Manual audit:

```text
python tools/audit_replay_visual.py replays_viewer_debug_smoke/replay_8.json --ticks 120 --substeps 3 --jump 38
```

Latest result on schema `0.7.2`: first `120` ticks at one-third-tick sampling
(`360` subframes) had `jumps>=38 = 0`. Remaining speed spikes are inside
battle-layer movement and should be reviewed visually, but they are no longer
large replay-position teleports.

## Recommended Next Steps

1. Run a proper 2D balance audit using `combat2d`, not the old 1D helper.
2. Diagnose scripted-vs-scripted runtime and death-time tendency.
3. Add persistent replay visual ids if the PixiJS viewer still has identity
   jumps.
4. Only after rules are stable, start fresh training on `train_obs_v1.0`.
5. Commit the current 2D engine and PixiJS viewer once accepted.
