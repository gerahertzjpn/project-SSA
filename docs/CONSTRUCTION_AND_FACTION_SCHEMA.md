# Construction And Faction Schema

Date: 2026-06-05

Status: construction v0 plus active three-faction roster implemented as a new
training interface:

```text
OBSERVATION_VERSION = train_obs_v3.1
ACTION_DIM = 447
OBSERVATION_DIM = 5188
```

Old `train_obs_v1.0`, `train_obs_v2.0`, and `train_obs_v2.2` checkpoints are
intentionally incompatible with this ruleset. This is a schema-lock step before
the next serious league training.

## 2026-06-07 Current Lock

Active pre-training lock:

```text
OBSERVATION_VERSION = train_obs_v3.1
OBSERVATION_DIM = 5188
ACTION_DIM = 447
controlled players = A1,A2,A3
action heads = produce + move + build
```

Current global combat constants:

```text
CORE: hp 80, atk 7 x 2 attacks/tick, range 5.0
LORD/BARON: hp 300, atk 13 x 3 primary + 3 x 5 secondary attacks/tick, range 4.6
NEUTRAL_GUARD: hp 4, atk 2 x 2 attacks/tick, range 3.5
movement map units per tick = 30.0 * unit speed
combat range to map units = 24.0 * attack range
```

Neutral control points still use map-defined garrison counts in the normal
`1..7` range. Labels like `neutral_10` belong to laboratory balance probes, not
to normal map generation.

Latest no-training safety snapshot:

```text
pytest: 276 passed
training preflight: ok true
100-match crash gate: seeds_with_issues 0, issue_summary {}
throughput: 5.665M world steps/hour, 16.994M controlled-agent decisions/hour
random-legal dry-run baseline: total_steps 0, invalid action rate 0
```

## Fog Of War

Fog of war is explicitly out of scope for this project.

Agents and viewers keep full-map information. Future work should not add hidden
information unless this document is revised first.

## Factions

Faction schema exists now:

```text
factions = humans / infected / robots
player.faction = humans
core_type = standard
```

All current generated matches still use the human faction by default unless a
test/tool explicitly passes `player_factions` into `make_initial_state`. The
schema now includes one unique base building per faction:

```text
humans: BOOSTER
robots: POWER
infected: LIFE
```

The full unit roster is now active in constants, production gating, RL masks,
and replay metadata.

## Buildings

Captured control points:

```text
1 building slot
allowed: TURRET
```

Player bases / core nodes:

```text
2 building slots
allowed:
- BOOSTER
- POWER
- LIFE
- ECONOMY_COLLECTOR
- TECH_1
- TECH_2
```

Rules:

- buildings can only be built on nodes owned by the player;
- neutral or contested nodes cannot start construction;
- Tech2 requires completed Tech1;
- duplicate building types on the same node are blocked;
- buildings under construction do not fight or provide income/unlocks;
- completed turrets join defensive battles as static combat units;
- BOOSTER is a human-only base building and increases the strategic movement
  speed of groups departing that player's core by +1.0;
- POWER is a robot-only base building and currently gives robot units +2 max HP
  when they materialize into battle;
- LIFE is an infected-only base building and spawns one free SWARMLING on the
  owner's core every 6 ticks after completion;
- base buildings are added before the core in core defense battles;
- core assaults are gated by base building slots: slot 2 is targetable before
  slot 1, and the core is targetable only after all live base buildings are
  down;
- lethal overkill into the active base-building slot does not spill into another
  building or the core during the same combat tick;
- building HP persists between battles;
- buildings cannot be manually repaired, replaced, demolished, or rebuilt over
  while their slot remains occupied;
- completed buildings at 0 HP stay in their slot, are temporarily disabled, and
  auto-regenerate after `BUILDING_REGEN_DELAY_TICKS *
  BUILDING_ZERO_HP_REGEN_DELAY_MULTIPLIER`, currently `8 * 2 = 16` ticks
  without damage, at +5 HP/tick;
- damaged completed buildings above 0 HP use the normal 8-tick no-damage delay
  before regeneration;
- disabled buildings do not fight, generate income, unlock tech units, apply
  BOOSTER/POWER effects, or spawn LIFE units until they regenerate above 0 HP;
- when a node changes owner, buildings not owned by the new owner are removed;
- destroyed/dead core nodes are neutralized and lose buildings permanently.

## Economy

`ECONOMY_COLLECTOR` adds 2 passive resources per tick after completion.

The base economy and captured-point income remain unchanged.

## Unit Rosters

Current active rosters:

```text
Humans:
  base: VANGUARD, RANGER, SKIRMISHER
  Tech1: RAIDER
  Tech2: BMP

Infected:
  base: SWARMLING, SPITTER, BRUTE
  Tech1: RAVAGER
  Tech2: BEHEMOTH

Robots:
  base: WALKER, LASER, DRONE
  Tech1: GUARDIAN_MECH
  Tech2: SIEGE_PLATFORM
```

Faction production is hard-gated: a player can only produce units from their
own faction. Tech1 requires a completed live `TECH_1`; Tech2 requires completed
live `TECH_1` and `TECH_2`.

## Tech Units

Tech units exist so the unlock path is real:

```text
RAIDER / RAVAGER / GUARDIAN_MECH require completed TECH_1
BMP / BEHEMOTH / SIEGE_PLATFORM require completed TECH_1 and completed TECH_2
```

`completed` here means finished and above 0 HP. A disabled tech building at
0 HP does not unlock production until it regenerates.

Current provisional stats:

```text
VANGUARD: hp 14, atk 3, range 1.0, speed 1.5, cost 3
RANGER: hp 4, atk 10, range 3.5, speed 1.5, cost 5
SKIRMISHER: hp 3, atk 4, range 1.0, speed 3.0, cost 2
RAIDER: hp 56, atk 11 x 2 attacks/tick, range 2.8, speed 3.2, cost 9
BMP: hp 230, atk 11 x 5 attacks/tick, range 4.0, speed 1.0, cost 26

SWARMLING: hp 2, atk 2, range 0.8, speed 3.6, cost 1
SPITTER: hp 3, atk 5, range 2.5, speed 2.2, cost 3
BRUTE: hp 24, atk 4, range 1.0, speed 1.1, cost 4
RAVAGER: hp 72, atk 11 x 2 attacks/tick, range 3.5, speed 3.4, cost 14
BEHEMOTH: hp 360, atk 12 x 6 attacks/tick, range 3.0, speed 0.8, cost 30

WALKER: hp 34, atk 4 x 2 attacks/tick, range 1.2, speed 1.0, cost 9
LASER: hp 9, atk 9 x 2 attacks/tick, range 4.0, speed 1.0, cost 11
DRONE: hp 6, atk 3 x 2 attacks/tick, range 1.1, speed 3.2, cost 6
GUARDIAN_MECH: hp 104, atk 8 x 3 attacks/tick, range 3.6, speed 1.3, cost 16
SIEGE_PLATFORM: hp 260, atk 12 x 6 attacks/tick, range 4.8, speed 0.6, cost 36
```

Design intent:

- Tech1 units should beat basic-unit compositions across factions in ordinary
  equal-budget fights, but still be expensive enough that rushing Tech1 has an
  economy/opportunity cost.
- Tech2 units should beat Tech1 and basic compositions, but they are expensive,
  slow to unlock, and should not erase map-play decisions.
- Infected pressure with numbers and cheap bodies.
- Humans stay the balanced baseline.
- Robots rely on fewer expensive bodies with high individual value.

Legacy enum names `SPECIALIST_1` and `SPECIALIST_2` remain as compatibility
aliases for RAIDER and BMP, but new replay/state data should use RAIDER/BMP.

## Building Stats

Current provisional stats:

```text
TURRET: hp 150, atk 3 x 5 attacks/tick, range 4.0, cost 28, build 14t
BOOSTER: hp 28, human-only, speed bonus +1.0, cost 18, build 10t
POWER: hp 34, robot-only, +2 HP combat bonus, cost 22, build 12t
LIFE: hp 34, infected-only, free unit every 6t, cost 22, build 12t
ECONOMY_COLLECTOR: hp 30, income 2, cost 30, build 20t
TECH_1: hp 36, cost 28, build 16t
TECH_2: hp 42, cost 40, build 22t
```

Turret/building numbers are explicitly provisional and must be balance-tested
before serious population training.

## 2026-06-06 Balance Patch

Applied before the next Sprint 3 training prep. Current live deltas:

```text
RAIDER:
  cost 3 -> 9
  hp 12 -> 56
  atk 8 -> 11 x 2 attacks/tick
  range 1.8 -> 2.8
  speed 3.0 -> 3.2

BMP:
  cost 16 -> 26
  hp 120 -> 230
  atk 10 x 2 -> 11 x 5 attacks/tick
  range 3.5 -> 4.0

TURRET:
  cost 12 -> 28
  build 8t -> 14t
  hp 30 -> 150
  atk 5 x 1 -> 3 x 5 attacks/tick

ENHANCED_TURRET:
  removed from the game

BOOSTER:
  added as the human unique base building

POWER:
  added as the robot unique base building

LIFE:
  added as the infected unique base building

BARON_HP: 200 -> 300
NEUTRAL_GUARD_HP: 2 -> 4
NEUTRAL_GUARD_ATK: 2 explicit
NEUTRAL_GUARD_ATTACK_RATE: 1 implied -> 2 explicit
LIFE_SPAWN_UNIT: SKIRMISHER -> SWARMLING
OBSERVATION_VERSION: train_obs_v2.2 -> train_obs_v3.1
ACTION_DIM: 2125 -> 447
OBSERVATION_DIM: 2708 -> 5188

ECONOMY_COLLECTOR:
  cost 24 -> 30
  build 16t -> 20t
  income unchanged at +2/tick

TECH_1:
  cost 18 -> 28
  build 12t -> 16t

TECH_2:
  cost 24 -> 40
  build 16t -> 22t
```

This patch intentionally invalidates older policy checkpoints. The next serious
training run should start from the new schema instead of trying to reuse the old
human-only action head.

Collector payback from game start is now 35 ticks:
20 build ticks plus 15 ticks to recover the 30 resource cost at +2/tick.

Scripted eval variants added:

```text
scripted_core
scripted_core_raider
scripted_core_bmp
```

These variants are intended as core-pressure evaluation baselines. The old
`scripted` bot remains available, but it is no longer a sufficient gate by
itself because it often proves path/targeting weakness rather than combat
strength.

## Population-Training Compatibility Note

The current schema is intentionally faction-aware and includes active human,
infected, and robot unit kits. Any future faction expansion that adds unit types
or production/build actions must bump the observation/action contract again.

## 2026-06-05 Crash And Balance Snapshot

This snapshot is historical. The 2026-06-06 balance patch above supersedes the
RAIDER and ECONOMY_COLLECTOR numbers in this section.

Crash tests after construction/faction visuals and tech-unit integration:

```text
python scripts/crash_test_simulation.py --matches 100 --seed-base 2026060600
result: seeds_with_issues 0, winners TEAM_A 50 / TEAM_B 46 / DRAW 4

python scripts/crash_test_simulation.py --matches 100 --seed-base 2026060500
result: seeds_with_issues 0, winners TEAM_A 46 / TEAM_B 49 / DRAW 5
```

Regression fixed during this pass:

- empty capture of a point now immediately removes unfinished/completed
  buildings from the previous owner;
- direct node-owner changes in the engine go through one helper, so future
  capture paths should not preserve hostile buildings accidentally.

Latest balance suite:

```text
python scripts/current_balance_suite.py --matches 100 --lord-matches 50 --seed 20260613 --output analysis/current_balance_suite_seed_20260613.json
```

Detailed construction/economy/tech audit:

```text
docs/CONSTRUCTION_BALANCE_DEEP_DIVE_20260606.md
analysis/construction_balance_deep_dive_20260606_50.json
analysis/construction_balance_core_focus_20260607_50.json
```

Random macro sample:

```text
TEAM_A 52 / TEAM_B 43 / DRAW 5
winner types: death_time 77 / normal 18 / draw_no_core_damage 5
ticks: mean 284.46, median 300
```

Lord trace sample:

```text
neutral Lord killed: 50 / 50
spawned after kill: 29 / 50
killed but no owner: 21 / 50
first kill tick: min 27, median 75.5, mean 81.02, max 174
spawned Lord destroyed a core: 24 / 29
spawned Lord died before core: 5 / 29
owner-team win rate after spawn: TEAM_A 82.35%, TEAM_B 83.33%
```

Equal-budget 50 micro checks:

```text
basic RPS:
VANGUARD beats SKIRMISHER
SKIRMISHER beats RANGER
RANGER beats VANGUARD

tech layer:
RAIDER beats VANGUARD/RANGER/SKIRMISHER
BMP beats VANGUARD/RANGER/SKIRMISHER/RAIDER
```

Objective thresholds from the current one-army tests:

```text
Lord:
RAIDER wins at 50 budget (16 raiders, 4 survive)
BMP wins at 50 budget (3 BMP, 2 survive)
VANGUARD first wins at 60 budget (20 vanguards, 3 survive)
SKIRMISHER at 60 budget is mutual death
RANGER still loses at 60 budget

Core:
RAIDER wins at 20 budget
BMP wins at 20 budget
SKIRMISHER is mutual death at 20 and wins from 30
VANGUARD wins from 30
RANGER wins from 40
```

Interpretation:

- construction and tech units are mechanically stable enough for viewer/testing;
- the current tech numbers are intentionally strong because RAIDER/BMP are gated
  behind Tech1/Tech2 building cost and build time;
- death-time remains very common in random macro after construction, so this
  ruleset still needs training/eval validation before league acceptance;
- the Lord is no longer hard to kill in random macro, but many Baron fights end
  with no surviving valid owner, which should be watched during policy training.

## RL Visibility And Actions

The observation vector exposes:

- player faction;
- node buildings;
- building owner relation;
- building type;
- building HP ratio;
- construction progress;
- completed/incomplete state through progress and complete-tick-derived data;
- special unit counts in stationed, moving, and battle features.
- current human/infected/robot faction one-hot.

The action mask exposes:

- `BUILD` actions only where legal;
- special-unit production only after the required completed tech building;
- three factorized heads: `produce`, `move`, and `build`.

Current action-head breakdown:

```text
produce: NONE + 15 unit types = 16
move: NONE + 338 source/destination/intent/lord actions = 339
build: NONE + 91 building/node actions = 92
flat mask/logit total = 447
```

This replaces the old single categorical `PRODUCE_MOVE` explosion. A policy can
now produce one unit, move one group, and start one building in the same world
tick. The game engine still resolves movement, then build, then production.

## Baseline Bots

Random and scripted bots can now build simple structures. This keeps replay,
balance, and smoke generation from ignoring the construction rules entirely.

Scripted behavior is intentionally simple and should not be treated as final
construction strategy.
