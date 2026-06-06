# Simple 2D Engine Project Plan

Date: 2026-06-04

Status: implementation in progress.

Implementation status:

- Phase 1 started on 2026-06-04.
- Added static 2D lane/zone metadata contract in `src/core/map2d.py`.
- Added map metadata tests in `tests/test_map2d.py`.
- Added `map2d_version` and corrected movement/fire rule metadata in replay
  rules.
- Moving group snapshots now expose `segment_src`, `segment_dst`, `lane_id`,
  and `approach_id` for replay/viewer and future RL projection checks.
- Phase 1 smoke throughput on 2026-06-04: `1227.8 steps/sec` with 16 workers
  in `TrainingEnv` random-vs-random mode, approximately `4.42M steps/hour`.
- Phase 2 visual projection started on 2026-06-04.
- Replay moving groups now include visual-only `visual` positions while
  `GameEngine.observe()` keeps those visual coordinates out of the RL path.
- Phase 2 smoke throughput on 2026-06-04: `1212.9 steps/sec` with 16 workers
  in `TrainingEnv` random-vs-random mode, approximately `4.37M steps/hour`.
- `viewer_game` now prefers replay-provided moving-group `visual` coordinates
  and falls back to the old browser-side interpolation only for older replays.
- Post-viewer bridge validation on 2026-06-04: `176 passed` in pytest, and
  `1216.8 steps/sec` with 16 workers in `TrainingEnv` random-vs-random mode,
  approximately `4.38M steps/hour`.
- Browser validation for `viewer_game` on 2026-06-04: `2 passed` in
  `tests/test_game_viewer_browser.py`.
- New replay files now include JSON-ready `meta.map2d` data with nodes, lanes,
  directed approaches, and threat-zone metadata for viewer/debug consumers.
- Replay-map metadata validation on 2026-06-04: `177 passed` in pytest.
- Moving group snapshots now expose `threat_zone_ids` based on directed
  approaches, using explicit ids such as `Lord` and `core:A1`.
- Threat-zone metadata validation on 2026-06-04: `178 passed` in pytest, and
  `1197.2 steps/sec` with 16 workers in `TrainingEnv` random-vs-random mode,
  approximately `4.31M steps/hour`; observation dimension remains `1080`.
- Phase 5 gameplay slice started on 2026-06-04: Lord/core threat zones now
  start real `approach_battle` encounters before a moving group fully enters
  Baron/core nodes.
- Replay schema version bumped to `0.6.0`; replay rules now include
  `approach_threat_battles: true`.
- Approach battles are projected as node-objective battles for RL features, so
  action space and observation dimension remain unchanged.
- Phase 5 validation on 2026-06-04: `181 passed` in pytest,
  `tests/test_game_viewer_browser.py` `2 passed`, `tests/test_viewer_browser.py`
  `1 passed`, and `1193.8 steps/sec` with 16 workers in `TrainingEnv`
  random-vs-random mode, approximately `4.30M steps/hour`.
- Fresh 20 random-vs-random viewer replays generated with seed `20260604`.
  Results: `TEAM_A 16`, `TEAM_B 4`; winner types: `normal 11`,
  `death_time 9`. Approach battle starts: `353` total across all 20 replays,
  including `Lord 140`, `core:B3 51`, `core:B2 44`, `core:B1 43`,
  `core:A2 30`, `core:A1 28`, `core:A3 17`.
- Node-vs-edge ranged gameplay slice added on 2026-06-04: hostile ranged
  garrisons can start `approach_battle` against moving groups on adjacent
  approaches before full node entry. Melee-only garrisons still wait for normal
  node entry.
- Replay schema version bumped to `0.6.1`; replay rules now include
  `node_ranged_approach_battles: true`.
- Node-ranged validation on 2026-06-04: `184 passed` in pytest,
  `tests/test_game_viewer_browser.py` `2 passed`, `tests/test_viewer_browser.py`
  `1 passed`, and `1216.3 steps/sec` with 16 workers in `TrainingEnv`
  random-vs-random mode, approximately `4.38M steps/hour`.
- Fresh 20 random-vs-random viewer replays regenerated with seed `20260604`
  after node-ranged approach battles. Results: `TEAM_A 9`, `TEAM_B 11`;
  winner types: `normal 6`, `death_time 14`. Approach battle starts: `980`
  total across all 20 replays, including `Lord 121`, core zones `211` total,
  and ranged garrison zones `648` total. This is a balance warning: ranged
  garrison pressure now strongly increases stalled/death-time outcomes.
- Map-distance movement added on 2026-06-04: movement segment duration now
  comes from rendered map distance instead of a flat edge cost. Edge movement
  visually and mechanically runs from source node boundary to destination node
  boundary; node exit/enter/transit phases cover the in-node distance.
- Replay schema version bumped to `0.6.2`; replay rules now include
  `map_distance_movement: true` and `movement_map_units_per_tick: 30.0`.
- Map-distance validation on 2026-06-04: `184 passed` in pytest,
  `tests/test_game_viewer_browser.py` `2 passed`, `tests/test_viewer_browser.py`
  `1 passed`, and `965.4 steps/sec` with 16 workers in `TrainingEnv`
  random-vs-random mode, approximately `3.48M steps/hour`; observation
  dimension remains `1080`.
- Fresh 20 random-vs-random viewer replays regenerated again with seed
  `20260604` after map-distance movement. Results: `TEAM_A 7`, `TEAM_B 13`;
  winner types: `normal 13`, `death_time 7`. Approach battle starts: `919`
  total across all 20 replays, including `Lord 143`, core zones `186` total,
  and ranged garrison zones `590` total.
- Phase 6 pivot started on 2026-06-04 after clarifying that the prior hybrid
  still resolved active fights on the legacy 1D combat line. Added
  `src/core/combat2d.py` and switched `GameEngine` active battles to real map
  coordinates: units keep `x/y`, choose targets by Euclidean map range, move at
  the same `MOVEMENT_MAP_UNITS_PER_TICK * unit.speed` rate used by empty-map
  movement, and stop moving when they already have an attack unless they are
  the allied pushing Lord.
- Replay schema version bumped to `0.7.2`; replay rules now include
  `combat_2d_map: true` and `combat_range_to_map_units: 24.0`. Battle
  formation rows keep the old compatible fields and add `x`, `y`, and
  `range_map`.
- Replay ticks now include `visual_units`, a replay-only stable identity track
  for stationed units, moving groups, battle units, neutral guards, and
  Lord/Baron. It is not exposed through `GameEngine.observe()` and training
  mode keeps visual counters/buckets empty.
- Viewer and `viewer_game` now prefer replay-provided battle `x/y` positions
  instead of reconstructing active fights from the old 1D `pos` value.
- 2D combat validation on 2026-06-04: `187 passed` in pytest,
  `tests/test_game_viewer_browser.py` `2 passed`, `tests/test_viewer_browser.py`
  `1 passed`, and `1082.0 steps/sec` with 16 workers in `TrainingEnv`
  random-vs-random mode, approximately `3.90M steps/hour`; observation
  dimension is now `1114`.
- Fresh 20 random-vs-random viewer replays regenerated with seed `20260604`
  after 2D active combat. Results: `TEAM_A 15`, `TEAM_B 5`; winner types:
  `normal 9`, `death_time 11`; average length `237.8` ticks. All `29594`
  battle formation rows in those replays contain real `x/y` coordinates.
- Training observation metadata bumped to `train_obs_v0.8` on 2026-06-04 so
  old checkpoints do not silently pass compatibility checks after the 2D combat
  and Lord-route visibility changes.
- Training interface bumped again to `train_obs_v0.9` on 2026-06-04:
  `OBSERVATION_DIM=1117`, `ACTION_DIM=342`. The action space now includes
  explicit `JOIN_LORD` and `HUNT_LORD` special destinations for resurrected
  Lord support/counterplay, and observations include waiting Lord support
  counts.
- Training interface bumped to `train_obs_v1.0` on 2026-06-04:
  `OBSERVATION_DIM=1117`, `ACTION_DIM=1356`. The agent can now issue one
  production command and one movement command in the same world tick through
  bounded `PRODUCE_MOVE` actions. This intentionally avoids unlimited
  per-tick command spam while removing the artificial "produce or move" choice.
  Movement is resolved before production, so a unit produced this tick remains
  at the core until a later command.
- Current active training interface after construction/factions and factorized
  actions is `train_obs_v3.1`: `OBSERVATION_DIM=5188`, `ACTION_DIM=447`.
  Actions are split into `produce`, `move`, and `build` heads.
- Resurrected Lord now waits 2 ticks before push. During this delay the replay
  and RL observation already expose owner, target, path, ready tick, and ETA.
- Current 2D status and open technical issues are summarized in
  `docs/CURRENT_2D_STATUS.md`.
- Experimental PixiJS replay viewer added in `viewer_pixi/` on 2026-06-04. It
  uses Pixi display objects and replay-provided battle `x/y` positions, while
  keeping replay data as the source of truth and staying out of the training
  loop.
- PixiJS replay viewer can now trigger `/api/generate-replays` directly with a
  manual UI cap of 100 matches. Successful generation replaces the replay list
  and auto-loads the first freshly generated replay.

This document collects the agreed direction for moving the game from the current
graph + 1D combat model toward a simple 2D-like rules engine and a readable RTS
viewer, while keeping reinforcement learning strategic, cheap, and fast.

## Core Goal

Build a simple 2D-like game engine where units visibly move through lanes,
approach nodes, fight near objectives, and get attacked by Lord/core threat
zones before they fully enter a node when rules allow it.

The new engine must not become StarCraft-level physics. It must be a fast
lane/zone rules engine with deterministic combat bubbles and aggregation back
to the current strategic state.

The RL agent must remain a strategic commander, not a micro-controller.

## Non-Negotiables

- Do not increase RL action complexity.
- Do not give RL per-unit micro-control.
- Do not make RL select individual attack targets.
- Do not require RL to reason about raw 2D coordinates directly.
- Do not put browser rendering, canvas animation, or UI code in the training
  loop.
- Preserve headless training.
- Preserve the generation/training speed target: do not drop below
  `3M steps/hour` without an explicit decision and benchmark evidence.
- Keep gameplay parity between training generation and replay generation.
- Viewer must display engine outcomes; viewer must not decide combat outcomes.
- Any real rules change invalidates old checkpoint quality until re-evaluated.

## What Stays Simple For RL

RL actions stay strategic:

```text
HOLD
PRODUCE Vanguard / Ranger / Skirmisher
MOVE source -> destination with intent
```

No new action dimensions are introduced for:

```text
unit id
exact x/y point
formation shape
attack target
kiting
focus fire
spell timing
manual retreat from a local fight
```

Routing remains automatic. The policy can choose a reachable destination, while
the engine chooses the shortest legal route and the corresponding approach
lanes.

## Important Clarification

"RL unchanged" means:

- same strategic command level;
- same broad action semantics;
- no per-unit control;
- no micro burden;
- same training pipeline shape where possible.

It does not mean old checkpoint skill remains valid. If engine rules change
enough that fights resolve differently, old checkpoints can still be loaded only
if interfaces match, but they must be re-evaluated and likely retrained.

## Why We Are Doing This

The current graph + 1D combat model is fast, but it creates hard-to-read and
hard-to-balance artifacts:

- fights depend too much on left/right line placement;
- approach direction is hard to see;
- Lord/core range cannot naturally affect units on nearby lanes;
- units can appear to teleport or change meaning in the viewer;
- visual debugging is weak because battles are compressed into cards;
- balance work sometimes tunes artifacts of the line instead of game behavior.

The new model should make cause and effect readable:

```text
units leave a node
units move along a lane
they enter a threat zone
an approach battle starts
some units die
survivors enter or fail to enter the objective
the objective changes owner or remains contested
```

## Simple 2D Model, Not Real Physics

Use a deterministic lane/zone model:

- nodes are circular or rectangular objective zones;
- edges are lanes/tunnels between zones;
- units have visual positions along lanes and inside zones;
- combat eligibility is decided by zones, lane progress, and ranges;
- movement uses fixed lane progress, not expensive pathfinding;
- collision/separation is visual or simplified, not a heavy physics solver;
- battle resolution remains tick-based and deterministic.

Avoid:

- full navmesh pathfinding in the training loop;
- continuous collision physics;
- per-unit steering as source of truth;
- projectile simulation as source of truth;
- browser/renderer-dependent gameplay.

## Map Concepts

The map keeps the existing strategic graph:

```text
cores: A1 A2 A3 B1 B2 B3
control points: CP1 CP2 CP3 CP4 CP5 CP6
central objective: Baron/Lord node
```

But each graph element gets 2D metadata:

```text
node center position
node capture radius
node combat radius
approach anchor per connected lane
lane centerline or simple polyline
lane width / tunnel flag
threat-zone radius for Lord and cores
```

The passages between cores and nearby points are treated as tunnels for
readability and routing:

```text
A cores -> CP1 / CP2
B cores -> CP5 / CP6
```

## Combat Model

The engine should support these combat locations:

```text
node battle      - units are inside the same objective zone
edge battle      - opposing groups meet on a lane
approach battle  - units on a lane are attacked by units/objective in a node
lord-zone battle - Lord attacks units approaching or inside his zone
core-zone battle - core attacks units approaching or inside its zone
```

Units cannot pass through an active battle on their lane. If a moving group
enters an already active edge/approach battle on its route, it automatically
joins from the correct approach side.

## Lord Rules In The 2D-Like Engine

Lord should not only attack units after they are fully inside the Baron node.
Lord should threaten the approaches to the Baron node.

Expected behavior:

```text
group moves CP1 -> Baron
Lord is alive and neutral
group enters Lord threat zone on that lane
Lord starts attacking before the group reaches Baron center
if group survives and reaches Baron zone, normal Lord battle continues
```

This can be implemented as a lane/zone check, not full 2D physics.

Lord range should be represented as a threat radius around the Lord's center.
Standing at center, Lord should cover the whole Baron objective and its intended
approach zones. If later we allow Lord to move, his threat zone moves with him.

## Core Rules In The 2D-Like Engine

Cores should also attack approaches, not only units already inside a core node.

Expected behavior:

```text
group moves from CP1/CP2 toward A core or from CP5/CP6 toward B core
enemy core threat zone starts shooting when the group enters the approach zone
survivors can continue into the core zone and attack the core
```

Core range should remain longer than Lord range by design if that balance rule
continues to be desired.

## Node-Vs-Edge Ranged Combat

Ranged units inside a node can fight units on an adjacent approach lane if their
range reaches the lane approach zone.

Example:

```text
Rangers hold CP1
enemy group moves along an adjacent lane toward CP1 or Baron
if the approach segment is in range, an approach battle starts
```

Melee units in a node do not magically hit lane units unless the approach battle
reaches melee/contact rules. Rangers, Lord, cores, and other ranged entities can
create threat before full node entry.

## RL Observation Requirements

The RL agent must not become more complex, but it must not be blind to why a
fight changes outcome based on approach direction.

The observation/projection must preserve or expose:

```text
moving group source node
current edge / approach id
destination node
final destination
intent
progress
ETA
composition by unit type
active battle id, if joined
entry approach / side
```

For each node/objective, the observation/projection should expose aggregated:

```text
owner
node type
units inside by owner/team/type
visiting units by owner/team/type
incoming groups by approach
neutral strength / income
core hp / core type
Lord hp / Lord state
active node battle summary
active approach battle summary
```

For each lane/approach, the projection should expose aggregated:

```text
which groups are on it
owner/team/type composition
ETA/progress buckets
whether the lane is in Lord/core/node threat range
whether an edge/approach battle is active
```

This is critical. If the policy sees only "my group goes to Baron" and "enemy is
in Baron", but not which approach is used, then two strategically different
situations become indistinguishable to RL.

## RL Interface Strategy

Preferred strategy:

```text
new 2D-like engine state
  -> compatibility projection
  -> GameState-like strategic snapshot
  -> existing observe_state / action masks where possible
```

The old `GameState` shape can be kept as a compatibility surface even if the
new engine has extra internal data.

If observation changes are unavoidable, version them explicitly:

```text
train_obs_v1.0 or later
OBSERVATION_DIM updated
checkpoint metadata must match
old checkpoints are stale until revalidated
```

Current v3.1 uses factorized `produce`, `move`, and `build` heads. PPO sums the
three head log-probs/entropies while keeping one shared value estimate.

## Training Speed Contract

Target throughput remains:

```text
>= 3M environment steps/hour
```

Training mode must:

- run headless;
- disable replay trace details unless explicitly requested;
- avoid visual interpolation;
- avoid canvas/browser;
- avoid per-frame visual state;
- use compact aggregated combat data;
- benchmark with the same worker count assumptions as Sprint 3, currently
  defaulting to 16 workers unless a new benchmark says otherwise.

Required benchmarks:

```text
old engine steps/hour baseline
new engine steps/hour headless
new engine replay generation speed
worker scaling 8/12/16 if needed
```

Acceptance target:

```text
new headless speed >= 3M steps/hour
or explicit redesign before training resumes
```

## Viewer Strategy

The viewer becomes a readable 2D RTS-style replay viewer.

Viewer responsibilities:

- draw a simple map, lanes, tunnels, nodes, cores, Lord;
- show individual visual units with stable visual ids;
- show movement along lanes;
- show approach/node/edge battles;
- show attacks, deaths, capture progress, core/Lord HP bars;
- allow clicking units/groups/nodes/battles for detail panels;
- reduce on-map text; move details to hover/side panels.

Viewer must not:

- decide winners;
- simulate hidden gameplay outcomes;
- change replay results;
- feed training observations.

Replay is the source of visual truth, but engine is the source of gameplay
truth.

## Replay Requirements

Replay should include enough data to reconstruct stable visuals:

```text
visual unit ids or deterministic visual id generation seed
group id
unit type / owner
route path
current lane / node / battle
join battle event
leave battle event
attack event
death event
capture event
Lord/core attack event
HP snapshots or deltas for key objects
```

Replay can be richer than training data. Training mode should not pay the full
replay cost unless replay generation is explicitly requested.

## Engine Implementation Phases

### Phase 0 - Contract And Baseline

- Keep this document as the migration contract.
- Record current headless training/generation benchmark.
- Record current replay generation benchmark.
- Record current unit balance and known issues.

Gate:

```text
benchmark and current behavior documented
```

### Phase 1 - Static 2D Map Metadata

- Add a simple 2D map metadata module.
- Define node centers, radii, lane polylines, tunnel flags, approach ids.
- No gameplay changes yet.
- Viewer can start reading this metadata.

Gate:

```text
tests verify graph connectivity and approach ids
no training speed change
```

### Phase 2 - Visual Projection Only

- Generate stable visual positions from existing graph/replay state.
- Do not change gameplay.
- Use it to improve viewer readability first.

Gate:

```text
old replay outcome unchanged
viewer shows stable unit movement without teleporting
```

### Phase 3 - 2D-Like Movement Executor

- Replace edge/node movement internals with lane progress and node-zone phases.
- Keep strategic commands unchanged.
- Keep aggregation back into `moving_groups`, `node.units`, and battle summaries.

Gate:

```text
RL action masks still work
movement ETA/progress visible
speed benchmark still acceptable
```

### Phase 4 - Combat Bubbles

Implement:

- node battles;
- edge battles;
- approach battles;
- joining existing lane battles;
- no passing through active lane battles.

Gate:

```text
deterministic tests for each battle type
approach direction preserved in state and observation/projection
```

### Phase 5 - Lord/Core Threat Zones

- Lord attacks units on Baron approaches.
- Core attacks units on core approaches.
- Threat zones are lane/zone checks, not full geometry physics.

Gate:

```text
Lord can shoot approaching groups before Baron node entry
core can shoot approaching groups before core node entry
RL observation/projection exposes the relevant threat
```

### Phase 6 - Training Compatibility

- Confirm existing action space still works.
- Confirm observation/action masks are not more complex for the policy.
- If observation version changes, update metadata and tests.
- Run random/scripted eval smoke tests.

Gate:

```text
training loop runs headless
speed >= 3M steps/hour
random/scripted match generation works
```

### Phase 7 - Full Viewer Rework

- Build the RTS-style replay viewer on top of the new replay/projection data.
- Add click/hover inspection.
- Keep generation UI separate from training.

Gate:

```text
manual replay inspection is readable
no viewer dependency in training
```

### Phase 8 - Rebalance And Retraining

- Re-run PvE and PvP balance probes.
- Re-run random/scripted baseline stats.
- Treat old checkpoints as stale until re-evaluated.
- Start fresh training only after speed and behavior gates pass.

Gate:

```text
fresh benchmark
fresh balance report
fresh eval baseline
explicit approval before long training
```

## Tests To Add

Core engine tests:

- group moves along lane with expected ETA;
- group enters node through correct approach;
- group cannot pass through active edge battle;
- reinforcement joins existing edge battle from correct side;
- node garrison can shoot an approaching lane group when in range;
- node garrison cannot shoot an out-of-range lane group;
- Lord shoots approaching Baron lane group;
- core shoots approaching core lane group;
- melee does not attack across approach unless contact rules allow it;
- ranged attack after movement follows active unit rules;
- battle result is deterministic under fixed seed.

RL/projection tests:

- action masks unchanged in shape/semantics where intended;
- moving group observation includes source/destination/approach/ETA/composition;
- active approach battle appears in observation/projection;
- B-side mirroring preserves approach meaning;
- policy can distinguish two identical node states with different incoming
  approaches if that distinction affects combat.

Performance tests:

- headless no-replay generation benchmark;
- replay generation benchmark;
- worker scaling benchmark;
- fail or warn if speed drops below the accepted threshold.

Viewer tests/checks:

- replay loads without simulating outcomes;
- visual unit ids stay stable;
- units do not visually teleport between edge/node phases;
- Lord/core attacks are visible from threat zones;
- clickable nodes/groups/battles expose detail data.

## Balance Concerns To Revisit After Engine Change

The current balance was tuned under the 1D combat model. It must be rechecked
after combat bubbles and threat zones exist.

Specific areas:

- Skirmisher PvE efficiency against Lord and cores;
- Lord's attack model against many low-HP units;
- neutral strength vs income payoff;
- core vulnerability once attackers reach core zone;
- ranged units shooting after movement;
- melee contact snap behavior if still relevant;
- impact of approach battles on defender advantage.

## Risks

### Speed Risk

The main project risk is making the 2D-like engine too physical. If the engine
uses heavy per-unit collision/pathfinding, training speed may fall below target.

Mitigation:

```text
lanes, zones, fixed progress, compact combat summaries
no renderer in training
benchmark after every major phase
```

### RL Blindness Risk

If approach direction affects outcomes but observation does not expose approach
data, RL will see different tactical situations as identical.

Mitigation:

```text
moving group approach id
lane battle summaries
incoming groups by approach
threat-zone flags
tests for distinguishability
```

### Checkpoint Validity Risk

Old checkpoints may run but not represent skill in the new ruleset.

Mitigation:

```text
metadata versioning
fresh eval
fresh training after gates
```

### Viewer Truth Risk

If viewer invents outcomes independently, debugging becomes misleading.

Mitigation:

```text
engine owns outcomes
viewer only interpolates known events
visual-only fields never feed RL
```

## Acceptance Criteria

The project is acceptable only if all are true:

- RL action complexity remains unchanged.
- Training remains headless.
- Training speed is at least `3M steps/hour`, or there is an explicit approved
  redesign before training resumes.
- Approach/lane information is visible to RL when it can affect outcomes.
- Lord can threaten Baron approaches in the engine, not only in the viewer.
- Core can threaten core approaches in the engine, not only in the viewer.
- Viewer is readable enough to inspect movement, fights, deaths, and objective
  control without relying on battle cards.
- Old checkpoints are not used as quality evidence without fresh eval.
- Fresh random/scripted baselines are generated after the engine change.

## Final Principle

The game can look and behave like a simple 2D RTS, but the AI must still play a
small strategic game:

```text
simple decisions for RL
fast deterministic engine for training
rich 2D replay for humans
```
