# Sprint 1 viewer check

## Scope

Replay viewer reads JSON only. It does not simulate combat, movement, income, battle resolution, or victory.

## Replay schema used

- `schema_version`: `0.5.1`
- `combat_trace_format`: `compact_v3`
- sample path: `replays_viewer_debug_smoke/replay_0.json`
- active battle formation rows are `[uid, team, unit_code, hp, max_hp, pos, range]`
- `BATTLE_TICK.summary.attacks` rows are `[attacker_uid, target_uid, damage, attacker_team, attacker_code, target_team, target_code]`

## Naming Canon

- Ranged unit display name is **Ranger**, code **R**.
- Ranger movement speed is `0.75`; mixed moving groups use the slowest unit speed.
- The legacy enum alias `UnitType.ARCHER` still exists for compatibility, but new production/replay data should use `RANGER`; viewer displays both `ARCHER` and old `A` rows as `R`.
- Central boss display name is **Lord**, code **L**.
- The internal map node id is still `Baron` and some engine events/constants still use `baron/BARON` as legacy compatibility names; viewer displays those as `Lord/L`.
- Do not introduce new user-facing `Archer/A` or `Baron/B` labels. `A` and `B` are reserved for player/team identifiers.

## Visual mapping

- The center panel is a map-first replay UI: nodes are rendered as rooms, edges are rendered as corridors, and core-to-CP1/CP2/CP5/CP6 passages are rendered as heavier tunnel corridors.
- `nodes[*].owner` controls room color: A players are red, B players are blue, neutral nodes are grey/gold.
- `nodes[*].units`, `nodes[*].visiting_units`, neutral guards, Lord guards, and moving groups currently inside a node phase are shown as separate force cards around the room, with fixed slots to avoid marker stacking.
- `nodes[*].unit_ready_tick` marks units that arrived/resolved this tick and cannot leave that node until the next tick.
- `nodes[*].unit_entry_side` records the last 1D combat side (`attacker`/left or `defender`/right) used by stationed units for each player.
- `nodes[*].neutral_hp` is shown on control points as a small neutral guard badge.
- `moving_groups` are shown as labeled route squad cards only while they are on an `edge` phase. Multiple squads on the same edge are offset into parallel lanes. During `node_exit`, `node_transit`, or `node_enter`, the group is visually counted inside that node instead of being drawn on the edge.
- `moving_groups[*].path` is the full planned route, while `src`/`dst` describe the current phase endpoints.
- `active_battles[*].location_kind == "node"` is shown as an in-map battle card near the node.
- `active_battles[*].location_kind == "edge"` is shown as an in-map battle card near the edge midpoint/position.
- `active_battles[*].formation` is shown as a compact 1D combat line inside each battle card.
- `BATTLE_TICK.summary.attacks` is shown as colored attack links plus a short attack log; ranged attacks are dashed.
- Recently resolved battles remain visible for a short replay lookback as faded cards, using the last active formation snapshot plus the final `BATTLE_TICK.summary`.
- `events` are shown in the tick event list; important recent map events such as captures, core deaths, Lord resolution, edge-battle starts, and unit deaths are also shown as short-lived map callouts.
- player alive/resources are shown from `players`.
- Lord state is visible through the central legacy `Baron` node owner and replay event list.
- Core HP is persistent and exposed as `nodes[*].core_hp` for core nodes and `players[*].core_hp`.
- The `State` tab shows active node state as visual force/order cards: stationed owner units, visitors, neutral/Lord center guards, ready locks, and incoming/outgoing movement orders with route progress.

## Simulation Notes

- Neutral guard and Lord combat units are centered exactly at combat position `10.0`, not placed on the defender/right side.
- Neutral guards do not regenerate after winning a fight; the node keeps only the surviving guard count.
- Neutral Lord stats are `120hp`, `16 atk`, and attack rate `3`.
- Neutral Lord HP persists across failed attempts to kill it.
- Killing the neutral Lord spawns an allied `LORD` unit with full HP on the same node. It automatically pushes along a shortest route toward a random living enemy core and fights enemies encountered on the way.
- The allied `LORD` keeps one persistent HP pool across all push fights. If it dies during the push, it does not resurrect again. If it destroys its first core, it disappears regardless of remaining HP.
- Neutral guards are represented as a stationary centered group with Ranger-equivalent range (`3.0`) and a small strength-based damage bonus. Once aggroed by an attack/capture, the whole neutral group attacks.
- Current neutral aggro is intent-based: `CAPTURE`/`ATTACK` starts neutral aggro immediately, while `TRANSIT` does not. Future damage-triggered aggro is tracked in `docs/SPRINT_2_TRAINING_PREP.md`.
- TEAM_A cores are placed on the left side of the combat line, TEAM_B cores on the right side. Cores have high attack range.
- Death Time starts at tick `270`: all living cores take fixed damage each tick through tick `299`, leaving a full-health core at `1hp`. At tick `300`, final lethal damage resolves the match by the number of own cores that survived until the final hit, then total core HP/damage tie-breakers. Draw is reserved for games with no non-death-time core damage.
- Combat approach movement uses unit speed. Speed `1.0` crosses the full combat line in `TRAVEL_TIME` world ticks; faster/slower units scale from that baseline. Attack cadence and damage are not accelerated.
- Route movement speed is based on the slowest unit in the moving group. Edge travel, empty node crossing, source exit, and final node entry all scale from that same speed.
- A source node exit or final node entry costs half of a full node crossing. Passing through an intermediate node costs one full node crossing.
- Neutral nodes are valid transit nodes even while neutral guards are alive; passing through does not aggro guards by itself.
- Units that arrive at a node or survive a node battle receive a one-tick ready lock. They can be observed on that node before they can be moved to the next edge.
- Node battles stay 1D, but player units now enter from the side implied by their route. Stationed owner/visitor units reuse their remembered `unit_entry_side` instead of always falling back to TEAM_A left / TEAM_B right.
- Edge battles block transit on that edge. Any moving group that reaches an active edge battle is automatically pulled into it from the side implied by its `src -> dst` route; moving through an edge battle is not allowed.
- Battle and event cards use fixed heights with internal scroll areas so UI height does not jump when event counts change.

## Current Control Model

- Action space is `HOLD`, `PRODUCE`, and routed `MOVE`.
- Sending units to hold a node is currently represented as `MOVE` to that node, then no follow-up `MOVE`; arrived units stand at the abstract node center and receive a one-tick ready lock.
- The current bots can leave units in place by choosing `HOLD`, but there is no dedicated high-level `DEFEND_POINT` or persistent garrison intent yet.
- A `MOVE` takes all currently controllable units for that player from the source node. There is no partial split or squad selection yet.
- A `MOVE` destination may be farther than one edge. The engine builds the shortest valid route automatically.
- Valid route destinations are neutral nodes, allied nodes, Lord, and enemy nodes that are adjacent to a neutral/allied transit node. Enemy nodes and living neutral Lord are valid targets, but not valid transit hubs.
- Active edge battle ids such as `A1|CP1` are not valid `MOVE` destinations. Units can only join an edge battle by already moving along that edge and reaching the battle position.
- Friendly reinforcements merge into node-level units; on a friendly node owned by another player, practical control transfers to the node owner. Visitor units on neutral/passive nodes keep player identity until moved or pulled into battle.
- Node combat is a 1D left/right abstraction derived from node order and remembered entry side. The map can have several graph entrances, but combat intentionally collapses them to exactly left or right; it does not preserve four separate entry channels.

## Manual check

1. Run the server from repo root:
   `python -m http.server 8001 --bind 127.0.0.1`
2. Open:
   `http://127.0.0.1:8001/`
3. Press `Load`.
4. Use play/pause, previous/next, seek slider, and speed selector.
5. Verify that ticks with `BATTLE_STARTED`, `BATTLE_TICK`, and `BATTLE_RESOLVED` have visible battle markers, matching side summaries, and stable formation rows.

## Automated checks

- `pytest -q`
- HTTP smoke:
  - `http://127.0.0.1:8001/`
  - `http://127.0.0.1:8001/viewer/index.html`
  - `http://127.0.0.1:8001/replays_viewer_debug_smoke/replay_0.json`

## Known limitation

Automated browser screenshot verification was not run in this environment because no browser executable was available in `PATH`. Static viewer wiring and HTTP delivery are covered by tests.
