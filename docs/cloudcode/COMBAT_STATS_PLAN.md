# Combat stats plan — per-player core kills / losses / assists

Goal: value/cards reflect REAL contribution so a carry outranks a passenger (fixes "один тащит всю
команду"). Feasibility verified against the engine. Backward-safe throughout (old worlds keep working).

Key fact: `UnitInstance.team` (`src/core/units.py:177`) is a `player_id`, not "TEAM_A" — so in any core
battle we already have the attacking units WITH their owner. Today that's collapsed to team level in
`_record_core_combat_damage` (`engine.py:4102`).

## Stage 1 — quick win (no merge-site change)

1. `src/core/state.py` (~147, GameState): add `core_damage_by_player`, `core_kills_by_player`,
   `core_losses_by_player`, `assists_by_player` = `field(default_factory=dict)`; init in
   `make_initial_state`. (Dicts on GameState, NOT PlayerState — PlayerState serializes into every replay
   frame; GameState dicts are read once at the end.)
2. `src/core/engine.py` `_finish_node_battle` core branch (~3348): pass `battle` to
   `_record_core_combat_damage`; attribute the core damage by each attacking unit's `max_hp` share
   (`unit.team` = player_id, source team only, exclude CORE/NEUTRAL). On core death: biggest contributor
   = core_kill; owner = core_loss; 2nd+ contributors = assists. Use `max_hp` (not current hp) so a
   sacrificial push isn't under-credited (record this choice in DECISIONS.md). DO NOT credit death-time/
   environment core deaths (`_apply_death_time_damage`/`_destroy_core_by_environment`) — timer, not a player.
3. `scripts/derive_player_cards.py` `play` (~109): read focal-slot (`A1`) per-player fields, backward-safe
   fallback to `core_damage_by_team["TEAM_B"]` if the engine is old. `attack` = personal core_dmg (recalibrate
   `REF_ATTACK` down on a fresh derive); new `teamplay`/assists attr (REF_ASSISTS ~1.5); strengthen `defense`
   via `core_losses`.
4. `src/world/valuation.py`: add `contribution_score(player)` from `soloq_matches` rows that carry the new
   fields (returns 0.0 if none → backward-safe); `skill_score *= (1 + 0.10*contribution_score)`. Carry vs
   passenger of equal SoloQ now diverge; `sporting_value` unchanged (already calls skill_score).
5. `src/world/net_match.py` `_slot_impacts` (~69): add per-player core_kills/losses/assists to the return
   (from `state.*_by_player[slot]`), and optionally fold a small term into impact so SoloQ reacts to it.
6. `src/world/soloq.py` `simulate_soloq` (~170): thread the per-player stats into the recorded
   `soloq_matches` entries so value/cards have data to read.

Tests: engine attribution unit test (a known battle → expected kills/assists); valuation carry>passenger;
derive backward-safe fallback.

## Stage 2 — full merge provenance (harder, separate)

Assists from armies merged BEFORE the battle aren't recoverable today: `_merge_visiting_units`
(`engine.py:3930`) buckets by `unit_type`, dropping donor identity. Minimal change: add
`visiting_provenance: dict` to `NodeState` (player→{donor_pid: count}), update at the merge site, and
read it at core-kill time. Stage 1's "2nd+ contributor in the killing battle" already covers team
core-finishes (the most valuable signal), so Stage 2 is optional polish.

Status: PLANNED (not yet implemented). Manager-depth work committed: 219fd69, 9d215b0.
