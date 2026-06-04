# Game Viewer

`viewer_game/` is the map-first game visualization for replay inspection.

It is intentionally separate from `viewer/`:

- `viewer/` remains the diagnostic replay UI with tabs, logs, generation controls, and exact text summaries.
- `viewer_game/` is the playfield-style UI: one canvas, camera pan/zoom, click selection, stable visual entities, and minimal overlay text.

## Visual Model

- The replay is still the source of truth. The game viewer does not simulate combat, movement, ownership, production, or victory.
- Replay schema `0.7.2` active battles use formation rows:
  `[uid, team, unit_code, hp, max_hp, legacy_pos, range, legacy_render_pos, x, y, range_map]`.
- Current combat uses real map `x/y` coordinates. `legacy_pos` and
  `legacy_render_pos` remain only for compatibility with older diagnostic UI
  paths and older replay files.
- Replay ticks include a replay-only `visual_units` layer with stable ids for
  stationed units, moving groups, active battle units, neutral guards, and the
  Lord/Baron. This layer is written only for replay generation and is not part
  of training observations.
- Older replays without `visual_units` still fall back to expanding aggregate
  counts such as `VANGUARD: 7` and handoff matching same-team/same-type
  entities within a radius.
- Node exit, edge movement, node transit, node entry, and battle formation are rendered by one canvas pipeline so movement appears continuous instead of switching between UI widgets.
- `viewer_pixi/` is the experimental PixiJS renderer. It uses Pixi display
  objects and prefers replay `visual_units`, while keeping the same rule that
  replay data is source of truth.
- Destroyed cores are shown as white dead nodes, not neutral/yellow nodes.
- Resurrected Lord has a two-tick `lord_delay` visual state on Baron before it
  starts moving; target, route, ready tick, and ETA are present in replay data
  immediately after resurrection.

## Usage

Run the normal viewer server:

`powershell -ExecutionPolicy Bypass -File .\start_viewer.ps1`

Open:

`http://127.0.0.1:8001/viewer_game/index.html`

Experimental PixiJS renderer:

`http://127.0.0.1:8001/viewer_pixi/index.html`

The old diagnostic viewer remains available at:

`http://127.0.0.1:8001/viewer/index.html`

## Current Limits

- `visual_units` is a viewer identity layer, not per-unit strategic simulation
  state. The game still resolves control, movement orders, production, and RL
  observations through aggregate units.
- The current PixiJS viewer improves rendering stability, but it remains a
  replay renderer. It does not run an independent physics simulation.
- Known large teleport sources fixed on 2026-06-04:
  stable visual backfill reset, direction-aware edge-battle continuation,
  node-enter approach reinforcement placement, and node-exit approach
  reinforcement placement.
