# Game Viewer

`viewer_game/` is the map-first game visualization for replay inspection.

It is intentionally separate from `viewer/`:

- `viewer/` remains the diagnostic replay UI with tabs, logs, generation controls, and exact text summaries.
- `viewer_game/` is the playfield-style UI: one canvas, camera pan/zoom, click selection, stable visual entities, and minimal overlay text.

## Visual Model

- The replay is still the source of truth. The game viewer does not simulate combat, movement, ownership, production, or victory.
- Active battles use true replay formation rows: `[uid, team, unit_code, hp, max_hp, pos, range]`.
- Movement and stationed forces outside battle are expanded visually from aggregate counts such as `VANGUARD: 7`.
- A visual tracker reuses same-team/same-type entities across representation changes within a handoff radius. This is for readability only; it does not create new simulation state.
- Node exit, edge movement, node transit, node entry, and battle formation are rendered by one canvas pipeline so movement appears continuous instead of switching between UI widgets.

## Usage

Run the normal viewer server:

`powershell -ExecutionPolicy Bypass -File .\start_viewer.ps1`

Open:

`http://127.0.0.1:8001/viewer_game/index.html`

The old diagnostic viewer remains available at:

`http://127.0.0.1:8001/viewer/index.html`

## Current Limits

- True persistent unit ids outside battle do not exist in the replay schema yet. The game viewer preserves visual continuity through handoff matching, not through engine-level unit identity.
- For a fully exact StarCraft-like replay, the engine/replay schema must eventually emit persistent visual/unit ids for production, stationed units, moving groups, battle entry, battle survival, and post-battle regrouping.
