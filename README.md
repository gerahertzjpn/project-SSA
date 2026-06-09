# Project SSA

3v3 autobattler simulation with replay generation, neural policy checkpoints,
and a Pixi-based match viewer/configurator.

The current supported UI is `viewer_pixi`. Older `viewer` and `viewer_game`
entrypoints are intentionally disabled and redirect to Pixi.

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run a simple match from the CLI:

```powershell
python main.py --runs 1 --seed 42 --bot random
```

Run several replay matches:

```powershell
python main.py --runs 20 --seed 20260602 --bot scripted --out replays
```

Start the local viewer server:

```powershell
.\start_viewer.ps1
```

Or double-click:

```text
start_viewer.bat
```

Open:

```text
http://127.0.0.1:8001/viewer_pixi/index.html
```

The startup script searches ports `8001..8010` and opens the first working
viewer API.

## Pixi Viewer

`viewer_pixi` is the main workflow for inspecting and generating matches.

It supports:

- replay loading and playback;
- compact/full configuration HUD;
- per-player model selection for `A1`, `A2`, `A3`, `B1`, `B2`, `B3`;
- per-player faction selection: `humans`, `infected`, `robots`;
- one-click replay generation through `/api/generate-replays`;
- default model list with only current/latest models;
- `All models` mode for older compatible checkpoints.

Important: match generation parameters are sent to the server and stored in
replay metadata as `player_models` and `player_factions`.

## Models and Slots

Baseline models:

- `Random Bot`
- `Scripted Bot`
- `Core Bot`
- `Raider Core`
- `BMP Core`

Policy models are loaded from compatible `.pt` checkpoints under
`training_runs/**/checkpoints`.

Most team checkpoints store three policy states:

```text
policy_states: A1, A2, A3
```

Those names identify the source policy state inside the checkpoint. They are
not a binding to the match position. The Pixi UI exposes each state as its own
model id, for example:

```text
Mix12 latest A1
Mix12 latest A2
Recover305 latest A3
```

Any of those entries can be assigned to any player slot. For example, `B3` can
run `Mix12 latest A2`; the generated replay metadata will store that exact
model id in `player_models.B3`.

So `Mix12` is a run/lineage family name, `latest` is the checkpoint, and
`A1/A2/A3` is the selected neural state inside that checkpoint. Legacy policy
ids without an explicit `#A1`/`#A2`/`#A3` state are still accepted for backward
compatibility and use the old slot fallback logic.

See also:

```text
docs/PLAYER_POOL_LINEAGE_AUDIT.md
```

Current audit note: the available serious policy population should be treated
as human-only diagnostic policies unless a newer run was explicitly trained with
varied factions.

## CLI Usage

`main.py` runs matches without the browser:

```powershell
python main.py
python main.py --runs 20
python main.py --seed 123
python main.py --bot scripted
python main.py --mode training --runs 100
```

Modes:

- `replay`: records replay JSON files.
- `training`: headless summaries, no replay JSON.

Replay output defaults to:

```text
replays/replay_0.json
```

## Tests

Run the test suite:

```powershell
python -m pytest -q
```

Browser tests use Playwright. If Chromium is missing:

```powershell
python -m playwright install chromium
```

## Repository Layout

```text
main.py                 CLI match runner
start_viewer.ps1        Windows viewer/API launcher
viewer/server.py        Local static server + generation API
viewer_pixi/            Supported Pixi replay viewer and match configurator
viewer/                 Disabled legacy diagnostics UI redirect
viewer_game/            Disabled legacy canvas UI redirect
src/core/               Game state, engine, combat, replay, runner
src/bots/               Random, scripted, and policy bots
src/training/           RL observation/action spaces and PPO training envs
scripts/                Training, evaluation, audit, and benchmark scripts
tests/                  Unit, integration, API, and browser tests
docs/                   Run logs, plans, and player-pool audit notes
training_runs/          Local checkpoints, evals, and training artifacts
replays*/               Generated replay JSON fixtures and debug output
```

## Development Notes

- Keep `viewer_pixi` as the primary UI surface.
- Legacy UI entrypoints should remain redirect-only unless intentionally
  revived.
- `viewer/server.py` is the API boundary for UI generation.
- `src/core/runner.py::run_match` is the main simulation entrypoint.
- Policy checkpoints must be compatible with the current observation/action
  metadata before they appear in the viewer model list.
