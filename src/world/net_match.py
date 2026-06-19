"""Connect the frozen population's nets to World matches.

World's SoloQ ladder and league season both take a pluggable match function. By default
those are rating-based stochastic models (no nets). This module supplies the REAL ones: a
3v3 GameEngine match via ``src.core.runner.run_match`` — the same runner regionals and the
viewer Match tab use — driven by each player's bound net checkpoint. So with a world that was
seeded from the frozen population, ratings (SoloQ) and league results come from actual net play.

Resolution: each World player carries a ``net_id``; the frozen manifest
(``population/frozen_population_v1.json``) maps that to a ``policy:<checkpoint>`` model id, which
run_match turns into a PolicyBot (LRU-cached checkpoint loads). Per-player SoloQ impact comes
from own core survival + economy in the final state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from src.core.constants import CORE_HP
from src.core.runner import run_match as _default_run_match

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "population" / "frozen_population_v1.json"
HOME_SLOTS = ("A1", "A2", "A3")
AWAY_SLOTS = ("B1", "B2", "B3")

_MANIFEST_CACHE: dict[str, dict] = {}


def load_manifest(path: Optional[str | Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_MANIFEST
    key = str(p.resolve())
    if key not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
    return _MANIFEST_CACHE[key]


def net_model_index(manifest: dict) -> dict[str, str]:
    """net_id -> ``policy:<checkpoint>`` model id consumable by run_match."""
    return {
        e["net_id"]: f"policy:{e['checkpoint']}"
        for group in ("descendants", "ancestors")
        for e in manifest.get(group, [])
    }


def _slot_impacts(state, slot_pids: list[tuple[str, str]]) -> dict[str, float]:
    """Per-player SoloQ impact in [0,1] (0.5 neutral) from own core survival + economy,
    scaled relative to the strongest economy in THIS match so teammates differ."""
    raw: dict[str, tuple[float, float]] = {}
    for slot, pid in slot_pids:
        p = state.players.get(slot)
        core = max(0.0, float(getattr(p, "core_hp", 0.0) or 0.0)) / float(CORE_HP)
        res = max(0.0, float(getattr(p, "resources", 0.0) or 0.0))
        raw[pid] = (min(1.0, core), res)
    max_res = max((r for _, r in raw.values()), default=0.0) or 1.0
    return {
        pid: max(0.0, min(1.0, 0.6 * core + 0.4 * (res / max_res)))
        for pid, (core, res) in raw.items()
    }


def _models_factions(world: dict, idx: dict[str, str],
                     team_a: list[str], team_b: list[str]):
    players = world["players"]
    models: dict[str, str] = {}
    factions: dict[str, str] = {}
    slot_pids: list[tuple[str, str]] = []
    for slots, team in ((HOME_SLOTS, team_a), (AWAY_SLOTS, team_b)):
        for slot, pid in zip(slots, team):
            p = players[pid]
            net_id = p.get("net_id")
            if not net_id or net_id not in idx:
                raise KeyError(f"player {pid} has no manifest net (net_id={net_id!r})")
            models[slot] = idx[net_id]
            factions[slot] = p["faction"]
            slot_pids.append((slot, pid))
    return models, factions, slot_pids


def make_soloq_match_fn(
    world: dict,
    manifest_path: Optional[str | Path] = None,
    *,
    run_match_fn: Callable = _default_run_match,
    orders_provider: Optional[Callable[[list, list], Optional[dict]]] = None,
) -> Callable:
    """Return a SoloQ ``match_fn(team_a, team_b, ratings, rng) -> (winner, impacts)`` that plays
    a real 3v3 net match. ``team_a``/``team_b`` are 3 World player ids each.

    ``orders_provider(team_a, team_b) -> orders|None`` lets a caller overlay manager game-plans
    (SoloQ normally has none — the nets play free)."""
    idx = net_model_index(load_manifest(manifest_path))

    def match_fn(team_a, team_b, ratings, rng):
        models, factions, slot_pids = _models_factions(world, idx, list(team_a), list(team_b))
        seed = rng.randrange(1, 2 ** 31 - 1)
        kwargs = dict(seed=seed, bot_type="league", mode="training", replay_path=None,
                      player_models=models, player_factions=factions)
        if orders_provider is not None:
            orders = orders_provider(list(team_a), list(team_b))
            if orders:
                kwargs["orders"] = orders
        result = run_match_fn(**kwargs)
        if result.winner == "TEAM_A":
            winner = "A"
        elif result.winner == "TEAM_B":
            winner = "B"
        else:  # genuine draw — no rating-flip bias, pick a side by the seeded rng
            winner = rng.choice(["A", "B"])
        return winner, _slot_impacts(result.state, slot_pids)

    return match_fn
