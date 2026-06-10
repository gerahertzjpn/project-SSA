"""Player cards: 18 unique players, each strictly bound to one trained net.

A card has two layers:

- **identity** (static): nickname, faction, archetype, scouting attributes.
  Attributes are derived once from real evaluation data
  (``analysis/population_balance_full.json`` per-net winrates + mechanics) and
  snapshotted into the persistent store, so cards stay stable even if the
  analysis file is regenerated or removed.
- **career** (dynamic): Elo rating (everyone starts at 1000), money, games,
  wins/losses, titles, rating history. Updated by the regionals engine.

The persistent store lives in ``data/regionals/players.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = ROOT / "data" / "regionals" / "players.json"
DEFAULT_REGISTRY_PATH = ROOT / "training_runs" / "population_18_v1" / "registry.json"
DEFAULT_ANALYSIS_PATH = ROOT / "analysis" / "population_balance_full.json"

START_RATING = 1000

NICKNAMES = {
    "humans_1": "Falcon", "humans_2": "Bishop", "humans_3": "Marshal",
    "humans_4": "Atlas", "humans_5": "Drifter", "humans_6": "Warden",
    "infected_1": "Spore", "infected_2": "Hivemind", "infected_3": "Carrion",
    "infected_4": "Latch", "infected_5": "Murk", "infected_6": "Vespid",
    "robots_1": "Vector", "robots_2": "Cipher", "robots_3": "Dynamo",
    "robots_4": "Relay", "robots_5": "Patch", "robots_6": "Omega",
}

ARCHETYPES = {
    "baseline": "Универсал",
    "low_entropy": "Дисциплина",
    "high_entropy": "Импровизатор",
    "fast_lr": "Спринтер",
    "slow_explore": "Стратег",
    "alt_mix": "Джокер",
}

# Fallback attribute profiles used only when no analysis data is available.
_RECIPE_FALLBACK = {
    "baseline":     {"attack": 60, "defense": 60, "economy": 60, "tempo": 60, "consistency": 70, "potential": 60},
    "low_entropy":  {"attack": 55, "defense": 70, "economy": 65, "tempo": 50, "consistency": 85, "potential": 55},
    "high_entropy": {"attack": 70, "defense": 45, "economy": 50, "tempo": 70, "consistency": 40, "potential": 65},
    "fast_lr":      {"attack": 70, "defense": 50, "economy": 50, "tempo": 80, "consistency": 50, "potential": 60},
    "slow_explore": {"attack": 50, "defense": 70, "economy": 70, "tempo": 45, "consistency": 75, "potential": 55},
    "alt_mix":      {"attack": 65, "defense": 55, "economy": 60, "tempo": 65, "consistency": 60, "potential": 65},
}

_CONSISTENCY_BASE = {
    "low_entropy": 85, "slow_explore": 75, "baseline": 70,
    "alt_mix": 60, "fast_lr": 50, "high_entropy": 40,
}


class PlayersError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Attribute derivation from real evaluation data
# --------------------------------------------------------------------------- #

def _scale(values: dict[str, float], lo: int = 20, hi: int = 95) -> dict[str, int]:
    """Min-max scale raw per-net metrics into a readable lo..hi card range."""
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    span = (vmax - vmin) or 1.0
    return {k: int(round(lo + (v - vmin) / span * (hi - lo))) for k, v in values.items()}


def _derive_attributes(registry_entries: list[dict], analysis: Optional[dict]) -> dict[str, dict]:
    """Per-net 0-100 attributes; grounded in eval data when available."""
    per_net = ((analysis or {}).get("experiment_b_per_net") or {})
    winrates = per_net.get("net_winrate") or {}
    mechanics = per_net.get("net_mechanics") or {}

    out: dict[str, dict] = {}
    if winrates and mechanics:
        raw_attack: dict[str, float] = {}
        raw_defense: dict[str, float] = {}
        raw_economy: dict[str, float] = {}
        raw_tempo: dict[str, float] = {}
        for entry in registry_entries:
            net_id = entry["net_id"]
            mech = mechanics.get(net_id) or {}
            decisions = max(1, int(mech.get("decisions", 1)))
            on_units = float(mech.get("resource_on_units", 0))
            on_buildings = float(mech.get("resource_on_buildings", 0))
            buildings = mech.get("buildings_built") or {}
            built_total = max(1, sum(buildings.values()))
            raw_attack[net_id] = on_units / max(1.0, on_units + on_buildings)
            raw_defense[net_id] = (buildings.get("TURRET", 0) / built_total
                                   + mech.get("hold", 0) / decisions)
            raw_economy[net_id] = (on_buildings / decisions
                                   + 2.0 * (buildings.get("BOOSTER", 0)
                                            + buildings.get("ECONOMY_COLLECTOR", 0)) / built_total)
            raw_tempo[net_id] = (mech.get("produce", 0) + mech.get("multi", 0)) / decisions
        attack = _scale(raw_attack)
        defense = _scale(raw_defense)
        economy = _scale(raw_economy)
        tempo = _scale(raw_tempo)
        potential = _scale({e["net_id"]: float(winrates.get(e["net_id"], 0.5))
                            for e in registry_entries}, lo=25, hi=95)
        for entry in registry_entries:
            net_id = entry["net_id"]
            consistency = _CONSISTENCY_BASE.get(entry.get("recipe", "baseline"), 60)
            out[net_id] = {
                "attack": attack.get(net_id, 50),
                "defense": defense.get(net_id, 50),
                "economy": economy.get(net_id, 50),
                "tempo": tempo.get(net_id, 50),
                "consistency": consistency,
                "potential": potential.get(net_id, 50),
            }
        return out

    # No analysis data: deterministic recipe-based fallback with a small
    # seed-derived jitter so cards still differ.
    for entry in registry_entries:
        profile = dict(_RECIPE_FALLBACK.get(entry.get("recipe", "baseline"),
                                            _RECIPE_FALLBACK["baseline"]))
        jitter = (int(entry.get("seed", 0)) % 7) - 3
        out[entry["net_id"]] = {k: max(1, min(99, v + jitter)) for k, v in profile.items()}
    return out


def _blurb(card: dict) -> str:
    parts = []
    attrs = card["attributes"]
    best = max(attrs, key=attrs.get)
    labels = {"attack": "давление и агрессия", "defense": "оборона и турели",
              "economy": "экономика и застройка", "tempo": "темп производства",
              "consistency": "стабильность решений", "potential": "общий потолок"}
    parts.append(f"{card['archetype']}; сильная сторона — {labels.get(best, best)}.")
    if attrs["consistency"] >= 75:
        parts.append("Редко ошибается.")
    elif attrs["consistency"] <= 50:
        parts.append("Нестабилен: может выдать и шедевр, и провал.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Persistent store
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save_player_store(store: dict, store_path: Optional[Path] = None) -> None:
    path = Path(store_path) if store_path else DEFAULT_STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=1)


def ensure_player_store(
    store_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    analysis_path: Optional[Path] = None,
) -> dict:
    """Load the persistent player store, creating/extending it from the net
    registry on first use. Enforces the 1 net = 1 player invariant."""
    store_path = Path(store_path) if store_path else DEFAULT_STORE_PATH
    registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH

    registry = _load_json(registry_path)
    if not registry or not registry.get("entries"):
        raise PlayersError(f"Population registry not found or empty: {registry_path}")
    entries = registry["entries"]

    seen_nets: set[str] = set()
    for entry in entries:
        net_id = entry.get("net_id")
        if not net_id or net_id in seen_nets:
            raise PlayersError(f"Registry violates 1-net-1-player invariant at {net_id!r}")
        seen_nets.add(net_id)

    store = _load_json(store_path) or {"version": 1, "players": {}}
    players = store.setdefault("players", {})

    missing = [e for e in entries if e["net_id"] not in players]
    if missing:
        analysis = _load_json(Path(analysis_path) if analysis_path else DEFAULT_ANALYSIS_PATH)
        attributes = _derive_attributes(entries, analysis)
        for entry in missing:
            net_id = entry["net_id"]
            players[net_id] = {
                "player_id": net_id,
                "net_id": net_id,
                "nickname": NICKNAMES.get(net_id, net_id),
                "faction": entry.get("faction", "humans"),
                "recipe": entry.get("recipe", "baseline"),
                "archetype": ARCHETYPES.get(entry.get("recipe", "baseline"), "Универсал"),
                "model": f"policy:{entry['checkpoint']}",
                "genotype_hash": entry.get("genotype_hash", ""),
                "attributes": attributes[net_id],
                "rating": START_RATING,
                "money": 0,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "titles": 0,
                "tournaments": 0,
                "rating_history": [],
            }
        save_player_store(store, store_path)
    return store


def player_cards(store: dict) -> list[dict]:
    cards = []
    for player in store.get("players", {}).values():
        card = dict(player)
        card["blurb"] = _blurb(card)
        cards.append(card)
    cards.sort(key=lambda c: (-c["rating"], c["player_id"]))
    return cards


def get_player(store: dict, player_id: str) -> dict:
    player = store.get("players", {}).get(player_id)
    if player is None:
        raise PlayersError(f"Unknown player: {player_id}")
    return player


def compact_card_line(card: dict) -> str:
    """One-line card representation for token-efficient LLM prompts."""
    a = card["attributes"]
    return (f"{card['player_id']}|{card['nickname']}|{card['faction']}|{card['archetype']}"
            f"|rating:{card['rating']}|atk:{a['attack']} def:{a['defense']} eco:{a['economy']}"
            f" tmp:{a['tempo']} con:{a['consistency']} pot:{a['potential']}"
            f"|career:{card['wins']}W-{card['losses']}L,titles:{card['titles']}")
