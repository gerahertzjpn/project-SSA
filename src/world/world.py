"""World data model + persistence + new-world creation.

A *world* is a clean-slate, persistent ecosystem. Players, teams and budgets
are created from scratch; the ONE shared resource across worlds is the trained
net pool (a net may reappear in another world under a different player name) —
so a player card carries an *optional* ``net_id`` link. In this foundation the
cards are synthesized (diverse, seeded) so the whole system runs with zero
trained nets; binding real nets + deriving attributes happens later.

Schemas are plain JSON-serializable dicts (matching the regionals store style).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_ROOT = ROOT / "data" / "world"

STORE_VERSION = 1
START_RATING = 1000.0
START_ELO = 1000.0
ATTR_KEYS = ("attack", "defense", "economy", "tempo", "consistency", "potential")

FACTIONS = ("humans", "infected", "robots")

# Name generation — world-scoped, diverse, deterministic by seed.
_NAME_PREFIX = (
    "Ash", "Bly", "Cor", "Dra", "Eko", "Fen", "Gax", "Hel", "Ira", "Jot",
    "Kor", "Lyx", "Mir", "Nyx", "Orb", "Pyr", "Quo", "Rho", "Syl", "Tau",
    "Umb", "Vex", "Wol", "Xan", "Ymir", "Zed",
)
_NAME_SUFFIX = (
    "an", "ix", "or", "us", "ai", "en", "ra", "os", "yn", "el", "ar", "io",
    "ux", "is", "ov", "ek", "ym", "ad", "ul", "ez",
)
_ARCHETYPES = (
    "Carry", "Anchor", "Tempo", "Macro", "Roamer", "Flex", "Pusher",
    "Defender", "Tech", "Wildcard",
)


class WorldError(Exception):
    pass


# --------------------------------------------------------------------------- #
# ID helpers
# --------------------------------------------------------------------------- #
def player_id(world_id: str, index: int) -> str:
    return f"{world_id}-p{index:03d}"


def team_id(world_id: str, franchise_id: str) -> str:
    return f"{world_id}-t-{franchise_id}"


# --------------------------------------------------------------------------- #
# Player / team factories
# --------------------------------------------------------------------------- #
def _unique_nicknames(n: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        name = rng.choice(_NAME_PREFIX) + rng.choice(_NAME_SUFFIX)
        if name not in seen:
            seen.add(name)
            out.append(name)
    # Fallback for collisions / tiny syllable space.
    i = 0
    while len(out) < n:
        cand = f"Prospect{i:02d}"
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
        i += 1
    return out


def _synth_attributes(rng: random.Random, tier: float) -> dict:
    """Diverse attribute card. ``tier`` in [0,1] shifts overall strength so the
    pool spans weak prospects to stars (drives SoloQ/draft/salary spread)."""
    base = 30 + tier * 45  # 30..75 center
    attrs = {}
    for key in ATTR_KEYS:
        val = base + rng.uniform(-14, 14)
        attrs[key] = int(max(15, min(98, round(val))))
    # Potential leans higher for low-tier (upside) — keeps young prospects valuable.
    attrs["potential"] = int(max(15, min(99, round(
        attrs["potential"] + (1.0 - tier) * 12))))
    return attrs


def make_player(
    world_id: str,
    index: int,
    nickname: str,
    rng: random.Random,
    *,
    tier: float,
    origin: str = "founder",
    parent_id: Optional[str] = None,
    generation: int = 0,
    net_id: Optional[str] = None,
) -> dict:
    pid = player_id(world_id, index)
    return {
        "player_id": pid,
        "net_id": net_id,                       # shared-pool link; None in foundation
        "nickname": nickname,
        "faction": rng.choice(FACTIONS),
        "archetype": rng.choice(_ARCHETYPES),
        "attributes": _synth_attributes(rng, tier),
        "soloq_rating": START_RATING,
        "soloq_history": [],
        "salary": 0.0,                          # set after first played season
        "salary_history": [],
        "status": "free_agent",                 # free_agent | rostered | rookie
        "team_id": None,
        "lineage": {"parent_id": parent_id, "origin": origin, "generation": int(generation)},
        "history": [],                          # [{season, team_id, role, result}]
        "trophies": [],                         # [{season, title}]
        "games": 0,
        "official_games": 0,
        "seasons_played": 0,
    }


def make_team(world_id: str, franchise: dict) -> dict:
    fid = franchise["franchise_id"]
    return {
        "team_id": team_id(world_id, fid),
        "franchise_id": fid,
        "name": franchise.get("name", fid),
        "abbrev": franchise.get("abbrev", fid[:3].upper()),
        "manager": franchise.get("manager", ""),
        "league": None,                         # "upper" | "lower" | None
        "group": None,                          # "A" | "Z" | None
        "treasury": 0,
        "elo": START_ELO,
        "elo_history": [],
        "roster": [],                           # [player_id], >= 3 in season
        "titles": 0,
        "trophies": [],                         # [{season, title}]
        "transfer_requests_used": 0,            # per transfer window (reset each window)
        "training_certificates": [],            # bankable gradeslot certs [{cert_id, steps, season}]
    }


# --------------------------------------------------------------------------- #
# New world
# --------------------------------------------------------------------------- #
def new_world(
    world_id: str,
    franchises: list[dict],
    *,
    name: Optional[str] = None,
    player_pool_size: Optional[int] = None,
    seed: int = 0,
    config: Optional[dict] = None,
) -> dict:
    """Create a clean-slate world: a diverse player pool (all free agents) and
    one team per franchise (no league assignment, no money — Season 0 gives
    neither). Rosters are filled later by the balanced draft (see season.py).

    ``player_pool_size`` defaults to enough for 3 per team + a free-agent buffer.
    """
    if len(franchises) < 2:
        raise WorldError("need at least 2 franchises")
    rng = random.Random(seed)

    teams = {t["team_id"]: t for t in (make_team(world_id, fr) for fr in franchises)}
    n_teams = len(teams)
    pool = player_pool_size if player_pool_size is not None else n_teams * 3 + max(12, n_teams)

    nicks = _unique_nicknames(pool, rng)
    players: dict[str, dict] = {}
    for i in range(pool):
        tier = (i + 0.5) / pool  # even spread of strength across the pool
        # shuffle tier->index so ids aren't sorted by strength
        p = make_player(world_id, i, nicks[i], rng, tier=tier)
        players[p["player_id"]] = p

    return {
        "version": STORE_VERSION,
        "world_id": world_id,
        "name": name or world_id,
        "season": 0,
        "month": 0,
        "phase": "created",          # created | season0 | regular | offseason
        "config": dict(config or {}),
        "teams": teams,
        "players": players,
        "free_agents": list(players.keys()),
        "rookies_pool": [],
        "calendar": [],
        "history": [],
    }


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def get_team(world: dict, tid: str) -> dict:
    t = world["teams"].get(tid)
    if t is None:
        raise WorldError(f"unknown team {tid!r}")
    return t


def get_player(world: dict, pid: str) -> dict:
    p = world["players"].get(pid)
    if p is None:
        raise WorldError(f"unknown player {pid!r}")
    return p


def teams_in_league(world: dict, league: str) -> list[dict]:
    return [t for t in world["teams"].values() if t["league"] == league]


def upper_teams(world: dict) -> list[dict]:
    return teams_in_league(world, "upper")


def lower_teams(world: dict) -> list[dict]:
    return teams_in_league(world, "lower")


def roster_players(world: dict, team: dict) -> list[dict]:
    return [get_player(world, pid) for pid in team["roster"]]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def world_path(world_id: str, world_root: Optional[Path] = None) -> Path:
    root = Path(world_root) if world_root else DEFAULT_WORLD_ROOT
    return root / world_id / "world.json"


def save_world(world: dict, world_root: Optional[Path] = None) -> Path:
    path = world_path(world["world_id"], world_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(world, fh, ensure_ascii=False, indent=1)
    return path


def load_world(world_id: str, world_root: Optional[Path] = None) -> dict:
    path = world_path(world_id, world_root)
    if not path.exists():
        raise WorldError(f"world not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_worlds(world_root: Optional[Path] = None) -> list[str]:
    root = Path(world_root) if world_root else DEFAULT_WORLD_ROOT
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "world.json").exists())
