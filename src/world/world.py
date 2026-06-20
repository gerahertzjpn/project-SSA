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

from src.core import rng as seeds

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


POSITIONS = ("A1", "A2", "A3")


def _synth_position_affinity(rng: random.Random) -> dict:
    """Per-position skill multiplier (latent): each player is clearly better at one
    position. Drives strength when fielded there; the OBSERVED per-position winrate
    accumulates separately in ``position_stats`` (data the manager/LLM scouts)."""
    aff = {p: round(rng.uniform(0.86, 1.0), 3) for p in POSITIONS}
    aff[rng.choice(POSITIONS)] = round(rng.uniform(1.06, 1.18), 3)
    return aff


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
    faction: Optional[str] = None,
    archetype: Optional[str] = None,
    role: str = "pro",
    signable: bool = True,
    card: Optional[dict] = None,
) -> dict:
    pid = player_id(world_id, index)
    # Real card (derived from the net's measured play, scripts/derive_player_cards.py) when the
    # player is net-bound; else synthesise from tier (foundation / nets without a card yet).
    attributes = ({k: int(card[k]) for k in ATTR_KEYS if k in card}
                  if card else _synth_attributes(rng, tier))
    return {
        "player_id": pid,
        "net_id": net_id,                       # shared-pool link; None in foundation
        # role/signable: "pro" = league player (drafted, signable); "streamer" = permanent
        # free agent bound to an ancestor net, plays SoloQ only and is NEVER signable.
        "role": role,
        "signable": signable,
        "nickname": nickname,
        "faction": faction or rng.choice(FACTIONS),
        "archetype": archetype or rng.choice(_ARCHETYPES),
        "attributes": attributes,
        # latent per-position skill multiplier (A1/A2/A3) — fielding a player at a
        # good position boosts strength; the LLM/manager picks positions + subs.
        "position_affinity": _synth_position_affinity(rng),
        # OBSERVED per-position record, filled as official matches are played:
        # {pos: {"games": n, "wins": w}}. Different players -> different position winrate.
        "position_stats": {},
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
        # The manager-LLM's contingency PLAYBOOK for the current tournament
        # (base strategy + IF-THEN overrides incl. substitution rules). None until set.
        "tournament_plan": None,
    }


# --------------------------------------------------------------------------- #
# New world
# --------------------------------------------------------------------------- #
def seed_population_players(world_id: str, manifest: dict, rng: random.Random):
    """Build the player pool FROM the frozen population manifest. Returns
    ``(players, pro_ids, streamer_ids)``:

    - DESCENDANTS -> 'pro' players: signable, draftable to league teams, bound to their net,
      with the net's real faction + archetype. (soloq_rating starts equal; real SoloQ matches
      establish it, so the synth attributes are cosmetic/scouting flavour.)
    - ANCESTORS -> 'streamer' players: permanent free agents bound to an ancestor net, SoloQ
      only, NEVER signable -> kept OUT of the draftable free-agent pool.
    """
    descendants = manifest.get("descendants", [])
    ancestors = manifest.get("ancestors", [])
    cards = _load_player_cards()                  # net_id -> {attack, defense, ...} or {}
    nicks = _unique_nicknames(len(descendants) + len(ancestors), rng)
    players: dict[str, dict] = {}
    pro_ids: list[str] = []
    streamer_ids: list[str] = []
    i = 0
    for e in descendants:
        tier = (i + 0.5) / max(1, len(descendants))  # cosmetic spread; real strength via matches
        p = make_player(world_id, i, nicks[i], rng, tier=tier, net_id=e["net_id"],
                        faction=e["faction"], archetype=e.get("archetype"),
                        role="pro", signable=True, card=cards.get(e["net_id"]))
        players[p["player_id"]] = p
        pro_ids.append(p["player_id"])
        i += 1
    for e in ancestors:
        p = make_player(world_id, i, nicks[i], rng, tier=0.6, net_id=e["net_id"],
                        faction=e["faction"], archetype=e.get("archetype") or "base",
                        origin="streamer", role="streamer", signable=False,
                        card=cards.get(e["net_id"]))
        players[p["player_id"]] = p
        streamer_ids.append(p["player_id"])
        i += 1
    return players, pro_ids, streamer_ids


_PLAYER_CARDS_PATH = ROOT / "population" / "player_cards_v1.json"
_PLAYER_CARDS_CACHE: Optional[dict] = None


def _load_player_cards() -> dict:
    """net_id -> 6-attribute card derived from the net's measured play; {} if absent."""
    global _PLAYER_CARDS_CACHE
    if _PLAYER_CARDS_CACHE is None:
        try:
            _PLAYER_CARDS_CACHE = json.loads(_PLAYER_CARDS_PATH.read_text(encoding="utf-8")).get("cards", {})
        except (OSError, ValueError):
            _PLAYER_CARDS_CACHE = {}
    return _PLAYER_CARDS_CACHE


def new_world(
    world_id: str,
    franchises: list[dict],
    *,
    name: Optional[str] = None,
    player_pool_size: Optional[int] = None,
    seed: Optional[int] = None,
    config: Optional[dict] = None,
    population: Optional[str | dict] = None,
) -> dict:
    """Create a clean-slate world: a diverse player pool (all free agents) and
    one team per franchise (no league assignment, no money — Season 0 gives
    neither). Rosters are filled later by the balanced draft (see season.py).

    ``player_pool_size`` defaults to enough for 3 per team + a free-agent buffer.
    ``seed`` is OPTIONAL: ``None`` (production default) draws a fresh OS-entropy
    seed so every world is unique; the resolved seed is RECORDED in ``world['seed']``
    for history. Tests pass an explicit seed for determinism (WORLD_MODE_SPEC §0-ter).
    """
    if len(franchises) < 2:
        raise WorldError("need at least 2 franchises")
    resolved_seed = seeds.resolve_seed(seed)
    rng = random.Random(resolved_seed)

    teams = {t["team_id"]: t for t in (make_team(world_id, fr) for fr in franchises)}
    n_teams = len(teams)
    cfg = dict(config or {})
    streamers: list[str] = []

    if population is not None:
        # Seed the pool from the frozen trained population: 90 pros (draftable) + 18
        # streamers (SoloQ-only free agents). Loaded via plain json so world.py stays
        # torch-free; the net-backed match_fn (src.world.net_match) is imported only when
        # matches actually run.
        manifest = population if isinstance(population, dict) else json.loads(
            Path(population).read_text(encoding="utf-8"))
        players, pro_ids, streamers = seed_population_players(world_id, manifest, rng)
        free_agents = list(pro_ids)
        cfg.setdefault("population_source", manifest.get("source_run") or str(population))
        cfg.setdefault("population_manifest",
                       population if isinstance(population, str) else "population/frozen_population_v1.json")
    else:
        pool = player_pool_size if player_pool_size is not None else n_teams * 3 + max(12, n_teams)
        nicks = _unique_nicknames(pool, rng)
        players = {}
        for i in range(pool):
            tier = (i + 0.5) / pool  # even spread of strength across the pool
            # shuffle tier->index so ids aren't sorted by strength
            p = make_player(world_id, i, nicks[i], rng, tier=tier)
            players[p["player_id"]] = p
        free_agents = list(players.keys())

    return {
        "version": STORE_VERSION,
        "world_id": world_id,
        "name": name or world_id,
        "seed": resolved_seed,       # recorded for history (unique per run; see §0-ter)
        "season": 0,
        "month": 0,
        "phase": "created",          # created | season0 | regular | offseason
        "config": cfg,
        "teams": teams,
        "players": players,
        "free_agents": free_agents,
        "streamers": streamers,      # ancestor-bound free agents: SoloQ only, never signable
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
