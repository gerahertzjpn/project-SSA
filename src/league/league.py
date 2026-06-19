"""League core: scheduling, simulation, standings and persistence.

All encounters are simulated up front at creation time, so the full result is
known immediately. The UI is gated by ``current_day``: only encounters whose day
is ``<= current_day`` are revealed (standings, history, detail). Advancing the
day reveals the next encounter without re-simulating anything.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Callable, Optional

from src.core.constants import FACTION_TYPES
from src.core.rng import Spawner, resolve_seed
from src.core.runner import run_match as _default_run_match

from .templates import get_template


ROOT = Path(__file__).resolve().parents[2]
LEAGUES_ROOT = ROOT / "leagues"

# slot mapping: the "home" team always occupies TEAM_A slots, "away" TEAM_B.
HOME_SLOTS = ["A1", "A2", "A3"]
AWAY_SLOTS = ["B1", "B2", "B3"]


class LeagueError(Exception):
    """Raised for invalid league configuration or lookup failures."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug or "league"


def _validate_teams(teams: list[dict], template: dict, allowed_models: set[str]) -> list[dict]:
    team_count = int(template["team_count"])
    players_per_team = int(template["players_per_team"])
    if not isinstance(teams, list) or len(teams) != team_count:
        raise LeagueError(f"Template {template['id']} expects exactly {team_count} teams, got {len(teams)}")

    normalized: list[dict] = []
    seen_names: set[str] = set()
    for index, team in enumerate(teams):
        name = str((team or {}).get("name") or "").strip() or f"Team {index + 1}"
        key = name.lower()
        if key in seen_names:
            raise LeagueError(f"Duplicate team name: {name!r}")
        seen_names.add(key)

        players = (team or {}).get("players") or []
        if len(players) != players_per_team:
            raise LeagueError(f"Team {name!r} must have {players_per_team} players, got {len(players)}")

        norm_players = []
        for slot_index, player in enumerate(players):
            player = player or {}
            pname = str(player.get("name") or "").strip() or f"P{slot_index + 1}"
            model = str(player.get("model") or "").strip()
            faction = str(player.get("faction") or "humans").strip()
            if allowed_models and model not in allowed_models:
                raise LeagueError(f"Team {name!r} player {pname!r} has unknown model: {model!r}")
            if faction not in set(FACTION_TYPES):
                choices = ", ".join(FACTION_TYPES)
                raise LeagueError(f"Team {name!r} player {pname!r} has unknown faction {faction!r}; expected: {choices}")
            norm_players.append({"name": pname, "model": model, "faction": faction})

        normalized.append({"id": f"t{index}", "name": name, "players": norm_players})
    return normalized


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #

def _round_robin_pairings(team_ids: list[str]) -> list[tuple[str, str]]:
    """Single round-robin using the circle method. Returns (home, away) pairs."""
    ids = list(team_ids)
    if len(ids) % 2 == 1:
        ids.append(None)  # bye marker
    n = len(ids)
    rounds: list[list[tuple[str, str]]] = []
    fixed = ids[0]
    rotating = ids[1:]
    for _ in range(n - 1):
        order = [fixed] + rotating
        pairs = []
        for i in range(n // 2):
            home, away = order[i], order[n - 1 - i]
            if home is not None and away is not None:
                pairs.append((home, away))
        rounds.append(pairs)
        rotating = [rotating[-1]] + rotating[:-1]
    return [pair for round_pairs in rounds for pair in round_pairs]


def _build_schedule(team_ids: list[str], template: dict, rng: random.Random) -> list[dict]:
    pairings = _round_robin_pairings(team_ids)
    # Randomize encounter order and home/away each time, as required.
    rng.shuffle(pairings)
    schedule: list[dict] = []
    per_day = max(1, int(template.get("encounters_per_day", 1)))
    for index, (home, away) in enumerate(pairings):
        if rng.random() < 0.5:
            home, away = away, home
        day = index // per_day + 1
        schedule.append({"day": day, "home": home, "away": away, "games": [], "result": None,
                         "home_game_wins": 0, "away_game_wins": 0})
    return schedule


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def _player_stats(state, slot: str, side: str, team: dict, player_index: int) -> dict:
    player = state.players.get(slot)
    sim_team = "TEAM_A" if side == "home" else "TEAM_B"
    team_player = team["players"][player_index]
    return {
        "slot": slot,
        "side": side,
        "team_id": team["id"],
        "team_name": team["name"],
        "player_name": team_player["name"],
        "model": team_player["model"],
        "faction": team_player["faction"],
        "core_hp": round(float(getattr(player, "core_hp", 0.0)), 2) if player else 0.0,
        "alive": bool(getattr(player, "alive", False)) if player else False,
        "resources": round(float(getattr(player, "resources", 0.0)), 2) if player else 0.0,
        "core_damage_dealt": round(float(state.core_damage_by_team.get(sim_team, 0.0)), 2),
    }


def _simulate_encounter(
    encounter: dict,
    teams_by_id: dict[str, dict],
    base_seed: int,
    best_of: int,
    league_dir: Path,
    run_match_fn: Callable,
    order_provider: Optional[Callable] = None,
) -> None:
    home = teams_by_id[encounter["home"]]
    away = teams_by_id[encounter["away"]]
    needed = best_of // 2 + 1  # wins required to clinch (Bo3 -> 2)

    player_models = {}
    player_factions = {}
    for index, slot in enumerate(HOME_SLOTS):
        player_models[slot] = home["players"][index]["model"]
        player_factions[slot] = home["players"][index]["faction"]
    for index, slot in enumerate(AWAY_SLOTS):
        player_models[slot] = away["players"][index]["model"]
        player_factions[slot] = away["players"][index]["faction"]

    home_wins = away_wins = 0
    games: list[dict] = []
    game_spawner = Spawner(base_seed)  # independent per-game seeds (not base+offset)
    for game_index in range(best_of):
        if home_wins >= needed or away_wins >= needed:
            break
        seed = game_spawner.seed()
        replay_abs = league_dir / "replays" / f"d{encounter['day']:02d}_g{game_index}.json"
        replay_abs.parent.mkdir(parents=True, exist_ok=True)
        run_kwargs = dict(
            seed=seed,
            bot_type="league",
            replay_path=str(replay_abs),
            mode="replay",
            player_models=player_models,
            player_factions=player_factions,
        )
        # Optional LLM-manager game plans: each side's manager sets a per-match
        # order for its three nets (showcase only; harmless for un-ordered nets).
        # Only passed when a provider is set, so plain run_match_fn doubles keep working.
        if order_provider is not None:
            run_kwargs["orders"] = order_provider(home, away, game_index)
        result = run_match_fn(**run_kwargs)
        try:
            rel_replay = replay_abs.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel_replay = replay_abs.as_posix()
        if result.winner == "TEAM_A":
            outcome = "home"
            home_wins += 1
        elif result.winner == "TEAM_B":
            outcome = "away"
            away_wins += 1
        else:
            outcome = "draw"

        state = result.state
        players_stats = {}
        for index, slot in enumerate(HOME_SLOTS):
            players_stats[slot] = _player_stats(state, slot, "home", home, index)
        for index, slot in enumerate(AWAY_SLOTS):
            players_stats[slot] = _player_stats(state, slot, "away", away, index)

        games.append({
            "index": game_index,
            "seed": int(result.seed),
            "winner": outcome,
            "winner_type": result.winner_type,
            "ticks": int(result.total_ticks),
            "replay": rel_replay,
            "players": players_stats,
        })

    encounter["games"] = games
    encounter["home_game_wins"] = home_wins
    encounter["away_game_wins"] = away_wins
    if home_wins > away_wins:
        encounter["result"] = "home"
    elif away_wins > home_wins:
        encounter["result"] = "away"
    else:
        encounter["result"] = "draw"


# --------------------------------------------------------------------------- #
# Creation / persistence
# --------------------------------------------------------------------------- #

def create_league(
    name: str,
    template_id: str,
    teams: list[dict],
    seed: int = 0,
    allowed_models: Optional[set[str]] = None,
    run_match_fn: Optional[Callable] = None,
    leagues_root: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Create, simulate (all encounters) and persist a league. Returns its data.

    ``progress_cb(done, total)`` is invoked after each encounter is simulated so
    long creations can be driven by a background job with live progress.
    """
    template = get_template(template_id)
    run_match_fn = run_match_fn or _default_run_match
    leagues_root = Path(leagues_root) if leagues_root else LEAGUES_ROOT

    norm_teams = _validate_teams(teams, template, set(allowed_models or set()))

    seed = resolve_seed(seed if seed else None)  # fresh OS entropy when not provided (recorded)
    rng = random.Random(seed)
    enc_spawner = Spawner(seed)  # independent per-encounter seeds (not base+offset)

    league_id = f"{_slugify(name)}-{int(time.time())}"
    league_dir = leagues_root / league_id
    (league_dir / "replays").mkdir(parents=True, exist_ok=True)

    team_ids = [team["id"] for team in norm_teams]
    teams_by_id = {team["id"]: team for team in norm_teams}
    schedule = _build_schedule(team_ids, template, rng)

    best_of = int(template["best_of"])
    total = len(schedule)
    if progress_cb:
        progress_cb(0, total)
    for index, encounter in enumerate(schedule):
        _simulate_encounter(
            encounter,
            teams_by_id,
            base_seed=enc_spawner.seed(),
            best_of=best_of,
            league_dir=league_dir,
            run_match_fn=run_match_fn,
        )
        if progress_cb:
            progress_cb(index + 1, total)

    total_days = max((enc["day"] for enc in schedule), default=0)
    league = {
        "id": league_id,
        "name": str(name).strip() or "League",
        "template": template,
        "seed": seed,
        "created_at": int(time.time()),
        "current_day": 0,
        "total_days": total_days,
        "teams": norm_teams,
        "schedule": schedule,
    }
    _save_league(league, leagues_root)
    return league


def _league_path(league_id: str, leagues_root: Path) -> Path:
    return leagues_root / str(league_id) / "league.json"


def _save_league(league: dict, leagues_root: Optional[Path] = None) -> None:
    leagues_root = Path(leagues_root) if leagues_root else LEAGUES_ROOT
    path = _league_path(league["id"], leagues_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(league, handle, ensure_ascii=False, separators=(",", ":"))


def load_league(league_id: str, leagues_root: Optional[Path] = None) -> dict:
    leagues_root = Path(leagues_root) if leagues_root else LEAGUES_ROOT
    path = _league_path(league_id, leagues_root)
    if not path.exists():
        raise LeagueError(f"League not found: {league_id}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_leagues(leagues_root: Optional[Path] = None) -> list[dict]:
    leagues_root = Path(leagues_root) if leagues_root else LEAGUES_ROOT
    if not leagues_root.exists():
        return []
    summaries = []
    for child in sorted(leagues_root.iterdir(), key=lambda p: p.name, reverse=True):
        path = child / "league.json"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                league = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append({
            "id": league.get("id", child.name),
            "name": league.get("name", child.name),
            "template": (league.get("template") or {}).get("name") or league.get("template_id"),
            "template_id": (league.get("template") or {}).get("id"),
            "current_day": league.get("current_day", 0),
            "total_days": league.get("total_days", 0),
            "team_count": len(league.get("teams", [])),
            "created_at": league.get("created_at", 0),
        })
    return summaries


def advance_day(league_id: str, leagues_root: Optional[Path] = None) -> dict:
    leagues_root = Path(leagues_root) if leagues_root else LEAGUES_ROOT
    league = load_league(league_id, leagues_root)
    if league["current_day"] < league["total_days"]:
        league["current_day"] += 1
        _save_league(league, leagues_root)
    return league_public_view(league)


# --------------------------------------------------------------------------- #
# Standings + public (result-gated) views
# --------------------------------------------------------------------------- #

def _compute_standings(league: dict) -> list[dict]:
    template = league["template"]
    points_win = int(template.get("points_win", 3))
    points_draw = int(template.get("points_draw", 1))
    points_loss = int(template.get("points_loss", 0))

    table: dict[str, dict] = {}
    for team in league["teams"]:
        table[team["id"]] = {
            "team_id": team["id"], "team_name": team["name"],
            "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "game_wins": 0, "game_losses": 0, "points": 0,
        }

    current_day = league["current_day"]
    for enc in league["schedule"]:
        if enc["day"] > current_day or enc.get("result") is None:
            continue
        home = table[enc["home"]]
        away = table[enc["away"]]
        home["played"] += 1
        away["played"] += 1
        home["game_wins"] += enc["home_game_wins"]
        home["game_losses"] += enc["away_game_wins"]
        away["game_wins"] += enc["away_game_wins"]
        away["game_losses"] += enc["home_game_wins"]
        if enc["result"] == "home":
            home["won"] += 1
            away["lost"] += 1
            home["points"] += points_win
            away["points"] += points_loss
        elif enc["result"] == "away":
            away["won"] += 1
            home["lost"] += 1
            away["points"] += points_win
            home["points"] += points_loss
        else:
            home["drawn"] += 1
            away["drawn"] += 1
            home["points"] += points_draw
            away["points"] += points_draw

    standings = list(table.values())
    standings.sort(key=lambda row: (
        -row["points"],
        -(row["game_wins"] - row["game_losses"]),
        -row["game_wins"],
        row["team_name"].lower(),
    ))
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
    return standings


def _public_encounter(enc: dict, teams_by_id: dict, revealed: bool) -> dict:
    base = {
        "day": enc["day"],
        "home": enc["home"],
        "away": enc["away"],
        "home_name": teams_by_id[enc["home"]]["name"],
        "away_name": teams_by_id[enc["away"]]["name"],
        "revealed": revealed,
    }
    if revealed:
        base.update({
            "result": enc.get("result"),
            "home_game_wins": enc.get("home_game_wins", 0),
            "away_game_wins": enc.get("away_game_wins", 0),
            "games_played": len(enc.get("games", [])),
        })
    return base


def league_public_view(league: dict) -> dict:
    """League state with unrevealed encounter results hidden."""
    teams_by_id = {team["id"]: team for team in league["teams"]}
    current_day = league["current_day"]
    schedule = []
    for index, enc in enumerate(league["schedule"]):
        revealed = enc["day"] <= current_day
        item = _public_encounter(enc, teams_by_id, revealed)
        item["encounter_index"] = index
        schedule.append(item)
    return {
        "id": league["id"],
        "name": league["name"],
        "template": league["template"],
        "current_day": current_day,
        "total_days": league["total_days"],
        "teams": league["teams"],
        "standings": _compute_standings(league),
        "schedule": schedule,
        "history": [item for item in schedule if item["revealed"]],
    }


def encounter_detail(league_id: str, encounter_index: int, leagues_root: Optional[Path] = None) -> dict:
    league = load_league(league_id, leagues_root)
    schedule = league["schedule"]
    index = int(encounter_index)
    if index < 0 or index >= len(schedule):
        raise LeagueError(f"Encounter index out of range: {encounter_index}")
    enc = schedule[index]
    if enc["day"] > league["current_day"]:
        raise LeagueError("Encounter result is not revealed yet")
    teams_by_id = {team["id"]: team for team in league["teams"]}
    detail = _public_encounter(enc, teams_by_id, revealed=True)
    detail["encounter_index"] = index
    detail["home_team"] = teams_by_id[enc["home"]]
    detail["away_team"] = teams_by_id[enc["away"]]
    detail["games"] = enc.get("games", [])
    return detail
