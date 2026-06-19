"""Regionals core: draft -> league -> playoff, Elo, prizes, persistence.

Everything is simulated up-front at creation time (league round-robin, then
the step-ladder playoff seeded by the final league standings). The UI is
gated by ``current_day``: encounters reveal one day at a time, the playoff
bracket reveals stage by stage, and in-tournament ratings are recomputed
"as of the revealed day" from stored per-encounter deltas.

Persistent side effects of a finished simulation (player Elo/career, franchise
money/titles/head-to-head) are applied to the registries immediately at
creation; the day gate is purely a presentation device.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Callable, Optional

from src.core.rng import Spawner, resolve_seed

from src.core.runner import run_match as _default_run_match
from src.league.league import (
    AWAY_SLOTS,
    HOME_SLOTS,
    _build_schedule,
    _compute_standings,
    _simulate_encounter,
)

from . import commentary as commentary_mod
from .draft import run_draft
from .elo import K_FACTOR, expected_score, rating_delta, team_average
from .franchises import (
    ensure_franchise_store,
    franchise_specs,
    record_head_to_head,
    save_franchise_store,
)
from .players import (
    ensure_player_store,
    player_cards,
    save_player_store,
)
from .templates import get_template

ROOT = Path(__file__).resolve().parents[2]
REGIONALS_ROOT = ROOT / "regionals"

STAGE_LABELS = {"league": "Лига", "semifinal": "Полуфинал", "final": "Гранд-финал"}


class RegionalsError(Exception):
    pass


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug or "regionals"


# --------------------------------------------------------------------------- #
# Elo bookkeeping per encounter
# --------------------------------------------------------------------------- #

def _game_impacts(game: dict, side: str) -> dict[str, float]:
    """Impact 0..1 per slot of one side, from per-player survival/HP/resources."""
    slots = HOME_SLOTS if side == "home" else AWAY_SLOTS
    scores = {}
    for slot in slots:
        stats = game["players"][slot]
        scores[slot] = (float(stats.get("core_hp", 0.0))
                        + (40.0 if stats.get("alive") else 0.0)
                        + float(stats.get("resources", 0.0)))
    total = sum(scores.values())
    if total <= 0:
        return {slot: 0.5 for slot in slots}
    n = len(slots)
    return {slot: max(0.0, min(1.0, 0.5 * scores[slot] * n / total)) for slot in slots}


def _apply_elo_for_encounter(enc: dict, teams_by_id: dict, ratings: dict[str, float]) -> None:
    """Per-game Elo updates; writes enc["rating_changes"] (per player) in place."""
    home = teams_by_id[enc["home"]]
    away = teams_by_id[enc["away"]]
    slot_player = {}
    for index, slot in enumerate(HOME_SLOTS):
        slot_player[slot] = home["players"][index]["player_id"]
    for index, slot in enumerate(AWAY_SLOTS):
        slot_player[slot] = away["players"][index]["player_id"]

    changes: dict[str, dict] = {
        pid: {"before": round(ratings[pid], 1), "delta": 0.0, "games": []}
        for pid in slot_player.values()
    }
    for game in enc.get("games", []):
        home_avg = team_average([ratings[slot_player[s]] for s in HOME_SLOTS])
        away_avg = team_average([ratings[slot_player[s]] for s in AWAY_SLOTS])
        impacts = {**_game_impacts(game, "home"), **_game_impacts(game, "away")}
        for slot, pid in slot_player.items():
            is_home = slot in HOME_SLOTS
            opp_avg = away_avg if is_home else home_avg
            is_draw = game["winner"] == "draw"
            if is_draw:
                # plain Elo draw: score 0.5, no impact modulation
                delta = K_FACTOR * (0.5 - expected_score(ratings[pid], opp_avg))
            else:
                won = (game["winner"] == "home") == is_home
                delta = rating_delta(ratings[pid], opp_avg, won, impacts[slot])
            ratings[pid] += delta
            changes[pid]["delta"] += delta
            changes[pid]["games"].append({
                "game": game["index"],
                "delta": round(delta, 1),
                "impact": round(impacts[slot], 2),
                "won": not is_draw and (game["winner"] == "home") == is_home,
                "draw": is_draw,
            })
    for pid, change in changes.items():
        change["delta"] = round(change["delta"], 1)
        change["after"] = round(ratings[pid], 1)
    enc["rating_changes"] = changes

    # top performers (for the commentator), by best summed impact on winning side
    performers = []
    for slot, pid in slot_player.items():
        impacts = [g["impact"] for g in changes[pid]["games"]]
        wins = sum(1 for g in changes[pid]["games"] if g["won"])
        if impacts:
            performers.append({
                "player_id": pid,
                "name": next(p["name"] for t in (home, away) for p in t["players"]
                             if p["player_id"] == pid),
                "impact": sum(impacts) / len(impacts),
                "game_wins": wins,
            })
    performers.sort(key=lambda p: (-p["game_wins"], -p["impact"]))
    enc["top_performers"] = [
        {"player_id": p["player_id"], "name": p["name"], "impact": round(p["impact"], 2)}
        for p in performers[:3]
    ]


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #

def create_regionals(
    name: str,
    template_id: str,
    seed: int = 0,
    llm: Optional[object] = None,
    run_match_fn: Optional[Callable] = None,
    regionals_root: Optional[Path] = None,
    players_store_path: Optional[Path] = None,
    franchises_store_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    analysis_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Draft, simulate the full tournament, persist it and update registries."""
    template = get_template(template_id)
    run_match_fn = run_match_fn or _default_run_match
    regionals_root = Path(regionals_root) if regionals_root else REGIONALS_ROOT

    player_store = ensure_player_store(players_store_path, registry_path, analysis_path)
    franchise_store = ensure_franchise_store(franchises_store_path)
    cards = player_cards(player_store)

    seed = resolve_seed(seed if seed else None)  # fresh OS entropy when not provided (recorded)
    rng = random.Random(seed)
    enc_spawner = Spawner(seed)  # independent per-encounter seeds (not base+offset)
    # Draw this tournament's teams from the franchise pool (25). team_count<pool
    # → a varying lineup each tournament; otherwise use the whole pool.
    pool = franchise_specs()
    team_count = int(template.get("team_count", len(pool)))
    specs = rng.sample(pool, team_count) if 0 < team_count < len(pool) else pool

    def report(done: int, total: int, phase: str) -> None:
        if progress_cb:
            progress_cb(done, total, phase)

    # ---- 1. Draft -------------------------------------------------------- #
    report(0, 1, "draft")
    rosters, draft_log = run_draft(template, specs, cards, rng, llm)
    teams = []
    for spec in specs:
        roster = rosters[spec["franchise_id"]]
        teams.append({
            "id": spec["franchise_id"],
            "name": spec["name"],
            "manager": spec["manager"],
            "players": [
                {
                    "name": card["nickname"],
                    "player_id": card["player_id"],
                    "model": card["model"],
                    "faction": card["faction"],
                    "archetype": card["archetype"],
                    "attributes": card["attributes"],
                    "rating_at_draft": card["rating"],
                }
                for card in roster
            ],
        })
    teams_by_id = {team["id"]: team for team in teams}

    regionals_id = f"{_slugify(name)}-{int(time.time())}"
    regionals_dir = regionals_root / regionals_id
    (regionals_dir / "replays").mkdir(parents=True, exist_ok=True)

    # ---- 2. League stage -------------------------------------------------- #
    league_rules = dict(template["league"])
    schedule = _build_schedule([t["id"] for t in teams], league_rules, rng)
    for enc in schedule:
        enc["stage"] = "league"
    league_days = max((enc["day"] for enc in schedule), default=0)
    league_best_of = int(league_rules["best_of"])
    playoff_best_of = int(template["playoff"]["best_of"])
    total_encounters = len(schedule) + 2

    for index, enc in enumerate(schedule):
        _simulate_encounter(enc, teams_by_id, base_seed=enc_spawner.seed(),
                            best_of=league_best_of, league_dir=regionals_dir,
                            run_match_fn=run_match_fn)
        report(index + 1, total_encounters, "league")

    league_table = _compute_standings({
        "template": league_rules,
        "teams": teams,
        "current_day": league_days,
        "schedule": schedule,
    })
    seed_order = [row["team_id"] for row in league_table]  # rank 1 first

    # ---- 3. Playoff (step-ladder top-3, drawn series go to higher seed) --- #
    def playoff_encounter(day: int, stage: str, home_id: str, away_id: str, index: int) -> dict:
        enc = {"day": day, "home": home_id, "away": away_id, "games": [], "result": None,
               "home_game_wins": 0, "away_game_wins": 0, "stage": stage}
        _simulate_encounter(enc, teams_by_id, base_seed=enc_spawner.seed(),
                            best_of=playoff_best_of, league_dir=regionals_dir,
                            run_match_fn=run_match_fn)
        return enc

    semifinal = playoff_encounter(league_days + 1, "semifinal",
                                  seed_order[1], seed_order[2], len(schedule))
    report(len(schedule) + 1, total_encounters, "playoff")
    # drawn series: the higher league seed advances
    sf_winner = semifinal["home"] if semifinal["result"] in ("home", "draw") else semifinal["away"]

    final = playoff_encounter(league_days + 2, "final",
                              seed_order[0], sf_winner, len(schedule) + 1)
    report(len(schedule) + 2, total_encounters, "playoff")
    champion_id = final["home"] if final["result"] in ("home", "draw") else final["away"]

    schedule.extend([semifinal, final])
    total_days = league_days + 2

    # ---- 4. Elo over all encounters in chronological order ---------------- #
    drafted_ids = [p["player_id"] for t in teams for p in t["players"]]
    ratings = {pid: float(player_store["players"][pid]["rating"]) for pid in drafted_ids}
    initial_ratings = {pid: round(r, 1) for pid, r in ratings.items()}
    for enc in schedule:
        _apply_elo_for_encounter(enc, teams_by_id, ratings)

    # ---- 5. Final placement + prize money ---------------------------------- #
    runner_up = final["away"] if champion_id == final["home"] else final["home"]
    third = semifinal["away"] if sf_winner == semifinal["home"] else semifinal["home"]
    placement_ids = [champion_id, runner_up, third]
    placement_ids.extend(tid for tid in seed_order if tid not in placement_ids)

    prizes = list(template["prizes"])
    champion_bonus = int(template["champion_bonus"])
    players_per_team = int(template["players_per_team"])
    placements = []
    for place_index, team_id in enumerate(placement_ids):
        prize = int(prizes[place_index]) if place_index < len(prizes) else 0
        bonus = champion_bonus if place_index == 0 else 0
        placements.append({
            "place": place_index + 1,
            "team_id": team_id,
            "team_name": teams_by_id[team_id]["name"],
            "prize": prize,
            "champion_bonus": bonus,
        })

    # ---- 6. Apply persistent effects -------------------------------------- #
    for placement in placements:
        franchise = franchise_store["franchises"][placement["team_id"]]
        franchise["money"] += placement["prize"] + placement["champion_bonus"]
        franchise["seasons"] += 1
        if placement["place"] == 1:
            franchise["titles"] += 1
        player_share = (placement["prize"] + placement["champion_bonus"]) // players_per_team
        for player in teams_by_id[placement["team_id"]]["players"]:
            entry = player_store["players"][player["player_id"]]
            entry["money"] += player_share
            entry["tournaments"] += 1
            if placement["place"] == 1:
                entry["titles"] += 1

    for enc in schedule:
        record_head_to_head(franchise_store, enc["home"], enc["away"], enc["result"])
        for pid, change in enc["rating_changes"].items():
            entry = player_store["players"][pid]
            for game in change["games"]:
                entry["games"] += 1
                if game["won"]:
                    entry["wins"] += 1
                elif not game.get("draw"):
                    entry["losses"] += 1

    for pid in drafted_ids:
        entry = player_store["players"][pid]
        entry["rating"] = round(ratings[pid], 1)
        entry["rating_history"].append({
            "tournament": regionals_id,
            "before": initial_ratings[pid],
            "after": round(ratings[pid], 1),
        })

    save_player_store(player_store, players_store_path)
    save_franchise_store(franchise_store, franchises_store_path)

    # ---- 7. Opening commentary -------------------------------------------- #
    commentary_list: list[dict] = []
    memory: list[dict] = []
    if llm is not None:
        try:
            text = llm.say(commentary_mod.system_prompt(memory),
                           commentary_mod.opening_prompt(name, teams, draft_log),
                           max_tokens=500)
        except Exception:  # noqa: BLE001
            text = commentary_mod.fallback_text("opening")
    else:
        text = commentary_mod.fallback_text("opening")
    entry = {"kind": "opening", "day": 0, "text": text}
    commentary_list.append(entry)
    memory.append(entry)

    llm_usage = dict(getattr(llm, "usage", {})) if llm is not None else {}

    data = {
        "id": regionals_id,
        "name": str(name).strip() or "Regionals",
        "template": template,
        "seed": seed,
        "created_at": int(time.time()),
        "current_day": 0,
        "total_days": total_days,
        "league_days": league_days,
        "teams": teams,
        "draft": draft_log,
        "schedule": schedule,
        "initial_ratings": initial_ratings,
        "final_ratings": {pid: round(r, 1) for pid, r in ratings.items()},
        "placements": placements,
        "champion": {"team_id": champion_id, "team_name": teams_by_id[champion_id]["name"]},
        "commentary": commentary_list,
        "commentator_memory": memory,
        "llm": {"used": llm is not None, "usage": llm_usage},
    }
    _save_regionals(data, regionals_root)
    return data


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def _regionals_path(regionals_id: str, regionals_root: Path) -> Path:
    return regionals_root / str(regionals_id) / "regionals.json"


def _save_regionals(data: dict, regionals_root: Optional[Path] = None) -> None:
    regionals_root = Path(regionals_root) if regionals_root else REGIONALS_ROOT
    path = _regionals_path(data["id"], regionals_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))


def load_regionals(regionals_id: str, regionals_root: Optional[Path] = None) -> dict:
    regionals_root = Path(regionals_root) if regionals_root else REGIONALS_ROOT
    path = _regionals_path(regionals_id, regionals_root)
    if not path.exists():
        raise RegionalsError(f"Regionals not found: {regionals_id}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_regionals(regionals_root: Optional[Path] = None) -> list[dict]:
    regionals_root = Path(regionals_root) if regionals_root else REGIONALS_ROOT
    if not regionals_root.exists():
        return []
    summaries = []
    for child in sorted(regionals_root.iterdir(), key=lambda p: p.name, reverse=True):
        path = child / "regionals.json"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append({
            "id": data.get("id", child.name),
            "name": data.get("name", child.name),
            "template_id": (data.get("template") or {}).get("id"),
            "template": (data.get("template") or {}).get("name"),
            "current_day": data.get("current_day", 0),
            "total_days": data.get("total_days", 0),
            "team_count": len(data.get("teams", [])),
            "created_at": data.get("created_at", 0),
            "llm": (data.get("llm") or {}).get("used", False),
        })
    return summaries


def advance_day(regionals_id: str, regionals_root: Optional[Path] = None) -> dict:
    regionals_root = Path(regionals_root) if regionals_root else REGIONALS_ROOT
    data = load_regionals(regionals_id, regionals_root)
    if data["current_day"] < data["total_days"]:
        data["current_day"] += 1
        _save_regionals(data, regionals_root)
    return regionals_public_view(data)


# --------------------------------------------------------------------------- #
# Public (day-gated) views
# --------------------------------------------------------------------------- #

def _public_encounter(enc: dict, index: int, teams_by_id: dict, revealed: bool) -> dict:
    base = {
        "encounter_index": index,
        "day": enc["day"],
        "stage": enc.get("stage", "league"),
        "stage_label": STAGE_LABELS.get(enc.get("stage", "league"), enc.get("stage")),
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


def _ratings_as_of_day(data: dict, day: int) -> dict[str, float]:
    ratings = {pid: float(value) for pid, value in data["initial_ratings"].items()}
    for enc in data["schedule"]:
        if enc["day"] > day:
            continue
        for pid, change in enc.get("rating_changes", {}).items():
            ratings[pid] += float(change["delta"])
    return {pid: round(value, 1) for pid, value in ratings.items()}


def regionals_public_view(data: dict) -> dict:
    teams_by_id = {team["id"]: team for team in data["teams"]}
    current_day = data["current_day"]
    league_days = data["league_days"]

    league_finished = current_day >= league_days
    semifinal_revealed = any(
        enc.get("stage") == "semifinal" and enc["day"] <= current_day
        for enc in data["schedule"])

    schedule_public = []
    for index, enc in enumerate(data["schedule"]):
        revealed = enc["day"] <= current_day
        item = _public_encounter(enc, index, teams_by_id, revealed)
        # playoff matchups are spoilers: mask them until the previous stage
        # is fully revealed (the league decides the SF, the SF decides the final)
        if item["stage"] == "semifinal" and not league_finished:
            item["home"] = item["away"] = None
            item["home_name"] = item["away_name"] = "?"
        if item["stage"] == "final":
            if not league_finished:
                item["home"] = item["away"] = None
                item["home_name"] = item["away_name"] = "?"
            elif not semifinal_revealed:
                item["away"] = None
                item["away_name"] = "?"
        schedule_public.append(item)

    league_encounters = [enc for enc in data["schedule"] if enc.get("stage") == "league"]
    standings = _compute_standings({
        "template": data["template"]["league"],
        "teams": data["teams"],
        "current_day": current_day,
        "schedule": league_encounters,
    })

    bracket = [item for item in schedule_public if item["stage"] != "league"]

    finished = current_day >= data["total_days"]
    ratings_now = _ratings_as_of_day(data, current_day)
    players_view = []
    for team in data["teams"]:
        for player in team["players"]:
            players_view.append({
                "player_id": player["player_id"],
                "name": player["name"],
                "faction": player["faction"],
                "archetype": player["archetype"],
                "team_id": team["id"],
                "team_name": team["name"],
                "attributes": player["attributes"],
                "rating_start": data["initial_ratings"][player["player_id"]],
                "rating_now": ratings_now[player["player_id"]],
            })
    players_view.sort(key=lambda p: -p["rating_now"])

    view = {
        "id": data["id"],
        "name": data["name"],
        "template": data["template"],
        "current_day": current_day,
        "total_days": data["total_days"],
        "league_days": league_days,
        "league_finished": league_finished,
        "finished": finished,
        "teams": data["teams"],
        "draft": data["draft"],
        "standings": standings,
        "schedule": schedule_public,
        "history": [item for item in schedule_public if item["revealed"]],
        "bracket": bracket,
        "players": players_view,
        "commentary": data.get("commentary", []),
        "llm": data.get("llm", {}),
    }
    if finished:
        view["placements"] = data["placements"]
        view["champion"] = data["champion"]
    return view


def encounter_detail(regionals_id: str, encounter_index: int,
                     regionals_root: Optional[Path] = None) -> dict:
    data = load_regionals(regionals_id, regionals_root)
    schedule = data["schedule"]
    index = int(encounter_index)
    if index < 0 or index >= len(schedule):
        raise RegionalsError(f"Encounter index out of range: {encounter_index}")
    enc = schedule[index]
    if enc["day"] > data["current_day"]:
        raise RegionalsError("Encounter result is not revealed yet")
    teams_by_id = {team["id"]: team for team in data["teams"]}
    detail = _public_encounter(enc, index, teams_by_id, revealed=True)
    detail["home_team"] = teams_by_id[enc["home"]]
    detail["away_team"] = teams_by_id[enc["away"]]
    detail["games"] = enc.get("games", [])
    detail["rating_changes"] = enc.get("rating_changes", {})
    detail["top_performers"] = enc.get("top_performers", [])
    detail["commentary"] = next(
        (c for c in data.get("commentary", [])
         if c.get("kind") == "match" and c.get("encounter_index") == index),
        None)
    return detail


# --------------------------------------------------------------------------- #
# Button-triggered commentary
# --------------------------------------------------------------------------- #

def add_commentary(
    regionals_id: str,
    target: dict,
    llm: Optional[object] = None,
    regionals_root: Optional[Path] = None,
) -> dict:
    """Generate (or return cached) commentary for an encounter or the finale.

    ``target`` is ``{"type": "encounter", "index": N}`` or ``{"type": "final"}``.
    """
    regionals_root = Path(regionals_root) if regionals_root else REGIONALS_ROOT
    data = load_regionals(regionals_id, regionals_root)
    commentary_list = data.setdefault("commentary", [])
    memory = data.setdefault("commentator_memory", [])

    target_type = str(target.get("type", ""))
    if target_type == "encounter":
        index = int(target.get("index", -1))
        if index < 0 or index >= len(data["schedule"]):
            raise RegionalsError(f"Encounter index out of range: {index}")
        enc = data["schedule"][index]
        if enc["day"] > data["current_day"]:
            raise RegionalsError("Encounter result is not revealed yet")
        cached = next((c for c in commentary_list
                       if c.get("kind") == "match" and c.get("encounter_index") == index), None)
        if cached:
            return cached
        teams_by_id = {team["id"]: team for team in data["teams"]}
        public = _public_encounter(enc, index, teams_by_id, revealed=True)
        public["top_performers"] = enc.get("top_performers", [])
        prompt = commentary_mod.encounter_prompt(
            public | {"games": enc.get("games", []), "top_performers": enc.get("top_performers", [])},
            STAGE_LABELS.get(enc.get("stage", "league"), "Лига"))
        kind, day = "match", enc["day"]
        extra = {"encounter_index": index}
    elif target_type == "final":
        if data["current_day"] < data["total_days"]:
            raise RegionalsError("Tournament is not finished yet")
        cached = next((c for c in commentary_list if c.get("kind") == "final"), None)
        if cached:
            return cached
        prompt = commentary_mod.final_prompt(
            data["name"], data["placements"], data["champion"]["team_name"])
        kind, day = "final", data["total_days"]
        extra = {}
    else:
        raise RegionalsError(f"Unknown commentary target: {target_type!r}")

    if llm is None:
        raise RegionalsError("LLM недоступна: установите ANTHROPIC_API_KEY и пакет anthropic")
    usage_before = dict(getattr(llm, "usage", {}) or {})
    try:
        text = llm.say(commentary_mod.system_prompt(memory), prompt,
                       max_tokens=500 if kind == "final" else 300)
    except Exception as error:  # noqa: BLE001 - surfaced to the client
        raise RegionalsError(f"Комментатор не ответил: {error}") from error

    entry = {"kind": kind, "day": day, "text": text, **extra}
    commentary_list.append(entry)
    memory.append(entry)
    usage_after = getattr(llm, "usage", {}) or {}
    if usage_after:
        usage = data.setdefault("llm", {}).setdefault("usage", {})
        for key, value in usage_after.items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + (value - int(usage_before.get(key, 0)))
    _save_regionals(data, regionals_root)
    return entry
