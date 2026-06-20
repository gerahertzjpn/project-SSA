from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import FACTION_TYPES, PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.core.replay import REPLAY_SCHEMA_VERSION
from src.core.rng import Spawner, resolve_seed
from src.core.runner import current_code_commit, rules_meta, run_match
from src.training.spaces import (
    STRATEGY_PRESETS, STRATEGY_PRESET_BY_ID, strategy_order_for_slot,
)


def strategy_vocab() -> dict:
    """Coherent strategy presets for the match-tab selector (one per player).
    "none" = no order -> the net plays freely (absent -> zeroed channel)."""
    return {
        "none": {"value": "none", "label": "— нет приказа —"},
        "presets": [{"value": p["id"], "label": p["label"]} for p in STRATEGY_PRESETS],
    }


def _team_slot_index(player_id: str) -> int:
    """0/1/2 position within the team (A1/B1->0, A2/B2->1, A3/B3->2)."""
    try:
        return max(0, int(player_id[1:]) - 1)
    except (ValueError, IndexError):
        return 0


def parse_player_orders(orders_in: dict) -> "dict | None":
    """Build the per-player order dict for run_match from a match request.

    Each player picks a coherent STRATEGY preset (id); it expands to a per-slot
    {tempo, win_path, role}. "no order" (or unset) -> {"absent": True} so the net
    plays freely. Returns None when NO player has a strategy (engine default).
    """
    if not isinstance(orders_in, dict):
        return None
    result: dict[str, dict] = {}
    has_strategy = False
    for player_id in PLAYER_IDS:
        entry = orders_in.get(player_id) or {}
        if not isinstance(entry, dict):
            entry = {}
        strategy = entry.get("strategy")
        if entry.get("absent") or str(entry.get("mode", "")) == "none" or not strategy:
            result[player_id] = {"absent": True}
            continue
        if strategy not in STRATEGY_PRESET_BY_ID:
            raise ValueError(f"Unknown strategy for {player_id}: {strategy}")
        result[player_id] = strategy_order_for_slot(strategy, _team_slot_index(player_id))
        has_strategy = True
    return result if has_strategy else None
from src.bots.policy_bot import (
    checkpoint_is_compatible,
    default_policy_checkpoint,
    default_team_policy_checkpoint,
    policy_model_id_for_checkpoint,
    policy_state_ids_for_checkpoint,
)
from src.league import (
    LeagueError,
    create_league,
    advance_day,
    league_public_view,
    load_league,
    list_leagues,
    encounter_detail,
    list_templates,
)
from src.regionals import (
    RegionalsError,
    add_commentary as regionals_add_commentary,
    advance_day as regionals_advance_day,
    create_regionals,
    default_llm as regionals_default_llm,
    encounter_detail as regionals_encounter_detail,
    ensure_franchise_store,
    ensure_player_store,
    list_regionals,
    list_templates as regionals_list_templates,
    llm_unavailable_reason,
    load_regionals,
    player_cards,
    regionals_public_view,
)
from src.regionals.franchises import emblem_svg, franchise_specs
from src.world import feature_flags as world_flags
from src.world import season as world_season
from src.world.world import list_worlds, load_world, new_world, save_world


WORLD_TEAM_COUNT = 22  # a World needs exactly 22 teams (lower 12 + upper 10)


def world_franchise_cards() -> list[dict]:
    """Every available franchise as a selectable card (name, manager, persona,
    colours, generated emblem, default strategy) for the new-World team picker."""
    cards = []
    for fr in franchise_specs():
        cards.append({
            "franchise_id": fr["franchise_id"],
            "name": fr["name"],
            "abbrev": fr["abbrev"],
            "manager": fr.get("manager", ""),
            "persona": fr.get("persona", ""),
            "color": fr.get("color", "#3a4757"),
            "accent": fr.get("accent", "#d8e4f0"),
            "default_plan": fr.get("default_plan", {}),
            "emblem": emblem_svg(fr),
        })
    return cards


def world_info() -> dict:
    """Lock state + constraints for the World tab. ui.locked=True means the tab is
    in-development: the UI shows it but blocks selection/start until one flag flips."""
    return {
        "team_count": WORLD_TEAM_COUNT,
        "available": len(franchise_specs()),
        "ui": world_flags.ui_state("ui_world_tab"),
        "worlds": list_worlds(),
    }


# A World creation + each season advance run REAL net matches (minutes), so they run on a
# background thread (same pattern as the league/regionals jobs) and the UI polls for status.
WORLD_JOBS: dict[str, dict] = {}
WORLD_JOBS_LOCK = threading.Lock()
FROZEN_POP = ROOT / "population" / "frozen_population_v1.json"


def _world_net_fns(world: dict):
    """(soloq_match_fn, season_play_fn) bound to the frozen population, or (None, None)
    when the manifest is absent (-> abstract rating model). EVERY net match is recorded as a
    replay under data/world/<world_id>/replays/ (owner: record all World matches)."""
    if not FROZEN_POP.exists():
        return None, None
    from src.world import net_match  # local: pulls torch only when a world actually runs
    replay_dir = ROOT / "data" / "world" / world["world_id"] / "replays"
    return net_match.world_match_fns(world, str(FROZEN_POP), replay_dir=replay_dir)


def _set_world_job(job_id: str, **fields) -> None:
    with WORLD_JOBS_LOCK:
        job = WORLD_JOBS.get(job_id)
        if job is not None:
            job.update(**fields)


def _team_name(world: dict, tid) -> str:
    t = world["teams"].get(tid)
    return t["name"] if t else str(tid)


def _world_llm():
    """Real Claude client for manager decisions when the llm_manager flag is ON and a key is
    available; None otherwise (-> heuristic manager, zero API calls). Model via WORLD_LLM_MODEL."""
    from src.world import llm_manager
    if not llm_manager.llm_enabled():
        return None
    return llm_manager.get_client()


def start_world_create_job(payload: dict) -> dict:
    """Validate the 22-team pick, then bootstrap a population-backed world on a worker
    thread (net-real SoloQ + league fill). Returns a job_id the UI polls."""
    world_flags.assert_enabled("ui_world_tab")
    selected = list(payload.get("franchises") or [])
    if len(set(selected)) != WORLD_TEAM_COUNT:
        raise ValueError(f"select exactly {WORLD_TEAM_COUNT} teams (got {len(set(selected))})")
    by_id = {f["franchise_id"]: f for f in franchise_specs()}
    unknown = [fid for fid in selected if fid not in by_id]
    if unknown:
        raise ValueError(f"unknown franchises: {unknown}")
    specs = [by_id[fid] for fid in selected]
    # Season-0 player selection (optional): owner-named pros (by net id) + a total to debut.
    entered_net_ids = list(payload.get("entered_net_ids") or []) or None
    n_enter = payload.get("n_enter")
    n_enter = int(n_enter) if n_enter else None

    job_id = uuid.uuid4().hex
    with WORLD_JOBS_LOCK:
        WORLD_JOBS[job_id] = {"status": "running", "kind": "create",
                              "phase": "Готовим мир…", "world_id": None, "error": None}

    def worker() -> None:
        try:
            world_id = "world-" + uuid.uuid4().hex[:8]
            seed = int(uuid.uuid4().int % (2 ** 31))
            use_pop = FROZEN_POP.exists()
            _set_world_job(job_id, phase="Привязываем нейронок к составам…")
            world = new_world(world_id, specs, name=world_id, seed=seed,
                              population=str(FROZEN_POP) if use_pop else None,
                              entered_net_ids=entered_net_ids, n_enter=n_enter)
            soloq_mf, play_fn = _world_net_fns(world) if use_pop else (None, None)
            llm = _world_llm()
            draft_fn = None
            if llm is not None:
                from src.world import llm_manager
                draft_fn = llm_manager.draft_pick_fn(world, llm=llm)  # L2: one LLM call per pick
            _set_world_job(job_id, phase="SoloQ + драфт (LLM) + лиги (реальные матчи)…"
                           if llm else "SoloQ + драфт + лиги (реальные матчи)…")
            world_season.bootstrap_world(world, seed=seed, play=play_fn, soloq_match_fn=soloq_mf,
                                         draft_pick_fn=draft_fn)
            _set_world_job(job_id, phase="Сохраняем…")
            save_world(world)
            _set_world_job(job_id, status="done", phase="Готово", world_id=world_id)
        except Exception as error:  # noqa: BLE001 — surfaced to the client
            traceback.print_exc()
            _set_world_job(job_id, status="error", error=str(error))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def start_world_advance_job(world_id: str) -> dict:
    """Play ONE regular season for an existing world on a worker thread (net-real groups +
    playoffs -> Elo/economy/promotion), roll to the next season, save. Returns a job_id."""
    world_flags.assert_enabled("ui_world_tab")
    job_id = uuid.uuid4().hex
    with WORLD_JOBS_LOCK:
        WORLD_JOBS[job_id] = {"status": "running", "kind": "advance", "world_id": world_id,
                              "phase": "Играем сезон (реальные матчи)…", "summary": None,
                              "error": None}

    def worker() -> None:
        try:
            world = load_world(world_id)
            _, play_fn = _world_net_fns(world)
            llm = _world_llm()
            if llm is not None:
                from src.world import llm_manager
                _set_world_job(job_id, phase="Менеджеры пишут плейбуки (LLM)…")
                for t in world["teams"].values():       # L1: one playbook call per team
                    if len(t.get("roster", [])) >= 3:
                        llm_manager.author_playbook(world, t, llm=llm)
            from src.world import commentary
            open_line = commentary.season_open(world, llm=llm)      # 📻 broadcast: season open
            _set_world_job(job_id, phase="Группы + плей-офф (реальные матчи)…")
            summary = world_season.run_regular_season(world, play=play_fn)
            close_line = commentary.season_close(world, summary, llm=llm)   # 📻 champions + promo/releg
            world["season"] = int(world.get("season", 0)) + 1  # roll to the next season
            save_world(world)
            _set_world_job(job_id, status="done", phase="Сезон сыгран", summary={
                "season": summary["season"],
                "upper_champion": _team_name(world, summary["upper_champion"]),
                "lower_champion": _team_name(world, summary["lower_champion"]),
                "promoted": [_team_name(world, t) for t in summary["promoted"]],
                "relegated": [_team_name(world, t) for t in summary["relegated"]],
                "commentary": [open_line, close_line],
            })
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            _set_world_job(job_id, status="error", error=str(error))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def _offseason_budgets(settings: dict):
    """Build the owner-set off-season knobs (rookies / free agents / auction / SoloQ rounds) from
    the UI settings payload. Empty/zero sections -> None (that stage trains nobody)."""
    from src.world import freeagents as fa, rookies as rk, soloq as sq, training_auction as ta
    from src.world.training_ops import MIN_TRAINING_STEPS as FLOOR
    s = settings or {}
    rs, fs, asec = s.get("rookies") or {}, s.get("free_agents") or {}, s.get("auction") or {}
    rookie_budget = (rk.RookieBudget(n_rookies=int(rs["n"]),
                                     steps_per_rookie=max(FLOOR, int(rs.get("steps") or FLOOR)))
                     if int(rs.get("n", 0)) > 0 else None)
    fa_budget = (fa.FreeAgentBudget(max_agents=int(fs["max"]),
                                    steps_per_agent=max(FLOOR, int(fs.get("steps") or FLOOR)))
                 if int(fs.get("max", 0)) > 0 else None)
    cfg = ta.TrainingConfig(steps_per_slot=max(FLOOR, int(asec.get("steps_per_slot") or ta.DEFAULT_SLOT_STEPS)))
    slots = int(asec["slots"]) if asec.get("slots") else None
    rounds = int(s.get("soloq_rounds") or sq.DEFAULT_ROUNDS)
    return rookie_budget, fa_budget, cfg, slots, rounds


def start_world_offseason_job(world_id: str, settings: dict | None = None) -> dict:
    """Run the OFF-SEASON on a worker thread with the OWNER's settings (rookie/free-agent budgets,
    auction slot count + size, SoloQ rounds): SoloQ -> rookies -> auction (LLM bids) -> spend ->
    transfers -> draft -> playbooks, then roll to the next season + save. Launches the won training
    jobs (tracked) and returns the full event CHRONICLE for the live-replay panel."""
    world_flags.assert_enabled("ui_world_tab")
    job_id = uuid.uuid4().hex
    with WORLD_JOBS_LOCK:
        WORLD_JOBS[job_id] = {"status": "running", "kind": "offseason", "world_id": world_id,
                              "phase": "Межсезонье…", "summary": None, "error": None}

    def worker() -> None:
        try:
            from src.world import offseason
            world = load_world(world_id)
            soloq_mf, _ = _world_net_fns(world)
            rookie_budget, fa_budget, cfg, slots, rounds = _offseason_budgets(settings)
            _set_world_job(job_id, phase="SoloQ + аукцион + трансферы (LLM-менеджер)…")
            llm = _world_llm()
            report = offseason.run_offseason(
                world, soloq_match_fn=soloq_mf, llm=llm, rookie_budget=rookie_budget,
                fa_budget=fa_budget, training_config=cfg, auction_slots=slots, soloq_rounds=rounds)
            from src.world import commentary
            commentary.offseason_note(world, report, llm=llm)       # 📻 off-season recap
            save_world(world)
            events = [e for e in (report.get("events") or []) if isinstance(e, dict) and e.get("msg")]
            chronicle = [{"seq": e.get("seq"), "kind": e.get("kind"), "msg": e.get("msg"),
                          "source": e.get("source"), "team": e.get("team_id")} for e in events][:200]
            st = report.get("stages", {})
            train_jobs = (st.get("training", {}) or {}).get("jobs", []) or []
            launched = [start_training_job(world_id, j, j.get("player_name"))
                        for j in train_jobs[:TRAINING_LAUNCH_CAP]]
            _set_world_job(job_id, status="done", phase="Межсезонье закрыто", summary={
                "next_season": report.get("next_season"),
                "rookies": (st.get("rookies") or {}).get("n", 0),
                "training_auctioned": ((st.get("training", {}) or {}).get("auction") or {}).get("awarded", 0),
                "training_launched": len(launched),
                "training_skipped": max(0, len(train_jobs) - len(launched)),
                "chronicle": chronicle,
                "log": [e["msg"] for e in events][:50],
            })
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            _set_world_job(job_id, status="error", error=str(error))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def world_job_status(job_id: str) -> dict:
    with WORLD_JOBS_LOCK:
        job = WORLD_JOBS.get(job_id)
        if job is None:
            raise KeyError(f"unknown world job {job_id}")
        return dict(job, job_id=job_id)


# --- background TRAINING jobs (real PPO) + a per-world progress tracker -------- #
TRAINING_JOBS: dict[str, dict] = {}
TRAINING_JOBS_LOCK = threading.Lock()
# Each training op spins up a many-worker PPO pool, so only ONE heavy run executes at a time;
# the rest queue (shown as "queued" in the tracker) and run as the slot frees.
TRAINING_SEM = threading.Semaphore(1)
# Cap how many manager-decided trainings one off-season launches (they queue + run serially);
# the remainder are reported as skipped rather than silently dropped.
TRAINING_LAUNCH_CAP = 6
TRAINING_RUN_DIR = str(FROZEN_POP.parent.parent / "training_runs" / "league_population_v1")
_OP_METHOD = {"skill_up": "continue_ppo", "meta_adapt": "continue_ppo", "style_spec": "continue_ppo",
              "obedience_tune": "continue_ppo", "teamplay": "cotrain", "create_player": "fork"}


def start_training_job(world_id: str, job: dict, player_name: str | None = None) -> str:
    """Queue a REAL training op (manager- or owner-decided) on a background thread, capped to one
    heavy run at a time, tracked for progress. ``job`` = {spend_type, target, total_steps}; the op
    dispatches to the right backend method (continue_ppo / cotrain / fork)."""
    op = job.get("spend_type", "skill_up")
    steps = int(job.get("total_steps", 800_000))
    target = job.get("target", {})
    player_name = player_name or job.get("player_name") or target.get("player_id") or op
    job_id = uuid.uuid4().hex
    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = {"job_id": job_id, "world_id": world_id, "op": op,
                                 "player": player_name, "done": 0, "total": steps,
                                 "status": "queued", "error": None}

    def progress(done: int, total: int) -> None:
        with TRAINING_JOBS_LOCK:
            j = TRAINING_JOBS.get(job_id)
            if j is not None:
                j["done"] = int(done); j["total"] = int(total)

    def worker() -> None:
        with TRAINING_SEM:                       # one heavy training at a time; others stay "queued"
            with TRAINING_JOBS_LOCK:
                if job_id in TRAINING_JOBS:
                    TRAINING_JOBS[job_id]["status"] = "running"
            try:
                from src.world import backends
                be = backends.RealTrainingBackend(run_dir=TRAINING_RUN_DIR, require_flag=False)
                ctx = {"target": target, "total_steps": steps, "op": op,
                       "net_ids": target.get("net_ids")}
                method = getattr(be, _OP_METHOD.get(op, "continue_ppo"))
                res = method(ctx, progress)
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[job_id].update(status="done", done=steps, result=res)
            except Exception as error:  # noqa: BLE001
                traceback.print_exc()
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[job_id].update(status="error", error=str(error))

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def training_jobs_status(world_id: str) -> dict:
    rank = {"running": 0, "queued": 1, "done": 2, "error": 3}
    with TRAINING_JOBS_LOCK:
        jobs = [dict(j) for j in TRAINING_JOBS.values() if j["world_id"] == world_id]
    jobs.sort(key=lambda j: (rank.get(j["status"], 9), j["player"]))
    return {"jobs": jobs}


def world_view_payload(world_id: str) -> dict:
    """Full state for the World view: leagues sorted by Power(Elo) with rosters (players +
    bound net/faction/archetype/SoloQ rating), the SoloQ ladder, season + champions history."""
    w = load_world(world_id)
    players = w["players"]
    pid_team = {pid: t for t in w["teams"].values() for pid in t.get("roster", [])}

    def pcard(pid: str) -> dict:
        p = players[pid]
        t = pid_team.get(pid)
        return {"nickname": p["nickname"], "net_id": p.get("net_id"), "faction": p["faction"],
                "archetype": p.get("archetype"), "rating": round(float(p["soloq_rating"])),
                "role": p.get("role", "pro"), "team": (t["abbrev"] if t else None)}

    def team_row(t: dict) -> dict:
        return {"team_id": t["team_id"], "name": t["name"], "abbrev": t["abbrev"],
                "league": t["league"], "group": t["group"], "power": round(float(t["elo"])),
                "treasury": round(float(t.get("treasury", 0))), "titles": t.get("titles", 0),
                "roster": [pcard(pid) for pid in t.get("roster", [])]}

    teams = [team_row(t) for t in w["teams"].values()]
    upper = sorted([t for t in teams if t["league"] == "upper"], key=lambda t: -t["power"])
    lower = sorted([t for t in teams if t["league"] == "lower"], key=lambda t: -t["power"])
    ladder = sorted((pcard(pid) for pid in players), key=lambda c: -c["rating"])[:25]
    history = [{"season": s["season"], "upper_champion": _team_name(w, s["upper_champion"]),
                "lower_champion": _team_name(w, s["lower_champion"])}
               for s in w.get("history", [])[-10:]]
    reserves = [pcard(pid) for pid in w.get("reserves", []) if pid in players]
    return {"world_id": w["world_id"], "name": w.get("name"), "season": w["season"],
            "phase": w.get("phase"), "upper": upper, "lower": lower, "ladder": ladder,
            "history": history, "reserves": reserves,
            "commentary": (w.get("commentary") or [])[-40:],
            "counts": {"players": len(players), "free_agents": len(w.get("free_agents", [])),
                       "streamers": len(w.get("streamers", [])), "rostered": len(pid_team),
                       "reserves": len(reserves)}}


# -- tabbed-UI read models (src/world/views.py) ---------------------------------- #
def world_players_payload(world_id: str) -> dict:
    from src.world import views
    w = load_world(world_id)
    return {"season": w["season"], "players": views.ranked_players(w)}


def world_leagues_payload(world_id: str) -> dict:
    from src.world import views
    w = load_world(world_id)
    return {"season": w["season"], "leagues": views.league_tables(w)}


def world_team_payload(world_id: str, team_id: str) -> dict:
    from src.world import views
    w = load_world(world_id)
    if team_id not in w["teams"]:
        raise KeyError(f"team {team_id!r} not in world")
    return {"season": w["season"], "team": views.team_view(w, team_id)}


def world_schedule_payload(world_id: str, season=None) -> dict:
    from src.world import views
    w = load_world(world_id)
    return views.schedule_view(w, season)


def world_draft_payload(world_id: str) -> dict:
    from src.world import views
    return views.draft_view(load_world(world_id))


def world_players_pool_payload() -> dict:
    """The frozen pool the owner picks from at world creation (Season-0 selection). Pros only,
    with their card overall + per-attribute, sorted strongest-first."""
    manifest = json.loads(FROZEN_POP.read_text(encoding="utf-8"))
    try:
        cards = json.loads((FROZEN_POP.parent / "player_cards_v1.json").read_text(encoding="utf-8")).get("cards", {})
    except (OSError, ValueError):
        cards = {}
    ATTRS = ("attack", "defense", "economy", "tempo", "consistency", "potential")
    pool = []
    for e in manifest.get("descendants", []):
        c = cards.get(e["net_id"], {})
        vals = [c[k] for k in ATTRS if isinstance(c.get(k), (int, float))]
        pool.append({"net_id": e["net_id"], "faction": e["faction"], "archetype": e.get("archetype"),
                     "overall": round(sum(vals) / len(vals)) if vals else 0,
                     "attrs": {k: c.get(k) for k in ATTRS}})
    pool.sort(key=lambda p: -p["overall"])
    return {"pool": pool, "total": len(pool), "n_teams": WORLD_TEAM_COUNT,
            "min_enter": WORLD_TEAM_COUNT * 3}


def introduce_reserves_action(world_id: str, player_ids: list) -> dict:
    """Debut held-out reserve players (any season) -> the rookies pool."""
    from src.world.world import introduce_reserves
    w = load_world(world_id)
    intro = introduce_reserves(w, list(player_ids or []))
    save_world(w)
    return {"introduced": len(intro), "reserves_left": len(w.get("reserves", []))}


REPLAY_DIR = ROOT / "replays"
MAX_UI_RUNS = 100
MAX_POLICY_CHECKPOINT_SCAN = 120
MAX_DISCOVERED_POLICY_MODELS = 30
REPLAY_NAME = re.compile(r"^replay_\d+\.json$")
# The current trained anchor population shown on the match tab = population_v1 (the live
# base-18). Older runs (base18_v2 / population_18_v1) are only a fallback if the current
# one is absent; any other discovered checkpoints are listed as legacy further down.
POPULATION_18_REGISTRY = ROOT / "training_runs/population_v1/registry.json"
for _legacy_registry in ("training_runs/base18_v2/registry.json",
                         "training_runs/population_18_v1/registry.json"):
    if not POPULATION_18_REGISTRY.exists():
        POPULATION_18_REGISTRY = ROOT / _legacy_registry

_RECIPE_LABEL = {
    "baseline": "Baseline",
    "low_entropy": "LowEnt",
    "high_entropy": "HighEnt",
    "fast_lr": "FastLR",
    "slow_explore": "SlowExp",
    "alt_mix": "AltMix",
}
_FACTION_LABEL = {
    "humans": "Humans",
    "infected": "Infected",
    "robots": "Robots",
}
SERVER_STARTED_AT = int(time.time())
SERVER_COMMIT = current_code_commit()
SERVER_SOURCE_FINGERPRINT = ""


def source_fingerprint() -> str:
    paths = [ROOT / "viewer/server.py"]
    src_root = ROOT / "src"
    if src_root.exists():
        paths.extend(src_root.rglob("*.py"))
    parts = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            parts.append(f"{path.relative_to(ROOT).as_posix()}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            continue
    return "|".join(parts)


SERVER_SOURCE_FINGERPRINT = source_fingerprint()


def available_models() -> list[dict]:
    models = [
        baseline_model("random", "Random Bot"),
        baseline_model("scripted", "Scripted Bot"),
    ]
    models.extend(population18_models())
    models.extend(legacy_models())
    return models


def population18_models() -> list[dict]:
    if not POPULATION_18_REGISTRY.exists():
        return []
    try:
        data = json.loads(POPULATION_18_REGISTRY.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
    except Exception:
        return []
    result = []
    for entry in entries:
        if entry.get("status") != "trained":
            continue
        checkpoint_path = ROOT / entry["checkpoint"]
        if not checkpoint_path.exists():
            continue
        net_id = entry["net_id"]
        faction = entry.get("faction", "")
        recipe = entry.get("recipe", "")
        num = net_id.rsplit("_", 1)[-1] if "_" in net_id else net_id
        faction_label = _FACTION_LABEL.get(faction, faction.title())
        recipe_label = _RECIPE_LABEL.get(recipe, recipe)
        result.append({
            "id": policy_model_id_for_checkpoint(checkpoint_path),
            "label": f"{faction_label} {num} · {recipe_label}",
            "kind": "policy",
            "checkpoint": entry["checkpoint"],
            "updated_at": int(checkpoint_path.stat().st_mtime),
            "is_latest": True,
            "show_by_default": True,
            "faction": faction,
        })
    return result


def legacy_models() -> list[dict]:
    models: list[dict] = [
        legacy_baseline("scripted_core", "Core Bot (legacy)"),
        legacy_baseline("scripted_core_raider", "Raider Core (legacy)"),
        legacy_baseline("scripted_core_bmp", "BMP Core (legacy)"),
    ]
    seen_checkpoints: set[Path] = set()
    team_checkpoint = default_team_policy_checkpoint()
    if team_checkpoint is not None:
        seen_checkpoints.add(team_checkpoint.resolve())
        models.append({
            "id": "ppo_team_scripted_v1",
            "label": "Team Best (legacy)",
            "kind": "legacy",
            "checkpoint": checkpoint_display_path(team_checkpoint),
            "is_latest": False,
            "show_by_default": False,
        })
    latest_team_checkpoint = ROOT / "training_runs/sprint3_team_scripted_warm_v1/checkpoints/latest.pt"
    if latest_team_checkpoint.exists() and checkpoint_is_compatible(latest_team_checkpoint):
        seen_checkpoints.add(latest_team_checkpoint.resolve())
        models.append({
            "id": "ppo_team_scripted_v1_latest",
            "label": "Team Latest (legacy)",
            "kind": "legacy",
            "checkpoint": checkpoint_display_path(latest_team_checkpoint),
            "updated_at": int(latest_team_checkpoint.stat().st_mtime),
            "is_latest": False,
            "show_by_default": False,
        })
    best_checkpoint = default_policy_checkpoint()
    if best_checkpoint is not None:
        seen_checkpoints.add(best_checkpoint.resolve())
        models.append({
            "id": "ppo_random_v1",
            "label": "Random Best (legacy)",
            "kind": "legacy",
            "checkpoint": checkpoint_display_path(best_checkpoint),
            "is_latest": False,
            "show_by_default": False,
        })
    latest_checkpoint = ROOT / "training_runs/sprint2_random_first_real_v1/checkpoints/latest.pt"
    if latest_checkpoint.exists() and checkpoint_is_compatible(latest_checkpoint):
        seen_checkpoints.add(latest_checkpoint.resolve())
        models.append({
            "id": "ppo_random_v1_latest",
            "label": "Random Latest (legacy)",
            "kind": "legacy",
            "checkpoint": checkpoint_display_path(latest_checkpoint),
            "updated_at": int(latest_checkpoint.stat().st_mtime),
            "is_latest": False,
            "show_by_default": False,
        })
    for checkpoint in discover_policy_checkpoints():
        resolved = checkpoint.resolve()
        if resolved in seen_checkpoints:
            continue
        seen_checkpoints.add(resolved)
        for m in policy_models_for_checkpoint(checkpoint):
            m["kind"] = "legacy"
            m["show_by_default"] = False
            m["label"] = m["label"] + " (legacy)"
            models.append(m)
    return models


def baseline_model(model_id: str, label: str) -> dict:
    return {
        "id": model_id,
        "label": label,
        "kind": "baseline",
        "is_latest": True,
        "show_by_default": True,
    }


def legacy_baseline(model_id: str, label: str) -> dict:
    return {
        "id": model_id,
        "label": label,
        "kind": "legacy",
        "is_latest": False,
        "show_by_default": False,
    }


def policy_models_for_checkpoint(checkpoint: Path) -> list[dict]:
    state_ids = policy_state_ids_for_checkpoint(checkpoint)
    if not state_ids:
        return [policy_model(checkpoint)]
    return [policy_model(checkpoint, policy_state_id=state_id) for state_id in state_ids]


def policy_model(checkpoint: Path, policy_state_id: str | None = None) -> dict:
    relative = checkpoint.relative_to(ROOT).as_posix()
    is_latest = checkpoint.name == "latest.pt"
    label = short_policy_label(checkpoint)
    if policy_state_id:
        label = f"{label} {policy_state_id}"
    model = {
        "id": policy_model_id_for_checkpoint(checkpoint, policy_state_id),
        "label": label,
        "kind": "policy",
        "checkpoint": relative,
        "updated_at": int(checkpoint.stat().st_mtime),
        "is_latest": is_latest,
        "show_by_default": is_latest,
    }
    if policy_state_id:
        model["policy_state"] = policy_state_id
    return model


def checkpoint_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def short_policy_label(checkpoint: Path) -> str:
    return f"{short_run_label(checkpoint.parent.parent.name)} {short_checkpoint_label(checkpoint)}"


def short_checkpoint_label(checkpoint: Path) -> str:
    if checkpoint.name == "latest.pt":
        return "latest"
    match = re.search(r"checkpoint_update_(\d+)", checkpoint.stem)
    if match:
        return f"u{match.group(1)}"
    return re.sub(r"[^A-Za-z0-9]+", "", checkpoint.stem)[:10] or "checkpoint"


def short_run_label(run_name: str) -> str:
    patterns = [
        (r"teammate_mix_v4_tmix(\d+)", "Mix{}"),
        (r"teammate_mix_tmix(\d+)", "Mix{}"),
        (r"selfplay_v3_recovery_spv(\d+)", "Recover{}"),
        (r"selfplay_v3_tmix(\d+)_vs_tmix(\d+)", "Tmix{}v{}"),
        (r"rotmix_mix(\d+)_peer\d+_stage(\d+)", "RotMix{}s{}"),
        (r"bridge_seed(\d+)_vs_seed(\d+)", "Bridge{}v{}"),
        (r"peer_seed(\d+)_vs_bridge(\d+)", "Peer{}v{}"),
        (r"seed(\d+)_scripted", "Seed{}"),
    ]
    for pattern, template in patterns:
        match = re.search(pattern, run_name)
        if match:
            values = [str(int(value)) if value.isdigit() else value for value in match.groups()]
            return template.format(*values)
    if "sprint3_team_scripted" in run_name:
        return "SlotScript"
    if "sprint2_random" in run_name:
        return "RandomPPO"
    words = [word for word in re.split(r"[^A-Za-z0-9]+", run_name) if word]
    useful = [word for word in words if word not in {"fresh", "population", "v31", "checkpoints"}]
    compact = "".join(word[:3].title() for word in useful[:2])
    return compact or "Policy"


def discover_policy_checkpoints() -> list[Path]:
    root = ROOT / "training_runs"
    if not root.exists():
        return []
    candidates = sorted(
        (path for path in root.glob("*/checkpoints/*.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    checkpoints: list[Path] = []
    for path in candidates[:MAX_POLICY_CHECKPOINT_SCAN]:
        if checkpoint_is_compatible(path):
            checkpoints.append(path)
            if len(checkpoints) >= MAX_DISCOVERED_POLICY_MODELS:
                break
    return checkpoints


def server_info() -> dict:
    current_commit = current_code_commit()
    current_fingerprint = source_fingerprint()
    fingerprint_equal = SERVER_SOURCE_FINGERPRINT == current_fingerprint
    stale = (
        SERVER_COMMIT != "unknown"
        and current_commit != "unknown"
        and SERVER_COMMIT != current_commit
    ) or not fingerprint_equal
    return {
        "commit": SERVER_COMMIT,
        "current_commit": current_commit,
        "stale": stale,
        "source_fingerprint_equal": fingerprint_equal,
        "source_fingerprint": SERVER_SOURCE_FINGERPRINT[:24],
        "current_source_fingerprint": current_fingerprint[:24],
        "started_at": SERVER_STARTED_AT,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "rules": rules_meta(),
    }


def replay_files() -> list[dict]:
    if not REPLAY_DIR.exists():
        return []
    files = []
    for path in sorted(REPLAY_DIR.iterdir(), key=replay_sort_key):
        if path.is_file() and REPLAY_NAME.match(path.name):
            files.append({
                "label": path.name,
                "path": f"../{REPLAY_DIR.name}/{path.name}",
            })
    return files


def replay_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    index = int(match.group(1)) if match else 0
    return index, path.name


def clear_replays() -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for path in REPLAY_DIR.iterdir():
        if path.is_file() and REPLAY_NAME.match(path.name):
            path.unlink()


def generate_replays(payload: dict) -> dict:
    info = server_info()
    if info["stale"]:
        raise RuntimeError(
            f"Viewer API is running old code ({info['commit']}); restart viewer for {info['current_commit']}"
        )

    runs = int(payload.get("runs", 1) or 1)
    if runs < 1 or runs > MAX_UI_RUNS:
        raise ValueError(f"runs must be between 1 and {MAX_UI_RUNS}")
    # Non-training randomness must be UNIQUE every run: fresh OS-entropy root by default
    # (recorded in the response), independent per-match children via spawn — never base+offset.
    raw_seed = payload.get("seed")
    root_seed = resolve_seed(int(raw_seed) if raw_seed not in (None, "") else None)
    match_seeds = Spawner(root_seed).seeds(runs)

    allowed = {model["id"] for model in available_models()}
    players = payload.get("players", {}) or {}
    player_models = {}
    for player_id in PLAYER_IDS:
        model_id = str(players.get(player_id, "random"))
        if model_id not in allowed:
            raise ValueError(f"Unknown model for {player_id}: {model_id}")
        player_models[player_id] = model_id

    allowed_factions = set(FACTION_TYPES)
    factions = payload.get("factions", {}) or {}
    player_factions = {}
    for player_id in PLAYER_IDS:
        faction = str(factions.get(player_id, "humans"))
        if faction not in allowed_factions:
            choices = ", ".join(FACTION_TYPES)
            raise ValueError(f"Unknown faction for {player_id}: {faction}; expected one of: {choices}")
        player_factions[player_id] = faction

    orders_arg = parse_player_orders(payload.get("orders", {}) or {})

    clear_replays()
    results = []
    winners: dict[str, int] = {}
    winner_types: dict[str, int] = {}
    for index in range(runs):
        replay_path = REPLAY_DIR / f"replay_{index + 1}.json"  # 1-based: first replay is #1
        result = run_match(
            seed=match_seeds[index],
            bot_type="mixed",
            replay_path=str(replay_path),
            mode="replay",
            player_models=player_models,
            player_factions=player_factions,
            orders=orders_arg,
        )
        winners[result.winner] = winners.get(result.winner, 0) + 1
        winner_types[result.winner_type] = winner_types.get(result.winner_type, 0) + 1
        results.append({
            "index": index + 1,
            "seed": result.seed,
            "winner": result.winner,
            "winner_type": result.winner_type,
            "ticks": result.total_ticks,
            "path": f"../{REPLAY_DIR.name}/replay_{index + 1}.json",
        })

    return {
        "runs": runs,
        "seed": root_seed,
        "players": player_models,
        "factions": player_factions,
        "orders": orders_arg or {},
        "winners": winners,
        "winner_types": winner_types,
        "results": results,
        "replays": replay_files(),
        "server": info,
    }


# --------------------------------------------------------------------------- #
# League endpoints
# --------------------------------------------------------------------------- #

LEAGUE_JOBS: dict[str, dict] = {}
LEAGUE_JOBS_LOCK = threading.Lock()
LEAGUE_ENDPOINT = re.compile(r"^/api/league/(?P<league_id>[^/]+)(?:/(?P<action>encounter|advance))?(?:/(?P<arg>\d+))?$")


def start_league_job(payload: dict) -> dict:
    info = server_info()
    if info["stale"]:
        raise RuntimeError(
            f"Viewer API is running old code ({info['commit']}); restart viewer for {info['current_commit']}"
        )
    name = str(payload.get("name") or "League").strip() or "League"
    template_id = str(payload.get("template") or "classic_6x3")
    teams = payload.get("teams") or []
    seed = int(payload.get("seed", 0) or 0)
    allowed = {model["id"] for model in available_models()}

    job_id = uuid.uuid4().hex
    with LEAGUE_JOBS_LOCK:
        LEAGUE_JOBS[job_id] = {"status": "running", "done": 0, "total": 0, "league_id": None, "error": None}

    def progress(done: int, total: int) -> None:
        with LEAGUE_JOBS_LOCK:
            job = LEAGUE_JOBS.get(job_id)
            if job is not None:
                job["done"] = done
                job["total"] = total

    def worker() -> None:
        try:
            league = create_league(
                name=name,
                template_id=template_id,
                teams=teams,
                seed=seed,
                allowed_models=allowed,
                progress_cb=progress,
            )
            with LEAGUE_JOBS_LOCK:
                LEAGUE_JOBS[job_id].update(status="done", league_id=league["id"])
        except Exception as error:  # noqa: BLE001 - surfaced to the client
            traceback.print_exc()
            with LEAGUE_JOBS_LOCK:
                LEAGUE_JOBS[job_id].update(status="error", error=str(error))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def league_job_status(job_id: str) -> dict:
    with LEAGUE_JOBS_LOCK:
        job = LEAGUE_JOBS.get(job_id)
        if job is None:
            raise LeagueError(f"Unknown league job: {job_id}")
        return dict(job, job_id=job_id)


# --------------------------------------------------------------------------- #
# Regionals endpoints
# --------------------------------------------------------------------------- #

REGIONALS_JOBS: dict[str, dict] = {}
REGIONALS_JOBS_LOCK = threading.Lock()
REGIONALS_ENDPOINT = re.compile(
    r"^/api/regionals/(?P<rid>[^/]+)(?:/(?P<action>encounter|advance|commentary))?(?:/(?P<arg>\d+))?$")


def start_regionals_job(payload: dict) -> dict:
    info = server_info()
    if info["stale"]:
        raise RuntimeError(
            f"Viewer API is running old code ({info['commit']}); restart viewer for {info['current_commit']}"
        )
    name = str(payload.get("name") or "Regionals").strip() or "Regionals"
    template_id = str(payload.get("template") or "regionals_v1")
    seed = int(payload.get("seed", 0) or 0)

    job_id = uuid.uuid4().hex
    with REGIONALS_JOBS_LOCK:
        REGIONALS_JOBS[job_id] = {"status": "running", "done": 0, "total": 0,
                                  "phase": "draft", "regionals_id": None, "error": None}

    def progress(done: int, total: int, phase: str) -> None:
        with REGIONALS_JOBS_LOCK:
            job = REGIONALS_JOBS.get(job_id)
            if job is not None:
                job["done"] = done
                job["total"] = total
                job["phase"] = phase

    def worker() -> None:
        try:
            llm = regionals_default_llm()
            data = create_regionals(
                name=name, template_id=template_id, seed=seed,
                llm=llm, progress_cb=progress,
            )
            with REGIONALS_JOBS_LOCK:
                REGIONALS_JOBS[job_id].update(status="done", regionals_id=data["id"])
        except Exception as error:  # noqa: BLE001 - surfaced to the client
            traceback.print_exc()
            with REGIONALS_JOBS_LOCK:
                REGIONALS_JOBS[job_id].update(status="error", error=str(error))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running", "llm": llm_unavailable_reason() is None}


def regionals_job_status(job_id: str) -> dict:
    with REGIONALS_JOBS_LOCK:
        job = REGIONALS_JOBS.get(job_id)
        if job is None:
            raise RegionalsError(f"Unknown regionals job: {job_id}")
        return dict(job, job_id=job_id)


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/models":
            self._send_json({
                "players": PLAYER_IDS,
                "teams": {
                    "TEAM_A": TEAM_A_PLAYERS,
                    "TEAM_B": TEAM_B_PLAYERS,
                },
                "factions": list(FACTION_TYPES),
                "models": available_models(),
                "strategies": strategy_vocab(),
                "max_ui_runs": MAX_UI_RUNS,
                "replays": replay_files(),
                "server": server_info(),
            })
            return
        if path == "/api/replays":
            self._send_json({"replays": replay_files(), "server": server_info()})
            return
        if path == "/api/league/templates":
            self._send_json({"templates": list_templates(), "models": available_models(),
                             "factions": list(FACTION_TYPES), "server": server_info()})
            return
        if path == "/api/league/list":
            self._send_json({"leagues": list_leagues(), "server": server_info()})
            return
        if path.startswith("/api/league/"):
            self._handle_league_get(path)
            return
        if path == "/api/regionals/templates":
            self._send_json({
                "templates": regionals_list_templates(),
                "llm_unavailable": llm_unavailable_reason(),
                "server": server_info(),
            })
            return
        if path == "/api/regionals/list":
            self._send_json({"regionals": list_regionals(), "server": server_info()})
            return
        if path == "/api/regionals/players":
            try:
                store = ensure_player_store()
                self._send_json({"players": player_cards(store)})
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=400)
            return
        if path == "/api/regionals/franchises":
            try:
                store = ensure_franchise_store()
                self._send_json({"franchises": list(store["franchises"].values())})
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=400)
            return
        if path == "/api/world/info":
            self._send_json({**world_info(), "server": server_info()})
            return
        if path == "/api/world/franchises":
            self._send_json({"franchises": world_franchise_cards(), "server": server_info()})
            return
        world_job = re.match(r"^/api/world/job/(?P<job_id>[0-9a-f]+)$", path)
        if world_job:
            try:
                self._send_json(world_job_status(world_job.group("job_id")))
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=404)
            return
        if path == "/api/world/players-pool":
            self._send_json(world_players_pool_payload())
            return
        world_train = re.match(r"^/api/world/(?P<wid>world-[0-9a-f]+)/training$", path)
        if world_train:
            self._send_json(training_jobs_status(world_train.group("wid")))
            return
        for pat, fn in (
            (r"^/api/world/(?P<wid>world-[0-9a-f]+)/players$", world_players_payload),
            (r"^/api/world/(?P<wid>world-[0-9a-f]+)/leagues$", world_leagues_payload),
            (r"^/api/world/(?P<wid>world-[0-9a-f]+)/schedule$", world_schedule_payload),
            (r"^/api/world/(?P<wid>world-[0-9a-f]+)/draft$", world_draft_payload),
        ):
            m = re.match(pat, path)
            if m:
                try:
                    self._send_json(fn(m.group("wid")))
                except Exception as error:  # noqa: BLE001
                    self._send_json({"error": str(error)}, status=404)
                return
        world_team = re.match(r"^/api/world/(?P<wid>world-[0-9a-f]+)/team/(?P<tid>[\w-]+)$", path)
        if world_team:
            try:
                self._send_json(world_team_payload(world_team.group("wid"), world_team.group("tid")))
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=404)
            return
        if path.startswith("/api/world/"):
            wid = path[len("/api/world/"):]
            try:
                self._send_json({"world": world_view_payload(wid)})
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=404)
            return
        if path.startswith("/api/regionals/"):
            self._handle_regionals_get(path)
            return
        super().do_GET()

    def _handle_regionals_get(self, path: str) -> None:
        if path.startswith("/api/regionals/job/"):
            job_id = path[len("/api/regionals/job/"):]
            try:
                self._send_json(regionals_job_status(job_id))
            except RegionalsError as error:
                self._send_json({"error": str(error)}, status=404)
            return
        match = REGIONALS_ENDPOINT.match(path)
        if not match:
            self.send_error(404, "Unknown regionals endpoint")
            return
        rid = match.group("rid")
        action = match.group("action")
        arg = match.group("arg")
        try:
            if action == "encounter" and arg is not None:
                self._send_json({"encounter": regionals_encounter_detail(rid, int(arg))})
            elif action is None:
                self._send_json({"regionals": regionals_public_view(load_regionals(rid))})
            else:
                self.send_error(404, "Unknown regionals endpoint")
        except RegionalsError as error:
            self._send_json({"error": str(error)}, status=404)
        except Exception as error:  # noqa: BLE001
            self._send_json({"error": str(error)}, status=400)

    def _handle_league_get(self, path: str) -> None:
        if path.startswith("/api/league/job/"):
            job_id = path[len("/api/league/job/"):]
            try:
                self._send_json(league_job_status(job_id))
            except LeagueError as error:
                self._send_json({"error": str(error)}, status=404)
            return
        match = LEAGUE_ENDPOINT.match(path)
        if not match:
            self.send_error(404, "Unknown league endpoint")
            return
        league_id = match.group("league_id")
        action = match.group("action")
        arg = match.group("arg")
        try:
            if action == "encounter" and arg is not None:
                self._send_json({"encounter": encounter_detail(league_id, int(arg))})
            elif action is None:
                self._send_json({"league": league_public_view(load_league(league_id))})
            else:
                self.send_error(404, "Unknown league endpoint")
        except LeagueError as error:
            self._send_json({"error": str(error)}, status=404)
        except Exception as error:  # noqa: BLE001
            self._send_json({"error": str(error)}, status=400)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/generate-replays":
            try:
                payload = self._read_json()
                self._send_json(generate_replays(payload))
            except RuntimeError as error:
                self._send_json({"error": str(error), "server": server_info()}, status=409)
            except Exception as error:
                self._send_json({"error": str(error), "server": server_info()}, status=400)
            return
        if path == "/api/league/create":
            try:
                payload = self._read_json()
                self._send_json(start_league_job(payload))
            except RuntimeError as error:
                self._send_json({"error": str(error), "server": server_info()}, status=409)
            except Exception as error:
                self._send_json({"error": str(error), "server": server_info()}, status=400)
            return
        league_advance = re.match(r"^/api/league/(?P<league_id>[^/]+)/advance$", path)
        if league_advance:
            try:
                self._send_json({"league": advance_day(league_advance.group("league_id"))})
            except LeagueError as error:
                self._send_json({"error": str(error)}, status=404)
            except Exception as error:
                self._send_json({"error": str(error)}, status=400)
            return
        if path == "/api/regionals/create":
            try:
                payload = self._read_json()
                self._send_json(start_regionals_job(payload))
            except RuntimeError as error:
                self._send_json({"error": str(error), "server": server_info()}, status=409)
            except Exception as error:
                self._send_json({"error": str(error), "server": server_info()}, status=400)
            return
        if path == "/api/world/create":
            try:
                payload = self._read_json()
                self._send_json(start_world_create_job(payload))
            except world_flags.FeatureLocked as error:
                self._send_json({"error": str(error), "locked": True,
                                 "server": server_info()}, status=403)
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error), "server": server_info()}, status=400)
            return
        world_intro = re.match(r"^/api/world/(?P<wid>world-[0-9a-f]+)/introduce$", path)
        if world_intro:
            try:
                body = self._read_json() if int(self.headers.get("Content-Length") or 0) else {}
                self._send_json(introduce_reserves_action(world_intro.group("wid"),
                                                          body.get("player_ids") or []))
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error), "server": server_info()}, status=400)
            return
        world_post = re.match(r"^/api/world/(?P<wid>world-[0-9a-f]+)/(?P<action>advance|offseason)$", path)
        if world_post:
            action = world_post.group("action")
            try:
                if action == "offseason":
                    settings = self._read_json() if int(self.headers.get("Content-Length") or 0) else None
                    self._send_json(start_world_offseason_job(world_post.group("wid"), settings))
                else:
                    self._send_json(start_world_advance_job(world_post.group("wid")))
            except world_flags.FeatureLocked as error:
                self._send_json({"error": str(error), "locked": True,
                                 "server": server_info()}, status=403)
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error), "server": server_info()}, status=400)
            return
        regionals_post = re.match(r"^/api/regionals/(?P<rid>[^/]+)/(?P<action>advance|commentary)$", path)
        if regionals_post:
            rid = regionals_post.group("rid")
            action = regionals_post.group("action")
            try:
                if action == "advance":
                    self._send_json({"regionals": regionals_advance_day(rid)})
                else:
                    payload = self._read_json()
                    llm = regionals_default_llm()
                    entry = regionals_add_commentary(rid, payload, llm)
                    self._send_json({"commentary": entry})
            except RegionalsError as error:
                self._send_json({"error": str(error)}, status=400)
            except Exception as error:
                self._send_json({"error": str(error)}, status=400)
            return
        self.send_error(404, "Unknown API endpoint")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local replay viewer server with generation API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ViewerRequestHandler)
    print(f"Pixi viewer: http://{args.host}:{args.port}/viewer_pixi/index.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
