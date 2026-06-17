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


def create_world_payload(payload: dict) -> dict:
    """Create + bootstrap a new World from exactly WORLD_TEAM_COUNT selected
    franchises. Gated by the ui_world_tab flag (raises FeatureLocked when locked)."""
    world_flags.assert_enabled("ui_world_tab")  # locked in UI until the flag flips
    selected = list(payload.get("franchises") or [])
    if len(set(selected)) != WORLD_TEAM_COUNT:
        raise ValueError(f"select exactly {WORLD_TEAM_COUNT} teams (got {len(set(selected))})")
    by_id = {f["franchise_id"]: f for f in franchise_specs()}
    unknown = [fid for fid in selected if fid not in by_id]
    if unknown:
        raise ValueError(f"unknown franchises: {unknown}")
    specs = [by_id[fid] for fid in selected]
    world_id = "world-" + uuid.uuid4().hex[:8]
    seed = int(uuid.uuid4().int % (2 ** 31))
    world = new_world(world_id, specs, name=world_id, seed=seed)
    world_season.bootstrap_world(world, seed=seed)
    save_world(world)
    return {
        "world_id": world_id,
        "season": world["season"],
        "upper": [t["name"] for t in world["teams"].values() if t["league"] == "upper"],
        "lower": [t["name"] for t in world["teams"].values() if t["league"] == "lower"],
    }


REPLAY_DIR = ROOT / "replays_viewer_debug_smoke"
MAX_UI_RUNS = 100
MAX_POLICY_CHECKPOINT_SCAN = 120
MAX_DISCOVERED_POLICY_MODELS = 30
REPLAY_NAME = re.compile(r"^replay_\d+\.json$")
# The current trained anchor population (base18_v1). Falls back to the old
# population_18_v1 only if base18 is absent.
POPULATION_18_REGISTRY = ROOT / "training_runs/base18_v2/registry.json"
if not POPULATION_18_REGISTRY.exists():
    POPULATION_18_REGISTRY = ROOT / "training_runs/population_18_v1/registry.json"

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
    seed = int(payload.get("seed", 20260602) or 20260602)
    if runs < 1 or runs > MAX_UI_RUNS:
        raise ValueError(f"runs must be between 1 and {MAX_UI_RUNS}")

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
        replay_path = REPLAY_DIR / f"replay_{index}.json"
        result = run_match(
            seed=seed + index,
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
            "index": index,
            "seed": result.seed,
            "winner": result.winner,
            "winner_type": result.winner_type,
            "ticks": result.total_ticks,
            "path": f"../{REPLAY_DIR.name}/replay_{index}.json",
        })

    return {
        "runs": runs,
        "seed": seed,
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
        if path.startswith("/api/world/"):
            wid = path[len("/api/world/"):]
            try:
                from src.world.world import roster_players  # local import
                w = load_world(wid)
                self._send_json({"world": {
                    "world_id": w["world_id"], "season": w["season"], "month": w["month"],
                    "teams": [{"name": t["name"], "abbrev": t["abbrev"], "league": t["league"],
                               "group": t["group"], "elo": t["elo"], "treasury": t["treasury"],
                               "titles": t["titles"]} for t in w["teams"].values()],
                }})
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
                self._send_json(create_world_payload(payload))
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
