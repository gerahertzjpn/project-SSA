from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.constants import PLAYER_IDS
from src.core.engine import GameEngine
from src.core.runner import current_code_commit, make_bots, rules_meta
from src.core.state import get_team, make_initial_state


def player_id(value: str | None) -> bool:
    return value in PLAYER_IDS


def numeric_stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "p10": round(percentile(ordered, 0.10), 3),
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.mean(ordered), 3),
        "p90": round(percentile(ordered, 0.90), 3),
        "max": round(ordered[-1], 3),
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def counter_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def merge_counter(target: Counter, values: dict | Counter) -> None:
    for key, value in (values or {}).items():
        target[str(key)] += int(value or 0)


def empty_lord_stats() -> dict[str, Any]:
    return {
        "neutral_killed": False,
        "kill_tick": None,
        "owner": None,
        "owner_team": None,
        "target": None,
        "spawn_path": [],
        "no_owner_after_kill": False,
        "baron_participants": [],
        "baron_damage_by_player": Counter(),
        "baron_damage_by_team": Counter(),
        "shots": 0,
        "damage": 0.0,
        "core_damage": 0.0,
        "damage_by_target_type": Counter(),
        "damage_by_target_team": Counter(),
        "battle_ticks": 0,
        "attack_locations": Counter(),
        "continued_nodes": [],
        "core_expended": False,
        "core_expended_tick": None,
        "core_expended_owner": None,
        "enemy_deaths_during_lord_ticks": 0,
        "enemy_deaths_during_lord_ticks_by_type": Counter(),
        "enemy_core_deaths_during_lord_ticks": 0,
    }


def run_one(seed: int) -> dict[str, Any]:
    state = make_initial_state(seed=seed)
    engine = GameEngine(
        state,
        make_bots("random", seed_offset=seed * 10),
        record_trace=True,
        record_replay=True,
    )
    winner = engine.run()
    return analyze_match(seed, winner, state, engine.replay_ticks)


def analyze_match(seed: int, winner: str, state, ticks: list[dict]) -> dict[str, Any]:
    lord = empty_lord_stats()

    for tick in ticks:
        tick_index = int(tick.get("tick", 0) or 0)
        for event in tick.get("events", []) or []:
            event_type = event.get("type")
            location_id = event.get("location_id") or event.get("node")
            if (
                event_type == "BATTLE_STARTED"
                and location_id == "Baron"
            ):
                lord["baron_participants"].extend(event.get("participants") or [])

            if event_type == "BATTLE_TICK":
                summary = event.get("summary") or {}
                if location_id == "Baron":
                    for player, damage in (summary.get("damage_by_team") or {}).items():
                        if not player_id(player):
                            continue
                        rounded_damage = int(round(float(damage or 0.0)))
                        lord["baron_damage_by_player"][player] += rounded_damage
                        lord["baron_damage_by_team"][get_team(player)] += rounded_damage
                consume_lord_tick(lord, event, summary, location_id)

            if (
                event_type == "BATTLE_RESOLVED"
                and location_id == "Baron"
                and event.get("baron_killed")
            ):
                lord["neutral_killed"] = True
                lord["kill_tick"] = tick_index
                spawned = event.get("lord_spawned") or {}
                if spawned:
                    owner = spawned.get("player")
                    lord["owner"] = owner
                    lord["owner_team"] = get_team(owner) if owner else None
                    lord["target"] = spawned.get("target")
                    lord["spawn_path"] = list(spawned.get("path") or [])
                else:
                    lord["no_owner_after_kill"] = True

            if event_type == "BATTLE_RESOLVED" and event.get("lord_continued"):
                continued = event.get("lord_continued") or {}
                lord["continued_nodes"].append({
                    "tick": tick_index,
                    "from": continued.get("src") or location_id,
                    "target": continued.get("target"),
                    "path": list(continued.get("path") or []),
                })

            if event_type == "BATTLE_RESOLVED" and event.get("lord_expended"):
                lord["core_expended"] = True
                lord["core_expended_tick"] = tick_index
                lord["core_expended_owner"] = event.get("core_owner")

    final_core_hp = {
        player_id_: round(float(player.core_hp), 3)
        for player_id_, player in sorted(state.players.items())
    }
    final_alive = {
        player_id_: bool(player.alive)
        for player_id_, player in sorted(state.players.items())
    }

    lord_state = {
        "alive": bool(state.baron.allied_lord_alive),
        "expended": bool(state.baron.allied_lord_expended),
        "hp": round(float(state.baron.allied_lord_hp), 3),
        "owner": state.baron.allied_lord_owner,
        "target": state.baron.allied_lord_target,
    }

    if lord["owner"] and not lord_state["alive"] and not lord_state["expended"] and not lord["core_expended"]:
        lord["died_before_core"] = True
    else:
        lord["died_before_core"] = False

    return {
        "seed": seed,
        "winner": winner,
        "winner_type": state.winner_type or "unknown",
        "ticks": int(state.tick),
        "core_combat_damage": round(float(state.core_combat_damage), 3),
        "core_damage_by_team": {
            team: round(float(value), 3)
            for team, value in sorted(state.core_damage_by_team.items())
        },
        "final_core_hp": final_core_hp,
        "final_alive": final_alive,
        "death_time_final_core_deaths": dict(state.death_time_final_core_deaths),
        "lord": serialize_lord(lord),
        "allied_lord_state": lord_state,
    }


def consume_lord_tick(lord: dict[str, Any], event: dict, summary: dict, location_id: str | None) -> None:
    attacks = summary.get("attacks") or []
    lord_attacks = [
        attack
        for attack in attacks
        if len(attack) >= 7 and player_id(attack[3]) and attack[4] == "L"
    ]
    alive = summary.get("alive") or {}
    lord_alive_sides = [
        side
        for side, side_info in alive.items()
        if player_id(side) and int((side_info.get("units") or {}).get("LORD", 0) or 0) > 0
    ]
    if not lord_attacks and not lord_alive_sides:
        return

    lord["battle_ticks"] += 1
    if location_id:
        lord["attack_locations"][location_id] += 1

    for attack in lord_attacks:
        damage = float(attack[2] or 0.0)
        target_team = str(attack[5])
        target_type = str(attack[6])
        lord["shots"] += 1
        lord["damage"] += damage
        lord["damage_by_target_team"][target_team] += int(round(damage))
        lord["damage_by_target_type"][target_type] += int(round(damage))
        if target_type == "C":
            lord["core_damage"] += damage

    owner_team = lord.get("owner_team")
    if not owner_team and lord_attacks:
        owner_team = get_team(lord_attacks[0][3])
    if not owner_team:
        return

    for dead_side, units in (summary.get("deaths") or {}).items():
        if not player_id(dead_side) or get_team(dead_side) == owner_team:
            continue
        for unit_type, count in (units or {}).items():
            value = int(count or 0)
            lord["enemy_deaths_during_lord_ticks"] += value
            lord["enemy_deaths_during_lord_ticks_by_type"][unit_type] += value
            if unit_type == "CORE":
                lord["enemy_core_deaths_during_lord_ticks"] += value


def serialize_lord(lord: dict[str, Any]) -> dict[str, Any]:
    result = dict(lord)
    for key in (
        "baron_damage_by_player",
        "baron_damage_by_team",
        "damage_by_target_type",
        "damage_by_target_team",
        "attack_locations",
        "enemy_deaths_during_lord_ticks_by_type",
    ):
        result[key] = counter_dict(result[key])
    result["damage"] = round(float(result["damage"]), 3)
    result["core_damage"] = round(float(result["core_damage"]), 3)
    result["baron_participants"] = sorted(set(result["baron_participants"]))
    return result


def chunks(items: list[int], workers: int) -> list[list[int]]:
    workers = max(1, min(workers, len(items)))
    return [items[index::workers] for index in range(workers)]


def run_chunk(seeds: list[int]) -> list[dict[str, Any]]:
    return [run_one(seed) for seed in seeds]


def aggregate(matches: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(matches)
    winners = Counter(match["winner"] for match in matches)
    winner_types = Counter(match["winner_type"] for match in matches)
    owner_players = Counter()
    owner_teams = Counter()
    target_counts = Counter()
    owner_team_outcomes = defaultdict(Counter)
    no_owner_seeds = []

    kill_ticks = []
    shot_counts = []
    lord_damage = []
    lord_core_damage = []
    lord_battle_ticks = []
    enemy_deaths = []
    continued_counts = []
    attack_locations = Counter()
    damage_by_target_type = Counter()
    enemy_deaths_by_type = Counter()
    baron_damage_by_team = Counter()
    core_expended = 0
    died_before_core = 0
    alive_at_end = 0
    spawned_count = 0
    killed_count = 0
    no_owner_count = 0

    for match in matches:
        lord = match["lord"]
        if lord["neutral_killed"]:
            killed_count += 1
            kill_ticks.append(lord["kill_tick"])
        if lord["no_owner_after_kill"]:
            no_owner_count += 1
            no_owner_seeds.append(match["seed"])
        owner = lord.get("owner")
        owner_team = lord.get("owner_team")
        if owner:
            spawned_count += 1
            owner_players[owner] += 1
            owner_teams[owner_team] += 1
            owner_team_outcomes[owner_team][match["winner"]] += 1
            target_counts[lord.get("target") or "?"] += 1
            shot_counts.append(lord["shots"])
            lord_damage.append(lord["damage"])
            lord_core_damage.append(lord["core_damage"])
            lord_battle_ticks.append(lord["battle_ticks"])
            enemy_deaths.append(lord["enemy_deaths_during_lord_ticks"])
            continued_counts.append(len(lord["continued_nodes"]))
            if lord["core_expended"]:
                core_expended += 1
            if lord["died_before_core"]:
                died_before_core += 1
            if match["allied_lord_state"]["alive"]:
                alive_at_end += 1
            merge_counter(attack_locations, lord["attack_locations"])
            merge_counter(damage_by_target_type, lord["damage_by_target_type"])
            merge_counter(enemy_deaths_by_type, lord["enemy_deaths_during_lord_ticks_by_type"])
        merge_counter(baron_damage_by_team, lord["baron_damage_by_team"])

    owner_outcomes = {
        team: {
            "winner_counts": counter_dict(counter),
            "owner_team_win_rate": round(counter.get(team, 0) / max(1, sum(counter.values())), 4),
        }
        for team, counter in sorted(owner_team_outcomes.items())
    }

    return {
        "matches": total,
        "winner_counts": counter_dict(winners),
        "winner_type_counts": counter_dict(winner_types),
        "win_rates": {
            key: round(value / max(1, total), 4)
            for key, value in sorted(winners.items())
        },
        "winner_type_rates": {
            key: round(value / max(1, total), 4)
            for key, value in sorted(winner_types.items())
        },
        "ticks": numeric_stats([match["ticks"] for match in matches]),
        "core_combat_damage": numeric_stats([match["core_combat_damage"] for match in matches]),
        "lord": {
            "neutral_lord_killed": killed_count,
            "neutral_lord_kill_rate": round(killed_count / max(1, total), 4),
            "spawned_after_kill": spawned_count,
            "spawn_rate": round(spawned_count / max(1, total), 4),
            "killed_but_no_owner": no_owner_count,
            "killed_but_no_owner_rate": round(no_owner_count / max(1, total), 4),
            "killed_but_no_owner_seeds_sample": no_owner_seeds[:20],
            "first_kill_tick": numeric_stats(kill_ticks),
            "owner_players": counter_dict(owner_players),
            "owner_teams": counter_dict(owner_teams),
            "targets": counter_dict(target_counts),
            "owner_team_outcomes": owner_outcomes,
            "core_expended": core_expended,
            "core_expended_rate_per_spawn": round(core_expended / max(1, spawned_count), 4),
            "died_before_core": died_before_core,
            "died_before_core_rate_per_spawn": round(died_before_core / max(1, spawned_count), 4),
            "alive_at_end": alive_at_end,
            "alive_at_end_rate_per_spawn": round(alive_at_end / max(1, spawned_count), 4),
            "shots": numeric_stats(shot_counts),
            "damage": numeric_stats(lord_damage),
            "core_damage": numeric_stats(lord_core_damage),
            "battle_ticks": numeric_stats(lord_battle_ticks),
            "continued_node_count": numeric_stats(continued_counts),
            "enemy_deaths_during_lord_ticks": numeric_stats(enemy_deaths),
            "attack_locations": counter_dict(attack_locations),
            "damage_by_target_type": counter_dict(damage_by_target_type),
            "enemy_deaths_during_lord_ticks_by_type": counter_dict(enemy_deaths_by_type),
            "baron_damage_by_team": counter_dict(baron_damage_by_team),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze random-vs-random lord and match stats")
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = [int(args.seed) + index for index in range(int(args.matches))]
    started_at = time.perf_counter()
    if int(args.workers) <= 1:
        matches = run_chunk(seeds)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(int(args.workers), len(seeds))) as pool:
            nested = pool.map(run_chunk, chunks(seeds, int(args.workers)))
        matches = [item for chunk in nested for item in chunk]
        matches.sort(key=lambda item: item["seed"])

    summary = aggregate(matches)
    payload = {
        "kind": "random_lord_stats",
        "config": {
            "matches": int(args.matches),
            "seed": int(args.seed),
            "workers": int(args.workers),
            "code_commit": current_code_commit(),
            "rules": rules_meta(),
        },
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "matches": matches,
    }

    out = Path(args.out or f"analysis/random_lord_stats_{int(args.seed)}_{int(args.matches)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    compact = {
        "kind": payload["kind"],
        "out": str(out),
        "elapsed_sec": payload["elapsed_sec"],
        "config": payload["config"],
        "summary": summary,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
