from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.policy import MaskedActorCritic
from src.training.ppo_trainer import summarize_action_distribution, summarize_training_episodes
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION, team_of
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig
from src.training.team_ppo_trainer import _team_episode_summary


CONTROLLED_PLAYERS = ("A1", "A2", "A3")
_CHECKPOINT_CACHE: dict[str, dict[str, Any]] = {}


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one value")
    return items


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def checkpoint_label(path: str | Path) -> str:
    path = Path(path)
    run = path.parents[1].name if len(path.parents) >= 2 else path.parent.name
    stem = path.stem.replace("checkpoint_update_", "u")
    return f"{run}_{stem}"


def load_team_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    cache_key = str(checkpoint_path.resolve())
    if cache_key not in _CHECKPOINT_CACHE:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("observation_version") != OBSERVATION_VERSION
            or int(checkpoint.get("observation_dim") or -1) != OBSERVATION_DIM
            or int(checkpoint.get("action_dim") or -1) != ACTION_DIM
        ):
            raise ValueError(
                f"Incompatible checkpoint {checkpoint_path}: "
                f"{checkpoint.get('observation_version')}, "
                f"{checkpoint.get('observation_dim')}, {checkpoint.get('action_dim')}"
            )
        policy_states = checkpoint.get("policy_states")
        if not isinstance(policy_states, dict):
            raise ValueError(f"Expected team checkpoint with policy_states: {checkpoint_path}")
        hidden_size = int((checkpoint.get("config") or {}).get("hidden_size", 128) or 128)
        _CHECKPOINT_CACHE[cache_key] = {
            "path": str(checkpoint_path),
            "label": checkpoint_label(checkpoint_path),
            "update": int(checkpoint.get("update", -1)),
            "total_steps": int(checkpoint.get("total_steps", 0) or 0),
            "hidden_size": hidden_size,
            "policy_states": policy_states,
        }
    return _CHECKPOINT_CACHE[cache_key]


def slot_assignment(checkpoint_path: str | Path, source_player: str) -> dict[str, str]:
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_label": checkpoint_label(checkpoint_path),
        "source_player": source_player,
    }


def native_team(checkpoint_path: str | Path) -> dict[str, Any]:
    label = checkpoint_label(checkpoint_path)
    return {
        "name": f"native__{label}",
        "kind": "native",
        "slots": {
            player_id: slot_assignment(checkpoint_path, player_id)
            for player_id in CONTROLLED_PLAYERS
        },
    }


def mixed_team(name: str, checkpoints: tuple[str, str, str], *, kind: str = "slot_mixed") -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "slots": {
            player_id: slot_assignment(checkpoint, player_id)
            for player_id, checkpoint in zip(CONTROLLED_PLAYERS, checkpoints)
        },
    }


def role_stress_team(checkpoint_path: str | Path) -> dict[str, Any]:
    label = checkpoint_label(checkpoint_path)
    return {
        "name": f"role_stress__{label}__A1_from_A2__A2_from_A3__A3_from_A1",
        "kind": "role_stress",
        "slots": {
            "A1": slot_assignment(checkpoint_path, "A2"),
            "A2": slot_assignment(checkpoint_path, "A3"),
            "A3": slot_assignment(checkpoint_path, "A1"),
        },
    }


def build_swap_suite(
    checkpoints: list[str],
    *,
    suite_kinds: set[str],
    max_teams: int = 0,
) -> list[dict[str, Any]]:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")

    teams: list[dict[str, Any]] = []
    if "native" in suite_kinds:
        teams.extend(native_team(path) for path in checkpoints)

    if "slot_mixed" in suite_kinds:
        if len(checkpoints) < 3:
            raise ValueError("slot_mixed suite requires at least three checkpoints")
        for index, checkpoint in enumerate(checkpoints):
            second = checkpoints[(index + 1) % len(checkpoints)]
            third = checkpoints[(index + 2) % len(checkpoints)]
            teams.append(mixed_team(
                "slot_mixed__"
                f"A1_{checkpoint_label(checkpoint)}__"
                f"A2_{checkpoint_label(second)}__"
                f"A3_{checkpoint_label(third)}",
                (checkpoint, second, third),
            ))
        for index, checkpoint in enumerate(checkpoints):
            second = checkpoints[(index + 2) % len(checkpoints)]
            third = checkpoints[(index + 1) % len(checkpoints)]
            teams.append(mixed_team(
                "slot_mixed_alt__"
                f"A1_{checkpoint_label(checkpoint)}__"
                f"A2_{checkpoint_label(second)}__"
                f"A3_{checkpoint_label(third)}",
                (checkpoint, second, third),
            ))

    if "role_stress" in suite_kinds:
        teams.extend(role_stress_team(path) for path in checkpoints)

    if max_teams > 0:
        natives = [team for team in teams if team["kind"] == "native"]
        others = [team for team in teams if team["kind"] != "native"]
        teams = (natives + others)[:max_teams]
    return teams


def load_policies_for_team(team_spec: dict[str, Any]) -> dict[str, MaskedActorCritic]:
    policies: dict[str, MaskedActorCritic] = {}
    for target_player, assignment in team_spec["slots"].items():
        checkpoint = load_team_checkpoint(assignment["checkpoint"])
        source_player = assignment["source_player"]
        policy_state = checkpoint["policy_states"][source_player]
        policy = MaskedActorCritic(hidden_size=int(checkpoint["hidden_size"]))
        policy.load_state_dict(policy_state)
        policy.eval()
        policies[target_player] = policy
    return policies


def _action_payload(action) -> list[int]:
    action_array = np.asarray(action, dtype=np.int64).reshape(-1)
    return [int(value) for value in action_array]


def sequence_hash(actions: list) -> str:
    payload = json.dumps(
        [_action_payload(action) for action in actions],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha1(payload).hexdigest()[:12]


def episode_specs(seed_bases: list[int], episodes_per_seed: int) -> list[dict[str, int]]:
    return [
        {"seed_block": int(seed_base), "env_seed": int(seed_base + episode_index)}
        for seed_base in seed_bases
        for episode_index in range(episodes_per_seed)
    ]


def chunks(items: list[Any], chunk_count: int) -> list[list[Any]]:
    chunk_count = max(1, min(int(chunk_count), len(items)))
    return [items[index::chunk_count] for index in range(chunk_count)]


def team_swap_eval_worker(request: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(request["torch_threads"]))
    np.random.seed(int(request["worker_seed"]))
    torch.manual_seed(int(request["worker_seed"]))

    team_spec = request["team_spec"]
    policies = load_policies_for_team(team_spec)
    controlled_players = tuple(request["controlled_players"])
    controlled_team = team_of(controlled_players[0])

    episodes = []
    actions_by_player = {player_id: [] for player_id in controlled_players}
    masks_by_player = {player_id: [] for player_id in controlled_players}
    entropies_by_player = {player_id: [] for player_id in controlled_players}
    sequence_hashes = []
    opening_hashes = []

    for spec in request["episodes"]:
        env_seed = int(spec["env_seed"])
        sample_seed = int(request["sample_seed_offset"]) + env_seed
        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed % (2**32 - 1))

        env = TeamTrainingEnv(TeamTrainingEnvConfig(
            controlled_players=controlled_players,
            seed=env_seed,
            opponent_bot=str(request["opponent_bot"]),
            ally_bot=str(request["ally_bot"]),
            opponent_policy_checkpoint=request.get("opponent_policy_checkpoint"),
            opponent_policy_deterministic=bool(request["opponent_policy_deterministic"]),
            max_ticks=int(request["max_ticks"]),
            record_replay=False,
            record_trace=False,
        ))
        observations, _ = env.reset(seed=env_seed)
        return_sums = {player_id: 0.0 for player_id in controlled_players}
        shaping_sums = {player_id: 0.0 for player_id in controlled_players}
        terminal_sums = {player_id: 0.0 for player_id in controlled_players}
        episode_actions: list = []
        steps = 0
        done = False

        while not done:
            masks = env.action_masks()
            selected_actions = {}
            for player_id in controlled_players:
                step = policies[player_id].act(
                    observations[player_id],
                    masks[player_id],
                    deterministic=bool(request["deterministic"]),
                    device="cpu",
                )
                selected_actions[player_id] = step.action
                actions_by_player[player_id].append(step.action)
                masks_by_player[player_id].append(masks[player_id])
                entropies_by_player[player_id].append(float(step.entropy))
                episode_actions.append(step.action)

            result = env.step(selected_actions)
            for player_id in controlled_players:
                reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
                return_sums[player_id] += float(result.rewards[player_id])
                shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
                terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)
            observations = result.observations
            steps += 1
            done = bool(result.terminal or result.truncated)

        episode = _team_episode_summary(
            return_sums=return_sums,
            shaping_sums=shaping_sums,
            terminal_sums=terminal_sums,
            steps=steps,
            winner=result.info.get("winner") or "DRAW",
            winner_type=result.info.get("winner_type") or "unknown",
            controlled_team=controlled_team,
        )
        episode["seed_block"] = int(spec["seed_block"])
        episode["env_seed"] = env_seed
        episode["sequence_hash"] = sequence_hash(episode_actions)
        episode["opening_hash"] = sequence_hash(episode_actions[:60])
        episodes.append(episode)
        sequence_hashes.append(episode["sequence_hash"])
        opening_hashes.append(episode["opening_hash"])

    return {
        "episodes": episodes,
        "actions_by_player": actions_by_player,
        "masks_by_player": masks_by_player,
        "entropies_by_player": entropies_by_player,
        "sequence_hashes": sequence_hashes,
        "opening_hashes": opening_hashes,
    }


def pool_map(requests: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(requests) <= 1:
        return [team_swap_eval_worker(request) for request in requests]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(requests))) as pool:
        return pool.map(team_swap_eval_worker, requests)


def add_counts(summary: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    winners: dict[str, int] = {}
    winner_types: dict[str, int] = {}
    for item in episodes:
        winners[item["winner"]] = winners.get(item["winner"], 0) + 1
        winner_types[item["winner_type"]] = winner_types.get(item["winner_type"], 0) + 1
    summary["winner_counts"] = dict(sorted(winners.items()))
    summary["winner_type_counts"] = dict(sorted(winner_types.items()))


def seed_block_summaries(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for seed_block in sorted({int(item["seed_block"]) for item in episodes}):
        block_items = [item for item in episodes if int(item["seed_block"]) == seed_block]
        summary = summarize_training_episodes(block_items)
        add_counts(summary, block_items)
        result[str(seed_block)] = summary
    return result


def aggregate_eval(
    results: list[dict[str, Any]],
    *,
    mode: str,
    controlled_players: tuple[str, ...],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    actions_by_player = {player_id: [] for player_id in controlled_players}
    masks_by_player = {player_id: [] for player_id in controlled_players}
    entropies_by_player = {player_id: [] for player_id in controlled_players}
    sequence_hashes: list[str] = []
    opening_hashes: list[str] = []
    for result in results:
        episodes.extend(result.get("episodes", []))
        sequence_hashes.extend(result.get("sequence_hashes", []))
        opening_hashes.extend(result.get("opening_hashes", []))
        for player_id in controlled_players:
            actions_by_player[player_id].extend(result.get("actions_by_player", {}).get(player_id, []))
            masks_by_player[player_id].extend(result.get("masks_by_player", {}).get(player_id, []))
            entropies_by_player[player_id].extend(result.get("entropies_by_player", {}).get(player_id, []))

    summary = summarize_training_episodes(episodes)
    add_counts(summary, episodes)
    summary["mode"] = mode
    summary["seed_blocks"] = seed_block_summaries(episodes)
    summary["agents"] = {}
    for player_id in controlled_players:
        summary["agents"][player_id] = {
            "action_distribution": summarize_action_distribution(
                actions_by_player[player_id],
                masks_by_player[player_id],
            ),
            "mean_policy_entropy": round(float(np.mean(entropies_by_player[player_id])), 6)
            if entropies_by_player[player_id]
            else 0.0,
        }
    if sequence_hashes:
        summary["unique_action_sequences"] = len(set(sequence_hashes))
        summary["sequence_diversity_rate"] = round(len(set(sequence_hashes)) / max(1, len(sequence_hashes)), 6)
    if opening_hashes:
        summary["unique_opening_sequences"] = len(set(opening_hashes))
        summary["opening_diversity_rate"] = round(len(set(opening_hashes)) / max(1, len(opening_hashes)), 6)
    return summary


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate non-native teammate combinations")
    parser.add_argument("--checkpoints", required=True, help="Comma-separated compatible team checkpoints")
    parser.add_argument("--suite", default="native,slot_mixed", help="native,slot_mixed,role_stress")
    parser.add_argument("--max-teams", type=int, default=0)
    parser.add_argument("--modes", default="argmax,stochastic")
    parser.add_argument("--seed-bases", default="20260604,20270604,20280604,20290604")
    parser.add_argument("--episodes-per-seed", type=int, default=25)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=1)
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random", "policy"], default="scripted")
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--opponent-policy-checkpoint", default=None)
    parser.add_argument("--opponent-policy-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.opponent_bot == "policy" and not args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint is required with --opponent-bot policy")
    if args.opponent_bot != "policy" and args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint requires --opponent-bot policy")

    checkpoints = parse_csv(args.checkpoints)
    for checkpoint in checkpoints:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        load_team_checkpoint(checkpoint)

    suite_kinds = set(parse_csv(args.suite))
    allowed = {"native", "slot_mixed", "role_stress"}
    unknown = suite_kinds - allowed
    if unknown:
        raise ValueError(f"Unknown suite kinds: {sorted(unknown)}")

    teams = build_swap_suite(checkpoints, suite_kinds=suite_kinds, max_teams=int(args.max_teams))
    modes = parse_csv(args.modes)
    seed_bases = parse_csv_ints(args.seed_bases)
    specs = episode_specs(seed_bases, int(args.episodes_per_seed))
    specs_chunks = chunks(specs, int(args.workers))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"
    if metrics_path.exists():
        metrics_path.unlink()

    config_record = {
        "kind": "teammate_swap_eval_started",
        "out_dir": str(out_dir),
        "checkpoints": checkpoints,
        "suite": sorted(suite_kinds),
        "teams": teams,
        "modes": modes,
        "seed_bases": seed_bases,
        "episodes_per_seed": int(args.episodes_per_seed),
        "total_episodes_per_eval": len(specs),
        "workers": int(args.workers),
        "opponent_bot": args.opponent_bot,
        "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
        "opponent_policy_mode": args.opponent_policy_mode,
        "ally_bot": args.ally_bot,
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
    }
    print(json.dumps(config_record, ensure_ascii=False, sort_keys=True), flush=True)
    write_jsonl(metrics_path, config_record)

    records: list[dict[str, Any]] = []
    for team_index, team_spec in enumerate(teams):
        for mode in modes:
            deterministic = mode == "argmax"
            if mode not in {"argmax", "stochastic"}:
                raise ValueError(f"Unknown mode: {mode}")
            requests = []
            for worker_index, chunk in enumerate(specs_chunks):
                requests.append({
                    "team_spec": team_spec,
                    "episodes": chunk,
                    "worker_seed": seed_bases[0] + team_index * 100_003 + worker_index * 9_973,
                    "sample_seed_offset": seed_bases[0] + team_index * 1_000_003,
                    "deterministic": deterministic,
                    "controlled_players": CONTROLLED_PLAYERS,
                    "opponent_bot": args.opponent_bot,
                    "ally_bot": args.ally_bot,
                    "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
                    "opponent_policy_deterministic": args.opponent_policy_mode == "argmax",
                    "max_ticks": int(args.max_ticks),
                    "torch_threads": int(args.torch_threads),
                })
            started_at = time.perf_counter()
            summary = aggregate_eval(
                pool_map(requests, int(args.workers)),
                mode=mode,
                controlled_players=CONTROLLED_PLAYERS,
            )
            record = {
                "kind": "teammate_swap_team_eval",
                "team": team_spec["name"],
                "team_kind": team_spec["kind"],
                "team_spec": team_spec,
                "mode": mode,
                "deterministic": deterministic,
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
                "summary": summary,
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
            write_jsonl(metrics_path, record)

    final = {
        "config": config_record,
        "records": records,
        "metrics_path": str(metrics_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({
        "kind": "teammate_swap_eval_finished",
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "records": len(records),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
