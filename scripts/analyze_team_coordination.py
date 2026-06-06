import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.ppo_trainer import summarize_action_distribution
from src.training.spaces import ACTION_SPECS, movable_sources, team_of
from src.training.team_env import TeamTrainingEnv
from src.training.team_ppo_trainer import TeamPPOConfig, _env_config, make_team_policies


def parse_players(value: str) -> tuple[str, ...]:
    players = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not players:
        raise argparse.ArgumentTypeError("Expected at least one player id")
    return players


def load_policies(checkpoint_path: Path, config: TeamPPOConfig) -> tuple[dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {}) or {}
    config.hidden_size = int(checkpoint_config.get("hidden_size", config.hidden_size))
    policies = make_team_policies(config)
    for player_id, policy_state in checkpoint["policy_states"].items():
        if player_id in policies:
            policies[player_id].load_state_dict(policy_state)
            policies[player_id].eval()
    return policies, checkpoint


def active_node_battle_locations(env: TeamTrainingEnv) -> set[str]:
    return {
        battle.location_id
        for battle in env.state.active_battles.values()
        if battle.location_kind == "node"
    }


def moving_unit_count_for_team(env: TeamTrainingEnv, team: str) -> int:
    return sum(
        int(group.total_units())
        for group in env.state.moving_groups
        if group.team == team
    )


def choose_policy_actions(policies: dict, env: TeamTrainingEnv, observations: dict, deterministic: bool) -> tuple[dict, dict, dict]:
    masks = env.action_masks()
    actions = {}
    entropies = {}
    for player_id, policy in policies.items():
        step = policy.act(observations[player_id], masks[player_id], deterministic=deterministic)
        actions[player_id] = int(step.action)
        entropies[player_id] = float(step.entropy)
    return actions, masks, entropies


def analyze_episode(policies: dict, config: TeamPPOConfig, seed: int, deterministic: bool) -> dict:
    env = TeamTrainingEnv(_env_config(config, seed))
    observations, _ = env.reset(seed=seed)
    controlled_team = env.controlled_team
    action_counts = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}
    entropies = {player_id: [] for player_id in config.controlled_players}
    duplicate_target_ticks = 0
    move_ticks = 0
    node_battle_move_ticks = 0
    idle_with_units = Counter()
    move_destinations = Counter()
    unique_move_destinations = []
    lord_kill_tick = None
    lord_owner = None
    lord_spawn_tick = None
    lord_expended_tick = None
    lord_alive_ticks = 0
    done = False

    while not done:
        raw = env.raw_observations()
        battle_nodes = active_node_battle_locations(env)
        actions, masks, step_entropies = choose_policy_actions(policies, env, observations, deterministic)
        destinations = []
        moved_to_battle = False
        for player_id, action_index in actions.items():
            spec = ACTION_SPECS[int(action_index)]
            action_counts[player_id].append(int(action_index))
            masks_by_player[player_id].append(masks[player_id])
            entropies[player_id].append(step_entropies[player_id])
            if spec.kind == "HOLD" and movable_sources(raw[player_id], player_id):
                idle_with_units[player_id] += 1
            if spec.kind == "MOVE" and spec.destination:
                destinations.append(spec.destination)
                move_destinations[spec.destination] += 1
                if spec.destination in battle_nodes:
                    moved_to_battle = True

        if destinations:
            move_ticks += 1
            unique_move_destinations.append(len(set(destinations)))
            if len(destinations) != len(set(destinations)):
                duplicate_target_ticks += 1
        if moved_to_battle:
            node_battle_move_ticks += 1

        result = env.step(actions)
        state = env.state
        if lord_kill_tick is None and state.baron.hp <= 0:
            lord_kill_tick = int(state.tick)
        if lord_spawn_tick is None and state.baron.allied_lord_owner:
            lord_spawn_tick = int(state.tick)
            lord_owner = state.baron.allied_lord_owner
        if state.baron.allied_lord_alive:
            lord_alive_ticks += 1
        if lord_expended_tick is None and state.baron.allied_lord_expended:
            lord_expended_tick = int(state.tick)

        observations = result.observations
        done = bool(result.terminal or result.truncated)

    final = env.state
    return {
        "seed": int(seed),
        "winner": final.winner or "DRAW",
        "winner_type": final.winner_type or "unknown",
        "win": 1 if final.winner == controlled_team else 0,
        "loss": 1 if final.winner not in {controlled_team, "DRAW", None} else 0,
        "draw": 1 if final.winner in {"DRAW", None} else 0,
        "ticks": int(final.tick),
        "duplicate_target_ticks": int(duplicate_target_ticks),
        "move_ticks": int(move_ticks),
        "duplicate_target_rate": round(duplicate_target_ticks / max(1, move_ticks), 6),
        "mean_unique_move_destinations": round(float(np.mean(unique_move_destinations)) if unique_move_destinations else 0.0, 6),
        "node_battle_move_ticks": int(node_battle_move_ticks),
        "node_battle_move_rate": round(node_battle_move_ticks / max(1, move_ticks), 6),
        "idle_with_units": dict(idle_with_units),
        "moving_team_units_final": moving_unit_count_for_team(env, controlled_team),
        "lord": {
            "neutral_killed": lord_kill_tick is not None,
            "kill_tick": lord_kill_tick,
            "spawned": lord_spawn_tick is not None,
            "spawn_tick": lord_spawn_tick,
            "owner": lord_owner,
            "owner_team": team_of(lord_owner) if lord_owner else None,
            "target": final.baron.allied_lord_target,
            "alive_final": bool(final.baron.allied_lord_alive),
            "expended": bool(final.baron.allied_lord_expended),
            "expended_tick": lord_expended_tick,
            "alive_ticks": int(lord_alive_ticks),
            "hp_final": round(float(final.baron.allied_lord_hp), 3),
        },
        "core_damage_by_team": {
            team: round(float(value), 3)
            for team, value in sorted(final.core_damage_by_team.items())
        },
        "final_core_hp": {
            player_id: round(float(player.core_hp), 3)
            for player_id, player in sorted(final.players.items())
        },
        "actions": {
            player_id: summarize_action_distribution(action_counts[player_id], masks_by_player[player_id])
            for player_id in config.controlled_players
        },
        "mean_entropy": {
            player_id: round(float(np.mean(values)) if values else 0.0, 6)
            for player_id, values in entropies.items()
        },
        "top_move_destinations": dict(move_destinations.most_common(8)),
    }


def aggregate(episodes: list[dict]) -> dict:
    winners = Counter(item["winner"] for item in episodes)
    winner_types = Counter(item["winner_type"] for item in episodes)
    lord_owner_teams = Counter(
        item["lord"]["owner_team"] or "NONE"
        for item in episodes
        if item["lord"]["neutral_killed"]
    )
    return {
        "episodes": len(episodes),
        "win_rate": round(sum(item["win"] for item in episodes) / max(1, len(episodes)), 6),
        "loss_rate": round(sum(item["loss"] for item in episodes) / max(1, len(episodes)), 6),
        "draw_rate": round(sum(item["draw"] for item in episodes) / max(1, len(episodes)), 6),
        "winner_counts": dict(sorted(winners.items())),
        "winner_type_counts": dict(sorted(winner_types.items())),
        "mean_ticks": round(float(np.mean([item["ticks"] for item in episodes])) if episodes else 0.0, 3),
        "duplicate_target_rate": round(float(np.mean([item["duplicate_target_rate"] for item in episodes])) if episodes else 0.0, 6),
        "mean_unique_move_destinations": round(float(np.mean([item["mean_unique_move_destinations"] for item in episodes])) if episodes else 0.0, 6),
        "node_battle_move_rate": round(float(np.mean([item["node_battle_move_rate"] for item in episodes])) if episodes else 0.0, 6),
        "lord": {
            "neutral_kill_rate": round(sum(1 for item in episodes if item["lord"]["neutral_killed"]) / max(1, len(episodes)), 6),
            "spawn_rate": round(sum(1 for item in episodes if item["lord"]["spawned"]) / max(1, len(episodes)), 6),
            "expended_rate": round(sum(1 for item in episodes if item["lord"]["expended"]) / max(1, len(episodes)), 6),
            "owner_team_counts": dict(sorted(lord_owner_teams.items())),
            "mean_kill_tick": round(float(np.mean([
                item["lord"]["kill_tick"]
                for item in episodes
                if item["lord"]["kill_tick"] is not None
            ])), 3) if any(item["lord"]["kill_tick"] is not None for item in episodes) else None,
            "mean_alive_ticks": round(float(np.mean([item["lord"]["alive_ticks"] for item in episodes])) if episodes else 0.0, 3),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze team-policy coordination and Lord impact")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--mode", choices=["argmax", "stochastic"], default="argmax")
    parser.add_argument("--controlled-players", type=parse_players, default=("A1", "A2", "A3"))
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    config = TeamPPOConfig(
        controlled_players=tuple(args.controlled_players),
        opponent_bot=args.opponent_bot,
        ally_bot=args.ally_bot,
        max_ticks=int(args.max_ticks),
        workers=1,
    )
    policies, checkpoint = load_policies(checkpoint_path, config)
    deterministic = args.mode == "argmax"
    episodes = [
        analyze_episode(policies, config, seed=int(args.seed) + index, deterministic=deterministic)
        for index in range(int(args.episodes))
    ]
    report = {
        "kind": "team_coordination_analysis",
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": int(checkpoint.get("update", -1)),
        "checkpoint_total_steps": int(checkpoint.get("total_steps", 0) or 0),
        "mode": args.mode,
        "seed": int(args.seed),
        "config": {
            "controlled_players": list(config.controlled_players),
            "opponent_bot": config.opponent_bot,
            "ally_bot": config.ally_bot,
            "max_ticks": config.max_ticks,
        },
        "summary": aggregate(episodes),
        "episodes": episodes,
    }
    out = Path(args.out or f"analysis/team_coordination_{args.mode}_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({
        "kind": "team_coordination_analysis_finished",
        "out": str(out),
        "summary": report["summary"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
