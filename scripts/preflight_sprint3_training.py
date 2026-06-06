from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig
from src.training.team_ppo_trainer import (
    TeamPPOConfig,
    load_team_checkpoint,
    make_team_optimizers,
    make_team_policies,
    save_team_checkpoint,
)


def _finite_vectors(observations: dict[str, np.ndarray]) -> bool:
    return all(
        vector.shape == (OBSERVATION_DIM,)
        and vector.dtype == np.float32
        and np.isfinite(vector).all()
        for vector in observations.values()
    )


def _env_preflight(
    seed: int,
    opponent_bot: str,
    ally_bot: str,
    max_ticks: int,
    opponent_policy_checkpoint: str | None,
    opponent_policy_deterministic: bool,
) -> dict:
    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        seed=seed,
        opponent_bot=opponent_bot,
        ally_bot=ally_bot,
        opponent_policy_checkpoint=opponent_policy_checkpoint,
        opponent_policy_deterministic=opponent_policy_deterministic,
        max_ticks=max_ticks,
        record_replay=False,
        record_trace=False,
    ))
    observations, info = env.reset(seed=seed)
    masks = env.action_masks()
    sampled_actions = env.sample_legal_actions()
    result = env.step(sampled_actions)

    mask_shapes_ok = all(mask.shape == (ACTION_DIM,) for mask in masks.values())
    mask_hold_ok = all(bool(mask[0]) for mask in masks.values())
    sampled_legal_ok = all(
        0 <= action < ACTION_DIM and bool(masks[player_id][action])
        for player_id, action in sampled_actions.items()
    )

    return {
        "controlled_players": list(observations.keys()),
        "controlled_team": info["controlled_team"],
        "initial_tick": info["tick"],
        "after_one_step_tick": result.info["tick"],
        "engine_tick": env.state.tick,
        "observation_shapes_ok": _finite_vectors(observations) and _finite_vectors(result.observations),
        "mask_shapes_ok": mask_shapes_ok,
        "mask_hold_ok": mask_hold_ok,
        "sampled_legal_ok": sampled_legal_ok,
        "invalid_actions_after_sample": result.info["invalid_actions"],
        "replay_ticks_recorded": len(env.engine.replay_ticks),
        "trace_disabled": not env.engine.record_trace,
        "terminal": result.terminal,
        "truncated": result.truncated,
    }


@torch.no_grad()
def _policy_preflight(seed: int, hidden_size: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = TeamPPOConfig(seed=seed, workers=1, rollout_steps=8, hidden_size=hidden_size)
    policies = make_team_policies(config)
    optimizers = make_team_optimizers(policies, config)
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=seed, max_ticks=30))
    observations, _ = env.reset(seed=seed)
    masks = env.action_masks()

    actions = {}
    values = {}
    entropies = {}
    for player_id, policy in policies.items():
        step = policy.act(observations[player_id], masks[player_id], deterministic=True, device="cpu")
        actions[player_id] = int(step.action)
        values[player_id] = float(step.value)
        entropies[player_id] = float(step.entropy)

    with tempfile.TemporaryDirectory(prefix="sprint3_preflight_") as tmp:
        checkpoint_path = Path(tmp) / "team_preflight.pt"
        save_team_checkpoint(
            checkpoint_path,
            policies,
            optimizers,
            config,
            update=0,
            total_steps=0,
        )
        loaded_policies = make_team_policies(config)
        loaded_optimizers = make_team_optimizers(loaded_policies, config)
        checkpoint = load_team_checkpoint(checkpoint_path, loaded_policies, loaded_optimizers)

    return {
        "controlled_players": list(config.controlled_players),
        "policy_forward_ok": all(0 <= action < ACTION_DIM for action in actions.values()),
        "policy_values_finite": all(np.isfinite(value) for value in values.values()),
        "policy_entropies_finite": all(np.isfinite(value) for value in entropies.values()),
        "argmax_actions": actions,
        "checkpoint_roundtrip_ok": checkpoint["update"] == 0 and checkpoint["total_steps"] == 0,
        "checkpoint_observation_version": checkpoint["observation_version"],
        "checkpoint_observation_dim": checkpoint["observation_dim"],
        "checkpoint_action_dim": checkpoint["action_dim"],
    }


def _checkpoint_compatibility(path: str | None, hidden_size: int) -> dict | None:
    if not path:
        return None
    checkpoint_path = Path(path)
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_hidden_size = int((raw.get("config") or {}).get("hidden_size", hidden_size) or hidden_size)
        config = TeamPPOConfig(hidden_size=checkpoint_hidden_size)
        policies = make_team_policies(config)
        checkpoint = load_team_checkpoint(checkpoint_path, policies, map_location="cpu")
    except Exception as exc:  # compatibility failure is useful preflight data
        return {
            "path": str(checkpoint_path),
            "compatible": False,
            "error": str(exc),
        }
    return {
        "path": str(checkpoint_path),
        "compatible": True,
        "update": int(checkpoint.get("update", -1)),
        "total_steps": int(checkpoint.get("total_steps", 0) or 0),
        "observation_version": checkpoint.get("observation_version"),
        "observation_dim": checkpoint.get("observation_dim"),
        "action_dim": checkpoint.get("action_dim"),
    }


def run_preflight(args: argparse.Namespace) -> dict:
    env_result = _env_preflight(
        seed=args.seed,
        opponent_bot=args.opponent_bot,
        ally_bot=args.ally_bot,
        max_ticks=args.max_ticks,
        opponent_policy_checkpoint=args.opponent_policy_checkpoint,
        opponent_policy_deterministic=args.opponent_policy_mode == "argmax",
    )
    policy_result = _policy_preflight(seed=args.seed, hidden_size=args.hidden_size)
    checkpoint_result = _checkpoint_compatibility(args.checkpoint, args.hidden_size)

    ok = (
        env_result["observation_shapes_ok"]
        and env_result["mask_shapes_ok"]
        and env_result["mask_hold_ok"]
        and env_result["sampled_legal_ok"]
        and env_result["after_one_step_tick"] == 1
        and env_result["engine_tick"] == 1
        and env_result["replay_ticks_recorded"] == 0
        and policy_result["policy_forward_ok"]
        and policy_result["policy_values_finite"]
        and policy_result["policy_entropies_finite"]
        and policy_result["checkpoint_roundtrip_ok"]
    )
    if checkpoint_result is not None:
        ok = ok and checkpoint_result["compatible"]

    return {
        "kind": "sprint3_training_preflight_no_learning",
        "ok": bool(ok),
        "seed": int(args.seed),
        "cpu_count": os.cpu_count(),
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "opponent_bot": args.opponent_bot,
        "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
        "opponent_policy_mode": args.opponent_policy_mode,
        "ally_bot": args.ally_bot,
        "max_ticks": int(args.max_ticks),
        "env": env_result,
        "policy": policy_result,
        "checkpoint": checkpoint_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 3 no-learning training preflight")
    parser.add_argument("--seed", type=int, default=20260604)
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random", "policy"], default="scripted")
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--opponent-policy-checkpoint", default=None)
    parser.add_argument("--opponent-policy-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--checkpoint", default=None, help="Optional team checkpoint compatibility check")
    args = parser.parse_args()
    if args.opponent_bot == "policy" and not args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint is required with --opponent-bot policy")
    if args.opponent_bot != "policy" and args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint requires --opponent-bot policy")

    result = run_preflight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
