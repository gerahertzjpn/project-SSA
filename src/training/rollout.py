from __future__ import annotations

from dataclasses import asdict

import numpy as np

from src.core.constants import TICK_LIMIT
from src.training.env import TrainingEnv, TrainingEnvConfig


def run_random_rollouts(
    episodes: int = 5,
    seed: int = 20260602,
    controlled_player: str = "A1",
    opponent_bot: str = "scripted",
    ally_bot: str = "scripted",
    max_ticks: int | None = None,
) -> dict:
    results = []
    rng = np.random.default_rng(seed)

    for episode in range(episodes):
        config = TrainingEnvConfig(
            controlled_player=controlled_player,
            seed=seed + episode,
            opponent_bot=opponent_bot,
            ally_bot=ally_bot,
            max_ticks=max_ticks or TICK_LIMIT,
            record_replay=False,
            record_trace=False,
        )
        env = TrainingEnv(config)
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        steps = 0
        terminal = False
        truncated = False

        while not (terminal or truncated):
            mask = env.action_mask()
            legal = np.flatnonzero(mask)
            action_index = int(rng.choice(legal)) if len(legal) else 0
            obs, reward, terminal, truncated, info = env.step(action_index)
            total_reward += reward
            steps += 1

        results.append({
            "episode": episode,
            "seed": seed + episode,
            "winner": info["winner"],
            "ticks": info["tick"],
            "steps": steps,
            "return": round(float(total_reward), 6),
            "final_legal_actions": info["legal_actions"],
            "observation_dim": int(obs.shape[0]),
        })

    winners = {}
    for item in results:
        winners[item["winner"]] = winners.get(item["winner"], 0) + 1

    return {
        "kind": "random_rollout_smoke_no_training",
        "episodes": episodes,
        "seed": seed,
        "controlled_player": controlled_player,
        "opponent_bot": opponent_bot,
        "ally_bot": ally_bot,
        "config": asdict(TrainingEnvConfig(
            controlled_player=controlled_player,
            seed=seed,
            opponent_bot=opponent_bot,
            ally_bot=ally_bot,
            max_ticks=max_ticks or TICK_LIMIT,
            record_replay=False,
            record_trace=False,
        )),
        "winners": winners,
        "mean_return": round(float(np.mean([item["return"] for item in results])) if results else 0.0, 6),
        "mean_ticks": round(float(np.mean([item["ticks"] for item in results])) if results else 0.0, 3),
        "results": results,
    }
