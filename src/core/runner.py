from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot
from src.bots.policy_bot import PolicyBot, default_policy_checkpoint
from src.core.constants import PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.core.engine import GameEngine
from src.core.replay import save_replay
from src.core.state import GameState, make_initial_state


@dataclass(frozen=True)
class GenerationProfile:
    mode: str
    record_trace: bool
    record_replay: bool
    save_replay: bool
    description: str


GENERATION_PROFILES: dict[str, GenerationProfile] = {
    "replay": GenerationProfile(
        mode="replay",
        record_trace=True,
        record_replay=True,
        save_replay=True,
        description="Classic generation for viewer playback and debugging.",
    ),
    "training": GenerationProfile(
        mode="training",
        record_trace=False,
        record_replay=False,
        save_replay=False,
        description="Headless generation for RL rollouts; simulation rules are unchanged.",
    ),
}


@dataclass
class MatchResult:
    seed: int
    bot_type: str
    mode: str
    winner: str
    winner_type: str
    total_ticks: int
    replay_path: Optional[str]
    replay_ticks_recorded: int
    record_trace: bool
    record_replay: bool
    state: GameState


def make_bots(bot_type: str, seed_offset: int = 0) -> dict:
    bots = {}
    for index, player_id in enumerate(PLAYER_IDS):
        team = "TEAM_A" if player_id in TEAM_A_PLAYERS else "TEAM_B"
        bots[player_id] = make_bot(bot_type, player_id, team, seed=index + seed_offset)
    return bots


def make_bot(model_id: str, player_id: str, team: str, seed: int = 0):
    if model_id == "scripted":
        return ScriptedBot(player_id, team)
    if model_id == "random":
        return RandomBot(player_id, team, seed=seed)
    if model_id in {"ppo_random_v1", "ppo_random_v1_latest"}:
        if model_id == "ppo_random_v1_latest":
            latest = Path("training_runs/sprint2_random_first_real_v1/checkpoints/latest.pt")
            checkpoint = latest if latest.exists() else None
        else:
            checkpoint = default_policy_checkpoint()
        if checkpoint is None:
            raise ValueError("PPO random v1 checkpoint is not available")
        return PolicyBot(player_id, team, checkpoint_path=checkpoint, deterministic=True)
    raise ValueError(f"Unknown bot model: {model_id}")


def make_bots_from_models(player_models: dict[str, str], seed_offset: int = 0) -> dict:
    bots = {}
    for index, player_id in enumerate(PLAYER_IDS):
        team = "TEAM_A" if player_id in TEAM_A_PLAYERS else "TEAM_B"
        model_id = player_models.get(player_id, "random")
        bots[player_id] = make_bot(model_id, player_id, team, seed=index + seed_offset)
    return bots


def generation_profile(mode: str) -> GenerationProfile:
    if mode not in GENERATION_PROFILES:
        choices = ", ".join(sorted(GENERATION_PROFILES))
        raise ValueError(f"Unknown generation mode {mode!r}; expected one of: {choices}")
    return GENERATION_PROFILES[mode]


def run_match(
    seed: int,
    bot_type: str,
    replay_path: Optional[str] = None,
    mode: str = "replay",
    pretty_replay: bool = False,
    player_models: Optional[dict[str, str]] = None,
) -> MatchResult:
    profile = generation_profile(mode)
    if profile.save_replay and replay_path is None:
        raise ValueError("replay_path is required in replay generation mode")

    state = make_initial_state(seed=seed)
    bots = (
        make_bots_from_models(player_models, seed_offset=seed * 10)
        if player_models is not None
        else make_bots(bot_type, seed_offset=seed * 10)
    )
    engine = GameEngine(
        state,
        bots,
        record_trace=profile.record_trace,
        record_replay=profile.record_replay,
    )
    winner = engine.run()

    if profile.save_replay and replay_path:
        save_replay(
            winner=winner,
            total_ticks=state.tick,
            ticks=engine.replay_ticks,
            path=replay_path,
            extra_meta={
                "generation_mode": profile.mode,
                "bot_type": bot_type,
                "player_models": dict(player_models or {}),
                "winner_type": state.winner_type,
                "core_combat_damage": round(state.core_combat_damage, 3),
                "core_damage_by_team": {
                    team: round(damage, 3)
                    for team, damage in sorted(state.core_damage_by_team.items())
                },
                "death_time_final_core_deaths": dict(state.death_time_final_core_deaths),
                "final_core_hp": {
                    player_id: round(player.core_hp, 3)
                    for player_id, player in sorted(state.players.items())
                },
                "allied_lord": {
                    "owner": state.baron.allied_lord_owner,
                    "hp": round(state.baron.allied_lord_hp, 3),
                    "target": state.baron.allied_lord_target,
                    "alive": state.baron.allied_lord_alive,
                    "expended": state.baron.allied_lord_expended,
                },
            },
            pretty=pretty_replay,
        )

    return MatchResult(
        seed=seed,
        bot_type=bot_type,
        mode=profile.mode,
        winner=winner,
        winner_type=state.winner_type or "unknown",
        total_ticks=state.tick,
        replay_path=replay_path if profile.save_replay else None,
        replay_ticks_recorded=len(engine.replay_ticks),
        record_trace=profile.record_trace,
        record_replay=profile.record_replay,
        state=state,
    )


def format_match_result(result: MatchResult) -> str:
    replay_part = f"Replay: {result.replay_path}" if result.replay_path else "Replay: disabled"
    return (
        f"[mode={result.mode} seed={result.seed}] Winner: {result.winner} "
        f"({result.winner_type}) | Ticks: {result.total_ticks} | {replay_part}"
    )
