from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot
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
        if bot_type == "scripted":
            bots[player_id] = ScriptedBot(player_id, team)
        else:
            bots[player_id] = RandomBot(player_id, team, seed=index + seed_offset)
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
) -> MatchResult:
    profile = generation_profile(mode)
    if profile.save_replay and replay_path is None:
        raise ValueError("replay_path is required in replay generation mode")

    state = make_initial_state(seed=seed)
    bots = make_bots(bot_type, seed_offset=seed * 10)
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
            extra_meta={"generation_mode": profile.mode},
            pretty=pretty_replay,
        )

    return MatchResult(
        seed=seed,
        bot_type=bot_type,
        mode=profile.mode,
        winner=winner,
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
        f"| Ticks: {result.total_ticks} | {replay_part}"
    )
