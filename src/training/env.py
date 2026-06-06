from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.bots.base import Bot
from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot, make_scripted_bot_variant
from src.core.constants import PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS, TICK_LIMIT, ActionType
from src.core.engine import Action, GameEngine
from src.core.state import make_initial_state
from src.training.spaces import (
    ACTION_DIM,
    OBSERVATION_VERSION,
    decode_action,
    encode_observation,
    legal_action_mask,
    team_of,
)


@dataclass
class TrainingEnvConfig:
    controlled_player: str = "A1"
    seed: int = 42
    opponent_bot: str = "scripted"
    ally_bot: str = "scripted"
    max_ticks: int = TICK_LIMIT
    record_replay: bool = False
    record_trace: bool = False
    terminal_win_reward: float = 1.0
    terminal_loss_reward: float = -1.0
    terminal_draw_reward: float = 0.0
    death_time_win_reward_multiplier: float = 0.5
    shaping_weight: float = 0.02


class ExternalActionBot(Bot):
    requires_observation = False

    def __init__(self, player_id: str, team: str):
        super().__init__(player_id, team)
        self.next_action = Action(ActionType.HOLD)
        self.last_action = self.next_action

    def set_action(self, action: Action) -> None:
        self.next_action = action

    def act(self, obs: dict) -> Action:
        action = self.next_action
        self.last_action = action
        self.next_action = Action(ActionType.HOLD)
        return action


class TrainingEnv:
    """Lightweight single-controlled-player training environment.

    This is deliberately Gymnasium-like without importing Gymnasium. The first
    training run can wrap this class, while tests can exercise it with only
    NumPy installed.
    """

    action_space_n = ACTION_DIM

    def __init__(self, config: Optional[TrainingEnvConfig] = None):
        self.config = config or TrainingEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.state = None
        self.engine: Optional[GameEngine] = None
        self.control_bot: Optional[ExternalActionBot] = None
        self._previous_score = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.config.seed = int(seed)
            self.rng = np.random.default_rng(self.config.seed)

        self.state = make_initial_state(seed=self.config.seed)
        bots = self._make_bots()
        self.engine = GameEngine(
            self.state,
            bots,
            record_trace=self.config.record_trace,
            record_replay=self.config.record_replay,
        )
        self._done = False
        obs = self.raw_observation()
        self._previous_score = self._score(obs)
        return encode_observation(obs, self.config.controlled_player), self._info(obs, reward_components={})

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.engine is None:
            self.reset()
        if self._done:
            obs = self.raw_observation()
            return encode_observation(obs, self.config.controlled_player), 0.0, True, False, self._info(obs, {})

        obs_before = self.raw_observation()
        action = decode_action(action_index, obs_before, self.config.controlled_player)
        assert self.control_bot is not None
        self.control_bot.set_action(action)

        self.engine.tick()
        if self.state.winner is None and self.state.tick >= TICK_LIMIT:
            self.engine.run()
        elif self.state.tick >= self.config.max_ticks and self.state.winner is None:
            self.state.winner = "DRAW"
            self.state.winner_type = "truncated"

        obs_after = self.raw_observation()
        shaped_score = self._score(obs_after)
        shaping = (shaped_score - self._previous_score) * self.config.shaping_weight
        self._previous_score = shaped_score

        terminal = self.state.winner in {"TEAM_A", "TEAM_B"} or (
            self.state.winner == "DRAW" and self.state.winner_type != "truncated"
        )
        truncated = self.state.winner_type == "truncated"
        terminal_reward = self._terminal_reward()
        reward = float(shaping + terminal_reward)
        self._done = terminal or truncated

        info = self._info(
            obs_after,
            reward_components={
                "shaping": float(shaping),
                "terminal": float(terminal_reward),
                "score": float(shaped_score),
            },
        )
        info["decoded_action"] = {
            "type": action.action_type.value,
            "source": action.source,
            "destination": action.destination,
            "intent": action.intent,
            "unit_type": action.unit_type.value if action.unit_type else None,
            "building_type": action.building_type.value if action.building_type else None,
        }
        info["action_index"] = int(action_index)
        return encode_observation(obs_after, self.config.controlled_player), reward, terminal, truncated, info

    def raw_observation(self) -> dict:
        if self.engine is None:
            raise RuntimeError("TrainingEnv.reset() must be called before observation")
        return self.engine.observe(self.config.controlled_player)

    def action_mask(self) -> np.ndarray:
        return legal_action_mask(self.raw_observation(), self.config.controlled_player)

    def sample_legal_action(self) -> int:
        mask = self.action_mask()
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return 0
        return int(self.rng.choice(legal))

    def _make_bots(self) -> dict:
        bots = {}
        controlled_team = team_of(self.config.controlled_player)
        for index, player_id in enumerate(PLAYER_IDS):
            team = "TEAM_A" if player_id in TEAM_A_PLAYERS else "TEAM_B"
            if player_id == self.config.controlled_player:
                self.control_bot = ExternalActionBot(player_id, team)
                bots[player_id] = self.control_bot
            elif team == controlled_team:
                bots[player_id] = self._make_baseline_bot(player_id, team, self.config.ally_bot, index)
            else:
                bots[player_id] = self._make_baseline_bot(player_id, team, self.config.opponent_bot, index)
        return bots

    def _make_baseline_bot(self, player_id: str, team: str, bot_type: str, index: int) -> Bot:
        if bot_type == "random":
            return RandomBot(player_id, team, seed=self.config.seed * 10 + index)
        scripted_variant = make_scripted_bot_variant(bot_type, player_id, team)
        if scripted_variant is not None:
            return scripted_variant
        raise ValueError(f"Unknown baseline bot type: {bot_type}")

    def _terminal_reward(self) -> float:
        winner = self.state.winner
        if winner is None:
            return 0.0
        if winner == "DRAW":
            return self.config.terminal_draw_reward
        return (
            self._terminal_win_reward()
            if winner == team_of(self.config.controlled_player)
            else self.config.terminal_loss_reward
        )

    def _terminal_win_reward(self) -> float:
        reward = self.config.terminal_win_reward
        if self.state.winner_type == "death_time":
            reward *= self.config.death_time_win_reward_multiplier
        return reward

    def _score(self, obs: dict) -> float:
        own_team = team_of(self.config.controlled_player)
        control = 0.0
        for node in (obs.get("nodes", {}) or {}).values():
            owner = node.get("owner", "NEUTRAL")
            if owner == "NEUTRAL":
                continue
            control += 1.0 if team_of(owner) == own_team else -1.0

        resources = 0.0
        alive = 0.0
        for player_id, player in (obs.get("players", {}) or {}).items():
            if not player.get("alive", True):
                alive += 1.0 if team_of(player_id) != own_team else -1.0
                continue
            sign = 1.0 if team_of(player_id) == own_team else -1.0
            resources += sign * float(player.get("resources", 0.0) or 0.0) / 100.0

        return control / 13.0 + resources / 6.0 + alive / 3.0

    def _info(self, obs: dict, reward_components: dict) -> dict:
        mask = legal_action_mask(obs, self.config.controlled_player)
        return {
            "tick": int(obs.get("tick", 0) or 0),
            "winner": self.state.winner if self.state is not None else None,
            "winner_type": self.state.winner_type if self.state is not None else None,
            "controlled_player": self.config.controlled_player,
            "legal_actions": int(mask.sum()),
            "reward_components": reward_components,
            "observation_version": OBSERVATION_VERSION,
        }
