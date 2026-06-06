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
from src.training.env import ExternalActionBot
from src.training.spaces import (
    ACTION_DIM,
    OBSERVATION_VERSION,
    decode_action,
    encode_observation,
    legal_action_mask,
    team_of,
)


DEFAULT_CONTROLLED_PLAYERS = ("A1", "A2", "A3")


@dataclass
class TeamTrainingEnvConfig:
    controlled_players: tuple[str, ...] = DEFAULT_CONTROLLED_PLAYERS
    seed: int = 42
    opponent_bot: str = "scripted"
    ally_bot: str = "scripted"
    opponent_policy_checkpoint: Optional[str] = None
    opponent_policy_deterministic: bool = False
    max_ticks: int = TICK_LIMIT
    record_replay: bool = False
    record_trace: bool = False
    terminal_win_reward: float = 1.0
    terminal_loss_reward: float = -1.0
    terminal_draw_reward: float = 0.0
    death_time_win_reward_multiplier: float = 0.5
    shaping_weight: float = 0.02
    per_player_shaping: bool = False

    def __post_init__(self) -> None:
        self.controlled_players = tuple(self.controlled_players)
        if not self.controlled_players:
            raise ValueError("TeamTrainingEnv requires at least one controlled player")
        teams = {team_of(player_id) for player_id in self.controlled_players}
        if len(teams) != 1 or "NEUTRAL" in teams:
            raise ValueError("All controlled players must be real players on the same team")


@dataclass
class TeamStepResult:
    observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    terminal: bool
    truncated: bool
    info: dict


class TeamTrainingEnv:
    """Synchronized multi-agent training environment for one controlled team.

    `step()` accepts one action per controlled player, installs all actions into
    external bots, and advances the engine by exactly one world tick. This keeps
    3v3 training from accidentally becoming three sequential single-agent turns.
    """

    action_space_n = ACTION_DIM

    def __init__(self, config: Optional[TeamTrainingEnvConfig] = None):
        self.config = config or TeamTrainingEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.state = None
        self.engine: Optional[GameEngine] = None
        self.control_bots: dict[str, ExternalActionBot] = {}
        self._previous_team_score = 0.0
        self._previous_player_scores: dict[str, float] = {}
        self._done = False
        self._cache_tick: Optional[int] = None
        self._raw_cache: Optional[dict[str, dict]] = None
        self._mask_cache: Optional[dict[str, np.ndarray]] = None
        self._obs_cache: Optional[dict[str, np.ndarray]] = None

    @property
    def controlled_players(self) -> tuple[str, ...]:
        return self.config.controlled_players

    @property
    def controlled_team(self) -> str:
        return team_of(self.controlled_players[0])

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self.config.seed = int(seed)
            self.rng = np.random.default_rng(self.config.seed)

        self.state = make_initial_state(seed=self.config.seed)
        self.control_bots = {}
        bots = self._make_bots()
        self.engine = GameEngine(
            self.state,
            bots,
            record_trace=self.config.record_trace,
            record_replay=self.config.record_replay,
        )
        self._done = False
        self._invalidate_cache()
        observations = self.observations()
        raw = self.raw_observations()
        self._previous_team_score = self._team_score(next(iter(raw.values())))
        self._previous_player_scores = {
            player_id: self._player_score(obs)
            for player_id, obs in raw.items()
        }
        masks = self.action_masks()
        return observations, self._info(raw, reward_components={}, invalid_actions={}, masks=masks)

    def step(self, actions: dict[str, int]) -> TeamStepResult:
        if self.engine is None:
            self.reset()
        if self._done:
            raw = self.raw_observations()
            return TeamStepResult(
                observations=self.observations_from_raw(raw),
                rewards={player_id: 0.0 for player_id in self.controlled_players},
                terminal=True,
                truncated=False,
                info=self._info(raw, reward_components={}, invalid_actions={}),
            )

        raw_before = self.raw_observations()
        masks_before = self.action_masks()
        decoded_actions: dict[str, Action] = {}
        invalid_actions: dict[str, int] = {}
        for player_id in self.controlled_players:
            action_index = int(actions.get(player_id, 0))
            mask = masks_before[player_id]
            is_invalid = (
                action_index < 0 or action_index >= ACTION_DIM or not bool(mask[action_index])
            )
            invalid_actions[player_id] = int(is_invalid)
            decoded = (
                decode_action(action_index, raw_before[player_id], player_id, assume_legal=True)
                if self._player_alive(player_id) and not is_invalid
                else Action(ActionType.HOLD)
            )
            decoded_actions[player_id] = decoded
            self.control_bots[player_id].set_action(decoded)

        self.engine.tick()
        self._invalidate_cache()
        if self.state.winner is None and self.state.tick >= TICK_LIMIT:
            self.engine.run()
            self._invalidate_cache()
        elif self.state.tick >= self.config.max_ticks and self.state.winner is None:
            self.state.winner = "DRAW"
            self.state.winner_type = "truncated"
            self._invalidate_cache()

        raw_after = self.raw_observations()
        masks_after = self.action_masks()
        team_score = self._team_score(next(iter(raw_after.values())))
        team_shaping = (team_score - self._previous_team_score) * self.config.shaping_weight
        self._previous_team_score = team_score

        player_shaping: dict[str, float] = {}
        for player_id, obs in raw_after.items():
            score = self._player_score(obs)
            player_shaping[player_id] = (score - self._previous_player_scores.get(player_id, 0.0)) * self.config.shaping_weight
            self._previous_player_scores[player_id] = score

        terminal = self.state.winner in {"TEAM_A", "TEAM_B"} or (
            self.state.winner == "DRAW" and self.state.winner_type != "truncated"
        )
        truncated = self.state.winner_type == "truncated"
        terminal_reward = self._terminal_reward()
        self._done = terminal or truncated

        rewards = {}
        reward_components = {}
        for player_id in self.controlled_players:
            shaping = player_shaping[player_id] if self.config.per_player_shaping else team_shaping
            rewards[player_id] = float(shaping + terminal_reward)
            reward_components[player_id] = {
                "shaping": float(shaping),
                "terminal": float(terminal_reward),
                "team_score": float(team_score),
            }

        info = self._info(
            raw_after,
            reward_components=reward_components,
            invalid_actions=invalid_actions,
            masks=masks_after,
        )
        info["decoded_actions"] = {
            player_id: {
                "type": action.action_type.value,
                "source": action.source,
                "destination": action.destination,
                "intent": action.intent,
                "unit_type": action.unit_type.value if action.unit_type else None,
                "building_type": action.building_type.value if action.building_type else None,
            }
            for player_id, action in decoded_actions.items()
        }
        info["action_indices"] = {
            player_id: int(actions.get(player_id, 0))
            for player_id in self.controlled_players
        }
        return TeamStepResult(
            observations=self.observations(),
            rewards=rewards,
            terminal=terminal,
            truncated=truncated,
            info=info,
        )

    def raw_observations(self) -> dict[str, dict]:
        if self.engine is None:
            raise RuntimeError("TeamTrainingEnv.reset() must be called before observation")
        tick = int(self.state.tick)
        if self._cache_tick == tick and self._raw_cache is not None:
            return self._raw_cache
        self._cache_tick = tick
        self._raw_cache = {
            player_id: self.engine.observe(player_id)
            for player_id in self.controlled_players
        }
        self._mask_cache = None
        self._obs_cache = None
        return self._raw_cache

    def observations(self) -> dict[str, np.ndarray]:
        tick = int(self.state.tick)
        if self._cache_tick == tick and self._obs_cache is not None:
            return self._obs_cache
        self._obs_cache = self.observations_from_raw(self.raw_observations())
        return self._obs_cache

    def observations_from_raw(self, raw_observations: dict[str, dict]) -> dict[str, np.ndarray]:
        return {
            player_id: encode_observation(obs, player_id)
            for player_id, obs in raw_observations.items()
        }

    def action_masks(self) -> dict[str, np.ndarray]:
        tick = int(self.state.tick)
        if self._cache_tick == tick and self._mask_cache is not None:
            return self._mask_cache
        raw = self.raw_observations()
        self._mask_cache = {
            player_id: self.action_mask(player_id, raw[player_id])
            for player_id in self.controlled_players
        }
        return self._mask_cache

    def action_mask(self, player_id: str, raw_observation: Optional[dict] = None) -> np.ndarray:
        if not self._player_alive(player_id):
            mask = np.zeros(ACTION_DIM, dtype=np.bool_)
            mask[0] = True
            return mask
        obs = raw_observation or self.engine.observe(player_id)
        return legal_action_mask(obs, player_id)

    def sample_legal_actions(self) -> dict[str, int]:
        actions = {}
        for player_id, mask in self.action_masks().items():
            legal = np.flatnonzero(mask)
            actions[player_id] = int(self.rng.choice(legal)) if len(legal) else 0
        return actions

    def _make_bots(self) -> dict[str, Bot]:
        bots: dict[str, Bot] = {}
        controlled_set = set(self.controlled_players)
        for index, player_id in enumerate(PLAYER_IDS):
            team = "TEAM_A" if player_id in TEAM_A_PLAYERS else "TEAM_B"
            if player_id in controlled_set:
                bot = ExternalActionBot(player_id, team)
                self.control_bots[player_id] = bot
                bots[player_id] = bot
            elif team == self.controlled_team:
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
        if bot_type == "policy":
            if not self.config.opponent_policy_checkpoint:
                raise ValueError("opponent_policy_checkpoint is required when opponent_bot='policy'")
            from src.bots.policy_bot import PolicyBot

            return PolicyBot(
                player_id,
                team,
                checkpoint_path=self.config.opponent_policy_checkpoint,
                deterministic=self.config.opponent_policy_deterministic,
                device="cpu",
            )
        raise ValueError(f"Unknown baseline bot type: {bot_type}")

    def _player_alive(self, player_id: str) -> bool:
        if self.state is None:
            return True
        return bool(self.state.players[player_id].alive)

    def _terminal_reward(self) -> float:
        winner = self.state.winner
        if winner is None:
            return 0.0
        if winner == "DRAW":
            return self.config.terminal_draw_reward
        return self._terminal_win_reward() if winner == self.controlled_team else self.config.terminal_loss_reward

    def _terminal_win_reward(self) -> float:
        reward = self.config.terminal_win_reward
        if self.state.winner_type == "death_time":
            reward *= self.config.death_time_win_reward_multiplier
        return reward

    def _team_score(self, obs: dict) -> float:
        own_team = self.controlled_team
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

    def _player_score(self, obs: dict) -> float:
        player_id = obs.get("player_id", "")
        resources = float(obs.get("resources", 0.0) or 0.0) / 100.0
        own_units = 0.0
        for node in (obs.get("nodes", {}) or {}).values():
            if node.get("owner") == player_id:
                own_units += sum(float(count or 0) for count in (node.get("units", {}) or {}).values()) / 50.0
            own_units += sum(
                float(count or 0) / 50.0
                for count in ((node.get("visiting_units", {}) or {}).get(player_id, {}) or {}).values()
            )
        return resources + own_units

    def _info(
        self,
        raw_observations: dict[str, dict],
        reward_components: dict,
        invalid_actions: dict[str, int],
        masks: Optional[dict[str, np.ndarray]] = None,
    ) -> dict:
        masks = masks or self.action_masks()
        legal_actions = {
            player_id: int(masks[player_id].sum())
            for player_id in self.controlled_players
        }
        return {
            "tick": int(next(iter(raw_observations.values())).get("tick", 0) or 0),
            "winner": self.state.winner if self.state is not None else None,
            "winner_type": self.state.winner_type if self.state is not None else None,
            "controlled_players": list(self.controlled_players),
            "controlled_team": self.controlled_team,
            "legal_actions": legal_actions,
            "invalid_actions": dict(invalid_actions),
            "reward_components": reward_components,
            "observation_version": OBSERVATION_VERSION,
        }

    def _invalidate_cache(self) -> None:
        self._cache_tick = None
        self._raw_cache = None
        self._mask_cache = None
        self._obs_cache = None
