from abc import ABC, abstractmethod
from ..core.engine import Action


class Bot(ABC):
    def __init__(self, player_id: str, team: str):
        self.player_id = player_id
        self.team = team

    @abstractmethod
    def act(self, obs: dict) -> Action:
        """Given an observation dict, return an Action."""
        ...
