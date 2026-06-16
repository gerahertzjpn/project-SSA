"""Free-agent maintenance — keep unsigned players in form between seasons.

Separate compute from rookies (owner-set, and by estimate the TOTAL free-agent
budget is smaller than the rookie budget). A free agent who plays no official
games would otherwise stagnate; a light maintenance run keeps their net sharp so
they remain a credible signing. These jobs join the SAME combined training launch
as team trainings and rookie forks (so the owner starts everything at once).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.world import valuation
from src.world.training_ops import MIN_TRAINING_STEPS

# Form maintenance sits at the floor — it is upkeep, not a real upgrade.
RECOMMENDED_STEPS_PER_AGENT = MIN_TRAINING_STEPS  # 800k


@dataclass
class FreeAgentBudget:
    """Owner-set free-agent upkeep. Keep the total below the rookie budget."""
    steps_per_agent: int = RECOMMENDED_STEPS_PER_AGENT
    max_agents: int | None = 6          # cap how many FAs train (None = all)
    select: str = "by_rating"           # which FAs: strongest keep form first

    def __post_init__(self):
        if self.steps_per_agent < MIN_TRAINING_STEPS:
            raise ValueError(f"steps_per_agent >= {MIN_TRAINING_STEPS} (absolute floor)")


def free_agents(world: dict) -> list[str]:
    return [pid for pid, p in world["players"].items()
            if p.get("status") == "free_agent" and p.get("team_id") is None]


def select_agents(world: dict, budget: FreeAgentBudget) -> list[str]:
    """Pick which free agents to maintain this off-season."""
    fas = free_agents(world)
    if budget.select == "by_rating":
        fas.sort(key=lambda pid: -valuation.skill_score(world["players"][pid]))
    if budget.max_agents is not None:
        fas = fas[: budget.max_agents]
    return fas


def maintenance_jobs(world: dict, budget: FreeAgentBudget) -> list[dict]:
    """Build the free-agent maintenance training jobs for the combined launch."""
    jobs = []
    for pid in select_agents(world, budget):
        jobs.append({
            "kind": "free_agent",
            "op": "fa_maintain",
            "team_id": None,
            "target": {"kind": "player", "player_id": pid},
            "total_steps": int(budget.steps_per_agent),
        })
    return jobs


def total_steps(jobs: list[dict]) -> int:
    return sum(j["total_steps"] for j in jobs)
