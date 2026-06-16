"""Training auction — BETA. STRUCTURE ONLY; real training is a HARD STUB.

The owner configures, between seasons, how many trainings exist and how big each
is (in training steps ~ real wall-clock). The N trainings are auctioned one at a
time; teams bid by choice; the winner spends the training on one of:

  - ``skill_up``      : extra PPO steps for one player's net
  - ``teamplay``      : joint training for a roster's three nets
  - ``create_player`` : a lineage fork (costs 4 trainings) -> a brand-new player
  - ``meta_adapt``    : adapt to the current meta (NEEDS DEFINITION)

``execute_training`` deliberately raises: the owner asked to DISCUSS the exact
execution model before it is implemented, and "train nobody right now" is in
force. The lineage fork machinery (src/training/lineage.py) is the intended
backbone for ``create_player`` / ``meta_adapt``. See WORLD_MODE_SPEC §7, §9 for
the open questions to resolve before wiring this.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

# Spend types and their training cost.
TRAINING_TYPES = {
    "skill_up": 1,
    "teamplay": 1,
    "meta_adapt": 1,
    "create_player": 4,   # a full lineage fork costs 4 trainings
}


@dataclass
class TrainingConfig:
    """Owner-set, per offseason. ``steps_per_training`` ~ real time; show ETA."""
    n_trainings: int = 10
    steps_per_training: int = 200_000
    est_minutes_per_training: float = 20.0   # TODO: derive from measured throughput

    def total_minutes(self) -> float:
        return self.n_trainings * self.est_minutes_per_training


@dataclass
class TrainingAuctionState:
    """Tracks the sequential auction so the LLM can see who took/offered what."""
    config: TrainingConfig
    remaining: int = 0
    awards: list[dict] = field(default_factory=list)   # [{team_id, price, bids:{...}}]

    def __post_init__(self):
        if not self.remaining:
            self.remaining = self.config.n_trainings

    def public_state(self) -> dict:
        """Compact view handed to managers: trainings left + award history."""
        return {
            "trainings_left": self.remaining,
            "awards": [{"team_id": a["team_id"], "price": a["price"]} for a in self.awards],
        }


def auction_one_training(world: dict, state: TrainingAuctionState,
                         bids: dict, *, seed: int = 0) -> Optional[dict]:
    """Run ONE training auction. ``bids``: ``{team_id: amount}`` (teams bid by
    choice). Highest bidder who can afford it wins and pays. Money moves; the
    *spend* is requested separately (request_spend). Returns the award or None."""
    if state.remaining <= 0:
        return None
    rng = random.Random(seed)
    valid = {tid: amt for tid, amt in bids.items() if amt and amt > 0
             and world["teams"].get(tid, {}).get("treasury", 0) >= amt}
    if not valid:
        state.remaining -= 1
        return None
    top = max(valid.values())
    winners = [tid for tid, amt in valid.items() if amt == top]
    winner = rng.choice(winners)
    world["teams"][winner]["treasury"] -= top
    state.remaining -= 1
    award = {"team_id": winner, "price": top, "bids": dict(bids)}
    state.awards.append(award)
    return award


def request_spend(state: TrainingAuctionState, team_id: str, spend_type: str,
                  target: Optional[dict] = None) -> dict:
    """Record how a winning team wants to spend its training credit(s).

    Validates the type and (for ``create_player``) that the team has accumulated
    enough credits. Does NOT execute — see ``execute_training``.
    """
    if spend_type not in TRAINING_TYPES:
        raise ValueError(f"unknown training type {spend_type!r}")
    cost = TRAINING_TYPES[spend_type]
    won = sum(1 for a in state.awards if a["team_id"] == team_id)
    if won < cost:
        raise ValueError(f"{spend_type} costs {cost} trainings; team has {won}")
    return {"team_id": team_id, "spend_type": spend_type, "cost": cost,
            "target": target or {}, "status": "pending_execution"}


def execute_training(world: dict, request: dict, config: TrainingConfig) -> None:
    """HARD STUB — do not implement until the owner confirms the execution model.

    Intended backbone (per WORLD_MODE_SPEC §9):
      - create_player : src/training/lineage.run_branch_select_cycle (fork+mini-train)
      - skill_up      : extra PPO steps for one net (metric of 'skill' TBD)
      - teamplay      : joint roster training (target metric TBD)
      - meta_adapt    : train vs current top snapshots (definition TBD)
    Open questions: training unit (steps<->minutes), what each type optimizes and
    how the effect is validated, sequential execution/ETA. NO TRAINING RIGHT NOW.
    """
    raise NotImplementedError(
        "training execution is a flagged stub — confirm the execution model "
        "(WORLD_MODE_SPEC §9) before wiring real training. No training right now."
    )
