"""Training auction — BETA. Gradeslot certificates; real training is a HARD STUB.

Between seasons the owner sets the **gradeslot** size in TRAINING STEPS
(``steps_per_slot``, default 1,000,000) and we show the estimated wall-clock per
slot. N gradeslots are auctioned one at a time; the winning team receives a
**certificate** worth that many steps. Because slot size can differ between
seasons, value is tracked in steps, not slot count — so a team can:

  - **bank** certificates and use them later (defer by choice),
  - **combine** several certificates (e.g. to reach >= 4,000,000 steps to fork a
    new player from a chosen parent), or
  - **spend** one/several on a training run.

Spend types: ``skill_up`` / ``teamplay`` / ``meta_adapt`` (any step budget) and
``create_player`` (a lineage fork; requires >= ``FORK_MIN_STEPS`` combined).

``execute_training`` deliberately raises: WHAT each type optimizes and HOW the
effect is validated is still being decided with the owner (see
docs/cloudcode/WORLD_TRAINING_OPTIONS.md). No training right now.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.world.training_ops import (  # A–F ops + the absolute step floor (single source)
    ENABLED_OPS as TRAINING_TYPES,
    MIN_TRAINING_STEPS,
)

DEFAULT_SLOT_STEPS = 1_000_000
FORK_MIN_STEPS = 4_000_000
# Step budget is unconstrained for most ops; create_player is gated by FORK_MIN_STEPS.


@dataclass
class TrainingConfig:
    """Owner-set per offseason. ``steps_per_slot`` is the gradeslot size; the
    throughput is used only to show an ETA (calibrate on the real GPU)."""
    steps_per_slot: int = DEFAULT_SLOT_STEPS
    # TODO: calibrate on RTX 4070 SUPER (CPU sim is the real bottleneck, not GPU).
    throughput_steps_per_min: float = 50_000.0

    def est_minutes(self, steps: int) -> float:
        return steps / max(1.0, self.throughput_steps_per_min)

    def est_minutes_per_slot(self) -> float:
        return self.est_minutes(self.steps_per_slot)


@dataclass
class TrainingAuctionState:
    """Sequential auction; managers can see who took/offered what (public_state)."""
    config: TrainingConfig
    season: int = 0
    remaining: int = 0
    issued: int = 0                                    # cert counter (deterministic ids)
    awards: list[dict] = field(default_factory=list)
    n_trainings: int = 10                              # how many slots are auctioned

    def __post_init__(self):
        if not self.remaining:
            self.remaining = self.n_trainings

    def public_state(self) -> dict:
        return {
            "trainings_left": self.remaining,
            "steps_per_slot": self.config.steps_per_slot,
            "est_minutes_per_slot": round(self.config.est_minutes_per_slot(), 1),
            "awards": [{"team_id": a["team_id"], "price": a["price"]} for a in self.awards],
        }


# --------------------------------------------------------------------------- #
# Auction
# --------------------------------------------------------------------------- #
def auction_one_training(world: dict, state: TrainingAuctionState, bids: dict, *,
                         seed: int = 0) -> Optional[dict]:
    """Run ONE gradeslot auction. ``bids``: ``{team_id: amount}``. Highest
    affordable bidder wins, pays, and receives a step-valued certificate (banked
    on the team). Returns the award, or None if nobody bid."""
    if state.remaining <= 0:
        return None
    valid = {tid: amt for tid, amt in bids.items() if amt and amt > 0
             and world["teams"].get(tid, {}).get("treasury", 0) >= amt}
    if not valid:
        state.remaining -= 1
        return None
    top = max(valid.values())
    # deterministic tiebreak: lowest team_id among the top bids
    winner = min(tid for tid, amt in valid.items() if amt == top)
    world["teams"][winner]["treasury"] -= top
    state.remaining -= 1
    state.issued += 1
    cert = {"cert_id": f"cert-s{state.season}-{state.issued}",
            "steps": int(state.config.steps_per_slot), "season": state.season}
    world["teams"][winner].setdefault("training_certificates", []).append(cert)
    award = {"team_id": winner, "price": top, "cert_id": cert["cert_id"],
             "steps": cert["steps"], "bids": dict(bids)}
    state.awards.append(award)
    return award


# --------------------------------------------------------------------------- #
# Certificates wallet
# --------------------------------------------------------------------------- #
def team_certificates(world: dict, team_id: str) -> list[dict]:
    return world["teams"][team_id].get("training_certificates", [])


def total_banked_steps(world: dict, team_id: str) -> int:
    return sum(c["steps"] for c in team_certificates(world, team_id))


# --------------------------------------------------------------------------- #
# Spend (consume certificates) -> a pending training job
# --------------------------------------------------------------------------- #
def request_spend(world: dict, team_id: str, spend_type: str, cert_ids: list[str],
                  *, target: Optional[dict] = None,
                  config: Optional[TrainingConfig] = None) -> dict:
    """Spend the chosen certificates on a training run.

    ``cert_ids`` selects which banked certificates to combine. Validates type +
    ownership; ``create_player`` requires the combined steps to reach
    ``FORK_MIN_STEPS``. Consumes the certificates (deferral = simply don't call
    this; certs persist across seasons). Returns a pending job — NOT executed.
    """
    if spend_type not in TRAINING_TYPES:
        raise ValueError(f"unknown training type {spend_type!r}")
    wallet = team_certificates(world, team_id)
    by_id = {c["cert_id"]: c for c in wallet}
    chosen = []
    for cid in cert_ids:
        if cid not in by_id:
            raise ValueError(f"certificate {cid!r} not owned by {team_id}")
        chosen.append(by_id[cid])
    if not chosen:
        raise ValueError("no certificates selected")
    total_steps = sum(c["steps"] for c in chosen)
    if total_steps < MIN_TRAINING_STEPS:
        raise ValueError(f"absolute minimum {MIN_TRAINING_STEPS} steps per training; got {total_steps}")
    if spend_type == "create_player" and total_steps < FORK_MIN_STEPS:
        raise ValueError(f"create_player needs >= {FORK_MIN_STEPS} steps; got {total_steps}")

    # consume
    keep = [c for c in wallet if c["cert_id"] not in set(cert_ids)]
    world["teams"][team_id]["training_certificates"] = keep

    cfg = config or TrainingConfig()
    return {
        "team_id": team_id,
        "spend_type": spend_type,
        "total_steps": total_steps,
        "est_minutes": round(cfg.est_minutes(total_steps), 1),
        "target": target or {},
        "status": "pending_execution",
    }


def execute_training(world: dict, job: dict, config: TrainingConfig, *, backend=None):
    """Execute a pending training job via the A–F operation dispatch.

    Operations and their cards live in src/world/training_ops.py. Real training
    runs only when an explicit ``backend`` (the PPO/lineage infra) is passed; with
    the default (None -> NullBackend) any attempt is refused, so NO TRAINING runs
    until the owner enables a backend. Returns a TrainingResult on success.
    """
    from src.world import training_ops  # local import keeps torch out of the hot path
    ctx = {"world": world, "team_id": job.get("team_id"), "target": job.get("target", {})}
    return training_ops.execute(job["spend_type"], ctx,
                                total_steps=job.get("total_steps", 0), backend=backend)
