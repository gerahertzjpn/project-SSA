"""Rookies — where new players come from between seasons, and how to keep them fair.

Source: a rookie is a **lineage fork of a NON-top parent**. Forking from current
top players is forbidden (would mint instant imbas), so the parent pool excludes
the strongest fraction. Rookies are LIGHTLY trained (recommended ~1M steps, raw)
and start with low current skill but high **potential** — upside, not immediate
dominance. That, plus the non-top parentage, is the balance.

The owner controls the rookie compute separately (count, steps per rookie, extra
minutes that ADD to the teams' requested training time, quality). The best
prospects become available to sign; lower-league teams draft them weakest-first.

In this foundation the rookie CARDS are created now (so drafting/economy work); the
actual net-fork is performed by the (gated) training backend — recorded as a plan.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.world import valuation
from src.world.training_ops import MIN_TRAINING_STEPS
from src.world.world import make_player

# A rookie fork is raw-but-playable; far cheaper than a team's polished 4M create_player.
RECOMMENDED_STEPS_PER_ROOKIE = 1_000_000
EXCLUDE_TOP_FRACTION = 0.25          # never fork from the strongest quarter (anti-imba)
ROOKIE_SEED_BASE = 900_000


@dataclass
class RookieBudget:
    """Owner-set rookie compute (separate from team training; minutes ADD on top)."""
    n_rookies: int
    steps_per_rookie: int = RECOMMENDED_STEPS_PER_ROOKIE
    extra_minutes: float = 0.0
    quality: str = "standard"        # mini-train depth label

    def __post_init__(self):
        if self.steps_per_rookie < MIN_TRAINING_STEPS:
            raise ValueError(f"steps_per_rookie >= {MIN_TRAINING_STEPS} (absolute floor)")

    def total_steps(self) -> int:
        return self.n_rookies * self.steps_per_rookie


def eligible_parents(world: dict, exclude_top_fraction: float = EXCLUDE_TOP_FRACTION) -> list[str]:
    """Parent pool for forks: everyone EXCEPT the strongest fraction (forbidden to
    fork from current top players)."""
    players = list(world["players"].values())
    if not players:
        return []
    ranked = sorted(players, key=valuation.skill_score, reverse=True)
    cut = int(len(ranked) * exclude_top_fraction)
    top = {p["player_id"] for p in ranked[:cut]}
    return [p["player_id"] for p in players if p["player_id"] not in top]


def _next_player_index(world: dict) -> int:
    best = -1
    for pid in world["players"]:
        try:
            best = max(best, int(pid.rsplit("p", 1)[-1]))
        except ValueError:
            continue
    return best + 1


def _make_rookie_card(world: dict, index: int, parent: dict, seed: int,
                      rng: random.Random) -> dict:
    """A raw, high-upside prospect derived from ``parent`` (attenuated skill,
    boosted potential). net_id stays None until the backend forks the real net."""
    wid = world["world_id"]
    p = make_player(wid, index, f"Rookie{index:03d}", rng, tier=0.35,
                    origin="rookie_fork", parent_id=parent["player_id"],
                    generation=parent.get("lineage", {}).get("generation", 0) + 1)
    p["status"] = "rookie"
    p["soloq_rating"] = 930.0        # raw: starts below average, must prove itself
    # inherit a faded echo of the parent's profile, with high upside
    pa = parent.get("attributes", {})
    for k in ("attack", "defense", "economy", "tempo", "consistency"):
        inherited = int(round(0.6 * pa.get(k, 50) + 0.4 * p["attributes"][k]))
        p["attributes"][k] = max(15, min(85, inherited - 6))  # capped + dampened = not OP
    p["attributes"]["potential"] = int(min(99, max(60, pa.get("potential", 60) + rng.randint(2, 10))))
    p["lineage"]["seed"] = seed
    return p


def generate_rookies(world: dict, budget: RookieBudget, *, seed: int = 0) -> dict:
    """Create ``budget.n_rookies`` rookie cards forked from non-top parents and add
    them to the world's rookie pool. Returns a report incl. the fork plan (executed
    later by the gated training backend) and the total compute."""
    rng = random.Random(seed)
    parents = eligible_parents(world)
    if not parents:
        return {"created": [], "n": 0, "note": "no eligible parents"}
    rng.shuffle(parents)

    created, plan, parent_of = [], [], {}
    idx = _next_player_index(world)
    for i in range(budget.n_rookies):
        parent = world["players"][parents[i % len(parents)]]
        rseed = ROOKIE_SEED_BASE + seed * 1000 + i      # unique seed per rookie
        card = _make_rookie_card(world, idx + i, parent, rseed, rng)
        world["players"][card["player_id"]] = card
        world.setdefault("rookies_pool", []).append(card["player_id"])
        created.append(card["player_id"])
        parent_of[card["player_id"]] = parent["player_id"]
        plan.append({"rookie_id": card["player_id"], "parent_id": parent["player_id"],
                     "seed": rseed, "steps": budget.steps_per_rookie})

    return {
        "created": created, "n": len(created),
        "steps_per_rookie": budget.steps_per_rookie,
        "total_steps": budget.total_steps(),
        "extra_minutes": budget.extra_minutes,
        "parents": parent_of,
        "fork_plan": plan,
        "note": "cards created (raw, high-upside, non-top parents); real net-fork via gated backend",
    }


def rookie_training_jobs(report: dict) -> list[dict]:
    """Convert a generate_rookies report's fork plan into training jobs for the
    SAME combined launch as team trainings + free-agent upkeep (so the owner
    starts everything once). Each is a lineage fork from the chosen non-top parent."""
    jobs = []
    for item in report.get("fork_plan", []):
        jobs.append({
            "kind": "rookie",
            "op": "rookie_fork",
            "team_id": None,
            "target": {"kind": "parent", "parent_id": item["parent_id"],
                       "rookie_id": item["rookie_id"], "seed": item["seed"]},
            "total_steps": int(item["steps"]),
        })
    return jobs


def top_prospects(world: dict, k: int = 5) -> list[str]:
    """The best available rookies (by upside) — the ones offered for signing."""
    pool = world.get("rookies_pool", [])
    ranked = sorted(pool, key=lambda pid: valuation.potential_score(world["players"][pid]),
                    reverse=True)
    return ranked[:k]
