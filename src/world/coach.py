"""Coach — the team's tactical brain (an RL net), distinct from the LLM manager.

Role split (owner decision):
  - COACH (this module, an RL net, hired like a player, trained like a player):
    in-match TACTICS — picks the starting three, can substitute mid-match when the
    roster has more than three, swaps positions, and sets the per-match order
    (tempo / win_path / roles) the player nets then obey per their baked obedience.
  - MANAGER (LLM, see transfers.py / training_auction.py): the front office — buy/
    sell players, hire coaches, bid in auctions, spend training certificates.

The coach's decisions here are a deterministic, in-character heuristic so the world
runs with no trained coach net. When the RL coach is trained, it replaces
``select_lineup`` / ``in_match_decision`` — the interfaces stay identical.
"""
from __future__ import annotations

from typing import Optional

from src.world import economy
from src.world.world import get_coach, get_player, team_coach

# Mirrors src.training.spaces ORDER_* (kept local to avoid heavy imports here).
ORDER_TEMPO = ("turtle", "balanced", "aggressive", "all_in")
ORDER_WIN_PATH = ("economy", "core_rush", "baron_lord", "outlast")
ORDER_ROLE = ("carry_push", "anchor_defend", "tech_eco", "roam_objective", "support_flex")
POSITIONS = ("A1", "A2", "A3")
COACH_BONUS_SCALE = 60.0  # max strength bonus (rating units) a great coach adds


class CoachError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Hiring (manager action; here as a rules helper)
# --------------------------------------------------------------------------- #
def hire_coach(world: dict, team_id: str, coach_id: str) -> dict:
    coach = get_coach(world, coach_id)
    if coach["status"] == "hired" and coach["team_id"] not in (None, team_id):
        raise CoachError(f"coach {coach_id} already hired by {coach['team_id']}")
    # release the team's current coach back to the pool
    team = world["teams"][team_id]
    if team.get("coach") and team["coach"] != coach_id:
        release_coach(world, team_id)
    coach["status"] = "hired"
    coach["team_id"] = team_id
    team["coach"] = coach_id
    if coach_id in world.get("free_coaches", []):
        world["free_coaches"].remove(coach_id)
    return coach


def release_coach(world: dict, team_id: str) -> Optional[str]:
    team = world["teams"][team_id]
    cid = team.get("coach")
    if not cid:
        return None
    coach = world["coaches"][cid]
    coach["status"] = "free_agent"
    coach["team_id"] = None
    team["coach"] = None
    world.setdefault("free_coaches", []).append(cid)
    return cid


def assign_default_coaches(world: dict) -> None:
    """Give every team a coach from the free pool (used at bootstrap so each team
    has a tactical brain). Pops the best-rated free coaches first."""
    for team in world["teams"].values():
        if team.get("coach"):
            continue
        free = world.get("free_coaches", [])
        if not free:
            return
        free.sort(key=lambda cid: -world["coaches"][cid]["rating"])
        hire_coach(world, team["team_id"], free[0])


# --------------------------------------------------------------------------- #
# Lineup + order (tactics)
# --------------------------------------------------------------------------- #
def _assign_roles(starters: list[str], world: dict) -> dict:
    """Map the three starters to positions A1/A2/A3 and order-roles by value."""
    ranked = sorted(starters, key=lambda pid: -economy.player_value(get_player(world, pid)))
    roles = ("carry_push", "anchor_defend", "support_flex")
    out = {}
    for i, pid in enumerate(ranked):
        out[pid] = {"position": POSITIONS[i], "role": roles[i]}
    return out


def select_lineup(world: dict, team_id: str) -> dict:
    """Pick the starting three (+ bench), positions and the per-match order.

    Heuristic stand-in for the RL coach: start the three highest-value players,
    bias tempo/win_path from the coach's character. Returns a dict consumable by
    the engine (``order`` is in the {pid: {tempo, win_path, role}} env format).
    """
    team = world["teams"][team_id]
    roster = list(team["roster"])
    if len(roster) < 3:
        raise CoachError(f"team {team_id} has < 3 players")
    ranked = sorted(roster, key=lambda pid: -economy.player_value(get_player(world, pid)))
    starters, bench = ranked[:3], ranked[3:]

    coach = team_coach(world, team)
    bias = (coach or {}).get("tactic_bias", {})
    tempo = bias.get("tempo", "balanced")
    win_path = bias.get("win_path", "economy")

    assigned = _assign_roles(starters, world)
    order = {pid: {"tempo": tempo, "win_path": win_path, "role": assigned[pid]["role"]}
             for pid in starters}
    positions = {pid: assigned[pid]["position"] for pid in starters}
    return {"team_id": team_id, "coach_id": team.get("coach"),
            "starters": starters, "bench": bench, "positions": positions, "order": order}


# --------------------------------------------------------------------------- #
# In-match decisions (subs / position swaps) — only when roster > 3
# --------------------------------------------------------------------------- #
def can_substitute(world: dict, team_id: str) -> bool:
    return len(world["teams"][team_id]["roster"]) > 3


def apply_substitution(lineup: dict, out_pid: str, in_pid: str) -> dict:
    """Swap a starter for a bench player (allowed only when there is a bench).
    Mutates and returns the lineup; the incoming player inherits the position/role."""
    if out_pid not in lineup["starters"]:
        raise CoachError(f"{out_pid} is not a starter")
    if in_pid not in lineup["bench"]:
        raise CoachError(f"{in_pid} is not on the bench")
    idx = lineup["starters"].index(out_pid)
    lineup["starters"][idx] = in_pid
    lineup["bench"][lineup["bench"].index(in_pid)] = out_pid
    pos = lineup["positions"].pop(out_pid)
    role = lineup["order"].pop(out_pid)
    lineup["positions"][in_pid] = pos
    lineup["order"][in_pid] = role
    return lineup


def swap_positions(lineup: dict, pid_a: str, pid_b: str) -> dict:
    """Swap two starters' positions (and the role tied to that position)."""
    for pid in (pid_a, pid_b):
        if pid not in lineup["starters"]:
            raise CoachError(f"{pid} is not a starter")
    lineup["positions"][pid_a], lineup["positions"][pid_b] = (
        lineup["positions"][pid_b], lineup["positions"][pid_a])
    lineup["order"][pid_a]["role"], lineup["order"][pid_b]["role"] = (
        lineup["order"][pid_b]["role"], lineup["order"][pid_a]["role"])
    return lineup


# --------------------------------------------------------------------------- #
# Coach influence on a match (foundation: a small competence bonus)
# --------------------------------------------------------------------------- #
def coach_bonus(world: dict, team_id: str) -> float:
    """Strength bonus (rating units) the coach adds, scaled by competence rating.
    A 1000 coach adds 0; a strong coach adds up to ~COACH_BONUS_SCALE."""
    coach = team_coach(world, world["teams"][team_id])
    if not coach:
        return 0.0
    return (coach["rating"] - 1000.0) / 500.0 * COACH_BONUS_SCALE
