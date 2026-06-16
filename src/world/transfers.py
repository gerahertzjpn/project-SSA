"""Transfer window — rules engine (deterministic) + manager-decision hook.

The *rules* here are deterministic and balance-critical, so they are implemented
and tested: market valuation with a realistic floor (no "top player for 1 yen"),
the 2-round structure, the <=3-requests-per-team limit, sale listings, free-agent
auctions and the lower-league rookie draft.

The *decisions* (which players a manager bids on, how much, whether to accept an
offer) are the LLM manager's job. They are injected via ``decide_*`` callables
that default to a simple, sane heuristic so the window runs end-to-end with no
LLM. Wiring the real LLM manager is a later step (see WORLD_MODE_SPEC §6).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.world import economy
from src.world.world import get_player, get_team

MAX_REQUESTS_PER_WINDOW = 3
ROUNDS = 2
# Market value of a player as a multiple of their seasonal salary value.
VALUE_TO_PRICE = 8.0
# You cannot buy below this fraction of fair value -> blocks absurd lowballs.
MIN_PRICE_FRACTION = 0.6


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #
def fair_value(player: dict, config: economy.EconomyConfig = economy.CONFIG) -> int:
    """Realistic market price of a player (teams trade in this currency)."""
    return int(round(economy.player_value(player, config) * config.salary_base * VALUE_TO_PRICE))


def min_price(player: dict, config: economy.EconomyConfig = economy.CONFIG) -> int:
    """Lowest acceptable price — a hard floor that prevents 1-yen sales/lowballs."""
    listed = player.get("min_price")
    floor = int(round(fair_value(player, config) * MIN_PRICE_FRACTION))
    return max(floor, int(listed)) if listed else floor


# --------------------------------------------------------------------------- #
# Window state
# --------------------------------------------------------------------------- #
@dataclass
class TransferWindow:
    round: int = 1
    requests_used: dict[str, int] = field(default_factory=dict)      # team_id -> count
    # (round, buyer_team, player_id) seen — blocks 2 requests for same player in one round.
    seen_round_player: set = field(default_factory=set)
    listings: dict[str, dict] = field(default_factory=dict)          # player_id -> {seller, asking}
    log: list[dict] = field(default_factory=list)

    def can_request(self, buyer_team: str, player_id: str) -> tuple[bool, str]:
        if self.requests_used.get(buyer_team, 0) >= MAX_REQUESTS_PER_WINDOW:
            return False, "request budget exhausted (<=3 per window)"
        if (self.round, buyer_team, player_id) in self.seen_round_player:
            return False, "already requested this player this round"
        return True, ""

    def register_request(self, buyer_team: str, player_id: str) -> None:
        self.requests_used[buyer_team] = self.requests_used.get(buyer_team, 0) + 1
        self.seen_round_player.add((self.round, buyer_team, player_id))

    def advance_round(self) -> None:
        self.round += 1


# --------------------------------------------------------------------------- #
# Listings (a team offers a player for sale — for BOTH rounds, never one)
# --------------------------------------------------------------------------- #
def list_for_sale(world: dict, window: TransferWindow, player_id: str,
                  asking: Optional[int] = None) -> dict:
    player = get_player(world, player_id)
    if player.get("team_id") is None:
        raise ValueError("cannot list a free agent / unrostered player")
    ask = int(asking) if asking is not None else fair_value(player)
    ask = max(ask, min_price(player))
    entry = {"seller": player["team_id"], "player_id": player_id, "asking": ask}
    window.listings[player_id] = entry
    window.log.append({"event": "list", **entry})
    return entry


# --------------------------------------------------------------------------- #
# Free-agent auction
# --------------------------------------------------------------------------- #
def resolve_free_agent_auction(world: dict, bids: list[dict], *, seed: int = 0) -> list[dict]:
    """``bids``: ``[{team_id, player_id, amount}]`` for FREE AGENTS only.
    Per player: the highest bid >= min_price wins and pays its amount. Seeded
    tiebreak. Returns the list of completed signings."""
    rng = random.Random(seed)
    by_player: dict[str, list[dict]] = {}
    for b in bids:
        by_player.setdefault(b["player_id"], []).append(b)

    signings = []
    for pid, plist in by_player.items():
        player = get_player(world, pid)
        if player.get("status") != "free_agent":
            continue
        floor = min_price(player)
        valid = [b for b in plist if b["amount"] >= floor]
        if not valid:
            continue
        top = max(b["amount"] for b in valid)
        contenders = [b for b in valid if b["amount"] == top]
        winner = rng.choice(contenders)
        team = get_team(world, winner["team_id"])
        if team["treasury"] < top:
            continue  # can't afford -> auction void for this player
        team["treasury"] -= top
        _move_player_to_team(world, pid, team["team_id"])
        signings.append({"player_id": pid, "team_id": team["team_id"], "price": top})
    return signings


# --------------------------------------------------------------------------- #
# Owned-player transfer request (buyer -> owner). Owner decides (LLM stub).
# --------------------------------------------------------------------------- #
AcceptFn = Callable[[dict, dict, int], bool]  # (world, player, offer) -> accept?


def _default_accept(world: dict, player: dict, offer: int) -> bool:
    """Heuristic stand-in for the selling manager's LLM decision: accept if the
    offer meets fair value (sellers don't give away talent)."""
    return offer >= fair_value(player)


def propose_transfer(world: dict, window: TransferWindow, buyer_team: str,
                     player_id: str, offer: int, *,
                     accept_fn: AcceptFn = _default_accept) -> dict:
    """A buyer requests an owned player at ``offer``. Enforces window limits and
    the realistic min-price floor; the owner accepts/rejects via ``accept_fn``."""
    ok, reason = window.can_request(buyer_team, player_id)
    if not ok:
        return {"status": "rejected", "reason": reason}
    window.register_request(buyer_team, player_id)

    player = get_player(world, player_id)
    owner = player.get("team_id")
    if owner is None:
        return {"status": "rejected", "reason": "player is a free agent (use auction)"}
    if owner == buyer_team:
        return {"status": "rejected", "reason": "already owns player"}
    if offer < min_price(player):
        return {"status": "rejected", "reason": "offer below min price (no lowballs)"}
    buyer = get_team(world, buyer_team)
    if buyer["treasury"] < offer:
        return {"status": "rejected", "reason": "insufficient treasury"}
    if len(buyer["roster"]) < 3:
        pass  # buying up is always allowed
    if not accept_fn(world, player, offer):
        return {"status": "declined", "reason": "owner declined"}

    # execute: money owner<-buyer, player moves.
    buyer["treasury"] -= offer
    get_team(world, owner)["treasury"] += offer
    _move_player_to_team(world, player_id, buyer_team)
    window.listings.pop(player_id, None)
    record = {"status": "completed", "player_id": player_id, "from": owner,
              "to": buyer_team, "price": offer}
    window.log.append({"event": "transfer", **record})
    return record


# --------------------------------------------------------------------------- #
# Rookie draft (lower league only, weakest first, optional)
# --------------------------------------------------------------------------- #
def rookie_draft(world: dict, *, want_fn: Optional[Callable[[dict, list], Optional[str]]] = None
                 ) -> list[dict]:
    """Lower-league teams draft from ``world['rookies_pool']`` weakest-team-first.
    Each may take one or pass (``want_fn`` returns a rookie id or None). Upper-
    league teams cannot draft rookies. Returns the list of picks."""
    pool = world.get("rookies_pool", [])
    lower = [t for t in world["teams"].values() if t["league"] == "lower"]
    # weakest first by Elo (proxy); stable.
    lower.sort(key=lambda t: (t["elo"], t["team_id"]))
    picks = []
    for team in lower:
        if not pool:
            break
        if want_fn is not None:
            choice = want_fn(team, [get_player(world, pid) for pid in pool])
        else:
            # default heuristic: weakest team takes the best available rookie
            choice = max(pool, key=lambda pid: economy.player_value(get_player(world, pid)))
        if choice is None or choice not in pool:
            continue
        pool.remove(choice)
        player = get_player(world, choice)
        player["status"] = "rookie"   # salary starts after first played season
        player["team_id"] = team["team_id"]
        team["roster"].append(choice)
        picks.append({"team_id": team["team_id"], "player_id": choice})
    world["rookies_pool"] = pool
    return picks


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _move_player_to_team(world: dict, player_id: str, team_id: str) -> None:
    player = get_player(world, player_id)
    old = player.get("team_id")
    if old and old in world["teams"]:
        roster = world["teams"][old]["roster"]
        if player_id in roster:
            roster.remove(player_id)
    if player_id in world.get("free_agents", []):
        world["free_agents"].remove(player_id)
    player["team_id"] = team_id
    player["status"] = "rostered"
    if player_id not in world["teams"][team_id]["roster"]:
        world["teams"][team_id]["roster"].append(player_id)


def open_window() -> TransferWindow:
    return TransferWindow()
