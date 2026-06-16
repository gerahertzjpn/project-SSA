"""Player valuation + manager decision logic (buy / sell / draft a rookie).

Answers the three manager questions with explicit, testable signals that the LLM
manager consults (and may override within its persona). Two distinct notions:

  - **sporting_value**: worth of a player TO A TEAM for sporting decisions — would
    they start, do they fill a role gap, what is their upside. Drives keep/sell and
    whether a target is an upgrade.
  - **market_price** (transfers.fair_value): the money price for buy/sell.

A good manager buys when a target is a real upgrade it can afford, sells surplus
players for a fair price (or under budget pressure), and drafts a rookie when its
upside beats the current weakest option — never overpays, never dumps talent.
"""
from __future__ import annotations

from src.world import economy, transfers
from src.world.world import get_player

ATTR_KEYS = economy.ATTR_KEYS
UPGRADE_MARGIN = 0.08      # a target must beat the weakest starter by this much
SELL_PRICE_FLOOR = 1.0     # accept a sale at >= fair value (×floor)
BUDGET_PRESSURE_RATIO = 0.6  # treasury below this × salary bill => sell pressure


def skill_score(player: dict) -> float:
    """Current playing strength (0..~2): SoloQ-dominated + attribute quality."""
    rating = float(player.get("soloq_rating", 1000.0))
    attrs = player.get("attributes", {})
    avg_attr = sum(attrs.get(k, 50) for k in ATTR_KEYS) / len(ATTR_KEYS) if attrs else 50.0
    return (rating / 1000.0) * (0.7 + 0.3 * (avg_attr / 60.0))


def potential_score(player: dict) -> float:
    """Upside (0..~1.6): leans on the potential attribute + youth (few seasons)."""
    pot = player.get("attributes", {}).get("potential", 50)
    seasons = player.get("seasons_played", 0)
    youth = max(0.0, 1.0 - 0.12 * seasons)
    return (pot / 60.0) * (0.7 + 0.3 * youth)


def sporting_value(world: dict, player: dict, team: dict | None = None) -> float:
    """Worth of a player for sporting decisions. With a team, adds a small bonus
    if they would START (top-3 by skill) and a role-need bonus; weights upside."""
    base = 0.75 * skill_score(player) + 0.25 * potential_score(player)
    if team is None:
        return round(base, 4)
    roster = [get_player(world, pid) for pid in team["roster"]]
    ranked = sorted(roster, key=skill_score, reverse=True)
    starters = {p["player_id"] for p in ranked[:3]}
    starter_bonus = 0.12 if player.get("player_id") in starters else 0.0
    return round(base + starter_bonus, 4)


def weakest_starter(world: dict, team: dict) -> dict | None:
    roster = [get_player(world, pid) for pid in team["roster"]]
    if len(roster) < 1:
        return None
    starters = sorted(roster, key=skill_score, reverse=True)[:3]
    return min(starters, key=skill_score) if starters else None


def squad_needs(world: dict, team: dict) -> dict:
    """A compact read of the squad: size, weakest starter strength, bench depth."""
    roster = [get_player(world, pid) for pid in team["roster"]]
    ws = weakest_starter(world, team)
    return {
        "size": len(roster),
        "needs_starter": len(roster) < 3,
        "weakest_starter_skill": round(skill_score(ws), 4) if ws else 0.0,
        "bench": max(0, len(roster) - 3),
        "salary_bill": round(sum(p.get("salary", 0.0) for p in roster), 2),
    }


def salary_bill(world: dict, team: dict) -> float:
    return sum(get_player(world, pid).get("salary", 0.0) for pid in team["roster"])


# --------------------------------------------------------------------------- #
# Decisions (heuristic; the LLM manager consults / may override in persona)
# --------------------------------------------------------------------------- #
def should_buy(world: dict, team: dict, target: dict, price: int) -> dict:
    """Buy if the target is a real upgrade on the weakest starter (or fills an
    empty slot) AND the price is affordable and not above fair value."""
    fair = transfers.fair_value(target)
    affordable = team["treasury"] >= price
    fair_priced = price <= fair * 1.25     # don't wildly overpay
    needs = squad_needs(world, team)
    if needs["needs_starter"]:
        upgrade, why = True, "fills an empty starter slot"
    else:
        ws = weakest_starter(world, team)
        upgrade = sporting_value(world, target) >= sporting_value(world, ws) + UPGRADE_MARGIN
        why = "upgrades the weakest starter" if upgrade else "not enough of an upgrade"
    decide = bool(upgrade and affordable and fair_priced)
    reason = (why if not decide else f"buy: {why}")
    if upgrade and not affordable:
        reason = "wanted but unaffordable"
    elif upgrade and not fair_priced:
        reason = "overpriced vs fair value"
    return {"decision": decide, "reason": reason,
            "fair_value": fair, "max_offer": int(min(team["treasury"], fair * 1.25))}


def should_sell(world: dict, team: dict, player: dict, offer: int) -> dict:
    """Sell if the player is surplus (not a starter) and the offer is fair, or if
    the team is under budget pressure and the offer clears the floor."""
    fair = transfers.fair_value(player)
    roster = [get_player(world, pid) for pid in team["roster"]]
    starters = {p["player_id"] for p in sorted(roster, key=skill_score, reverse=True)[:3]}
    is_surplus = player.get("player_id") not in starters
    under_pressure = team["treasury"] < BUDGET_PRESSURE_RATIO * salary_bill(world, team)
    fair_offer = offer >= fair * SELL_PRICE_FLOOR
    if len(roster) <= 3 and not under_pressure:
        return {"decision": False, "reason": "cannot drop below 3 players", "fair_value": fair}
    decide = bool(fair_offer and (is_surplus or under_pressure))
    if not fair_offer:
        reason = "offer below fair value"
    elif decide:
        reason = "sell: surplus player at fair price" if is_surplus else "sell: budget pressure"
    else:
        reason = "keep: starter and no pressure"
    return {"decision": decide, "reason": reason, "fair_value": fair}


def should_draft_rookie(world: dict, team: dict, rookie: dict) -> dict:
    """Take a rookie (free, salary deferred) if its upside beats the current
    weakest roster option, or the squad is short. Upper-league teams cannot."""
    if team.get("league") == "upper":
        return {"decision": False, "reason": "upper-league teams cannot draft rookies"}
    roster = [get_player(world, pid) for pid in team["roster"]]
    if len(roster) < 3:
        return {"decision": True, "reason": "squad short — take the prospect"}
    weakest = min(roster, key=skill_score)
    # rookies are raw but high-upside: compare on a blend favouring potential
    rookie_proj = 0.5 * skill_score(rookie) + 0.5 * potential_score(rookie)
    weak_proj = 0.5 * skill_score(weakest) + 0.5 * potential_score(weakest)
    decide = rookie_proj > weak_proj
    return {"decision": bool(decide),
            "reason": "prospect upside beats weakest" if decide else "weakest is better — pass",
            "rookie_projection": round(rookie_proj, 4)}
