"""World manager — the LLM, the club's SINGLE brain (front office + playbook).

Per WORLD_MODE_SPEC §0-bis there is NO coach. The LLM manager owns everything:
  - the tournament PLAYBOOK (base strategy + IF-THEN incl. substitutions) — see
    ``playbook.py``; the manager authors it, the engine executes it match by match;
  - buy / sell players, bid in auctions (free agents + training gradeslots),
  - decide how to spend training certificates, draft rookies (per-pick),
  - all in service of one goal: the upper-league title.

A manager's personality is a **management profile** (how aggressively it trades, how
it spends, youth vs proven, loyalty…) plus a tactical tilt of its playbook base.
Profiles are NATURAL — drawn from each franchise's own id (a STABLE identity that
must not change run-to-run, hence a controlled franchise-hash seed, NOT entropy) —
so the field spreads on its own. The heuristics here are the decision scaffold the
LLM consults and may override in character.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from src.world import playbook, transfers, valuation
from src.world.world import get_player

MANAGER_SYSTEM_RULES = (
    "Ты МЕНЕДЖЕР клуба — единственный мозг (тренера НЕТ). Перед турниром ты задаёшь "
    "ПЛЕЙБУК: базовую стратегию и правила IF-THEN (включая замены, напр. «у соперника "
    "роботы в защите → 2 людей в раш»); дальше движок исполняет его сам. Твоя работа "
    "также: покупать/продавать игроков, участвовать в аукционах (тренировки и "
    "свободные агенты), тратить сертификаты тренировок, драфтить новичков. Цель "
    "ВСЕГДА — чемпионство в верхней лиге. Действуй в характере, но без абсурда "
    "(никаких «топ за 1 иену»)."
)


@dataclass
class ManagerProfile:
    """Management character (0..1 traits). The playbook (playbook.py) carries the
    in-match tactics; these traits drive the front office + the base-strategy tilt."""
    aggression: float      # how hard it chases deals / bids; tilts a sharper base
    frugality: float       # how tightly it holds money
    youth_focus: float     # rookies/potential vs proven veterans
    risk: float            # willingness to gamble on upside / overpay
    patience: float        # keep the core vs churn the roster; tilts a safer base
    loyalty: float         # keep own players vs sell for profit

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def natural_profile(franchise_id: str) -> ManagerProfile:
    """Diverse management profile drawn from the franchise's own id — a STABLE
    identity (controlled hash seed, never entropy, so a club's character is the same
    every run; see §0-ter)."""
    rng = random.Random(zlib.crc32(franchise_id.encode("utf-8")) ^ 0x5EED)
    r = lambda: round(rng.uniform(0.0, 1.0), 3)  # noqa: E731
    return ManagerProfile(aggression=r(), frugality=r(), youth_focus=r(), risk=r(),
                          patience=r(), loyalty=r())


def manager_persona(franchise: dict, profile: ManagerProfile) -> str:
    """A management-character blurb (reworked: front office, not playstyle)."""
    def hi(v): return v > 0.62
    def lo(v): return v < 0.38
    bits = []
    if hi(profile.aggression):
        bits.append("агрессивен на рынке, идёт в громкие сделки")
    elif lo(profile.aggression):
        bits.append("спокоен на рынке, без суеты")
    if hi(profile.frugality):
        bits.append("бережлив, копит бюджет")
    elif lo(profile.frugality):
        bits.append("тратит легко")
    if hi(profile.youth_focus):
        bits.append("ставит на молодёжь и потенциал")
    elif lo(profile.youth_focus):
        bits.append("доверяет проверенным ветеранам")
    if hi(profile.loyalty):
        bits.append("лоялен своим игрокам")
    elif lo(profile.loyalty):
        bits.append("продаст любого за хорошую цену")
    name = franchise.get("manager") or "Менеджер"
    style = "; ".join(bits) if bits else "сбалансированный управленец"
    return (f"Ты {name}, менеджер {franchise.get('name', '')}. Характер управленца: "
            f"{style}. Ты единственный мозг клуба: задаёшь плейбук на турнир и ведёшь "
            f"фронт-офис; твоя цель — титул верхней лиги.")


# --------------------------------------------------------------------------- #
# Front-office decisions (profile-weighted heuristics; LLM consults/overrides)
# --------------------------------------------------------------------------- #
def max_offer(profile: ManagerProfile, fair_value: int, treasury: int) -> int:
    """The most this manager would pay for a player worth ``fair_value``.
    Aggressive/risky pay over the odds; frugal stay under it. Never absurd."""
    mult = 1.0 + 0.45 * profile.aggression + 0.2 * profile.risk - 0.25 * profile.frugality
    mult = max(0.7, min(1.7, mult))
    return int(min(treasury, fair_value * mult))


def auction_bid(profile: ManagerProfile, item_value: int, treasury: int) -> int:
    """Bid for an auctioned item (training gradeslot / free agent), in character."""
    mult = 0.45 + 0.6 * profile.aggression + 0.2 * profile.risk - 0.3 * profile.frugality
    mult = max(0.0, mult)
    return int(max(0, min(treasury, round(item_value * mult))))


def wants_to_buy(world: dict, team: dict, target: dict, price: int,
                 profile: ManagerProfile) -> dict:
    """Manager's buy call: valuation signal, widened/narrowed by character and
    biased toward youth if that's the manager's thing."""
    fair = transfers.fair_value(target)
    ceiling = max_offer(profile, fair, team["treasury"])
    base = valuation.should_buy(world, team, target, price)
    youth_pull = profile.youth_focus > 0.6 and valuation.potential_score(target) > 1.05
    decision = bool((base["decision"] or youth_pull) and price <= ceiling
                    and team["treasury"] >= price)
    reason = base["reason"] if not youth_pull else "buy: young, high upside"
    if decision:
        reason = "buy: " + ("young, high upside" if youth_pull and not base["decision"]
                            else base["reason"].replace("buy: ", ""))
    elif price > ceiling:
        reason = "above what this manager will pay"
    return {"decision": decision, "reason": reason, "max_offer": ceiling, "fair_value": fair}


def wants_to_sell(world: dict, team: dict, player: dict, offer: int,
                  profile: ManagerProfile) -> dict:
    """Manager's sell call: base valuation, but a loyal manager demands more and a
    frugal one parts with talent more readily under pressure."""
    base = valuation.should_sell(world, team, player, offer)
    fair = base["fair_value"]
    demand = fair * (1.0 + 0.4 * profile.loyalty)   # loyalty raises the asking price
    if base["decision"] and offer < demand and profile.loyalty > 0.6:
        return {"decision": False, "reason": "loyal: holding out for more", "fair_value": fair}
    return base


def certificate_intent(profile: ManagerProfile, has_fork_budget: bool) -> str:
    """Which training op this manager favours spending a certificate on."""
    if has_fork_budget and profile.youth_focus > 0.6:
        return "create_player"
    if profile.aggression > 0.6:
        return "meta_adapt"
    if profile.patience > 0.6:
        return "teamplay"
    return "skill_up"


# --------------------------------------------------------------------------- #
# Tournament playbook (the manager's single pre-event tactical call) — heuristic
# scaffold; the LLM authors the same shape (see playbook.py, llm_manager.py).
# --------------------------------------------------------------------------- #
def set_tournament_playbook(world: dict, team: dict,
                            profile: ManagerProfile | None = None) -> dict:
    """Build + store the team's playbook for the tournament: base strategy (tilted
    by the manager's character) + the standard contingency/substitution overrides.
    The LLM manager overrides this in persona; this is the fallback."""
    if profile is None:
        profile = natural_profile(team["franchise_id"])
    tid = team["team_id"]
    plan = playbook.build_default_plan(world, tid)
    if profile.aggression > 0.66:
        plan["base"]["strategy"] = playbook.aggressive_comeback(world, tid)
    elif profile.patience > 0.66 and profile.aggression < 0.4:
        plan["base"]["strategy"] = playbook.safest(world, tid)
    return playbook.set_plan(world, tid, plan)


def run_planning_stage(world: dict) -> list[dict]:
    """Every manager sets its tournament playbook (heuristic; the LLM authors the
    same shape). Returns the chosen base strategy per team."""
    out = []
    for team in world["teams"].values():
        if len(team.get("roster", [])) < 3:
            continue
        plan = set_tournament_playbook(world, team)
        out.append({"team_id": team["team_id"], "base": plan["base"]["strategy"]})
    return out


# --------------------------------------------------------------------------- #
# LLM prompt context (front-office framing)
# --------------------------------------------------------------------------- #
def build_manager_situation(world: dict, team: dict) -> str:
    """Compact situation for the manager LLM — roster + budget. The manager owns
    both the front office and the tournament playbook (no coach)."""
    roster = ", ".join(f"{get_player(world, pid)['nickname']}"
                       f"({valuation.skill_score(get_player(world, pid)):.2f})"
                       for pid in team["roster"])
    return (f"Клуб {team['name']} ({team.get('league')}). Казна: {team['treasury']}.\n"
            f"Состав: {roster}.\n"
            f"Реши: плейбук на турнир + фронт-офис (трансферы/аукционы/сертификаты). "
            f"Цель — верхняя лига.")
