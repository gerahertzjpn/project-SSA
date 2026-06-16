"""World manager — the LLM front office (NOT tactics).

Role split (owner decision): the **coach** (an RL net, see coach.py) owns everything
in-match — lineup, substitutions, positions and the per-match order. The **manager**
no longer touches the game. The manager's job, and therefore its CHARACTER, is the
front office:

  - hire / replace COACHES (a new core duty — the manager picks the tactical brain),
  - buy / sell players, bid in auctions (free agents + training gradeslots),
  - decide how to spend training certificates, draft rookies,
  - all in service of one goal: the upper-league title.

So a manager's personality is a **management profile** (how aggressively it trades,
how it spends, youth vs proven, how much it invests in coaches, loyalty…), NOT a
playstyle. Profiles are NATURAL — drawn from each franchise's own id — so the field
spreads on its own rather than being a hand-set ladder. The heuristics here are the
decision scaffold the LLM consults and may override in character.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from src.world import coach, transfers, valuation
from src.world.world import get_player

MANAGER_SYSTEM_RULES = (
    "Ты МЕНЕДЖЕР клуба (фронт-офис). ТАКТИКУ в матче ставит ТРЕНЕР (нейросеть) — "
    "ты в саму игру не вмешиваешься. Твоя работа: нанимать/менять тренеров, "
    "покупать и продавать игроков, участвовать в аукционах (тренировки и свободные "
    "агенты), решать как тратить сертификаты тренировок, драфтить новичков. "
    "Цель ВСЕГДА — чемпионство в верхней лиге. Действуй в рамках своего "
    "управленческого характера, но без абсурда (никаких «топ за 1 иену»)."
)


@dataclass
class ManagerProfile:
    """Management character (0..1 traits). No playstyle here — that's the coach."""
    aggression: float      # how hard it chases deals / bids
    frugality: float       # how tightly it holds money
    youth_focus: float     # rookies/potential vs proven veterans
    risk: float            # willingness to gamble on upside / overpay
    patience: float        # keep the core vs churn the roster
    coach_priority: float  # how much it invests in hiring strong coaches
    loyalty: float         # keep own players vs sell for profit

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def natural_profile(franchise_id: str) -> ManagerProfile:
    """Diverse management profile drawn from the franchise's own id (stable,
    process-independent). Natural spread, not a hand-tuned table."""
    rng = random.Random(zlib.crc32(franchise_id.encode("utf-8")) ^ 0x5EED)
    r = lambda: round(rng.uniform(0.0, 1.0), 3)  # noqa: E731
    return ManagerProfile(aggression=r(), frugality=r(), youth_focus=r(), risk=r(),
                          patience=r(), coach_priority=r(), loyalty=r())


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
    if hi(profile.coach_priority):
        bits.append("вкладывается в сильных тренеров")
    elif lo(profile.coach_priority):
        bits.append("экономит на тренерах")
    if hi(profile.loyalty):
        bits.append("лоялен своим игрокам")
    elif lo(profile.loyalty):
        bits.append("продаст любого за хорошую цену")
    name = franchise.get("manager") or "Менеджер"
    style = "; ".join(bits) if bits else "сбалансированный управленец"
    return (f"Ты {name}, менеджер {franchise.get('name', '')}. Характер управленца: "
            f"{style}. В игру не вмешиваешься — это работа тренера; твоя цель — титул "
            f"верхней лиги через трансферы, тренеров и тренировки.")


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


def pick_coach(world: dict, team: dict, candidate_ids: list[str],
               profile: ManagerProfile) -> str | None:
    """Choose a coach to hire from ``candidate_ids`` (free coaches). High
    coach_priority managers grab the strongest available; low-priority managers
    only bother if it clearly beats their current coach."""
    if not candidate_ids:
        return None
    best = max(candidate_ids, key=lambda cid: world["coaches"][cid]["rating"])
    best_rating = world["coaches"][best]["rating"]
    current = coach.team_coach(world, team)
    current_rating = current["rating"] if current else -1.0
    # threshold to switch: low priority managers need a bigger upgrade
    needed = current_rating + (1.0 - profile.coach_priority) * 40.0
    if current is None or best_rating > needed:
        return best
    return None


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
# Coaching stage (managers pick coaches) — heuristic; LLM later
# --------------------------------------------------------------------------- #
def run_coaching_stage(world: dict) -> list[dict]:
    """Each manager considers hiring a better free coach, in character. Returns the
    moves made. (Heuristic stand-in for the LLM manager's coach decision.)"""
    moves = []
    for team in world["teams"].values():
        profile = natural_profile(team["franchise_id"])
        free = list(world.get("free_coaches", []))
        choice = pick_coach(world, team, free, profile)
        if choice and choice != team.get("coach"):
            coach.hire_coach(world, team["team_id"], choice)
            moves.append({"team_id": team["team_id"], "hired_coach": choice})
    return moves


# --------------------------------------------------------------------------- #
# LLM prompt context (front-office framing)
# --------------------------------------------------------------------------- #
def build_manager_situation(world: dict, team: dict) -> str:
    """Compact front-office situation for the manager LLM — roster, budget, coach,
    free coaches available. No tactics (that's the coach)."""
    roster = ", ".join(f"{get_player(world, pid)['nickname']}"
                       f"({valuation.skill_score(get_player(world, pid)):.2f})"
                       for pid in team["roster"])
    cur_coach = coach.team_coach(world, team)
    coach_str = f"{cur_coach['name']} (rating {cur_coach['rating']:.0f})" if cur_coach else "нет"
    n_free = len(world.get("free_coaches", []))
    return (f"Клуб {team['name']} ({team.get('league')}). Казна: {team['treasury']}. "
            f"Тренер: {coach_str}. Свободных тренеров: {n_free}.\n"
            f"Состав: {roster}.\n"
            f"Реши фронт-офис: тренер/трансферы/аукционы/сертификаты. Цель — верхняя лига.")
