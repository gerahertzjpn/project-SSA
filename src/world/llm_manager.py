"""Gated LLM decision layer for the World manager (the club's single brain).

Every decision goes through a GATE:
  - if the ``llm_manager`` feature flag is enabled AND a real Claude client is
    available → ask the LLM;
  - otherwise (locked / no SDK / no key / a malformed answer / any error) → run the
    deterministic heuristic and EMIT A FALLBACK NOTIFICATION into the event feed —
    never silent (owner requirement).

So the world runs IDENTICALLY with or without the LLM; the LLM only adds character.
Model is env-switchable (``WORLD_LLM_MODEL`` → ``REGIONALS_LLM_MODEL`` → opus-4-8 default) so
the same world can be replayed on Haiku/Sonnet/Opus and compared by eye.

Decision surfaces (WORLD_MODE_SPEC §0-bis):
  - draft pick      — L2, one call per pick (``draft_pick_fn``);
  - playbook        — L1, one call per tournament (``author_playbook``);
  - market plan     — L1, buy/sell/spend (``market_plan``);
  - auction bids    — L1, max-bid per team (``auction_max_bids``); plus the optional
                      🔴 L2 live auction, secondary + explicit-confirm (``assert_live_auction``).
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

from src.world import (events, feature_flags, manager, playbook, scouting, training_auction,
                       transfers, valuation)
from src.world.world import get_player

DEFAULT_MODEL = (os.environ.get("WORLD_LLM_MODEL")
                 or os.environ.get("REGIONALS_LLM_MODEL")
                 or "claude-opus-4-8")      # newest Opus by default; WORLD_LLM_MODEL env can override


# --------------------------------------------------------------------------- #
# Client + gate
# --------------------------------------------------------------------------- #
def get_client(model: Optional[str] = None):
    """Best-effort real Claude client (reuses the regionals wrapper). Returns None
    when the SDK or API key is unavailable — callers then run heuristics."""
    try:
        from src.regionals.llm import AnthropicLLM, LLMUnavailable  # noqa: PLC0415
    except Exception:
        return None
    try:
        return AnthropicLLM(model=model or DEFAULT_MODEL)
    except LLMUnavailable:
        return None
    except Exception:
        return None


def llm_enabled() -> bool:
    """The LLM manager is active only when the owner has unlocked the flag."""
    return feature_flags.is_enabled("llm_manager")


def _active(llm) -> bool:
    return bool(llm) and llm_enabled()


def _char_blurb(profile) -> str:
    """A one-line character read so each manager's market/training/playbook decisions diverge by
    personality (an aggressive club overpays + plays all-in; a frugal one saves; a youth club builds)."""
    if profile is None:
        return ""
    g = lambda a: getattr(profile, a, 0.0)  # noqa: E731 - terse local
    return (f" ТВОЙ ХАРАКТЕР: агрессия {g('aggression'):.0%}, риск {g('risk'):.0%}, "
            f"бережливость {g('frugality'):.0%}, ставка на молодёжь {g('youth_focus'):.0%}, "
            f"лояльность к составу {g('loyalty'):.0%}.")


# --------------------------------------------------------------------------- #
# Multi-season outlook — once per off-season the manager fixes a 2-3 season STANCE
# (build / contend / rebuild) + focus, stored on team['roster_plan']; the market + draft
# prompts read it so signings/picks serve a coherent vision, not just this-season opportunism.
# --------------------------------------------------------------------------- #
def _heuristic_stance(world: dict, team: dict) -> tuple:
    import statistics
    roster = [get_player(world, pid) for pid in team.get("roster", [])]
    if not roster:
        return "rebuild", "состав почти пуст — собираем заново"
    avg_seasons = statistics.mean(p.get("seasons_played", 0) for p in roster)
    avg_skill = statistics.mean(valuation.skill_score(p) for p in roster)
    if team.get("league") == "upper" and avg_skill > 1.05:
        return "contend", "состав силён — бьёмся за титул сейчас"
    if avg_seasons < 1.5:
        return "build", "молодой состав — растим под будущее, копим ядро"
    if avg_skill < 0.92:
        return "rebuild", "слабеем — перестройка: омоложение + точечное усиление"
    return "contend", "идём за стабильным результатом"


def season_outlook(world: dict, *, llm=None, feed: Optional[list] = None) -> dict:
    """Once per off-season each manager fixes a 2-3 season stance + focus on ``team['roster_plan']``.
    The market/draft prompts read it (``_roster_plan_blurb``) so a club's buying + picks follow a
    coherent multi-season vision. LLM when active; character heuristic fallback."""
    for team in world["teams"].values():
        if len(team.get("roster", [])) < 1:
            continue
        stance, focus = _heuristic_stance(world, team)
        if _active(llm):
            try:
                profile = manager.natural_profile(team["franchise_id"])
                stance, focus = _llm_season_outlook(world, team, llm, profile, (stance, focus))
            except Exception as exc:
                events.fallback(feed, f"outlook:{team['team_id']}", str(exc))
        team["roster_plan"] = {"season": world.get("season", 0), "stance": stance, "focus": focus}
    return {tid: t.get("roster_plan") for tid, t in world["teams"].items() if t.get("roster_plan")}


def _llm_season_outlook(world: dict, team: dict, llm, profile, default: tuple) -> tuple:
    from src.world import needs
    roster = [get_player(world, pid) for pid in team.get("roster", [])]
    ages = ", ".join(f"{p.get('nickname')}({int(p.get('soloq_rating', 1000))}, "
                     f"{p.get('seasons_played', 0)}сез)" for p in sorted(roster, key=lambda x: -x.get('soloq_rating', 0)))
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"stance": {"type": "string", "enum": ["build", "contend", "rebuild"]},
                             "focus": {"type": "string"}}, "required": ["stance", "focus"]}
    prompt = (f"Клуб {team['name']} ({'высшая' if team.get('league') == 'upper' else 'низшая'} лига)."
              f"{_char_blurb(profile)}\nСостав (рейтинг, сезонов сыграно): {ages}\n"
              f"ДЫРЫ: {needs.team_needs_brief(world, team)}\n--- РЕШЕНИЕ ---\n"
              f"Задай КУРС клуба на 2-3 сезона: build (растим молодёжь под будущее), contend (бьёмся за "
              f"титул сейчас) или rebuild (перестройка). focus — одна фраза: кого добираем, кого продаём, "
              f"на что копим. Это направит твои покупки и драфт.")
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, prompt, schema, max_tokens=160)
    stance = ans.get("stance")
    if stance not in ("build", "contend", "rebuild"):
        return default
    return stance, (ans.get("focus") or default[1])


def _roster_plan_blurb(team: dict) -> str:
    rp = team.get("roster_plan") or {}
    return f" КУРС КЛУБА (2-3 сезона): {rp['stance']} — {rp.get('focus', '')}." if rp.get("stance") else ""


def _budget_blurb(world: dict, team: dict) -> str:
    """Budget discipline: treasury vs the recurring wage bill + a runway read, so a manager weighs a
    signing against future sustainability, not just this season's cash."""
    bill = float(valuation.salary_bill(world, team))
    treas = float(team.get("treasury", 0))
    runway = (treas / bill) if bill > 0 else 99.0
    note = "тесно" if runway < 1.5 else ("комфортно" if runway > 4 else "норм")
    return f" БЮДЖЕТ: казна {int(treas)}, зарплаты/сезон {int(bill)} (запас ~{runway:.1f} сез., {note})."


# --------------------------------------------------------------------------- #
# Draft — L2 (one LLM call per pick; heuristic fallback inside balanced_draft)
# --------------------------------------------------------------------------- #
def draft_pick_fn(world: dict, *, llm=None, feed: Optional[list] = None
                  ) -> Optional[Callable[[dict, list], Optional[str]]]:
    """A ``pick_fn(team, available)`` for ``season.balanced_draft`` (L2). Returns
    ``None`` when the LLM is off → the draft uses its best-by-value heuristic with no
    per-pick noise. When on, each pick is an LLM call; on ANY failure the pick falls
    back (returns None) and a notification is emitted."""
    if not _active(llm):
        return None

    def pick_fn(team: dict, available: list) -> Optional[str]:
        ids = [p["player_id"] for p in available]
        try:
            schema = {
                "type": "object",
                "properties": {
                    "player_id": {"type": "string", "enum": ids[:200]},
                    "reason": {"type": "string"},
                },
                "required": ["player_id", "reason"], "additionalProperties": False,
            }
            ans = llm.structured(manager.MANAGER_SYSTEM_RULES,
                                 _draft_prompt(world, team, available), schema)   # 900 floor (was 200 -> truncated)
            pid = ans.get("player_id")
            if pid in ids:
                events.emit(feed, "draft", f"{team['name']}: ЛЛМ выбрал "
                            f"{world['players'][pid]['nickname']} — {ans.get('reason', '')}")
                return pid
            raise ValueError(f"LLM returned unknown id {pid!r}")
        except Exception as exc:
            print(f"[draft] LLM pick FAILED for {team['team_id']}: {exc}", file=sys.stderr)  # was silent
            events.fallback(feed, f"draft:{team['team_id']}", str(exc))
            return None     # -> balanced_draft picks best by value

    return pick_fn


def _draft_prompt(world: dict, team: dict, available: list) -> str:
    have = ", ".join(world["players"][pid]["nickname"] for pid in team["roster"]) or "пусто"
    pool = "; ".join(f"{p['player_id']}:{p['nickname']}({p['faction']},"
                     f"SoloQ{int(p['soloq_rating'])})" for p in available[:40])
    return (f"Драфт. Клуб {team['name']}.{_roster_plan_blurb(team)} Уже взял: {have}. Доступны: {pool}.\n"
            f"Выбери ОДНОГО игрока под свой состав, КУРС клуба и цель — титул верхней лиги.")


# --------------------------------------------------------------------------- #
# Playbook — L1 (one LLM call per tournament; heuristic fallback)
# --------------------------------------------------------------------------- #
def author_playbook(world: dict, team: dict, *, llm=None, feed: Optional[list] = None,
                    profile: Optional[manager.ManagerProfile] = None) -> dict:
    """Author + store the team's tournament playbook. LLM when active; else (or on
    failure) the heuristic ``manager.set_tournament_playbook``. Emits one chronicle
    event either way; a fallback notification when an active LLM fails."""
    plan = None
    src = "эвристика"
    profile = profile or manager.natural_profile(team["franchise_id"])
    if _active(llm):
        try:
            plan = playbook.set_plan(world, team["team_id"], _llm_playbook(world, team, llm, profile))
            src = "ЛЛМ"
        except Exception as exc:
            # LOUD: a silent fallback here is how every playbook quietly became heuristic before.
            print(f"[playbook] LLM authoring FAILED for {team['team_id']}: {exc}", file=sys.stderr)
            events.fallback(feed, f"playbook:{team['team_id']}", str(exc))
    if plan is None:
        plan = manager.set_tournament_playbook(world, team, profile)
    plan["source"] = "llm" if src == "ЛЛМ" else "heuristic"   # persisted so the UI/replan can tell them apart
    events.emit(feed, "playbook", f"{team['name']}: плейбук ({src}), база "
                f"«{plan['base']['strategy']}»", team_id=team["team_id"], source=src)
    return plan


# The ONLY conditions the LLM may use (the schema enum is these keys). Each maps to a check that
# _condition_matches actually implements, so the LLM can never author a rule that won't fire — an
# unknown condition is rejected by _playbook_from_llm (-> heuristic fallback). Covers BOTH the
# opponent (faction + their last call, all 6 strategies) AND the series state (score/streak).
_CONDITION_MAP = {
    # --- by OPPONENT faction ---
    "opp_robots": {"opp_faction_majority": "robots"},
    "opp_humans": {"opp_faction_majority": "humans"},
    "opp_infected": {"opp_faction_majority": "infected"},
    # --- by OPPONENT's last-known strategy (all six) ---
    "opp_plays_standard": {"opp_last_strategy": "standard"},
    "opp_plays_core_rush": {"opp_last_strategy": "core_rush"},
    "opp_plays_all_in": {"opp_last_strategy": "all_in"},
    "opp_plays_turtle": {"opp_last_strategy": "turtle"},
    "opp_plays_macro": {"opp_last_strategy": "macro"},
    "opp_plays_baron": {"opp_last_strategy": "baron"},
    # --- by SERIES state ---
    "behind_in_series": {"series_behind": True},
    "ahead_in_series": {"series_ahead": True},
    "series_lost1": {"series_lost": 1},
    "playoff_behind2": {"stage": "playoff", "playoff_losing_streak": 2},
    "playoff_ahead": {"stage": "playoff", "series_ahead": True},
}


def _llm_playbook(world: dict, team: dict, llm, profile=None) -> dict:
    strat_enum = list(playbook.STRATEGY_IDS)
    roster = list(team["roster"])
    # conditional substitution lineup (overrides): which player at A1/A2/A3 — PARTIAL allowed
    lineup_obj = {"type": "object", "additionalProperties": False,
                  "properties": {p: {"type": "string", "enum": roster}
                                 for p in playbook.POSITIONS}}
    # base lineup: the LLM MUST assign all 3 starter positions itself (no heuristic) — every
    # position is required, each filled by a CURRENT-roster player (so newly drafted/bought players
    # get an LLM-decided slot every season).
    base_lineup_obj = {"type": "object", "additionalProperties": False,
                       "properties": {p: {"type": "string", "enum": roster}
                                      for p in playbook.POSITIONS},
                       "required": list(playbook.POSITIONS)}
    schema = {
        "type": "object",
        "properties": {
            "base": {"type": "object", "additionalProperties": False,
                     "properties": {"strategy": {"type": "string", "enum": strat_enum},
                                    "lineup": base_lineup_obj},
                     "required": ["strategy", "lineup"]},
            "overrides": {
                # NB: structured-output API rejects array constraints (maxItems) — cap in code instead.
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "condition": {"type": "string", "enum": list(_CONDITION_MAP)},
                        "strategy": {"type": "string", "enum": strat_enum},
                        "lineup": lineup_obj,        # conditional substitution / placement
                        "note": {"type": "string"},
                    },
                    "required": ["condition", "strategy"],
                },
            },
        },
        "required": ["base", "overrides"], "additionalProperties": False,
    }
    # 6 overrides × {condition, strategy, lineup, RU note} can exceed 700 tokens (Cyrillic is
    # token-heavy) — too small a budget truncates the JSON mid-string -> parse error -> heuristic fallback.
    # Measured worst case ~950 (Haiku) / ~700 (Opus, terser); 3000 = ~3x margin. A high cap is free
    # (billed on ACTUAL output), so over-provision rather than risk truncation. Opus needs NO more here:
    # `structured` sends no `thinking` param, so there are no thinking tokens competing for the budget.
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, _playbook_prompt(world, team, profile),
                         schema, max_tokens=3000)
    return _playbook_from_llm(ans, roster)


def _clean_lineup(lineup, roster: list):
    """Keep only valid {position: rostered-pid} entries (partial allowed)."""
    if not lineup:
        return None
    out = {pos: pid for pos, pid in lineup.items()
           if pos in playbook.POSITIONS and pid in roster}
    return out or None


def _playbook_from_llm(ans: dict, roster: list) -> dict:
    """Translate the LLM answer into the internal playbook shape, validating the
    strategy vocabulary + lineups (raises on anything unknown → caller falls back).
    Honors the LLM authoring positions/substitutions (per-position player + bench)."""
    base = ans["base"]
    bstrat = base["strategy"]
    if bstrat not in playbook.STRATEGY_IDS:
        raise ValueError(f"unknown base strategy {bstrat!r}")
    overrides = []
    for ov in ans.get("overrides", [])[:6]:           # cap at 6 here (schema can't constrain array length)
        when = _CONDITION_MAP.get(ov.get("condition"))
        strat = ov.get("strategy")
        if when is None or strat not in playbook.STRATEGY_IDS:
            raise ValueError(f"bad override {ov!r}")
        overrides.append({"when": when, "strategy": strat,
                          "lineup": _clean_lineup(ov.get("lineup"), roster),
                          "note": ov.get("note", ov["condition"])})
    return {"base": {"strategy": bstrat, "lineup": _clean_lineup(base.get("lineup"), roster)},
            "overrides": overrides}


def _roster_intel(world: dict, team: dict) -> str:
    """Per-player STRENGTH so the manager can field the best 3 + substitute intelligently: faction,
    SoloQ rating, key attributes, and the strategy each player executes best. Without this the LLM saw
    only names (in roster order, starters first) and just re-picked the old starters — so signings and
    rookies, however strong, never made the lineup. Sorted by rating so the depth chart is obvious."""
    rows = []
    for pid in sorted(team["roster"], key=lambda x: -world["players"].get(x, {}).get("soloq_rating", 0)):
        p = world["players"].get(pid) or {}
        a = p.get("attributes") or {}
        best = max(playbook.STRATEGY_IDS, key=lambda s: playbook.player_exec(p, s))
        tag = " НОВИЧОК" if (p.get("lineage") or {}).get("origin") in ("rookie_fork", "create_player") else ""
        rows.append(f"{p.get('nickname', '?')}({pid}) {p.get('faction', '?')} "
                    f"R{int(p.get('soloq_rating', 1000))} "
                    f"атк{a.get('attack', 0)}/защ{a.get('defense', 0)}/эко{a.get('economy', 0)}/"
                    f"темп{a.get('tempo', 0)} лучш:{best}{tag}")
    return "\n  ".join(rows)


def _playbook_prompt(world: dict, team: dict, profile=None) -> str:
    """Playbook prompt GROUNDED in past games (B): own past calls + winrate, league
    meta, measured counters, per-position winrate, H2H vs league rivals — PLUS the roster's
    per-player strength so lineup + substitution choices are informed, not blind to name order,
    PLUS the manager's character so different managers author different plans."""
    league = team.get("league")
    opponents = [t["team_id"] for t in world["teams"].values()
                 if t.get("league") == league and t["team_id"] != team["team_id"]][:6]
    intel = scouting.render_intel(scouting.intel_report(world, team["team_id"], opponents))
    n = len(team["roster"])
    return (f"Клуб {team['name']}.{_char_blurb(profile)}\n"
            f"Состав ({n} игроков на 3 стартовых слота), по убыванию рейтинга:\n  "
            f"{_roster_intel(world, team)}\n"
            f"--- РАЗВЕДКА (прошлые игры) ---\n{intel}\n"
            f"--- ЗАДАЧА ---\nСоставь ПЛЕЙБУК на турнир:\n"
            f"1) Базовая стратегия из {list(playbook.STRATEGY_IDS)} — по РАЗВЕДКЕ (что реально работает "
            f"у нас и в мете) И по тому, что твои игроки исполняют лучше всего (поле «лучш»).\n"
            f"2) Стартовый состав base.lineup: 3 РАЗНЫХ игрока на A1/A2/A3 — выбери СИЛЬНЕЙШИХ под свою "
            f"базу (рейтинг + профильные атрибуты + синергия фракций). Состав НЕ священен: если новичок "
            f"или подписанный игрок сильнее старого старта — ставь его, иначе зачем он в клубе.\n"
            f"3) Условные ЗАМЕНЫ (overrides): используй ГЛУБИНУ — под конкретного соперника/фракцию/счёт "
            f"серии выводи со скамейки нужного специалиста (свой lineup в override), а не только меняй "
            f"стратегию. Глубину состава покупают именно ради таких замен.\n"
            f"Цель — титул верхней лиги.")


# --------------------------------------------------------------------------- #
# Market — L1 (one call: buy/sell/spend). Heuristic fallback.
# --------------------------------------------------------------------------- #
def market_plan(world: dict, team: dict, *, llm=None, feed: Optional[list] = None,
                profile: Optional[manager.ManagerProfile] = None) -> dict:
    """A team's L1 market plan: ``{buy:[{player_id,max_price}], sell:[{player_id,
    asking}], spend: <op>}``. Code executes it (ties → highest bidder)."""
    profile = profile or manager.natural_profile(team["franchise_id"])
    if _active(llm):
        try:
            return _llm_market(world, team, llm, profile)
        except Exception as exc:
            events.fallback(feed, f"market:{team['team_id']}", str(exc))
    return _heuristic_market(world, team, profile)


def _heuristic_market(world: dict, team: dict, profile) -> dict:
    roster = sorted(team["roster"], key=lambda pid: world["players"][pid]["soloq_rating"])
    sell = []
    if len(roster) > 3 and profile.loyalty < 0.4:        # churny manager lists its weakest
        weakest = world["players"][roster[0]]
        sell.append({"player_id": weakest["player_id"], "asking": transfers.fair_value(weakest)})
    return {"buy": [], "sell": sell,
            "spend": manager.certificate_intent(profile, has_fork_budget=False)}


def _llm_market(world: dict, team: dict, llm, profile) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "sell": {"type": "array", "items": {"type": "object", "properties": {
                "player_id": {"type": "string", "enum": list(team["roster"])},
                "asking": {"type": "integer"}}, "required": ["player_id", "asking"],
                "additionalProperties": False}},
            "spend": {"type": "string", "enum": ["skill_up", "meta_adapt", "teamplay",
                                                 "create_player"]},
        },
        "required": ["sell", "spend"], "additionalProperties": False,
    }
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES,
                         manager.build_manager_situation(world, team), schema, max_tokens=400)
    return {"buy": [], "sell": ans.get("sell", []),
            "spend": ans.get("spend") or manager.certificate_intent(profile, has_fork_budget=False)}


# --------------------------------------------------------------------------- #
# Training AUCTION — L1. The manager (LLM) decides whether/how much to BID for a
# gradeslot; winners bank a step-valued certificate. Then it decides how to SPEND a
# won certificate (which op + target). Both have a notified heuristic fallback.
# (Economy lives in training_auction.py; the backend executes via training_ops.)
# --------------------------------------------------------------------------- #
def auction_max_bids(world: dict, team_ids: list[str], item_value: int, *,
                     llm=None, feed: Optional[list] = None) -> dict:
    """L1: each team's max bid for a training gradeslot. LLM decides whether/how much to bid
    (0 = pass) when active; else (or on failure) the character heuristic ``manager.auction_bid``,
    with a fallback notification. The CODE then runs the sealed auction (auction_one_training)."""
    teams = [world["teams"][t] for t in team_ids]

    def _roster_str(t: dict) -> float:
        rs = [world["players"][p]["soloq_rating"] for p in t.get("roster", []) if p in world["players"]]
        return (sum(rs) / len(rs)) if rs else 1000.0

    treas = [float(t.get("treasury", 0)) for t in teams] or [0.0]
    rstr = [_roster_str(t) for t in teams] or [1000.0]
    league_ctx = {"avg_treasury": sum(treas) / len(treas), "avg_roster": sum(rstr) / len(rstr)}

    bids = {}
    for tid in team_ids:
        team = world["teams"][tid]
        profile = manager.natural_profile(team["franchise_id"])
        bid = None
        if _active(llm):
            try:
                bid = _llm_auction_bid(world, team, llm, item_value, profile, league_ctx, _roster_str(team))
            except Exception as exc:
                events.fallback(feed, f"train-bid:{tid}", str(exc))
        if bid is None:
            bid = manager.auction_bid(profile, item_value, team["treasury"])
        bids[tid] = max(0, min(int(bid), int(team.get("treasury", 0))))
    return bids


def _llm_auction_bid(world: dict, team: dict, llm, item_value: int, profile, league_ctx: dict,
                     my_roster: float) -> int:
    """A STRATEGIC, per-club bid. Each manager judges its OWN situation (wealth, roster strength,
    league, character) so bids genuinely VARY — a weak/ambitious/rich club splurges, a strong/frugal
    one saves or passes — instead of everyone copying one shared anchor."""
    roster = ", ".join(f"{world['players'][pid]['nickname']}"
                       f"(SoloQ{int(world['players'][pid]['soloq_rating'])})"
                       for pid in team["roster"]) or "пусто"
    avg_t, avg_r = league_ctx["avg_treasury"], league_ctx["avg_roster"]
    treasury = float(team.get("treasury", 0))
    wealth = ("богатая" if treasury > avg_t * 1.15 else "бедная" if treasury < avg_t * 0.85 else "средняя")
    strength = ("сильный" if my_roster > avg_r + 25 else "слабый" if my_roster < avg_r - 25 else "средний")
    lg = "высшая" if team.get("league") == "upper" else "низшая"
    # manager character drives WILLINGNESS to pay (aggressive/risky overpay, frugal save)
    char = (f"агрессия {profile.aggression:.0%}, риск {profile.risk:.0%}, "
            f"бережливость {profile.frugality:.0%}")
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"max_bid": {"type": "integer"}, "reason": {"type": "string"}},
              "required": ["max_bid"]}
    prompt = (f"Закрытый аукцион тренировочных слотов (1 слот = {training_auction.DEFAULT_SLOT_STEPS:,} "
              f"шагов PPO для одного из твоих игроков). Клуб {team['name']} — {lg} лига.\n"
              f"ТВОЯ СИТУАЦИЯ: казна {int(treasury)} ({wealth}; средняя по лиге ~{int(avg_t)}). "
              f"Состав {strength} (твой ср. SoloQ {int(my_roster)} vs лига ~{int(avg_r)}): {roster}.\n"
              f"ТВОЙ ХАРАКТЕР: {char}.\n"
              f"РЕШИ САМ, сколько слот стоит ИМЕННО ТЕБЕ сейчас, исходя из твоей ситуации и амбиций — "
              f"НЕ копируй чужой ориентир и не ставь шаблонный процент от казны. Слабому/амбициозному "
              f"клубу тренировка ценнее (ставь выше, можно перебить рынок); сильному/бережливому — "
              f"сэкономь или вовсе пропусти (0). Рынок примерно ~{item_value}, но это лишь грубый "
              f"ориентир. max_bid — целое, 0 = пропуск.")
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, prompt, schema, max_tokens=300)
    return int(ans.get("max_bid", 0))


def spend_certificates(world: dict, team: dict, *, llm=None, feed: Optional[list] = None,
                       config: Optional["training_auction.TrainingConfig"] = None) -> Optional[dict]:
    """A team that WON certificate(s) at the auction decides how to spend them: WHICH op, WHICH
    target (player / 3 starters / fork parent), and WHICH certs to combine — or DEFER (bank them).
    LLM when active, else the character heuristic, both validated through request_spend. Returns a
    pending training job or None (deferred / nothing to spend)."""
    tid = team["team_id"]
    wallet = training_auction.team_certificates(world, tid)
    if not wallet:
        return None
    netted = [p for p in (get_player(world, pid) for pid in team.get("roster", [])) if p.get("net_id")]
    if len(netted) < 3:
        return None
    decision = None
    if _active(llm):
        try:
            decision = _llm_spend(world, team, netted, wallet, llm)
        except Exception as exc:
            events.fallback(feed, f"train-spend:{tid}", str(exc))
    if decision is None:
        decision = _heuristic_spend(team, netted, wallet)
    if decision is None:                                  # defer: keep the certs banked
        events.emit(feed, "train", f"{team['name']}: сертификаты в банке (тренировку отложили)",
                    team_id=tid)
        return None
    op, target, cert_ids = decision
    try:
        job = training_auction.request_spend(world, tid, op, cert_ids, target=target, config=config)
    except ValueError as exc:                             # e.g. fork needs >= FORK_MIN_STEPS
        events.fallback(feed, f"train-spend:{tid}", str(exc))
        return None
    job["player_name"] = target.get("player_name") or _spend_name(world, team, op, target)
    events.emit(feed, "train", f"{team['name']}: тренировка {op} → {job['player_name']} "
                f"({job['total_steps']:,} шагов)", team_id=tid, source="ЛЛМ" if _active(llm) else "эвристика")
    return job


def _spend_name(world: dict, team: dict, op: str, target: dict) -> str:
    if op == "teamplay":
        return f"{team['name']} · тройка"
    pid = target.get("player_id")
    nick = world["players"][pid]["nickname"] if pid in world.get("players", {}) else pid
    return f"{team['name']} · новичок от {nick}" if op == "create_player" else nick


def _heuristic_spend(team: dict, netted: list, wallet: list):
    """Fallback: spend on the character op for the WEAKEST netted player, combining ALL certs
    (fork only if the combined steps reach the fork floor)."""
    profile = manager.natural_profile(team["franchise_id"])
    cert_ids = [c["cert_id"] for c in wallet]
    total = sum(c["steps"] for c in wallet)
    allow_fork = total >= training_auction.FORK_MIN_STEPS
    op = manager.certificate_intent(profile, has_fork_budget=allow_fork)
    weakest = min(netted, key=valuation.skill_score)
    if op == "teamplay":
        starters = sorted(netted, key=valuation.skill_score, reverse=True)[:3]
        target = {"kind": "team", "net_ids": [p["net_id"] for p in starters],
                  "player_ids": [p["player_id"] for p in starters]}
    else:
        target = {"kind": "parent" if op == "create_player" else "player",
                  "net_id": weakest["net_id"], "player_id": weakest["player_id"]}
    return op, target, cert_ids


def _llm_spend(world: dict, team: dict, netted: list, wallet: list, llm):
    by_pid = {p["player_id"]: p for p in netted}
    ids = list(by_pid)
    cert_ids = [c["cert_id"] for c in wallet]
    total = sum(c["steps"] for c in wallet)
    ops = ["skill_up", "meta_adapt", "style_spec", "obedience_tune", "teamplay"]
    if total >= training_auction.FORK_MIN_STEPS:
        ops.append("create_player")
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "spend": {"type": "boolean"},
            "op": {"type": "string", "enum": ops},
            "player_id": {"type": "string", "enum": ids},
            "player_ids": {"type": "array", "items": {"type": "string", "enum": ids}},  # capped to 3 in code below
            "reason": {"type": "string"},
        }, "required": ["spend"],
    }
    rows = "; ".join(f"{p['player_id']}:{p['nickname']}({p['faction']},SoloQ{int(p['soloq_rating'])},"
                     f"скилл{valuation.skill_score(p):.2f})" for p in netted)
    menu = ("skill_up=общий буст; meta_adapt=ПРОТИВ топов меты; style_spec=специализация в стиле; "
            "obedience_tune=послушание приказам; teamplay=сыгранность тройки; "
            "create_player=ФОРК нового игрока от родителя")
    from src.world import needs
    profile = manager.natural_profile(team["franchise_id"])
    prompt = (f"Клуб {team['name']}.{_char_blurb(profile)}{_roster_plan_blurb(team)} Выиграл сертификаты "
              f"тренировки: {len(wallet)} шт = {total:,} шагов.\n"
              f"ТВОИ ДЫРЫ: {needs.team_needs_brief(world, team)}\n"
              f"Игроки с нейронками:\n{rows}.\nВиды: {menu}.\n--- РЕШЕНИЕ ---\n"
              f"Тратить сейчас (spend) или отложить в банк? Заземли выбор в свои ДЫРЫ: качай слабого "
              f"стартера (skill_up), бери meta_adapt если проигрываешь мете, create_player если нужен "
              f"новый игрок старт-уровня, teamplay для сыгранности тройки. Выбери вид (op) и КОГО "
              f"(player_id; teamplay — троих в player_ids; create_player — родитель для форка в player_id).")
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, prompt, schema, max_tokens=300)
    if not ans.get("spend"):
        return None
    op = ans.get("op")
    if op not in ops:
        raise ValueError(f"LLM returned bad spend op {op!r}")
    if op == "teamplay":
        pids = [pid for pid in (ans.get("player_ids") or []) if pid in by_pid][:3]
        if len(pids) < 3:
            pids = [p["player_id"] for p in sorted(netted, key=valuation.skill_score, reverse=True)[:3]]
        target = {"kind": "team", "net_ids": [by_pid[pid]["net_id"] for pid in pids], "player_ids": pids}
    else:
        pid = ans.get("player_id")
        if pid not in by_pid:
            raise ValueError(f"LLM returned bad player_id {pid!r}")
        p = by_pid[pid]
        target = {"kind": "parent" if op == "create_player" else "player",
                  "net_id": p["net_id"], "player_id": pid}
    return op, target, cert_ids


# --------------------------------------------------------------------------- #
# Transfer market — each club with bench space bids for ONE free agent to add
# DEPTH (so mid-series subs have someone to bring in). Multi-club contention is
# resolved by the sealed FA auction (transfers.resolve_free_agent_auction).
# --------------------------------------------------------------------------- #
def transfer_bids(world: dict, *, llm=None, feed: Optional[list] = None) -> list[dict]:
    """Bids ``[{team_id, player_id, amount}]`` for the free-agent auction — each club below the
    roster cap with money bids for one free agent (LLM decides whether/who/how much; heuristic
    fallback). The auction then resolves contention (several clubs can want the same player)."""
    from src.world import freeagents, transfers
    from src.world.world import MAX_ROSTER
    fa_ids = freeagents.free_agents(world)
    if not fa_ids:
        return []
    fas = [get_player(world, pid) for pid in fa_ids]
    bids = []
    for team in world["teams"].values():
        if len(team.get("roster", [])) >= MAX_ROSTER or team.get("treasury", 0) <= 0:
            continue
        profile = manager.natural_profile(team["franchise_id"])
        bid = None
        if _active(llm):
            try:
                bid = _llm_transfer_bid(world, team, fas, llm, profile)
            except Exception as exc:
                events.fallback(feed, f"transfer:{team['team_id']}", str(exc))
        if bid is None:
            bid = _heuristic_transfer_bid(world, team, fas, profile)
        if bid:
            bids.append(bid)
    return bids


def _affordable(team: dict, fas: list, *, by_value=True):
    from src.world import transfers
    aff = [p for p in fas if team["treasury"] >= transfers.signing_bonus(p)]
    aff.sort(key=lambda p: valuation.skill_score(p), reverse=by_value)
    return aff


def _heuristic_transfer_bid(world: dict, team: dict, fas: list, profile):
    from src.world import transfers
    aff = _affordable(team, fas)
    if not aff:
        return None
    p = aff[0]                                   # best affordable free agent
    amount = int(round(transfers.fair_value(p) * (1.0 + 0.15 * profile.aggression)))
    amount = min(amount, int(team["treasury"]))
    return {"team_id": team["team_id"], "player_id": p["player_id"], "amount": amount}


def _llm_transfer_bid(world: dict, team: dict, fas: list, llm, profile=None):
    from src.world import transfers, needs
    from src.world.world import MAX_ROSTER
    aff = _affordable(team, fas)[:30]
    if not aff:
        return None
    ids = [p["player_id"] for p in aff]
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"sign": {"type": "boolean"},
                             "player_id": {"type": "string", "enum": ids},
                             "intent": {"type": "string", "enum": ["усиление-старта", "глубина"]},
                             "max_bid": {"type": "integer"}, "reason": {"type": "string"}},
              "required": ["sign"]}
    prompt = (f"Клуб {team['name']}. Ростер {len(team['roster'])}/{MAX_ROSTER}.{_char_blurb(profile)}"
              f"{_roster_plan_blurb(team)}{_budget_blurb(world, team)}\n"
              f"ТВОИ ДЫРЫ: {needs.team_needs_brief(world, team)}\n"
              f"РЫНОК (свободные агенты, отранжированы ПРОТИВ твоего слабейшего старта):\n"
              f"{needs.market_board(world, team, aff)}\n--- РЕШЕНИЕ ---\n"
              f"Сначала смотри на УСИЛЕНИЕ: если есть игрок с тегом АПГРЕЙД-СТАРТА по карману, который "
              f"реально усиливает старт или закрывает твою дыру (слабый стартер / перекос фракций / "
              f"пробел под мету) — ПОДПИШИ его (intent=усиление-старта). Это важнее, чем копить. Если "
              f"апгрейда нет — можешь взять «глубину» под замены (intent=глубина) или пропустить "
              f"(sign=false). max_bid — бонус сгорает; несколько клубов могут хотеть одного, победит "
              f"высшая ставка. Бид под силу казне.")
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, prompt, schema, max_tokens=240)
    if not ans.get("sign") or ans.get("player_id") not in ids:
        return None
    amount = int(ans.get("max_bid") or transfers.fair_value(get_player(world, ans["player_id"])))
    return {"team_id": team["team_id"], "player_id": ans["player_id"],
            "amount": min(amount, int(team["treasury"]))}


def llm_club_transfer_bids(world: dict, *, llm=None, feed: Optional[list] = None) -> list[dict]:
    """Club-to-club bids — the LLM decides whether to BUY a player from ANOTHER club to strengthen the
    first team (a guarded starter-upgrade) or add depth, reasoned from team needs + a board of buyable
    players. Heuristic fallback per buyer. Returns ``[{team_id, player_id, amount}]`` for
    ``transfers.resolve_transfer_auction`` (pays the selling club). This is the conscious LLM market
    decision that was missing — club transfers used to be heuristic-only."""
    from src.world.world import MAX_ROSTER
    bids = []
    for buyer in world["teams"].values():
        if len(buyer.get("roster", [])) >= MAX_ROSTER or buyer.get("treasury", 0) <= 0:
            continue
        cands = transfers.club_transfer_candidates(world, buyer)
        if not cands:
            continue
        bid = None
        if _active(llm):
            try:
                profile = manager.natural_profile(buyer["franchise_id"])
                bid = _llm_club_bid(world, buyer, cands, llm, profile)
            except Exception as exc:
                events.fallback(feed, f"club:{buyer['team_id']}", str(exc))
        if bid is None:                                  # heuristic: best candidate, upgrade-preferred
            best = max(cands, key=lambda c: (c["value"] + (1.0 if c["upgrade"] else 0.0)))
            bid = {"team_id": buyer["team_id"], "player_id": best["player_id"], "amount": best["price"]}
        bids.append(bid)
    return bids


def _llm_club_bid(world: dict, buyer: dict, cands: list, llm, profile):
    from src.world import needs
    by_pid = {c["player_id"]: c for c in cands}
    ids = list(by_pid)
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"buy": {"type": "boolean"},
                             "player_id": {"type": "string", "enum": ids},
                             "max_bid": {"type": "integer"}, "reason": {"type": "string"}},
              "required": ["buy"]}
    prompt = (f"Клуб {buyer['name']}.{_char_blurb(profile)}{_roster_plan_blurb(buyer)}"
              f"{_budget_blurb(world, buyer)}\nТВОИ ДЫРЫ: {needs.team_needs_brief(world, buyer)}\n"
              f"ИГРОКИ ДРУГИХ КЛУБОВ, доступные к покупке (цена платится клубу-продавцу):\n"
              f"{needs.club_market_board(world, buyer, cands)}\n--- РЕШЕНИЕ ---\n"
              f"Купить ли игрока у другого клуба (buy), КОГО (player_id) и макс. цену (max_bid)? Бери "
              f"того, кто с тегом АПГРЕЙД-СТАРТА закрывает твою дыру (слабый стартер / перекос фракций / "
              f"пробел под мету) — за реальное усиление старта переплатить не жалко. Глубину можно, но "
              f"усиление важнее. Не покупать (buy=false) — тоже нормально.")
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, prompt, schema, max_tokens=240)
    if not ans.get("buy") or ans.get("player_id") not in by_pid:
        return None
    cand = by_pid[ans["player_id"]]
    amount = max(int(ans.get("max_bid") or cand["price"]), cand["price"])   # meet the (premium) price
    if amount > int(buyer["treasury"]):
        return None
    return {"team_id": buyer["team_id"], "player_id": ans["player_id"], "amount": amount}


# --------------------------------------------------------------------------- #
# Auctions — 🔴 L2 live bidding is secondary + explicit-confirm
# --------------------------------------------------------------------------- #


def live_auction_available() -> bool:
    return feature_flags.is_enabled("live_auction_l2")


def assert_live_auction(confirmed: bool) -> str:
    """🔴 L2 live bidding is SECONDARY (default is L1). It requires BOTH the
    ``live_auction_l2`` flag AND an explicit owner confirmation, because it costs a
    lot of time and money. Returns the warning to show; raises otherwise."""
    if not feature_flags.is_enabled("live_auction_l2"):
        raise feature_flags.FeatureLocked(
            "live_auction_l2 — второстепенная фича, по умолчанию заблокирована (дефолт L1)")
    if not confirmed:
        raise PermissionError(
            "🔴 Живые торги L2 займут МНОГО времени и денег. Нужно явное подтверждение "
            "(нажми «ДА», передай confirmed=True).")
    return "🔴 Запущены живые торги L2 — это долго и дорого."
