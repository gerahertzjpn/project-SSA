"""Gated LLM decision layer for the World manager (the club's single brain).

Every decision goes through a GATE:
  - if the ``llm_manager`` feature flag is enabled AND a real Claude client is
    available → ask the LLM;
  - otherwise (locked / no SDK / no key / a malformed answer / any error) → run the
    deterministic heuristic and EMIT A FALLBACK NOTIFICATION into the event feed —
    never silent (owner requirement).

So the world runs IDENTICALLY with or without the LLM; the LLM only adds character.
Model is env-switchable (``WORLD_LLM_MODEL`` → ``REGIONALS_LLM_MODEL`` → opus-4-8) so
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
from typing import Callable, Optional

from src.world import events, feature_flags, manager, playbook, scouting, transfers, valuation
from src.world.world import get_player

DEFAULT_MODEL = (os.environ.get("WORLD_LLM_MODEL")
                 or os.environ.get("REGIONALS_LLM_MODEL")
                 or "claude-opus-4-8")


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
                                 _draft_prompt(world, team, available), schema, max_tokens=200)
            pid = ans.get("player_id")
            if pid in ids:
                events.emit(feed, "draft", f"{team['name']}: ЛЛМ выбрал "
                            f"{world['players'][pid]['nickname']} — {ans.get('reason', '')}")
                return pid
            raise ValueError(f"LLM returned unknown id {pid!r}")
        except Exception as exc:
            events.fallback(feed, f"draft:{team['team_id']}", str(exc))
            return None     # -> balanced_draft picks best by value

    return pick_fn


def _draft_prompt(world: dict, team: dict, available: list) -> str:
    have = ", ".join(world["players"][pid]["nickname"] for pid in team["roster"]) or "пусто"
    pool = "; ".join(f"{p['player_id']}:{p['nickname']}({p['faction']},"
                     f"SoloQ{int(p['soloq_rating'])})" for p in available[:40])
    return (f"Драфт. Клуб {team['name']}. Уже взял: {have}. Доступны: {pool}.\n"
            f"Выбери ОДНОГО игрока под свой состав и цель — титул верхней лиги.")


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
    if _active(llm):
        try:
            plan = playbook.set_plan(world, team["team_id"], _llm_playbook(world, team, llm))
            src = "ЛЛМ"
        except Exception as exc:
            events.fallback(feed, f"playbook:{team['team_id']}", str(exc))
    if plan is None:
        plan = manager.set_tournament_playbook(world, team, profile)
    events.emit(feed, "playbook", f"{team['name']}: плейбук ({src}), база "
                f"«{plan['base']['strategy']}»", team_id=team["team_id"], source=src)
    return plan


_CONDITION_MAP = {
    "opp_robots": {"opp_faction_majority": "robots"},
    "opp_humans": {"opp_faction_majority": "humans"},
    "opp_infected": {"opp_faction_majority": "infected"},
    "opp_plays_macro": {"opp_last_strategy": "macro"},
    "opp_plays_core_rush": {"opp_last_strategy": "core_rush"},
    "opp_plays_turtle": {"opp_last_strategy": "turtle"},
    "playoff_behind2": {"stage": "playoff", "playoff_losing_streak": 2},
    "playoff_ahead": {"stage": "playoff", "series_ahead": True},
    "series_lost1": {"series_lost": 1},
}


def _llm_playbook(world: dict, team: dict, llm) -> dict:
    strat_enum = list(playbook.STRATEGY_IDS)
    roster = list(team["roster"])
    # an optional lineup/substitution: which player at A1/A2/A3 (partial allowed)
    lineup_obj = {"type": "object", "additionalProperties": False,
                  "properties": {p: {"type": "string", "enum": roster}
                                 for p in playbook.POSITIONS}}
    schema = {
        "type": "object",
        "properties": {
            "base": {"type": "object", "additionalProperties": False,
                     "properties": {"strategy": {"type": "string", "enum": strat_enum},
                                    "lineup": lineup_obj},
                     "required": ["strategy"]},
            "overrides": {
                "type": "array", "maxItems": 6,
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
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, _playbook_prompt(world, team),
                         schema, max_tokens=700)
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
    for ov in ans.get("overrides", []):
        when = _CONDITION_MAP.get(ov.get("condition"))
        strat = ov.get("strategy")
        if when is None or strat not in playbook.STRATEGY_IDS:
            raise ValueError(f"bad override {ov!r}")
        overrides.append({"when": when, "strategy": strat,
                          "lineup": _clean_lineup(ov.get("lineup"), roster),
                          "note": ov.get("note", ov["condition"])})
    return {"base": {"strategy": bstrat, "lineup": _clean_lineup(base.get("lineup"), roster)},
            "overrides": overrides}


def _playbook_prompt(world: dict, team: dict) -> str:
    """Playbook prompt GROUNDED in past games (B): own past calls + winrate, league
    meta, measured counters, per-position winrate, H2H vs league rivals."""
    league = team.get("league")
    opponents = [t["team_id"] for t in world["teams"].values()
                 if t.get("league") == league and t["team_id"] != team["team_id"]][:6]
    intel = scouting.render_intel(scouting.intel_report(world, team["team_id"], opponents))
    roster = "; ".join(f"{world['players'][pid]['nickname']}({pid})" for pid in team["roster"])
    return (f"Клуб {team['name']}. Ростер: {roster}.\n"
            f"--- РАЗВЕДКА (прошлые игры) ---\n{intel}\n"
            f"--- ЗАДАЧА ---\n"
            f"Составь ПЛЕЙБУК на турнир: базовую стратегию из {list(playbook.STRATEGY_IDS)} "
            f"(опираясь на РАЗВЕДКУ — что реально работает у нас и в мете) + правила "
            f"IF-THEN (по сопернику/стадии/счёту серии). Можешь задать состав по "
            f"позициям A1/A2/A3 и условные ЗАМЕНЫ (поставить другого игрока на позицию "
            f"при условии). Цель — титул верхней лиги.")


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
# Training — L1 (one call per team: train? which op? which player(s)/parent?). The
# manager (LLM) is the single brain: it DECIDES whether to buy training and picks the
# op + target. manager.decide_training is the deterministic FALLBACK (notified).
# --------------------------------------------------------------------------- #
def training_plan(world: dict, team: dict, *, llm=None, feed: Optional[list] = None,
                  profile: Optional[manager.ManagerProfile] = None,
                  steps: Optional[int] = None, allow_fork: bool = False) -> Optional[dict]:
    """The manager's off-season training decision. LLM when active (it chooses WHETHER to train,
    WHICH op, and WHO/whom-to-fork); else (or on any failure) the heuristic
    ``manager.decide_training`` with a fallback notification. Returns a job dict or None."""
    profile = profile or manager.natural_profile(team["franchise_id"])
    if _active(llm):
        try:
            job = _llm_training(world, team, llm, steps=steps, allow_fork=allow_fork)
            msg = (f"{op_label(job['spend_type'])} → {job['player_name']}" if job
                   else "без тренировки в это межсезонье")
            events.emit(feed, "training", f"{team['name']}: ЛЛМ — {msg}",
                        team_id=team["team_id"], source="ЛЛМ")
            return job
        except Exception as exc:
            events.fallback(feed, f"training:{team['team_id']}", str(exc))
    return manager.decide_training(world, team, profile, steps=steps, allow_fork=allow_fork)


def run_training_decisions(world: dict, *, llm=None, feed: Optional[list] = None,
                           steps: Optional[int] = None, allow_fork: bool = False) -> list[dict]:
    """Every manager decides its off-season training (LLM, heuristic fallback). Returns the chosen
    jobs, weakest-team-first so a capped launch trains those who need it most."""
    jobs = [j for team in world["teams"].values()
            if (j := training_plan(world, team, llm=llm, feed=feed, steps=steps,
                                   allow_fork=allow_fork)) is not None]
    jobs.sort(key=lambda j: j["priority"], reverse=True)
    return jobs


_OP_LABELS = {"skill_up": "общий буст", "meta_adapt": "против меты", "style_spec": "специализация",
              "obedience_tune": "послушание", "teamplay": "сыгранность", "create_player": "новичок"}


def op_label(op: str) -> str:
    return _OP_LABELS.get(op, op)


def _llm_training(world: dict, team: dict, llm, *, steps=None, allow_fork=False) -> Optional[dict]:
    from src.world import training_ops
    netted = [p for p in (get_player(world, pid) for pid in team.get("roster", [])) if p.get("net_id")]
    if len(netted) < 3:
        return None
    by_pid = {p["player_id"]: p for p in netted}
    ids = list(by_pid)
    ops = ["skill_up", "meta_adapt", "style_spec", "obedience_tune", "teamplay"]
    if allow_fork:
        ops.append("create_player")
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "train": {"type": "boolean"},
            "op": {"type": "string", "enum": ops},
            "player_id": {"type": "string", "enum": ids},      # trainee, or parent for create_player
            "player_ids": {"type": "array", "items": {"type": "string", "enum": ids}, "maxItems": 3},
            "reason": {"type": "string"},
        },
        "required": ["train"],
    }
    ans = llm.structured(manager.MANAGER_SYSTEM_RULES, _training_prompt(world, team, netted, allow_fork),
                         schema, max_tokens=300)
    if not ans.get("train"):
        return None
    op = ans.get("op")
    if op not in ops:
        raise ValueError(f"LLM returned bad training op {op!r}")
    steps = int(steps or training_ops.MIN_TRAINING_STEPS)
    priority = round(-min(valuation.skill_score(p) for p in netted), 4)
    if op == "teamplay":
        pids = [pid for pid in (ans.get("player_ids") or []) if pid in by_pid][:3]
        if len(pids) < 3:                              # LLM gave a partial set -> top 3 by skill
            pids = [p["player_id"] for p in sorted(netted, key=valuation.skill_score, reverse=True)[:3]]
        target = {"kind": "team", "net_ids": [by_pid[pid]["net_id"] for pid in pids], "player_ids": pids}
        name = f"{team['name']} · тройка"
    else:
        pid = ans.get("player_id")
        if pid not in by_pid:
            raise ValueError(f"LLM returned bad player_id {pid!r}")
        p = by_pid[pid]
        target = {"kind": "parent" if op == "create_player" else "player",
                  "net_id": p["net_id"], "player_id": pid}
        name = f"{team['name']} · новичок от {p['nickname']}" if op == "create_player" else p["nickname"]
    return {"team_id": team["team_id"], "spend_type": op, "target": target, "total_steps": steps,
            "player_name": name, "priority": priority}


def _training_prompt(world: dict, team: dict, netted: list, allow_fork: bool) -> str:
    rows = "; ".join(f"{p['player_id']}:{p['nickname']}({p['faction']},SoloQ{int(p['soloq_rating'])},"
                     f"скилл{valuation.skill_score(p):.2f})" for p in netted)
    menu = ("skill_up=общий буст; meta_adapt=тренировка ПРОТИВ топов меты; "
            "style_spec=специализация в своём стиле; obedience_tune=послушание приказам; "
            "teamplay=сыгранность тройки; create_player=ФОРК нового игрока от выбранного родителя")
    forkline = "" if allow_fork else "\nВ это межсезонье create_player (форк) НЕдоступен — нет бюджета."
    return (f"Клуб {team['name']}. Казна: {team['treasury']}. Игроки с нейронками:\n{rows}.\n"
            f"Виды тренировки: {menu}.{forkline}\n--- РЕШЕНИЕ ---\n"
            f"Реши как менеджер: покупать ли тренировку в это межсезонье (train). Если да — выбери "
            f"вид (op) и КОГО тренировать (player_id; для teamplay — троих в player_ids; для "
            f"create_player player_id = родитель для форка). Усиливай клуб к титулу; можешь и "
            f"отказаться, если состав силён или казну важнее сберечь.")


# --------------------------------------------------------------------------- #
# Auctions — L1 by default; 🔴 L2 live bidding is secondary + explicit-confirm
# --------------------------------------------------------------------------- #
def auction_max_bids(world: dict, team_ids: list[str], item_value: int, *,
                     llm=None, feed: Optional[list] = None) -> dict:
    """L1: each team's max bid for an auction item — the manager sends an 'assistant'
    with a plan, the CODE runs the sealed auction. Heuristic = ``manager.auction_bid``
    weighted by character (the LLM may override priorities when active)."""
    bids = {}
    for tid in team_ids:
        team = world["teams"][tid]
        profile = manager.natural_profile(team["franchise_id"])
        bids[tid] = manager.auction_bid(profile, item_value, team["treasury"])
    return bids


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
