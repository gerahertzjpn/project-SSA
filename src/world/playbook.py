"""Playbook — the manager-LLM's single tactical brain (replaces the old coach).

Per WORLD_MODE_SPEC §0-bis: there is NO coach. Once, before a tournament, the LLM
manager authors ONE *playbook* for the whole event; the CODE then executes it match
by match — there are no per-match LLM calls. A playbook is:

    {
      "base":      {"strategy": <id>, "lineup": [pid, pid, pid] | None},
      "overrides": [                       # checked in order; FIRST match wins
        {"when": {<conditions>}, "strategy": <id>, "lineup": [...]|None, "note": str},
        ...
      ],
    }

``when`` conditions cover the tournament/series state AND the opponent — so a rule
like *"opponent fields a robot majority → switch to core_rush with 2 rushers"* is
expressible and fires automatically. Supported keys:
  - stage:                 "group" | "playoff"
  - series_lost / series_won:        int (>= match)
  - playoff_losing_streak:           int (>= match)
  - series_ahead / series_behind:    bool
  - opp_faction_majority:            "humans" | "infected" | "robots"
  - opp_has_faction:                 faction present on the opponent roster
  - opp_strategy:                    the opponent's expected base strategy

The effect is REAL: ``matchup_modifiers`` feeds a strength delta into the match sim
(``season.make_play_fn``), so the manager's choices change win probability via:
  1. a rock-paper-scissors STRATEGY counter (±COUNTER_BONUS rating),
  2. how well the roster EXECUTES the chosen strategy (±EXEC_BONUS, from attributes
     / measured per-strategy effectiveness),
  3. a LINEUP trade-off (a deliberate non-best-3 lineup trades raw rating for fit).
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

from src.world.world import POSITIONS, get_player

# Strategy vocabulary — mirrors src.training.spaces.STRATEGY_PRESETS ids (kept local
# so the world package stays independent of the training stack).
STRATEGY_IDS = ("standard", "core_rush", "all_in", "turtle", "macro", "baron")

# Which game attribute drives execution of each strategy (15..98 attribute scale).
STRATEGY_TO_ATTR = {
    "standard": "consistency",
    "core_rush": "attack",
    "all_in": "attack",
    "turtle": "defense",
    "macro": "economy",
    "baron": "tempo",
}

# Rock-paper-scissors: BEATS[a] = the strategy `a` hard-counters. A clean 5-cycle;
# "standard" is neutral (no counter either way).
BEATS = {
    "core_rush": "macro",     # rush punishes greed
    "macro": "turtle",        # economy out-scales passivity
    "turtle": "all_in",       # defense outlasts the overcommit
    "all_in": "baron",        # the all-in bursts before the baron timing
    "baron": "core_rush",     # objective control answers the rush
}

# Strategy-counter strength is now MEASURED (see counter_points below); the prior
# magnitude is set by PRIOR_WIN_COUNTER (≈ +60 pts at zero data).
EXEC_BONUS = 40.0      # rating pts for perfect vs worst execution of the chosen plan


# --------------------------------------------------------------------------- #
# Execution effectiveness (how well a roster runs a strategy)
# --------------------------------------------------------------------------- #
def player_exec(player: dict, strategy: str) -> float:
    """How well a player executes ``strategy`` in [0,1]. Uses measured per-strategy
    effectiveness when present, else derives it from the driving attribute."""
    eff = player.get("strategy_effectiveness") or {}
    if strategy in eff:
        return max(0.0, min(1.0, float(eff[strategy])))
    attr = STRATEGY_TO_ATTR.get(strategy, "consistency")
    val = (player.get("attributes") or {}).get(attr, 50)
    return max(0.0, min(1.0, (float(val) - 15.0) / (98.0 - 15.0)))


def starters_best3(world: dict, team_id: str) -> list[str]:
    """The 3 highest-SoloQ players (the default starters)."""
    roster = world["teams"][team_id]["roster"]
    return sorted(roster, key=lambda pid: -get_player(world, pid)["soloq_rating"])[:3]


def team_exec(world: dict, team_id: str, strategy: str,
              starters: Optional[list[str]] = None) -> float:
    """Average execution of ``strategy`` across the (chosen or best-3) starters."""
    starters = starters or starters_best3(world, team_id)
    if not starters:
        return 0.5
    return sum(player_exec(get_player(world, pid), strategy) for pid in starters) / len(starters)


def best_strategy(world: dict, team_id: str) -> str:
    """The strategy this roster executes best (the natural base)."""
    return max(STRATEGY_IDS, key=lambda s: team_exec(world, team_id, s))


def _argmax_exec(world: dict, team_id: str, among: tuple) -> str:
    return max(among, key=lambda s: team_exec(world, team_id, s))


def aggressive_comeback(world: dict, team_id: str) -> str:
    return _argmax_exec(world, team_id, ("all_in", "core_rush", "baron"))


def safest(world: dict, team_id: str) -> str:
    return _argmax_exec(world, team_id, ("turtle", "macro", "standard"))


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #
def counter_sign(a: str, b: str) -> int:
    """+1 if strategy ``a`` counters ``b``, -1 if ``b`` counters ``a``, else 0."""
    if BEATS.get(a) == b:
        return 1
    if BEATS.get(b) == a:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# (C) MEASURED counter strength — learns from past games, shrinking toward the
# designed prior so it turns on GRADUALLY (0 games → pure prior == current design;
# lots of games → the real measured winrate). No flag needed.
# --------------------------------------------------------------------------- #
PRIOR_WEIGHT = 25.0            # games of evidence needed to half-trust the data
PRIOR_WIN_COUNTER = 0.585      # prior winrate of a clean counter (≈ +60 rating pts)
COUNTER_CLAMP = 120.0          # max rating swing a matchup edge can be worth

def _strategy_tally(world: dict) -> dict:
    """{(winner_strategy, loser_strategy): count} over all PLAYED calendar matches
    whose calls are recorded. Recomputed per call (the calendar is small — group +
    playoff ≈ 85 series/season — so this is cheap and always correct)."""
    tally: dict = {}
    for e in world.get("calendar", []):
        if not e.get("played"):
            continue
        hs, as_ = e.get("home_strategy"), e.get("away_strategy")
        if not hs or not as_:
            continue
        win_strat = hs if e["winner"] == e["home"] else as_
        lose_strat = as_ if e["winner"] == e["home"] else hs
        tally[(win_strat, lose_strat)] = tally.get((win_strat, lose_strat), 0) + 1
    return tally


def measured_winrate(world: dict, a: str, b: str) -> tuple:
    """Shrunk winrate of strategy ``a`` vs ``b`` and the sample size: blends the
    measured calendar winrate with the designed prior by evidence (returns (p, n))."""
    tally = _strategy_tally(world)
    a_wins = tally.get((a, b), 0)
    b_wins = tally.get((b, a), 0)
    n = a_wins + b_wins
    sign = counter_sign(a, b)
    prior = 0.5 + (PRIOR_WIN_COUNTER - 0.5) * sign
    if n == 0:
        return prior, 0
    measured = a_wins / n
    w = n / (n + PRIOR_WEIGHT)
    return w * measured + (1.0 - w) * prior, n


def counter_points(world: dict, a: str, b: str) -> float:
    """Rating points the strategy matchup ``a`` vs ``b`` is worth (measured, shrunk
    toward the designed counter prior). Antisymmetric: counter_points(b,a) = -this."""
    p, _ = measured_winrate(world, a, b)
    p = min(0.999, max(0.001, p))
    return max(-COUNTER_CLAMP, min(COUNTER_CLAMP, 400.0 * math.log10(p / (1.0 - p))))


# --------------------------------------------------------------------------- #
# Positions & lineup (the manager/LLM places players + can substitute)
# --------------------------------------------------------------------------- #
def position_fit(player: dict, pos: str) -> float:
    """Latent skill multiplier of a player at a position (drives strength)."""
    return float((player.get("position_affinity") or {}).get(pos, 1.0))


def position_winrate(player: dict, pos: str) -> Optional[float]:
    """OBSERVED winrate of a player at a position (None until enough games)."""
    st = (player.get("position_stats") or {}).get(pos)
    if not st or st.get("games", 0) < 1:
        return None
    return st["wins"] / st["games"]


def optimal_lineup(world: dict, team_id: str) -> dict:
    """Default position assignment: the best 3 (by SoloQ) placed to MAXIMISE total
    SoloQ×fit (tries the 6 permutations). Returns {pos: pid}."""
    best3 = starters_best3(world, team_id)
    if len(best3) < 3:
        return {POSITIONS[i]: pid for i, pid in enumerate(best3)}
    best_map, best_val = None, -1.0
    for perm in itertools.permutations(best3):
        m = {POSITIONS[i]: perm[i] for i in range(3)}
        val = lineup_strength(world, m)
        if val > best_val:
            best_map, best_val = m, val
    return best_map


def resolve_lineup(world: dict, team_id: str, lineup) -> dict:
    """Normalise an authored lineup into a {pos: pid} map (3 starters). Accepts a
    list (ordered A1,A2,A3) or a {pos: pid} dict; validates against the roster and
    fills any gap from the best unused player. ``None`` → the optimal default."""
    roster = world["teams"][team_id]["roster"]
    if not lineup:
        return optimal_lineup(world, team_id)
    raw = (dict(zip(POSITIONS, lineup)) if isinstance(lineup, (list, tuple))
           else dict(lineup))
    out, used = {}, set()
    for pos in POSITIONS:
        pid = raw.get(pos)
        if pid in roster and pid not in used:
            out[pos] = pid
            used.add(pid)
    # fill empty positions from the best unused players
    spare = [p for p in sorted(roster, key=lambda x: -get_player(world, x)["soloq_rating"])
             if p not in used]
    for pos in POSITIONS:
        if pos not in out and spare:
            out[pos] = spare.pop(0)
    return out


def lineup_strength(world: dict, lineup_map: dict) -> float:
    """Position-adjusted strength of a {pos: pid} lineup: avg of SoloQ×fit. A player
    at a strong position is worth more — so placement and substitutions matter."""
    vals = [get_player(world, pid)["soloq_rating"] * position_fit(get_player(world, pid), pos)
            for pos, pid in lineup_map.items()]
    return sum(vals) / len(vals) if vals else 1000.0


def _best3_soloq_avg(world: dict, team_id: str) -> float:
    best3 = starters_best3(world, team_id)
    if not best3:
        return 1000.0
    return sum(get_player(world, pid)["soloq_rating"] for pid in best3) / len(best3)


def record_position_results(world: dict, lineup_map: dict, won: bool) -> None:
    """Attribute a played series to each starter AT THEIR POSITION (builds the
    observed per-position winrate the scouting layer reads)."""
    for pos, pid in lineup_map.items():
        st = get_player(world, pid).setdefault("position_stats", {}).setdefault(
            pos, {"games": 0, "wins": 0})
        st["games"] += 1
        st["wins"] += 1 if won else 0


# --------------------------------------------------------------------------- #
# Opponent scouting context
# --------------------------------------------------------------------------- #
def _last_strategy(world: dict, team_id: str) -> Optional[str]:
    """The most recent strategy this team actually CALLED (from the calendar)."""
    for e in reversed(world.get("calendar", [])):
        if not e.get("played"):
            continue
        if e["home"] == team_id and e.get("home_strategy"):
            return e["home_strategy"]
        if e["away"] == team_id and e.get("away_strategy"):
            return e["away_strategy"]
    return None


def opponent_context(world: dict, team_id: str) -> dict:
    """What a manager can scout about a team: faction make-up, expected base, and the
    opponent's LAST actual call (so you can prepare for them — and they for you)."""
    starters = starters_best3(world, team_id)
    factions = [get_player(world, pid).get("faction") for pid in starters]
    majority = None
    if factions:
        majority = max(set(factions), key=factions.count)
        if factions.count(majority) * 2 <= len(factions):   # no real majority
            majority = None
    return {"team_id": team_id, "factions": factions, "faction_majority": majority,
            "strategy": best_strategy(world, team_id),
            "last_strategy": _last_strategy(world, team_id)}


# --------------------------------------------------------------------------- #
# The playbook: build (heuristic default / LLM authors the same shape), resolve
# --------------------------------------------------------------------------- #
def build_default_plan(world: dict, team_id: str) -> dict:
    """Heuristic playbook (the LLM produces the SAME shape). Base = best-executed
    strategy + sensible contingencies, including an opponent-conditional rule."""
    base = best_strategy(world, team_id)
    agg = aggressive_comeback(world, team_id)
    safe = safest(world, team_id)
    return {
        "base": {"strategy": base, "lineup": None},
        "overrides": [
            {"when": {"opp_faction_majority": "robots"}, "strategy": "core_rush",
             "lineup": None, "note": "раш до того как роботы раскачают экономику"},
            {"when": {"stage": "playoff", "playoff_losing_streak": 2},
             "strategy": agg, "lineup": None, "note": "must-win: идём ва-банк"},
            {"when": {"stage": "playoff", "series_ahead": True},
             "strategy": safe, "lineup": None, "note": "ведём в серии — играем безопасно"},
        ],
    }


def get_plan(world: dict, team_id: str) -> dict:
    """The team's stored playbook, or a freshly-built heuristic default."""
    return world["teams"][team_id].get("tournament_plan") or build_default_plan(world, team_id)


def set_plan(world: dict, team_id: str, plan: dict) -> dict:
    world["teams"][team_id]["tournament_plan"] = plan
    return plan


def _condition_matches(when: dict, state: dict, opp: dict) -> bool:
    s, o = state or {}, opp or {}
    won, lost = int(s.get("series_won", 0)), int(s.get("series_lost", 0))
    for key, val in when.items():
        if key == "stage" and s.get("stage") != val:
            return False
        if key == "series_lost" and lost < int(val):
            return False
        if key == "series_won" and won < int(val):
            return False
        if key == "playoff_losing_streak" and int(s.get("playoff_losing_streak", 0)) < int(val):
            return False
        if key == "series_ahead" and bool(won > lost) != bool(val):
            return False
        if key == "series_behind" and bool(lost > won) != bool(val):
            return False
        if key == "opp_faction_majority" and o.get("faction_majority") != val:
            return False
        if key == "opp_has_faction" and val not in (o.get("factions") or []):
            return False
        if key == "opp_strategy" and o.get("strategy") != val:
            return False
        if key == "opp_last_strategy" and o.get("last_strategy") != val:
            return False
        if key == "vs_team" and o.get("team_id") != val:   # (D) prep for a specific club
            return False
    return True


def resolve_active(plan: dict, state: dict, opp: dict) -> dict:
    """Active {strategy, lineup, note} from the playbook + match state + opponent.
    First matching override wins; else the base."""
    for ov in plan.get("overrides", []):
        if _condition_matches(ov.get("when", {}), state, opp):
            return {"strategy": ov.get("strategy"), "lineup": ov.get("lineup"),
                    "note": ov.get("note", "override")}
    base = plan.get("base") or {}
    return {"strategy": base.get("strategy", "standard"), "lineup": base.get("lineup"),
            "note": "base"}


# --------------------------------------------------------------------------- #
# The EFFECT: strength deltas fed into the match sim
# --------------------------------------------------------------------------- #
def _side_strength_delta(world: dict, team_id: str, active: dict, lineup_map: dict,
                         counter_pts: float) -> float:
    """Delta for one side: (measured) strategy counter + execution quality + the
    position-aware LINEUP value vs the team's best-3 baseline (placement & subs)."""
    exec_q = team_exec(world, team_id, active["strategy"], list(lineup_map.values()))
    lineup_d = lineup_strength(world, lineup_map) - _best3_soloq_avg(world, team_id)
    return counter_pts + EXEC_BONUS * (exec_q - 0.5) * 2.0 + lineup_d


def resolve_match(world: dict, home: str, away: str, best_of: int,
                  *, state_home: Optional[dict] = None,
                  state_away: Optional[dict] = None) -> dict:
    """Resolve BOTH playbooks against each other for THIS match: each side's active
    strategy, its {pos: pid} lineup (placement + conditional substitutions), and the
    resulting strength delta. The opponent's playbook resolves simultaneously — so a
    team can prepare for you and you for it (mutual adaptation).

    One series per call → per-game series score is unknown here; ``state`` carries the
    stage (Bo5 → playoff, Bo3 → group). The real match-by-match engine passes
    ``state_home`` / ``state_away`` with live series score so the comeback/closeout
    IF-THEN rules also fire."""
    stage = "playoff" if best_of >= 5 else "group"
    sh = {"stage": stage, **(state_home or {})}
    sa = {"stage": stage, **(state_away or {})}

    home_opp = opponent_context(world, away)   # what HOME scouts about AWAY
    away_opp = opponent_context(world, home)

    ha = resolve_active(get_plan(world, home), sh, home_opp)
    aa = resolve_active(get_plan(world, away), sa, away_opp)
    home_lineup = resolve_lineup(world, home, ha.get("lineup"))
    away_lineup = resolve_lineup(world, away, aa.get("lineup"))

    sign = counter_sign(ha["strategy"], aa["strategy"])
    cp = counter_points(world, ha["strategy"], aa["strategy"])   # measured, shrunk
    dh = _side_strength_delta(world, home, ha, home_lineup, cp)
    da = _side_strength_delta(world, away, aa, away_lineup, -cp)
    return {
        "home": {"strategy": ha["strategy"], "note": ha["note"], "lineup": home_lineup,
                 "delta": round(dh, 1)},
        "away": {"strategy": aa["strategy"], "note": aa["note"], "lineup": away_lineup,
                 "delta": round(da, 1)},
        "counter": sign, "stage": stage, "delta_home": dh, "delta_away": da,
    }


def matchup_modifiers(world: dict, home: str, away: str, best_of: int,
                      *, state_home: Optional[dict] = None,
                      state_away: Optional[dict] = None) -> tuple:
    """Thin wrapper over ``resolve_match`` returning ``(delta_home, delta_away, meta)``
    for ``season.make_play_fn``. ``meta`` carries each side's strategy + lineup so the
    sim can record per-position results."""
    m = resolve_match(world, home, away, best_of,
                      state_home=state_home, state_away=state_away)
    return m["delta_home"], m["delta_away"], m
