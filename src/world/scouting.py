"""Scouting / intel (B) — compiles what a manager knows from PAST games so its
playbook calls are GROUNDED in data, not a hand-written guess.

Everything here reads already-stored data: the calendar (recorded CALLS + results,
see calendar.record_series), rosters + per-position stats (playbook.record_position_
results), team Elo, and the measured strategy matrix (playbook, C). The output feeds
the LLM manager's playbook prompt (llm_manager._playbook_prompt) and is also useful
for a UI scouting panel.
"""
from __future__ import annotations

from typing import Optional

from src.world import playbook
from src.world.world import get_player


def _played(world: dict):
    return [e for e in world.get("calendar", []) if e.get("played")]


def team_strategy_record(world: dict, team_id: str) -> dict:
    """How each strategy has performed FOR this team: {strategy: {games, wins}} —
    i.e. the team's own past CALLS and how they fared."""
    rec: dict = {}
    for e in _played(world):
        for side, skey in (("home", "home_strategy"), ("away", "away_strategy")):
            if e[side] != team_id or not e.get(skey):
                continue
            r = rec.setdefault(e[skey], {"games": 0, "wins": 0})
            r["games"] += 1
            r["wins"] += 1 if e["winner"] == team_id else 0
    return rec


def head_to_head(world: dict, team_id: str, opp_id: str) -> dict:
    """Record vs a specific opponent + which strategies they used against this team."""
    g = w = 0
    opp_strats: dict = {}
    for e in _played(world):
        if {e["home"], e["away"]} != {team_id, opp_id}:
            continue
        g += 1
        if e["winner"] == team_id:
            w += 1
        opp_strat = e["away_strategy"] if e["home"] == team_id else e["home_strategy"]
        if opp_strat:
            opp_strats[opp_strat] = opp_strats.get(opp_strat, 0) + 1
    return {"games": g, "wins": w, "opp_strategies": opp_strats}


def league_meta(world: dict) -> dict:
    """Leaguewide winrate by strategy (the current meta): {strategy: {games, winrate}}."""
    rec: dict = {}
    for e in _played(world):
        for skey, side in (("home_strategy", "home"), ("away_strategy", "away")):
            strat = e.get(skey)
            if not strat:
                continue
            r = rec.setdefault(strat, {"games": 0, "wins": 0})
            r["games"] += 1
            r["wins"] += 1 if e["winner"] == e[side] else 0
    for r in rec.values():
        r["winrate"] = round(r["wins"] / r["games"], 3) if r["games"] else None
    return rec


def position_record(world: dict, team_id: str) -> dict:
    """Per-player OBSERVED winrate by position (drives 'who plays where')."""
    out: dict = {}
    for pid in world["teams"][team_id]["roster"]:
        p = get_player(world, pid)
        st = p.get("position_stats") or {}
        out[pid] = {"nickname": p["nickname"],
                    "positions": {pos: {"games": s["games"],
                                        "winrate": round(s["wins"] / s["games"], 2)}
                                  for pos, s in st.items() if s.get("games")}}
    return out


def intel_report(world: dict, team_id: str,
                 opponents: Optional[list] = None) -> dict:
    """Everything the manager should weigh before authoring its playbook (B)."""
    return {
        "own_strategy_record": team_strategy_record(world, team_id),
        "league_meta": league_meta(world),
        "position_record": position_record(world, team_id),
        "head_to_head": {opp: head_to_head(world, team_id, opp) for opp in (opponents or [])},
        "measured_matrix": measured_matrix(world),
    }


def measured_matrix(world: dict) -> dict:
    """The C matrix, rendered: {"a_vs_b": {winrate, n}} for every ordered pair with
    any evidence — the measured strategy counters learned so far."""
    out: dict = {}
    for a in playbook.STRATEGY_IDS:
        for b in playbook.STRATEGY_IDS:
            if a == b:
                continue
            p, n = playbook.measured_winrate(world, a, b)
            if n > 0:
                out[f"{a}_vs_{b}"] = {"winrate": round(p, 3), "n": n}
    return out


def render_intel(report: dict, *, max_h2h: int = 4) -> str:
    """Token-lean text block for the LLM playbook prompt."""
    lines = []
    own = report.get("own_strategy_record") or {}
    if own:
        lines.append("Твои прошлые стратегии (winrate): " + ", ".join(
            f"{s}:{r['wins']}/{r['games']}" for s, r in sorted(own.items())))
    meta = report.get("league_meta") or {}
    if meta:
        lines.append("Мета лиги (winrate): " + ", ".join(
            f"{s}:{r['winrate']}" for s, r in sorted(meta.items()) if r["winrate"] is not None))
    mm = report.get("measured_matrix") or {}
    if mm:
        top = sorted(mm.items(), key=lambda kv: -kv[1]["winrate"])[:6]
        lines.append("Замеренные контры: " + ", ".join(
            f"{k}={v['winrate']}(n{v['n']})" for k, v in top))
    pos = report.get("position_record") or {}
    pos_bits = []
    for info in pos.values():
        best = [f"{pos_}:{d['winrate']}" for pos_, d in info["positions"].items()]
        if best:
            pos_bits.append(f"{info['nickname']}[{','.join(best)}]")
    if pos_bits:
        lines.append("Винрейт по позициям: " + "; ".join(pos_bits[:8]))
    h2h = report.get("head_to_head") or {}
    shown = [(opp, r) for opp, r in h2h.items() if r["games"]][:max_h2h]
    for opp, r in shown:
        opp_str = ", ".join(f"{s}x{c}" for s, c in r["opp_strategies"].items())
        lines.append(f"H2H vs {opp}: {r['wins']}/{r['games']}; их стратегии: {opp_str}")
    return "\n".join(lines) if lines else "истории игр пока нет (первый сезон)"
