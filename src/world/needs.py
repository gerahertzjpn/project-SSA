"""Team-needs analysis — the read a real GM uses to decide WHO to buy / train / field.

Composes signals that already exist (valuation skill/sporting/weakest-starter, playbook per-strategy
execution + faction mix, scouting league meta) into two compact briefs for the manager market/bid
prompts. No new game state — pure read. The point: the manager should understand its OWN squad's
holes and rank targets AGAINST its weakest starter, instead of blindly signing the best affordable.
"""
from __future__ import annotations

from src.world import playbook, scouting, transfers, valuation
from src.world.world import get_player


def _starters(world: dict, team: dict) -> list[dict]:
    roster = [get_player(world, pid) for pid in team.get("roster", [])]
    return sorted(roster, key=valuation.skill_score, reverse=True)[:3]


def team_needs_brief(world: dict, team: dict) -> str:
    """A GM's read of the squad's holes: weakest starter, faction balance, best/worst executed
    strategy, and a meta gap (a strategy that WINS in the league but this squad runs badly → it
    should acquire/train a specialist for it)."""
    roster = [get_player(world, pid) for pid in team.get("roster", [])]
    if not roster:
        return "Ростер пуст — нужны игроки в старт."
    starters = _starters(world, team)
    ws = min(starters, key=valuation.skill_score)
    fac: dict = {}
    for p in roster:
        fac[p.get("faction", "?")] = fac.get(p.get("faction", "?"), 0) + 1
    fac_str = ", ".join(f"{f}×{n}" for f, n in sorted(fac.items()))
    texec = {s: sum(playbook.player_exec(p, s) for p in starters) / max(1, len(starters))
             for s in playbook.STRATEGY_IDS}
    best_s = max(texec, key=texec.get)
    worst_s = min(texec, key=texec.get)
    meta = scouting.league_meta(world) or {}
    meta_gap = ""
    for s in sorted(playbook.STRATEGY_IDS, key=lambda x: -((meta.get(x) or {}).get("winrate") or 0)):
        wr = (meta.get(s) or {}).get("winrate")
        if wr is not None and wr > 0.55 and texec[s] < 0.5:
            meta_gap = (f"; мета любит «{s}» (wr {wr}), а мы его исполняем слабо ({texec[s]:.2f}) → "
                        f"нужен специалист под «{s}»")
            break
    return (f"Слабейший старт: {ws.get('nickname')} (скилл {valuation.skill_score(ws):.2f}, "
            f"{ws.get('faction')}). Старт-скиллы: "
            f"{[round(valuation.skill_score(p), 2) for p in starters]}. Фракции ростера: {fac_str}. "
            f"Лучше всего играем «{best_s}», хуже всего «{worst_s}»{meta_gap}.")


def market_board(world: dict, team: dict, candidates: list, *, limit: int = 18) -> str:
    """Rank available players AGAINST our weakest starter so the manager sees who is a real UPGRADE vs
    mere depth vs a pass. This is the 'who do we actually want' board."""
    ws = valuation.weakest_starter(world, team)
    ws_val = valuation.sporting_value(world, ws) if ws else 0.0
    ranked = sorted(candidates, key=lambda p: valuation.sporting_value(world, p), reverse=True)[:limit]
    rows = []
    for p in ranked:
        gap = valuation.sporting_value(world, p) - ws_val
        verdict = ("АПГРЕЙД-СТАРТА" if gap >= valuation.UPGRADE_MARGIN
                   else "глубина" if gap >= -0.15 else "мимо")
        a = p.get("attributes") or {}
        best = max(playbook.STRATEGY_IDS, key=lambda s: playbook.player_exec(p, s))
        rows.append(f"{p['player_id']}:{p.get('nickname')}({p.get('faction', '?')},"
                    f"SoloQ{int(p.get('soloq_rating', 1000))},"
                    f"атк{a.get('attack', 0)}/защ{a.get('defense', 0)}/эко{a.get('economy', 0)},"
                    f"лучш:{best},цена~{transfers.fair_value(p)},vs_старт{gap:+.2f}→{verdict})")
    return "; ".join(rows) if rows else "нет доступных кандидатов по карману"
