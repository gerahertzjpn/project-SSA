"""Derive REAL player cards by testing each net UNDER ORDERS + free play (owner's confirmed scheme).

Each net is the FOCAL player (slot A1) with a FIXED mid-strength reference cast (same 2 teammates +
3 opponents, same seeds, same N) for every net -> identical context -> cards are comparable.
GLOBAL normalization (cross-faction) + ABSOLUTE scales (a 99 means near a real ceiling, leaving
head-room a net can train into; owner's call).

Per attribute (raw -> 0-100 vs an ABSOLUTE reference, clamped):
  attack   [Олл-ин]  = core damage dealt to enemy cores / 240 (3 cores x 80 = all broken)
  defense  [Черепаха] = own core_hp kept / 80 (full core)
  tempo    [Олл-ин]  = core damage per tick / 3.0 (240 over 80 ticks = a fast rush)
  economy  [free]    = win-weighted tech tier (TECH_1=.5, TECH_2=1) + eco buildings + resources
  consistency [all]  = 1 - coefficient-of-variation of core damage across games (stable = high)
  potential          = head-room from steps trained (1M ~ high, ~100M ~ spent)

attack/tempo are TEAM-level (engine has no per-player core damage) -> the fixed cast makes them
fair-RELATIVE; defense/economy are clean per-net (own core_hp / own buildings).

    python scripts/derive_player_cards.py --smoke          # 6 nets, N=4
    python scripts/derive_player_cards.py --games 15       # full 108-net run
"""
from __future__ import annotations
import argparse, json, math, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.core.runner import run_match
from src.training.spaces import strategy_order_for_slot

MANIFEST = ROOT / "population" / "frozen_population_v1.json"
OUT = ROOT / "analysis" / "player_cards_v1.json"
LEAGUE_REG = ROOT / "training_runs" / "league_population_v1" / "registry.json"

# ABSOLUTE references (tunable after eyeballing the smoke).
REF_CORE_TOTAL = 240.0     # all 3 enemy cores broken (3 x CORE_HP 80)
REF_OWN_CORE = 80.0        # full own core
REF_TEMPO_RATE = 3.0       # core dmg / tick of a fast rush (240/80)
REF_ECO_BLD = 3.0          # booster/collector count that reads as "heavy economy"
REF_RES = 30.0            # resources held that reads as "rich"
POT_MIN_STEPS, POT_MAX_STEPS = 1_000_000.0, 200_000_000.0
ECO_BUILDINGS = {"BOOSTER", "ECONOMY_COLLECTOR"}

HOME, AWAY = ("A1", "A2", "A3"), ("B1", "B2", "B3")


def _idx(man):
    return {e["net_id"]: ("policy:" + e["checkpoint"], e["faction"])
            for g in ("descendants", "ancestors") for e in man[g]}


def _order(strategy_id):
    if not strategy_id:
        return {}
    out = {}
    for i, s in enumerate(HOME):  # order only the focal-side slots
        o = strategy_order_for_slot(strategy_id, i)
        out[s] = {"tempo": o["tempo"], "win_path": o["win_path"], "role": o["role"]}
    return out


def play(idx, focal, teammates, opponents, strategy_id, seed):
    models, factions = {}, {}
    for s, n in zip(HOME, [focal, *teammates]): models[s], factions[s] = idx[n]
    for s, n in zip(AWAY, opponents): models[s], factions[s] = idx[n]
    r = run_match(seed=seed, bot_type="league", mode="training", replay_path=None,
                  player_models=models, player_factions=factions, orders=_order(strategy_id) or None)
    st = r.state
    p = st.players.get("A1")
    nodes = st.nodes if isinstance(st.nodes, list) else list(st.nodes.values())
    tier, eco = 0, 0
    for nd in nodes:
        for b in (getattr(nd, "buildings", []) or []):
            if getattr(b, "owner", None) != "A1":
                continue
            bt = str(getattr(b, "building_type", "") or getattr(b, "type", ""))
            if bt == "TECH_2": tier = max(tier, 2)
            elif bt == "TECH_1": tier = max(tier, 1)
            if bt in ECO_BUILDINGS: eco += 1
    return {
        "won": 1.0 if r.winner == "TEAM_A" else 0.0,
        "core_dmg": float(st.core_damage_by_team.get("TEAM_B", 0.0)),       # dmg to enemy cores
        "own_hp": max(0.0, float(getattr(p, "core_hp", 0.0) or 0.0)),
        "res": max(0.0, float(getattr(p, "resources", 0.0) or 0.0)),
        "tier": tier, "eco": eco, "ticks": max(1, int(r.total_ticks)),
    }


def clamp100(x):
    return int(round(max(0.0, min(1.0, x)) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--games", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    N = 4 if args.smoke else args.games

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    idx = _idx(man)
    role_by = {e["net_id"]: ("streamer" if g == "ancestors" else "pro")
               for g in ("descendants", "ancestors") for e in man[g]}
    steps = {}
    if LEAGUE_REG.exists():
        for e in json.loads(LEAGUE_REG.read_text(encoding="utf-8"))["entries"]:
            steps[e["net_id"]] = int(e.get("steps_trained", 0) or 0)

    teammates = ("humans_4", "infected_4")                 # FIXED mid-strength cast
    opponents = ("robots_3", "humans_3", "infected_3")
    fixed = set(teammates) | set(opponents)
    nets = [n for n in idx if n not in fixed]
    if args.smoke: nets = nets[:6]
    elif args.limit: nets = nets[:args.limit]

    print(f"deriving {len(nets)} cards — N={N}/order (Олл-ин/Черепаха/free) vs fixed cast ...")
    cards, t0 = {}, time.time()
    for i, net in enumerate(nets):
        push = [play(idx, net, teammates, opponents, "all_in", 1000 + g) for g in range(N)]
        turt = [play(idx, net, teammates, opponents, "turtle", 2000 + g) for g in range(N)]
        free = [play(idx, net, teammates, opponents, None, 3000 + g) for g in range(N)]
        allg = push + turt + free
        attack = statistics.mean(p["core_dmg"] for p in push) / REF_CORE_TOTAL
        defense = statistics.mean(p["own_hp"] for p in turt) / REF_OWN_CORE
        tempo = statistics.mean(p["core_dmg"] / p["ticks"] for p in push) / REF_TEMPO_RATE
        tech_ww = statistics.mean((g["tier"] / 2.0) * (1.0 if g["won"] else 0.5) for g in free)
        eco_bld = statistics.mean(g["eco"] for g in free) / REF_ECO_BLD
        res = statistics.mean(g["res"] for g in free) / REF_RES
        economy = 0.5 * tech_ww + 0.3 * eco_bld + 0.2 * res
        # consistency = stability of a per-game performance score (dmg dealt + own core kept)
        # WITHIN each order condition (NOT across conditions — those differ by design), averaged.
        cvs = []
        for cond in (push, turt, free):
            scores = [g["core_dmg"] + g["own_hp"] for g in cond]
            md = statistics.mean(scores)
            if md > 1 and len(scores) > 1:
                cvs.append(statistics.pstdev(scores) / md)
        cv = statistics.mean(cvs) if cvs else 1.0
        consistency = 1.0 - min(1.0, cv)
        st = steps.get(net, POT_MAX_STEPS)
        pot = 1.0 - (math.log10(max(st, 1) / POT_MIN_STEPS) / math.log10(POT_MAX_STEPS / POT_MIN_STEPS))
        cards[net] = {"attack": clamp100(attack), "defense": clamp100(defense),
                      "economy": clamp100(economy), "tempo": clamp100(tempo),
                      "consistency": clamp100(consistency), "potential": clamp100(pot)}
        if args.smoke or (i + 1) % 10 == 0:
            c = cards[net]
            print(f"  [{i+1}/{len(nets)}] {net:<26} {role_by.get(net,''):<8} "
                  f"atk{c['attack']:>3} def{c['defense']:>3} eco{c['economy']:>3} "
                  f"tmp{c['tempo']:>3} con{c['consistency']:>3} pot{c['potential']:>3}  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "player_cards/v1",
        "method": "order-conditioned gauntlet, GLOBAL + ABSOLUTE scales",
        "games_per_order": N, "reference_teammates": teammates, "reference_opponents": opponents,
        "absolute_refs": {"core_total": REF_CORE_TOTAL, "own_core": REF_OWN_CORE,
                          "tempo_rate": REF_TEMPO_RATE, "eco_buildings": REF_ECO_BLD, "resources": REF_RES,
                          "potential_steps": [POT_MIN_STEPS, POT_MAX_STEPS]},
        "cards": cards,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(cards)} cards -> {OUT.relative_to(ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
