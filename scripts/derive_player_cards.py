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
OUT = ROOT / "population" / "player_cards_v1.json"   # World reads cards from here
LEAGUE_REG = ROOT / "training_runs" / "league_population_v1" / "registry.json"

# ABSOLUTE references (tunable; raw values are saved so re-scaling needs no re-run).
REF_ATTACK = 150.0         # core dmg of a DOMINANT attacking game (~2 enemy cores); 100 = this, head-room above
REF_OWN_CORE = 80.0        # full own core
REF_TEMPO_RATE = 1.5       # core dmg / tick of a strong push
REF_ECO_BLD = 3.0          # booster/collector count that reads as "heavy economy"
REF_RES = 30.0            # resources held that reads as "rich"
POT_MIN_STEPS, POT_MAX_STEPS = 1_000_000.0, 200_000_000.0
ECO_BUILDINGS = {"BOOSTER", "ECONOMY_COLLECTOR"}

# consistency = how EVENLY the net performs across DIFFERENT opponents (low winrate spread = stable).
CONSISTENCY_TRIOS = (
    ("robots_3", "humans_3", "infected_3"),
    ("robots_5", "humans_5", "infected_5"),
    ("robots_6", "humans_6", "infected_6"),
    ("robots_2", "humans_1", "infected_2"),
)

HOME, AWAY = ("A1", "A2", "A3"), ("B1", "B2", "B3")

# Reference cast for the gauntlet (focal net = A1). The SAME cast must be used for the initial
# derivation AND every re-derivation after training, else cards drift (a trained net would be scored
# against a different opponent set). `reference_cast` enforces this by reusing whatever an existing
# card file recorded.
DEF_TEAM, DEF_OPP = ("humans_4", "infected_4"), ("robots_3", "humans_3", "infected_3")
ALT_TEAM, ALT_OPP = ("humans_5", "infected_5"), ("robots_5", "humans_6", "infected_6")


def reference_cast(alt_cast: bool, existing_meta: Optional[dict]):
    """Pick the (teammates, opponents) gauntlet cast. Re-deriving into an EXISTING card set MUST
    reuse that set's recorded cast so trained nets are re-carded under IDENTICAL conditions; a fresh
    derive uses ALT/DEFAULT per the flag. Returns ((t1,t2), (o1,o2,o3))."""
    if existing_meta:
        rt, ro = existing_meta.get("reference_teammates"), existing_meta.get("reference_opponents")
        if rt and ro:
            return tuple(rt), tuple(ro)
    return (ALT_TEAM, ALT_OPP) if alt_cast else (DEF_TEAM, DEF_OPP)


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
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nets", default="", help="comma net_ids to (re)derive; merges into OUT")
    ap.add_argument("--alt-cast", action="store_true", help="alternate reference cast (to card the default cast)")
    args = ap.parse_args()
    N = 4 if args.smoke else args.games
    N_c = max(2, N // 3)          # games per opponent-trio for the consistency spread

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    idx = _idx(man)
    role_by = {e["net_id"]: ("streamer" if g == "ancestors" else "pro")
               for g in ("descendants", "ancestors") for e in man[g]}
    steps = {}
    if LEAGUE_REG.exists():
        for e in json.loads(LEAGUE_REG.read_text(encoding="utf-8"))["entries"]:
            steps[e["net_id"]] = int(e.get("steps_trained", 0) or 0)

    # Re-deriving into an existing card file? Reuse its EXACT recorded cast + games/order so a trained
    # net is re-carded under the same conditions it was first carded (no drift).
    existing_meta = None
    if args.nets and OUT.exists():
        try:
            existing_meta = json.loads(OUT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing_meta = None
    teammates, opponents = reference_cast(args.alt_cast, existing_meta)
    if existing_meta and not args.smoke and existing_meta.get("games_per_order"):
        N = int(existing_meta["games_per_order"])
        N_c = max(2, N // 3)
    fixed = set(teammates) | set(opponents)
    if args.nets:
        nets = [n.strip() for n in args.nets.split(",") if n.strip()]
    else:
        nets = [n for n in idx if n not in fixed]
        if args.smoke: nets = nets[:6]
        elif args.limit: nets = nets[:args.limit]

    print(f"deriving {len(nets)} cards — N={N}/order (Олл-ин/Черепаха/free) vs fixed cast ...")
    cards, t0 = {}, time.time()
    for i, net in enumerate(nets):
        push = [play(idx, net, teammates, opponents, "all_in", 1000 + g) for g in range(N)]
        turt = [play(idx, net, teammates, opponents, "turtle", 2000 + g) for g in range(N)]
        free = [play(idx, net, teammates, opponents, None, 3000 + g) for g in range(N)]
        # consistency: winrate vs each of several DIFFERENT opponent trios; low spread = stable.
        trio_wr = []
        for ti, opp in enumerate(CONSISTENCY_TRIOS):
            gms = [play(idx, net, teammates, opp, None, 4000 + ti * 100 + g) for g in range(N_c)]
            trio_wr.append(statistics.mean(g["won"] for g in gms))

        raw_attack = statistics.mean(p["core_dmg"] for p in push)
        raw_defense = statistics.mean(p["own_hp"] for p in turt)
        raw_tempo = statistics.mean(p["core_dmg"] / p["ticks"] for p in push)
        tech_ww = statistics.mean((g["tier"] / 2.0) * (1.0 if g["won"] else 0.5) for g in free)
        raw_eco = statistics.mean(g["eco"] for g in free)
        raw_res = statistics.mean(g["res"] for g in free)
        economy = 0.5 * tech_ww + 0.3 * (raw_eco / REF_ECO_BLD) + 0.2 * (raw_res / REF_RES)
        spread = statistics.pstdev(trio_wr) if len(trio_wr) > 1 else 0.0
        consistency = 1.0 - min(1.0, 2.0 * spread)         # spread 0 -> 100, spread .5 -> 0
        st = steps.get(net, POT_MAX_STEPS)
        pot = 1.0 - (math.log10(max(st, 1) / POT_MIN_STEPS) / math.log10(POT_MAX_STEPS / POT_MIN_STEPS))
        cards[net] = {"attack": clamp100(raw_attack / REF_ATTACK), "defense": clamp100(raw_defense / REF_OWN_CORE),
                      "economy": clamp100(economy), "tempo": clamp100(raw_tempo / REF_TEMPO_RATE),
                      "consistency": clamp100(consistency), "potential": clamp100(pot),
                      "raw": {"attack_dmg": round(raw_attack, 1), "defense_hp": round(raw_defense, 1),
                              "tempo_rate": round(raw_tempo, 3), "tech_ww": round(tech_ww, 3),
                              "eco_bld": round(raw_eco, 2), "res": round(raw_res, 1),
                              "trio_winrates": [round(w, 2) for w in trio_wr],
                              "steps": int(st)}}
        if args.smoke or (i + 1) % 10 == 0:
            c = cards[net]
            print(f"  [{i+1}/{len(nets)}] {net:<26} {role_by.get(net,''):<8} "
                  f"atk{c['attack']:>3} def{c['defense']:>3} eco{c['economy']:>3} "
                  f"tmp{c['tempo']:>3} con{c['consistency']:>3} pot{c['potential']:>3}  {time.time()-t0:.0f}s")

    out_cards = cards
    if args.nets and OUT.exists():                 # partial (re)derive -> merge into existing
        try:
            out_cards = json.loads(OUT.read_text(encoding="utf-8")).get("cards", {})
        except (OSError, ValueError):
            out_cards = {}
        out_cards.update(cards)                     # re-derived nets override
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "player_cards/v1",
        "method": "order-conditioned gauntlet, GLOBAL + ABSOLUTE scales",
        "games_per_order": N, "reference_teammates": teammates, "reference_opponents": opponents,
        "absolute_refs": {"attack_dmg": REF_ATTACK, "own_core": REF_OWN_CORE,
                          "tempo_rate": REF_TEMPO_RATE, "eco_buildings": REF_ECO_BLD, "resources": REF_RES,
                          "potential_steps": [POT_MIN_STEPS, POT_MAX_STEPS]},
        "cards": out_cards,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(out_cards)} cards (+{len(cards)} this run) -> {OUT.relative_to(ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
