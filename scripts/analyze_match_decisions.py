"""Decision-level match analysis for the current base-18 nets.

Runs concrete free-play matches (current nets vs current nets, argmax = each net's
*best* choice, reproducible), decodes every tick's action to human-readable form, tracks
the game state, and flags signs of weak play. Answers: "how do the nets actually play
right now, and are there traces they're still weak?"

    python scripts/analyze_match_decisions.py --ticks 160 --narrate
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from src.training.policy import MaskedActorCritic  # noqa: E402
from src.training.population import (  # noqa: E402
    CONTROLLED_SLOTS, OPPONENT_SLOTS, build_default_population, write_frozen_pool,
)
from src.training.ppo_trainer import action_label, action_kind_label  # noqa: E402
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig  # noqa: E402

RUN_DIR = ROOT / "training_runs" / "population_v1"


def _load_state(nid: str) -> dict:
    return torch.load(RUN_DIR / "nets" / nid / "latest.pt", map_location="cpu",
                      weights_only=False)["policy_state"]


def _team_core_hp(state, slots) -> float:
    tot = 0.0
    for slot in slots:
        p = state.players.get(slot)
        if p is not None:
            tot += max(0.0, float(p.core_hp))
    return round(tot, 1)


def _node_counts(state):
    own = enemy = neutral = 0
    for n in state.nodes.values():
        owner = str(n.owner)
        if owner.startswith("A"):
            own += 1
        elif owner.startswith("B"):
            enemy += 1
        else:
            neutral += 1
    return own, enemy, neutral


def run_match(ctrl_ids, opp_ids, specs, pool_path, ticks, seed, narrate):
    nets = {}
    for nid in ctrl_ids:
        net = MaskedActorCritic(hidden_size=specs[nid].hidden_size)
        net.load_state_dict(_load_state(nid)); net.eval()
        nets[nid] = net
    slot_to_net = {s: nid for s, nid in zip(CONTROLLED_SLOTS, ctrl_ids)}
    player_factions = {**{s: specs[nid].faction for s, nid in zip(CONTROLLED_SLOTS, ctrl_ids)},
                       **{s: specs[nid].faction for s, nid in zip(OPPONENT_SLOTS, opp_ids)}}
    opp_state_ids = {s: nid for s, nid in zip(OPPONENT_SLOTS, opp_ids)}

    cfg = TeamTrainingEnvConfig(
        controlled_players=CONTROLLED_SLOTS, seed=seed,
        opponent_bot="policy", opponent_policy_checkpoint=str(pool_path),
        opponent_policy_deterministic=True, opponent_policy_state_ids=opp_state_ids,
        player_factions=player_factions, max_ticks=ticks,
        record_replay=False, record_trace=False, core_progress_weight=0.5,
        sample_orders=False, order_shaping_weight=0.0, no_order_prob=1.0)  # pure free play
    env = TeamTrainingEnv(cfg)
    obs, _ = env.reset(seed=seed, player_factions=player_factions)

    kinds = {s: Counter() for s in CONTROLLED_SLOTS}
    action_idx = {s: Counter() for s in CONTROLLED_SLOTS}
    holds = {s: 0 for s in CONTROLLED_SLOTS}
    produced = {s: 0 for s in CONTROLLED_SLOTS}
    built = {s: 0 for s in CONTROLLED_SLOTS}
    moved = {s: 0 for s in CONTROLLED_SLOTS}
    events = []
    prev_own = _team_core_hp(env.state, CONTROLLED_SLOTS)
    prev_enemy = _team_core_hp(env.state, OPPONENT_SLOTS)
    start_own, start_enemy = prev_own, prev_enemy
    t = 0
    done = False
    while not done:
        masks = env.action_masks()
        actions, labels = {}, {}
        for s in CONTROLLED_SLOTS:
            step = nets[slot_to_net[s]].act(obs[s], masks[s], deterministic=True)
            actions[s] = step.action
            lbl = action_label(step.action)
            labels[s] = lbl
            kind = action_kind_label(step.action)
            kinds[s][kind] += 1
            action_idx[s][lbl] += 1
            if kind == "HOLD":
                holds[s] += 1
            if "PRODUCE" in kind:
                produced[s] += 1
            if "BUILD" in kind:
                built[s] += 1
            if "MOVE" in kind:
                moved[s] += 1
        result = env.step(actions)
        obs = result.observations
        done = bool(result.terminal or result.truncated)
        t += 1
        own = _team_core_hp(env.state, CONTROLLED_SLOTS)
        enemy = _team_core_hp(env.state, OPPONENT_SLOTS)
        if enemy < prev_enemy - 1:
            events.append((t, f"enemy core -{round(prev_enemy-enemy,1)} (now {enemy})",
                           {s: labels[s] for s in CONTROLLED_SLOTS}))
        if own < prev_own - 1:
            events.append((t, f"OWN core -{round(prev_own-own,1)} (now {own})",
                           {s: labels[s] for s in CONTROLLED_SLOTS}))
        prev_own, prev_enemy = own, enemy

    info = result.info or {}
    winner = info.get("winner") or "DRAW"
    won = winner == env.controlled_team
    o_own, o_enemy, o_neu = _node_counts(env.state)

    print(f"\n=== MATCH  A={ctrl_ids}  vs  B={opp_ids}  (seed {seed}) ===")
    print(f"  result: {'WIN' if won else ('DRAW' if winner=='DRAW' else 'LOSS')} for A   "
          f"ticks={t}  core A {start_own}->{prev_own}  B {start_enemy}->{prev_enemy}  "
          f"nodes A/B/neu {o_own}/{o_enemy}/{o_neu}")
    for s, nid in slot_to_net.items():
        hr = holds[s] / max(1, t)
        dom = action_idx[s].most_common(1)[0] if action_idx[s] else ("-", 0)
        print(f"  {nid:<12} HOLD {hr:4.0%}  produce {produced[s]:3d}  move {moved[s]:3d}  "
              f"build {built[s]:3d}  unique {len(action_idx[s]):3d}  "
              f"dom {dom[1]/max(1,t):4.0%} [{dom[0][:38]}]  kinds={dict(kinds[s])}")
    # weakness flags
    flags = []
    for s, nid in slot_to_net.items():
        if holds[s] / max(1, t) > 0.5:
            flags.append(f"{nid}: PASSIVE ({holds[s]/max(1,t):.0%} HOLD)")
        if built[s] == 0:
            flags.append(f"{nid}: never builds (no tech/eco)")
        if len(action_idx[s]) <= 3:
            flags.append(f"{nid}: narrow policy ({len(action_idx[s])} distinct actions)")
        if action_idx[s] and action_idx[s].most_common(1)[0][1] / max(1, t) > 0.6:
            flags.append(f"{nid}: repetitive ({action_idx[s].most_common(1)[0][1]/t:.0%} one action)")
    if prev_enemy >= start_enemy - 0.5 and prev_own >= start_own - 0.5:
        flags.append("STALEMATE: neither core damaged (no decisive aggression)")
    print(f"  WEAKNESS FLAGS: {flags or ['none obvious']}")
    if narrate and events:
        print("  key moments (core-damage ticks + the decisions that tick):")
        for tk, what, decs in events[:10]:
            ds = "  ".join(f"{s}:{decs[s].split(':',1)[-1][:26]}" for s in CONTROLLED_SLOTS)
            print(f"    t{tk:>3} {what:<28} | {ds}")
    return {"won": won, "winner": winner, "ticks": t, "flags": flags}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=160)
    ap.add_argument("--narrate", action="store_true")
    ap.add_argument("--seed", type=int, default=20260618)
    args = ap.parse_args()

    specs = {s.net_id: s for s in build_default_population()}
    states = {nid: _load_state(nid) for nid in specs}
    pool = RUN_DIR / "_match_pool.pt"
    write_frozen_pool(pool, specs, states, update=0)

    # a few faction-mixed matchups across different nets/recipes
    matchups = [
        (["humans_1", "infected_1", "robots_1"], ["humans_2", "infected_2", "robots_2"]),
        (["humans_5", "infected_3", "robots_2"], ["humans_4", "infected_6", "robots_5"]),
        (["robots_6", "humans_3", "infected_4"], ["robots_1", "humans_6", "infected_5"]),
    ]
    results = []
    for i, (a, b) in enumerate(matchups):
        results.append(run_match(a, b, specs, pool, args.ticks, args.seed + i * 101, args.narrate))
    pool.unlink(missing_ok=True)

    allflags = [f for r in results for f in r["flags"]]
    print(f"\n=== SUMMARY over {len(results)} matches ===")
    print(f"  decisive (non-draw): {sum(1 for r in results if r['winner']!='DRAW')}/{len(results)}")
    print(f"  flag tally: {dict(Counter(f.split(':')[1].strip() if ':' in f else f for f in allflags))}")


if __name__ == "__main__":
    main()
