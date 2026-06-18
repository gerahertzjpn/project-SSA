"""Order-adherence tracker — are the nets actually FOLLOWING the strategy orders?

base-18 trains with orders in the observation (`train_obs_v4_orders`) and a potential-
based order-alignment reward (`order_shaping_weight`). But the training logs never
aggregate how WELL nets follow orders. This eval fills that gap.

Signal: with `order_shaping_weight=1.0` the env's per-step order reward equals the change
in the alignment potential Φ (telescoping). Summed over an episode it = how far the net
steered its game-state toward the COMMANDED style. We read it straight from
`info["reward_components"][slot]["order"]` (env-computed, no re-derivation), bucket by the
strategy preset the env sampled, per faction, and compare CURRENT nets vs their own selves
~LOOKBACK updates ago. A positive, growing lift = nets are LEARNING to obey orders.

    python scripts/track_orders.py --episodes 24 --lookback 1500
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
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
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig  # noqa: E402
from src.training.spaces import STRATEGY_PRESETS  # noqa: E402

FACTIONS = ("humans", "infected", "robots")
PRESET_IDS = tuple(p["id"] for p in STRATEGY_PRESETS)


def _load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["policy_state"]


def _metric_rows(run_dir: Path):
    return [json.loads(l) for l in open(run_dir / "metrics.jsonl", encoding="utf-8")
            if l.strip() and '"kind": "population_update"' in l]


def _nearest_milestone(net_dir: Path, target: int):
    ups = sorted(int(p.stem.split("_")[1]) for p in net_dir.glob("update_*.pt"))
    le = [u for u in ups if u <= target]
    return le[-1] if le else (ups[0] if ups else None)


def _run_ordered_episodes(ctrl_ids, opp_ids, states, specs, pool_path, episodes, seed):
    """Return {preset_id: [per-net order-reward sums]} for the controlled team."""
    nets = {}
    for nid in ctrl_ids:
        net = MaskedActorCritic(hidden_size=specs[nid].hidden_size)
        net.load_state_dict(states[nid]); net.eval()
        nets[nid] = net
    slot_to_net = {slot: nid for slot, nid in zip(CONTROLLED_SLOTS, ctrl_ids)}
    player_factions = {**{slot: specs[nid].faction for slot, nid in zip(CONTROLLED_SLOTS, ctrl_ids)},
                       **{slot: specs[nid].faction for slot, nid in zip(OPPONENT_SLOTS, opp_ids)}}
    opp_state_ids = {slot: nid for slot, nid in zip(OPPONENT_SLOTS, opp_ids)}

    by_preset = defaultdict(list)
    for e in range(episodes):
        cfg = TeamTrainingEnvConfig(
            controlled_players=CONTROLLED_SLOTS, seed=seed + e * 17,
            opponent_bot="policy", opponent_policy_checkpoint=str(pool_path),
            opponent_policy_deterministic=False, opponent_policy_state_ids=opp_state_ids,
            player_factions=player_factions, max_ticks=300,
            record_replay=False, record_trace=False, core_progress_weight=0.5,
            sample_orders=True, order_shaping_weight=1.0, no_order_prob=0.0)
        env = TeamTrainingEnv(cfg)
        obs, _ = env.reset(seed=seed + e * 17, player_factions=player_factions)
        order0 = (env._order or {}).get(CONTROLLED_SLOTS[0]) or {}
        preset = order0.get("strategy")
        if not preset:                       # safety: skip the rare no-order episode
            continue
        sums = {slot: 0.0 for slot in CONTROLLED_SLOTS}
        done = False
        while not done:
            masks = env.action_masks()
            actions = {slot: nets[slot_to_net[slot]].act(obs[slot], masks[slot],
                                                         deterministic=False).action
                       for slot in CONTROLLED_SLOTS}
            result = env.step(actions)
            rc = (result.info or {}).get("reward_components", {})
            for slot in CONTROLLED_SLOTS:
                sums[slot] += float((rc.get(slot, {}) or {}).get("order", 0.0))
            obs = result.observations
            done = bool(result.terminal or result.truncated)
        for slot in CONTROLLED_SLOTS:
            by_preset[preset].append(sums[slot])
    return by_preset


def _faction_scores(states, specs, pool_path, episodes, seed):
    """Per-faction {preset: mean order-reward} using each faction's first 3 nets."""
    opp_ids = ["humans_6", "infected_6", "robots_6"]   # fixed mixed frozen opponents
    out = {}
    for fi, fac in enumerate(FACTIONS):
        ctrl = [f"{fac}_{i}" for i in (1, 2, 3)]
        bp = _run_ordered_episodes(ctrl, opp_ids, states, specs, pool_path,
                                   episodes, seed + fi * 1009)
        out[fac] = {p: round(statistics.mean(v), 4) for p, v in bp.items() if v}
    return out


def _overall(scores):
    """Mean over all (faction, preset) cells."""
    vals = [v for fac in scores.values() for v in fac.values()]
    return round(statistics.mean(vals), 4) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="training_runs/population_v1")
    ap.add_argument("--episodes", type=int, default=24, help="ordered episodes per faction")
    ap.add_argument("--lookback", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=909)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    specs = {s.net_id: s for s in build_default_population()}
    net_ids = list(specs)
    rows = _metric_rows(run_dir)
    cur_update, total_steps = rows[-1]["update"], rows[-1]["total_steps"]
    past_update = _nearest_milestone(run_dir / "nets" / net_ids[0], max(0, cur_update - args.lookback))

    cur_states = {n: _load_state(run_dir / "nets" / n / "latest.pt") for n in net_ids}
    past_states = {n: _load_state(run_dir / "nets" / n / f"update_{past_update}.pt") for n in net_ids}

    cur_pool = run_dir / "_orders_cur_pool.pt"; write_frozen_pool(cur_pool, specs, cur_states, cur_update)
    past_pool = run_dir / "_orders_past_pool.pt"; write_frozen_pool(past_pool, specs, past_states, past_update)

    print(f"order-adherence @update {cur_update} (~{total_steps/1e6:.1f}M) vs past @update {past_update}")
    cur = _faction_scores(cur_states, specs, cur_pool, args.episodes, args.seed)
    past = _faction_scores(past_states, specs, past_pool, args.episodes, args.seed)
    cur_pool.unlink(missing_ok=True); past_pool.unlink(missing_ok=True)

    # per-preset overall (mean over factions), current vs past
    preset_cur = {p: round(statistics.mean([cur[f][p] for f in FACTIONS if p in cur[f]]), 4)
                  for p in PRESET_IDS if any(p in cur[f] for f in FACTIONS)}
    preset_past = {p: round(statistics.mean([past[f][p] for f in FACTIONS if p in past[f]]), 4)
                   for p in PRESET_IDS if any(p in past[f] for f in FACTIONS)}
    fac_cur = {f: round(statistics.mean(cur[f].values()), 4) for f in FACTIONS if cur[f]}
    fac_past = {f: round(statistics.mean(past[f].values()), 4) for f in FACTIONS if past[f]}
    overall_cur, overall_past = _overall(cur), _overall(past)
    lift = round(overall_cur - overall_past, 4)

    record = {"ts": int(time.time()), "update": cur_update, "total_steps": total_steps,
              "past_update": past_update, "overall_cur": overall_cur, "overall_past": overall_past,
              "lift": lift, "faction_cur": fac_cur, "faction_past": fac_past,
              "preset_cur": preset_cur, "preset_past": preset_past}
    with open(run_dir / "orders_progress.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print("\n=== ORDER ADHERENCE (order-reward = Phi movement toward commanded style) ===")
    print(f"  OVERALL: current {overall_cur:+.3f}  past {overall_past:+.3f}  "
          f"lift {lift:+.3f}  (>0 = learning to obey)")
    print(f"  by faction (current): {fac_cur}")
    print(f"           (vs past):   {fac_past}")
    print("  by strategy preset (current -> follows which orders best):")
    for p in PRESET_IDS:
        if p in preset_cur:
            d = round(preset_cur[p] - preset_past.get(p, 0.0), 3)
            print(f"    {p:<10} {preset_cur[p]:+.3f}  (lift {d:+.3f})")


if __name__ == "__main__":
    main()
