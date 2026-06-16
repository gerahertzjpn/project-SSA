"""Absolute-progress eval for the base-18 training (faction-matched, reuses the
exact training machinery — no scripted/faction hacks).

For each net X we measure its win rate as the primary controlled player against a
frozen BASELINE pool (baseline_u923), with two weight sets:
  - current weights  -> wr_cur
  - baseline weights -> wr_base   (the control: a baseline net in a baseline match)
improvement(X) = wr_cur - wr_base. Aggregated per faction. >0 means the net got
stronger than its baseline since update ~923.

Free-play (no order) is used so the order channel doesn't confound raw skill.

    python scripts/eval_base18_progress.py --run-dir training_runs/base18_v1 \
        --baseline baseline_u923 --episodes 24 --workers 8
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.training.lineage import _collect_episodes_for_net  # noqa: E402
from src.training.policy import MaskedActorCritic  # noqa: E402
from src.training.population import (  # noqa: E402
    build_default_population, write_frozen_pool,
)


def _load_state(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["policy_state"]


def _win_rate(net, spec, ally_ids, opp_ids, baseline_states, specs, pool_path,
              config_view, episodes, seed):
    _, summaries = _collect_episodes_for_net(
        net, spec, ally_ids, opp_ids, episodes, baseline_states, specs,
        str(pool_path), config_view, episode_seed_base=seed)
    n = len(summaries) or 1
    return sum(s["win"] for s in summaries) / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Absolute progress vs a frozen baseline")
    ap.add_argument("--run-dir", default="training_runs/base18_v1")
    ap.add_argument("--baseline", default="baseline_u923", help="subdir with <net_id>.pt")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--nets", default="", help="comma net_ids to limit (debug)")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    specs = {s.net_id: s for s in build_default_population()}
    net_ids = list(specs)
    if args.nets:
        net_ids = [n for n in net_ids if n in set(args.nets.split(","))]

    base_dir = run_dir / args.baseline
    baseline_states = {nid: _load_state(base_dir / f"{nid}.pt") for nid in specs}
    pool_path = run_dir / "_eval_baseline_pool.pt"
    write_frozen_pool(pool_path, specs, baseline_states, update=923)

    # free-play eval: no order, no order shaping (raw skill)
    config_view = {"order_shaping_weight": 0.0, "opponent_deterministic": False,
                   "max_ticks": 300, "core_progress_weight": 0.5,
                   "sample_orders": True, "no_order_prob": 1.0}

    rows = []
    for i, nid in enumerate(net_ids):
        others = [n for n in specs if n != nid]
        ally_ids = [others[i % len(others)], others[(i + 6) % len(others)]]
        opp_ids = [others[(i + 1) % len(others)], others[(i + 7) % len(others)],
                   others[(i + 12) % len(others)]]
        cur = MaskedActorCritic(hidden_size=specs[nid].hidden_size)
        cur.load_state_dict(_load_state(run_dir / "nets" / nid / "latest.pt")); cur.eval()
        base = MaskedActorCritic(hidden_size=specs[nid].hidden_size)
        base.load_state_dict(baseline_states[nid]); base.eval()
        wr_cur = _win_rate(cur, specs[nid], ally_ids, opp_ids, baseline_states, specs,
                           pool_path, config_view, args.episodes, args.seed + i)
        wr_base = _win_rate(base, specs[nid], ally_ids, opp_ids, baseline_states, specs,
                            pool_path, config_view, args.episodes, args.seed + i)
        rows.append({"net": nid, "faction": specs[nid].faction,
                     "wr_cur": wr_cur, "wr_base": wr_base, "impr": wr_cur - wr_base})
        print(f"  {nid:11} cur {wr_cur:.2f} base {wr_base:.2f} impr {wr_cur - wr_base:+.2f}")

    print("\n=== per-faction improvement (current vs baseline_u923) ===")
    for fac in ("humans", "infected", "robots"):
        fr = [r["impr"] for r in rows if r["faction"] == fac]
        cr = [r["wr_cur"] for r in rows if r["faction"] == fac]
        if fr:
            print(f"  {fac:9} mean impr {statistics.mean(fr):+.2f} | mean cur-winrate {statistics.mean(cr):.2f}")
    allimpr = [r["impr"] for r in rows]
    print(f"  OVERALL   mean impr {statistics.mean(allimpr):+.2f} (n={len(rows)})")
    pool_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
