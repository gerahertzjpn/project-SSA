"""Benchmark LeagueTrainer throughput across configs to find the off-season speed gap.

SAFE: calls run_update directly (NO save_checkpoints / write_registry) so on-disk net checkpoints are
never overwritten — only in-memory net state changes, discarded at process exit. Cleans up any pool_v*.pt
files it creates. Use it to compare single-net (1 triple) vs saturated (many triples) throughput.

  python scripts/bench_training_throughput.py            # default sweep
  python scripts/bench_training_throughput.py --updates 6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.training.league_trainer import (  # noqa: E402
    LeagueTrainer, LeagueConfig, FACTIONS, _init_worker)
from src.training.population_roster import build_population  # noqa: E402

RUN = "training_runs/league_population_v1"


def _pick(n_per_fac: int):
    pop = build_population()
    by_fac = {f: [d for d in pop if d.faction == f] for f in FACTIONS}
    ds = []
    for f in FACTIONS:
        ds += by_fac[f][:n_per_fac]
    return ds


def bench(n_per_fac: int, ept: int, workers: int, n_updates: int) -> dict:
    """n_per_fac nets per faction -> n_per_fac balanced triples. ept episodes/triple."""
    ds = _pick(n_per_fac)
    pre_pools = {p.name for p in Path(RUN).glob("pool_v*.pt")}
    cfg = LeagueConfig(run_dir=RUN, workers=workers, episodes_per_triple=ept,
                       target_steps_per_net=10**12, ancestor_opponent_prob=0.7,
                       descendant_opponent_prob=0.0)
    tr = LeagueTrainer(cfg, descendants=ds)
    pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker, initargs=(1,))
    try:
        tr.run_update(0, pool=pool)                     # warm-up: pay the spawn here, not in timing
        s0 = sum(tr.steps_by_net.values()); t0 = time.time()
        per = []
        for u in range(1, n_updates + 1):
            a = time.time(); r = tr.run_update(u, pool=pool); per.append(time.time() - a)
        dt = time.time() - t0
        steps = sum(tr.steps_by_net.values()) - s0
        nets = len(ds)
        return {"triples": n_per_fac, "nets": nets, "ept": ept, "workers": workers,
                "total_steps_per_s": round(steps / dt),
                "per_net_steps_per_s": round(steps / dt / nets),
                "upd_s_median": round(statistics.median(per), 2),
                "upd_s_min": round(min(per), 2), "upd_s_max": round(max(per), 2),
                "episodes_per_upd": r.get("episodes"), "mean_ep_steps": r.get("mean_episode_steps")}
    finally:
        pool.close(); pool.join()
        for p in Path(RUN).glob("pool_v*.pt"):          # delete ONLY pool files this bench created
            if p.name not in pre_pools:
                try:
                    p.unlink()
                except OSError:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    W = args.workers
    # (triples_per_faction, episodes_per_triple, workers)
    sweep = [
        (1, 30, W),    # current single-net off-season: 1 triple, ept=workers
        (1, 60, W),    # 1 triple, 2x episodes (load-balancing buffer test)
        (1, 90, W),    # 1 triple, 3x episodes
        (8, 4, W),     # saturated, standalone-like (24 nets, few ep/triple)
        (8, 8, W),     # saturated, more ep/triple
        (10, 6, W),    # 30 nets
        (8, 4, max(2, W - 6)),   # fewer workers (oversubscription / contention check)
    ]
    print(f"machine workers default={W}; updates/timing={args.updates}\n", flush=True)
    for c in sweep:
        try:
            print(bench(*c, args.updates), flush=True)
        except Exception as e:  # noqa: BLE001
            print({"config": c, "ERROR": repr(e)}, flush=True)


if __name__ == "__main__":
    main()
