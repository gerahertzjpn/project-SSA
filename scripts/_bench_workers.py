"""Throwaway: CORRECT worker-count sweep 16..32 (step 1) at the REAL training stage.

Each worker count starts from the SAME trained nets (copied from population_v1), so
episode length (~101 steps, trained) is identical across configs — unlike the old
benchmark which used fresh nets (long ~200-step random games). Uses the real config:
ep4, frozen-pool refresh every snapshot_interval (NO pool rebuild — #3), full update
loop (rollout + PPO). Warmup discards pool-spawn + first-refresh; then measure.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.training.population import (
    PopulationConfig, PopulationTrainer, build_default_population,
)

REFERENCE = ROOT / "training_runs" / "population_v1"
WORKERS = list(range(16, 33))          # 16..32 step 1
EPISODES_PER_NET = 4
SNAPSHOT_INTERVAL = 5
N_WARMUP = 4                           # discarded: pool spawn + warmup + first refresh
N_MEASURE = 12                        # measured (>= 2 refreshes at interval 5)
RESULT = ROOT / "training_runs" / "_bench_workers_result.json"


def _init_worker(n_threads: int) -> None:
    import torch as _t
    _t.set_num_threads(n_threads)


def _copy_reference_nets(dst: Path) -> None:
    for net_dir in (REFERENCE / "nets").iterdir():
        latest = net_dir / "latest.pt"
        if latest.exists():
            (dst / "nets" / net_dir.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest, dst / "nets" / net_dir.name / "latest.pt")


def run_one(workers: int) -> dict:
    torch.set_num_threads(1)
    tmp = Path(tempfile.mkdtemp(prefix=f"bw{workers}_"))
    try:
        _copy_reference_nets(tmp)
        config = PopulationConfig(
            run_dir=str(tmp), nets=build_default_population(),
            updates=N_WARMUP + N_MEASURE + 1, episodes_per_net_per_update=EPISODES_PER_NET,
            workers=workers, max_ticks=300, snapshot_interval=SNAPSHOT_INTERVAL,
            checkpoint_interval=9999, torch_threads=1, device="auto",
        )
        trainer = PopulationTrainer(config)
        trainer.resume()                                   # load the trained weights
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers, initializer=_init_worker, initargs=(1,))
        pool_path = trainer.refresh_frozen_pool(version=0)
        last_refresh = 0
        u = trainer.start_update

        for _ in range(N_WARMUP):                          # warmup (no rebuild — #3)
            if u - last_refresh >= SNAPSHOT_INTERVAL:
                pool_path = trainer.refresh_frozen_pool(version=u); last_refresh = u
            trainer.run_update(u, pool_path, pool); u += 1

        t0 = time.perf_counter(); steps = 0; mes = []
        for _ in range(N_MEASURE):
            if u - last_refresh >= SNAPSHOT_INTERVAL:
                pool_path = trainer.refresh_frozen_pool(version=u); last_refresh = u
            s = trainer.run_update(u, pool_path, pool)
            steps += s.get("steps_this_update", 0); mes.append(s.get("mean_episode_steps", 0)); u += 1
        dt = time.perf_counter() - t0
        pool.close(); pool.join()
        return {"workers": workers, "steps_per_sec": round(steps / dt, 0),
                "sec_per_update": round(dt / N_MEASURE, 3),
                "mean_ep_steps": round(sum(mes) / len(mes), 0)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print(f"Worker sweep 16..32 (step 1) at trained stage | ep{EPISODES_PER_NET} | "
          f"warmup {N_WARMUP} measure {N_MEASURE}", flush=True)
    results = []
    for w in WORKERS:
        try:
            r = run_one(w)
            results.append(r)
            print(f"  w{w:<2}: {r['steps_per_sec']:>6.0f} st/s   {r['sec_per_update']:>6.3f}s/upd   "
                  f"mean_ep {r['mean_ep_steps']:.0f}", flush=True)
        except Exception as e:
            print(f"  w{w:<2}: ERROR {e}", flush=True)
    results.sort(key=lambda x: -x["steps_per_sec"])
    RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== ranked ===", flush=True)
    best = results[0]["steps_per_sec"] if results else 1
    for r in results:
        print(f"  w{r['workers']:<2}: {r['steps_per_sec']:>6.0f} st/s  ({r['steps_per_sec']/best:.3f})", flush=True)
    if results:
        print(f"\nBEST: {results[0]['workers']} workers @ {results[0]['steps_per_sec']:.0f} st/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
