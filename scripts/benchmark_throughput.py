"""
Benchmark main-training throughput for different (workers, episodes_per_net,
worker_torch_threads) configs.

Run:
    python scripts/benchmark_throughput.py

Each config: N_WARMUP throw-away updates, then N_MEASURE timed updates.
Reports steps/sec, sec/update, total_steps.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import multiprocessing as mp
import torch

from src.training.population import (
    PopulationConfig,
    PopulationTrainer,
    build_default_population,
)


# --------------------------------------------------------------------------- #
# Configs to benchmark: (workers, episodes_per_net, worker_torch_threads, label)
# --------------------------------------------------------------------------- #
CONFIGS = [
    # --- baseline: current production settings (workers inherit 8 torch threads)
    (16, 4, 8, "w16-ep4-t8  [current]"),
    # --- fix torch threads to 1 per worker via pool initializer
    (16, 4, 1, "w16-ep4-t1"),
    # --- vary worker count (torch_threads=1)
    ( 8, 4, 1, "w8-ep4-t1"),
    (12, 4, 1, "w12-ep4-t1"),
    (14, 4, 1, "w14-ep4-t1"),
    (18, 4, 1, "w18-ep4-t1"),
    # --- vary episodes at best-candidate worker count
    (16, 2, 1, "w16-ep2-t1"),
    (16, 6, 1, "w16-ep6-t1"),
]

N_WARMUP  = 2   # discarded updates (pool startup + JIT warmup)
N_MEASURE = 4   # measured updates


# --------------------------------------------------------------------------- #
# Worker initialiser
# --------------------------------------------------------------------------- #

def _init_worker(torch_threads: int) -> None:
    import torch as _torch
    _torch.set_num_threads(torch_threads)


# --------------------------------------------------------------------------- #
# Single config run
# --------------------------------------------------------------------------- #

def run_config(
    workers: int,
    episodes_per_net: int,
    worker_torch_threads: int,
    label: str,
) -> dict:
    torch.set_num_threads(1)  # main process always uses 1 thread for PPO

    tmp_dir = Path(tempfile.mkdtemp(prefix="bench_"))
    try:
        config = PopulationConfig(
            run_dir=str(tmp_dir),
            nets=build_default_population(),
            updates=N_WARMUP + N_MEASURE + 1,
            episodes_per_net_per_update=episodes_per_net,
            workers=max(1, workers),
            max_ticks=300,
            snapshot_interval=1,
            checkpoint_interval=9999,
            torch_threads=1,
            device="auto",
        )

        trainer = PopulationTrainer(config)
        trainer.save_all(update=0, milestone=False)

        ctx = mp.get_context("spawn")
        pool = None
        if workers > 1:
            pool = ctx.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(worker_torch_threads,),
            )

        pool_path = trainer.refresh_frozen_pool(version=0)

        # Warmup
        for i in range(1, N_WARMUP + 1):
            trainer.run_update(i, pool_path, pool)

        # Timed measurement
        t0 = time.perf_counter()
        total_steps = 0
        elapsed_per_update: list[float] = []

        for i in range(N_WARMUP + 1, N_WARMUP + N_MEASURE + 1):
            t1 = time.perf_counter()
            summary = trainer.run_update(i, pool_path, pool)
            dt = time.perf_counter() - t1
            total_steps += summary.get("steps_this_update", 0)
            elapsed_per_update.append(dt)

        total_elapsed = time.perf_counter() - t0

        return {
            "label": label,
            "workers": workers,
            "episodes_per_net": episodes_per_net,
            "worker_torch_threads": worker_torch_threads,
            "steps_per_sec": round(total_steps / total_elapsed, 1),
            "sec_per_update": round(total_elapsed / N_MEASURE, 3),
            "avg_steps_per_update": round(total_steps / N_MEASURE, 0),
        }

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    print(f"CPU cores: {os.cpu_count()} logical")
    print(f"PyTorch default threads: {torch.get_num_threads()}")
    print(f"Benchmarking {len(CONFIGS)} configs x {N_MEASURE} updates each")
    print(f"Warmup: {N_WARMUP} updates per config\n")

    results: list[dict] = []
    for workers, ep, threads, label in CONFIGS:
        print(f"  {label:<30}", end=" ", flush=True)
        try:
            r = run_config(workers, ep, threads, label)
            results.append(r)
            print(f"  {r['steps_per_sec']:>8.0f} steps/s   {r['sec_per_update']:>6.2f}s/upd")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 65)
    print(f"{'Config':<32} {'steps/sec':>10} {'sec/upd':>9} {'steps/upd':>10}")
    print("-" * 65)
    results.sort(key=lambda x: -x["steps_per_sec"])
    baseline = next((r for r in results if "current" in r["label"]), results[-1])
    for r in results:
        gain = r["steps_per_sec"] / baseline["steps_per_sec"]
        marker = " << BEST" if r is results[0] else (f" ({gain:.1f}x)" if r is not baseline else " << baseline")
        print(
            f"{r['label']:<32} {r['steps_per_sec']:>10.0f} "
            f"{r['sec_per_update']:>9.3f} {r['avg_steps_per_update']:>10.0f}"
            f"{marker}"
        )

    print(f"\nBest config:  {results[0]['label'].strip()}")
    print(f"  workers={results[0]['workers']}  episodes_per_net={results[0]['episodes_per_net']}  worker_torch_threads={results[0]['worker_torch_threads']}")
    baseline_sps = baseline["steps_per_sec"]
    best_sps = results[0]["steps_per_sec"]
    print(f"  {best_sps:.0f} steps/sec  vs baseline {baseline_sps:.0f}  ({best_sps/baseline_sps:.1f}x speedup)")
