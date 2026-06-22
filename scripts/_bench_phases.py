"""Throwaway: split one training update into ROLLOUT (parallel workers) vs PPO
(main process / GPU, workers idle). The PPO fraction is the CPU 'dip' and the
hard ceiling of what pipelining could ever hide. Loads the trained population_v1.
"""
from __future__ import annotations

import multiprocessing as mp
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.training.policy import to_cpu_state_dict
from src.training.population import (
    PopulationConfig, PopulationTrainer, build_default_population,
    build_episode_plans, _rollout_worker, _merge_net_rollout, _PPOUpdateView,
)
from src.training.ppo_trainer import update_policy

REF = ROOT / "training_runs" / "population_v1"
WORKERS = 24
N_WARMUP = 3
N_MEASURE = 8


def _init_worker(n):
    import torch as _t
    _t.set_num_threads(n)


def build_requests(trainer, config, update):
    plans = build_episode_plans(config, update)
    live_states = {nid: to_cpu_state_dict(net) for nid, net in trainer.nets.items()}
    specs = {nid: asdict(sp) for nid, sp in trainer.spec_by_id.items()}
    cv = {"opponent_deterministic": config.opponent_deterministic, "max_ticks": config.max_ticks,
          "core_progress_weight": config.core_progress_weight, "sample_orders": config.sample_orders,
          "order_shaping_weight": config.order_shaping_weight}
    wc = max(1, config.workers)
    pd = [{"seed": p.seed, "controlled": list(p.controlled), "opponents": list(p.opponents)} for p in plans]
    chunks = [c for c in (pd[i::wc] for i in range(wc)) if c]
    reqs = []
    for wi, ch in enumerate(chunks):
        needed = sorted({nid for p in ch for nid in p["controlled"]})
        reqs.append({"torch_threads": config.torch_threads,
                     "worker_seed": config.master_seed + update * 7919 + wi * 17,
                     "specs": specs, "config_view": cv, "frozen_pool_path": str(trainer._fpp),
                     "live_states": {nid: live_states[nid] for nid in needed}, "plans": ch})
    return reqs


def ppo_phase(trainer, config, worker_results):
    for net_id in trainer.nets:
        sp = trainer.spec_by_id[net_id]
        merged = _merge_net_rollout(worker_results, net_id, config.gamma, sp.gae_lambda)
        if merged is None:
            continue
        view = _PPOUpdateView(ppo_epochs=config.ppo_epochs, minibatch_size=config.minibatch_size,
                              clip_coef=sp.clip_coef, value_coef=config.value_coef,
                              entropy_coef=sp.entropy_coef, max_grad_norm=config.max_grad_norm)
        update_policy(trainer.nets[net_id], trainer.optimizers[net_id], merged, view, trainer.device)


def main():
    torch.set_num_threads(1)
    config = PopulationConfig(run_dir=str(REF), nets=build_default_population(),
                              updates=999, episodes_per_net_per_update=4, workers=WORKERS,
                              max_ticks=300, snapshot_interval=20, checkpoint_interval=99999,
                              torch_threads=1, device="auto")
    trainer = PopulationTrainer(config)
    trainer.resume()
    trainer._fpp = trainer.refresh_frozen_pool(version=0)
    pool = mp.get_context("spawn").Pool(processes=WORKERS, initializer=_init_worker, initargs=(1,))
    u = trainer.start_update
    for _ in range(N_WARMUP):
        trainer.run_update(u, trainer._fpp, pool); u += 1

    rolls, ppos = [], []
    for _ in range(N_MEASURE):
        reqs = build_requests(trainer, config, u)
        t = time.perf_counter()
        wr = pool.map(_rollout_worker, reqs)
        rolls.append(time.perf_counter() - t)
        t = time.perf_counter()
        ppo_phase(trainer, config, wr)
        ppos.append(time.perf_counter() - t)
        u += 1
    pool.close(); pool.join()

    r, p = statistics.mean(rolls), statistics.mean(ppos)
    tot = r + p
    print(f"workers={WORKERS}  measured {N_MEASURE} updates")
    print(f"  ROLLOUT (parallel, workers busy): {r:.3f}s  ({r/tot:.0%})")
    print(f"  PPO     (main/GPU, workers idle): {p:.3f}s  ({p/tot:.0%})  <- the CPU dip")
    print(f"  total/update: {tot:.3f}s")
    print(f"  pipelining ceiling (max hideable): {p/tot:.0%}  -> up to {tot/r:.2f}x if fully overlapped")


if __name__ == "__main__":
    raise SystemExit(main())
