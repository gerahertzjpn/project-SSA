"""
Per-lineage branching and selection for Population-18 draft pool.

Each of the 18 players is a permanent root.  Selection / branching happens
ONLY within one player's own lineage tree — never across players.  The pool
always contains exactly 18 unique models.

Mechanic (per player, independently):
  1. Clone parent weights → n_branches copies with small Gaussian perturbation.
  2. Train each branch for mini_updates PPO updates vs the shared frozen pool.
  3. Evaluate ALL branches against the SAME fixed opponents/allies with the
     SAME episode seeds (fairest possible comparison — only weights differ).
  4. Select winner (highest eval win rate).
  5. Winner's weights overwrite the live net in trainer.  latest.pt updated.
  6. Losing branches are discarded.  Registry records lineage.

Root preservation guarantee:
  - net_id strings never change.
  - All 18 nets are always present in trainer and on disk.
  - Lineage tracked via 'generation' counter in checkpoint payload.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.training.policy import MaskedActorCritic, to_cpu_state_dict
from src.training.ppo_trainer import update_policy
from src.training.population import (
    NetSpec,
    PopulationConfig,
    _PPOUpdateView,
    _empty_buffer,
    _merge_net_rollout,
    _run_episode_collect,
    EpisodePlan,
    CONTROLLED_SLOTS,
    OPPONENT_SLOTS,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BranchingConfig:
    """Hyperparameters for the per-lineage branch-train-eval-select cycle."""

    # Branches created per player per cycle.
    n_branches: int = 3

    # PPO updates per branch before evaluation.
    mini_updates: int = 3

    # Episodes per mini PPO update.
    mini_episodes_per_update: int = 3

    # Evaluation episodes per branch.
    # SAME seeds for all branches → only weights differ.
    eval_episodes: int = 12

    # Gaussian weight-perturbation std applied to parent weights.
    mutation_scale: float = 0.004

    # Trigger branching every N main training updates.
    branch_interval: int = 50

    # Reproducibility base seed for evaluation episode generation.
    eval_seed_base: int = 20260609_0000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _perturb_state_dict(state_dict: dict, scale: float, seed: int) -> dict:
    """Return a new state_dict with Gaussian noise added to all float params."""
    gen = torch.Generator()
    gen.manual_seed(int(seed) % (2 ** 32 - 1))
    out: dict = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            out[k] = v + torch.randn(v.shape, generator=gen) * scale
        else:
            out[k] = v.clone()
    return out


def _quick_state_hash(state_dict: dict) -> str:
    h = hashlib.md5()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        h.update(state_dict[k].cpu().numpy().tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Single-net episode collection
# --------------------------------------------------------------------------- #

def _collect_episodes_for_net(
    net: MaskedActorCritic,
    net_spec: NetSpec,
    ally_ids: list[str],
    opp_ids: list[str],
    n_episodes: int,
    frozen_pool_states: dict[str, dict],
    all_specs: dict[str, NetSpec],
    frozen_pool_path: str,
    config_view: dict,
    episode_seed_base: int,
) -> tuple[dict, list[dict]]:
    """
    Run n_episodes with `net` as the primary controlled player.

    The two ally slots are filled with frozen-pool weights (participate but
    never updated here).  B-team loaded via PolicyBot / frozen-pool path.

    Returns
    -------
    agent_data : dict (numpy arrays) for net_spec.net_id, or {} if empty.
    episode_summaries : list of per-episode dicts with win/loss/draw fields.
    """
    # Load frozen ally nets (read-only).
    ally_nets: dict[str, MaskedActorCritic] = {}
    for aid in ally_ids:
        n = MaskedActorCritic(hidden_size=all_specs[aid].hidden_size)
        n.load_state_dict(frozen_pool_states[aid])
        n.eval()
        ally_nets[aid] = n

    all_nets = {net_spec.net_id: net, **ally_nets}
    buffers = {nid: _empty_buffer() for nid in all_nets}

    rng = np.random.default_rng(episode_seed_base)
    episode_summaries: list[dict] = []

    for ep_i in range(n_episodes):
        controlled_order: list[str] = [net_spec.net_id] + list(ally_ids)
        rng.shuffle(controlled_order)
        plan = EpisodePlan(
            seed=int(episode_seed_base + ep_i * 997 + 13),
            controlled=tuple(controlled_order),
            opponents=tuple(opp_ids),
        )
        summary = _run_episode_collect(
            plan, all_nets, all_specs, frozen_pool_path, config_view, buffers
        )
        episode_summaries.append(summary)

    buf = buffers[net_spec.net_id]
    if not buf["dones"]:
        return {}, episode_summaries

    agent_data = {
        "observations": np.asarray(buf["observations"], dtype=np.float32),
        "masks":        np.asarray(buf["masks"],        dtype=np.bool_),
        "actions":      np.asarray(buf["actions"],      dtype=np.int64),
        "logprobs":     np.asarray(buf["logprobs"],     dtype=np.float32),
        "values":       np.asarray(buf["values"],       dtype=np.float32),
        "rewards":      np.asarray(buf["rewards"],      dtype=np.float32),
        "dones":        np.asarray(buf["dones"],        dtype=np.bool_),
        "entropies":    np.asarray(buf["entropies"],    dtype=np.float32),
    }
    return agent_data, episode_summaries


# --------------------------------------------------------------------------- #
# Mini-training a single branch
# --------------------------------------------------------------------------- #

def _mini_train_branch(
    branch_net: MaskedActorCritic,
    spec: NetSpec,
    frozen_pool_states: dict[str, dict],
    all_specs: dict[str, NetSpec],
    frozen_pool_path: str,
    config_view: dict,
    pop_config: PopulationConfig,
    mini_updates: int,
    mini_episodes_per_update: int,
    device: str,
    seed_base: int,
) -> None:
    """Train branch_net in-place for mini_updates PPO steps vs frozen opponents."""
    other_ids = [nid for nid in frozen_pool_states if nid != spec.net_id]
    optimizer = torch.optim.Adam(branch_net.parameters(), lr=spec.learning_rate)

    ppo_cfg = _PPOUpdateView(
        ppo_epochs=pop_config.ppo_epochs,
        minibatch_size=pop_config.minibatch_size,
        clip_coef=spec.clip_coef,
        value_coef=pop_config.value_coef,
        entropy_coef=spec.entropy_coef,
        max_grad_norm=pop_config.max_grad_norm,
    )

    rng = np.random.default_rng(seed_base)

    for mini_i in range(mini_updates):
        ally_ids = list(rng.choice(other_ids, size=2, replace=False))
        opp_ids  = list(rng.choice(other_ids, size=3, replace=False))

        agent_data, _ = _collect_episodes_for_net(
            net=branch_net,
            net_spec=spec,
            ally_ids=ally_ids,
            opp_ids=opp_ids,
            n_episodes=mini_episodes_per_update,
            frozen_pool_states=frozen_pool_states,
            all_specs=all_specs,
            frozen_pool_path=frozen_pool_path,
            config_view=config_view,
            episode_seed_base=int(seed_base + mini_i * 10007 + 3),
        )
        if not agent_data:
            continue

        rollout = _merge_net_rollout(
            [{"agents": {spec.net_id: agent_data}}],
            spec.net_id,
            gamma=pop_config.gamma,
            gae_lambda=spec.gae_lambda,
        )
        if rollout is None:
            continue

        branch_net.train()
        update_policy(branch_net, optimizer, rollout, ppo_cfg, device)
        branch_net.eval()


# --------------------------------------------------------------------------- #
# Per-player cycle (worker-safe, no trainer reference)
# --------------------------------------------------------------------------- #

def _branch_one_player_worker(args: tuple) -> tuple:
    """
    Picklable worker function: branch-train-eval-select for one player.
    All inputs are primitive/picklable — no PopulationTrainer reference.
    Returns (net_id, winner_state_dict, player_result_dict).
    """
    torch.set_num_threads(1)  # prevent CPU oversubscription in parallel workers

    (net_id, parent_state, spec, all_specs,
     frozen_pool_path, branching_config, pop_config, cycle_num) = args

    # Load frozen pool inside the worker to avoid serialising 18×~50 MB of
    # tensors through the pickle channel (disk read is cheaper here).
    pool_ckpt = torch.load(frozen_pool_path, map_location="cpu", weights_only=False)
    frozen_pool_states: dict[str, dict] = pool_ckpt["policy_states"]

    parent_hash = _quick_state_hash(parent_state)
    other_ids = [nid for nid in frozen_pool_states if nid != net_id]
    rng = np.random.default_rng(
        branching_config.eval_seed_base + cycle_num * 997 + abs(hash(net_id)) % 9999
    )

    eval_ally_ids = list(rng.choice(other_ids, size=2, replace=False))
    eval_opp_ids  = list(rng.choice(other_ids, size=3, replace=False))
    eval_seed = int(
        branching_config.eval_seed_base + cycle_num * 10007 + abs(hash(net_id)) % 9999
    )
    config_view = {
        "opponent_deterministic": pop_config.opponent_deterministic,
        "max_ticks": pop_config.max_ticks,
    }

    branch_results: list[dict] = []
    for branch_idx in range(branching_config.n_branches):
        perturb_seed = (
            cycle_num * 100_000
            + abs(hash(net_id)) % 99_999
            + branch_idx * 1_000
        )
        branch_state = _perturb_state_dict(
            parent_state, branching_config.mutation_scale, perturb_seed
        )
        branch_net = MaskedActorCritic(hidden_size=spec.hidden_size)
        branch_net.load_state_dict(branch_state)
        branch_net.cpu()
        branch_net.eval()

        train_seed = (
            cycle_num * 200_000
            + abs(hash(net_id)) % 99_999
            + branch_idx * 5_000
        )
        _mini_train_branch(
            branch_net=branch_net,
            spec=spec,
            frozen_pool_states=frozen_pool_states,
            all_specs=all_specs,
            frozen_pool_path=frozen_pool_path,
            config_view=config_view,
            pop_config=pop_config,
            mini_updates=branching_config.mini_updates,
            mini_episodes_per_update=branching_config.mini_episodes_per_update,
            device="cpu",
            seed_base=train_seed,
        )

        branch_net.eval()
        _, eval_summaries = _collect_episodes_for_net(
            net=branch_net,
            net_spec=spec,
            ally_ids=eval_ally_ids,
            opp_ids=eval_opp_ids,
            n_episodes=branching_config.eval_episodes,
            frozen_pool_states=frozen_pool_states,
            all_specs=all_specs,
            frozen_pool_path=frozen_pool_path,
            config_view=config_view,
            episode_seed_base=eval_seed,
        )

        wins = sum(1 for s in eval_summaries if s.get("win") == 1)
        win_rate = wins / len(eval_summaries) if eval_summaries else 0.0
        branch_results.append({
            "branch_idx": branch_idx,
            "win_rate": win_rate,
            "wins": wins,
            "eval_episodes": len(eval_summaries),
            "state_dict": to_cpu_state_dict(branch_net),
        })

    winner = max(branch_results, key=lambda r: r["win_rate"])
    player_result = {
        "net_id": net_id,
        "parent_hash": parent_hash,
        "winner_branch_idx": winner["branch_idx"],
        "winner_win_rate": round(winner["win_rate"], 4),
        "branch_win_rates": [round(r["win_rate"], 4) for r in branch_results],
    }
    return net_id, winner["state_dict"], player_result


def _branch_one_player(
    net_id: str,
    trainer,
    frozen_pool_states: dict[str, dict],
    branching_config: BranchingConfig,
    frozen_pool_path: str,
    cycle_num: int,
) -> dict:
    """Single-player wrapper used by tests; delegates to the worker function."""
    parent_state = to_cpu_state_dict(trainer.nets[net_id])
    device = trainer.device

    _, winner_state, player_result = _branch_one_player_worker((
        net_id, parent_state, trainer.spec_by_id[net_id],
        trainer.spec_by_id, frozen_pool_path,
        branching_config, trainer.config, cycle_num,
    ))

    trainer.nets[net_id].load_state_dict(
        {k: v.to(device) for k, v in winner_state.items()}
    )
    trainer.nets[net_id].eval()
    return player_result


# --------------------------------------------------------------------------- #
# Full cycle across all 18 players
# --------------------------------------------------------------------------- #

def run_branch_select_cycle(
    trainer,
    branching_config: BranchingConfig,
    frozen_pool_path: str,
    cycle_num: int,
    update: int,
    n_parallel: int = 0,
) -> dict:
    """
    Branch-train-eval-select for every player in the population.

    Players are processed in parallel (one worker per CPU core by default).
    Each worker is independent: loads the frozen pool from disk, runs
    branch/train/eval, and returns the winner state dict.

    Parameters
    ----------
    trainer       : PopulationTrainer — live nets updated in-place after workers finish.
    n_parallel    : worker count (0 = auto, uses all logical CPU cores).
    """
    t0 = time.time()

    # Extract parent states in the main process before spawning workers.
    parent_states = {
        spec.net_id: to_cpu_state_dict(trainer.nets[spec.net_id])
        for spec in trainer.config.nets
    }

    args_list = [
        (spec.net_id, parent_states[spec.net_id], spec,
         trainer.spec_by_id, frozen_pool_path,
         branching_config, trainer.config, cycle_num)
        for spec in trainer.config.nets
    ]

    n_workers = n_parallel if n_parallel > 0 else (os.cpu_count() or 4)
    n_workers = min(n_workers, len(args_list))

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            raw_results = pool.map(_branch_one_player_worker, args_list)
    else:
        raw_results = [_branch_one_player_worker(a) for a in args_list]

    # Load winner weights back into live trainer nets (main process only).
    player_results: list[dict] = []
    for net_id, winner_state, player_result in raw_results:
        trainer.nets[net_id].load_state_dict(
            {k: v.to(trainer.device) for k, v in winner_state.items()}
        )
        trainer.nets[net_id].eval()
        player_results.append(player_result)

    trainer.save_all(update, milestone=False)

    elapsed = time.time() - t0
    winner_wrs = [r["winner_win_rate"] for r in player_results]
    winner_branch_counts = [0] * branching_config.n_branches
    for r in player_results:
        winner_branch_counts[r["winner_branch_idx"]] += 1

    return {
        "kind": "branch_select_cycle",
        "cycle_num": cycle_num,
        "update": update,
        "n_players": len(player_results),
        "n_branches": branching_config.n_branches,
        "mini_updates_per_branch": branching_config.mini_updates,
        "eval_episodes_per_branch": branching_config.eval_episodes,
        "n_parallel_workers": n_workers,
        "avg_winner_win_rate": round(float(np.mean(winner_wrs)), 4),
        "max_winner_win_rate": round(float(max(winner_wrs)), 4),
        "min_winner_win_rate": round(float(min(winner_wrs)), 4),
        "winner_branch_counts": winner_branch_counts,
        "elapsed_sec": round(elapsed, 1),
        "player_results": player_results,
    }
