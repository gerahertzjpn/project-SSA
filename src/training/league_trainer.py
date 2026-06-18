"""League-population trainer (parallel) — trains the 90 descendants from the 18 ancestors
against a DIVERSE opponent pool, to break the rush monoculture.

Why diverse opponents: a controlled experiment (scripted) showed pure rush is NOT dominant
— a tech-while-pushing bot beats it 58:42. The base nets nonetheless converged to rush
because self-play is a rush echo-chamber (everyone rushes → no pressure to tech). So each
descendant trains against a mix of: frozen ANCESTORS (rushers — the "train from ancestors"
base) + scripted opponents (tech/raider/rush/turtle/eco). The tech bots (which beat rush)
create the pressure to learn the deeper strategy.

Reuses the proven base-18 PPO machinery unchanged (population._merge_net_rollout,
_PPOUpdateView, _empty_buffer, write_frozen_pool; ppo_trainer.update_policy) and the same
multiprocessing worker-pool pattern. base-18 code is untouched.

Each episode: a same-archetype descendant triple (one per faction) on team A learns; team B
is either 3 frozen ancestors or 3 scripted bots. Diversity axes: per-net recipe (PPO), per-
archetype reward shaping (env), curriculum (ancestor sampling), strategy niche (recorded).

Fork-init = each descendant starts as a copy of its parent ancestor's latest weights, then
diverges under its own recipe + the diverse opponents.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.training.policy import MaskedActorCritic, to_cpu_state_dict
from src.training.population import (
    NetSpec, build_default_population, write_frozen_pool,
    _merge_net_rollout, _PPOUpdateView, _empty_buffer,
    CONTROLLED_SLOTS, OPPONENT_SLOTS,
)
from src.training.population_roster import DescendantSpec, build_population
from src.training.ppo_trainer import update_policy
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig

FACTIONS = ("humans", "infected", "robots")
ANCESTOR_RUN = "training_runs/population_v1"
# Non-ancestor scripted opponents. bmp/raider/eco PRESSURE tech-learning (they punish pure
# rush); core adds rush variety; turtle adds a passive holder. Weighted in _pick_scripted.
SCRIPTED_OPPONENTS = (
    "scripted_core_bmp", "scripted_core_bmp", "scripted_core_raider", "scripted_eco",
    "scripted_core", "scripted_turtle",
)


@dataclass
class LeagueConfig:
    run_dir: str
    ancestor_run_dir: str = ANCESTOR_RUN
    master_seed: int = 20260618
    target_steps_per_net: int = 1_400_000
    episodes_per_triple: int = 2
    workers: int = 24
    torch_threads: int = 1
    max_ticks: int = 300
    gamma: float = 0.99
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    sample_orders: bool = True
    no_order_prob: float = 0.2
    opponent_deterministic: bool = False
    scripted_opponent_prob: float = 0.45    # fraction of episodes vs scripted (else ancestors)
    device: str = "auto"
    checkpoint_interval: int = 25
    max_triples_per_update: int = 0         # 0 = all 30; >0 caps for smoke


# --------------------------------------------------------------------------- #
# Worker (module-level, picklable for spawn)
# --------------------------------------------------------------------------- #
def _buffers_to_agents(buffers: dict) -> dict:
    agents = {}
    for net_id, buf in buffers.items():
        if not buf["dones"]:
            continue
        agents[net_id] = {
            "observations": np.asarray(buf["observations"], dtype=np.float32),
            "masks": np.asarray(buf["masks"], dtype=np.bool_),
            "actions": np.asarray(buf["actions"], dtype=np.int64),
            "logprobs": np.asarray(buf["logprobs"], dtype=np.float32),
            "values": np.asarray(buf["values"], dtype=np.float32),
            "rewards": np.asarray(buf["rewards"], dtype=np.float32),
            "dones": np.asarray(buf["dones"], dtype=np.bool_),
            "entropies": np.asarray(buf["entropies"], dtype=np.float32),
        }
    return agents


def _run_episode_vs(plan, nets, faction_by_id, pool_path, common, buffers):
    """One episode: controlled descendant triple vs either frozen ancestors or a scripted
    bot. Collects the controlled nets' trajectories. Mirrors population._run_episode_collect
    but with a configurable opponent."""
    ctrl = plan["controlled"]
    kind = plan["opponent_kind"]
    slot_to_net = {slot: nid for slot, nid in zip(CONTROLLED_SLOTS, ctrl)}
    controlled_factions = {slot: faction_by_id[nid] for slot, nid in zip(CONTROLLED_SLOTS, ctrl)}
    player_factions = {**controlled_factions, **plan["opponent_factions"]}

    cfg_kwargs = dict(
        controlled_players=CONTROLLED_SLOTS, seed=plan["seed"],
        player_factions=player_factions, max_ticks=int(common["max_ticks"]),
        record_replay=False, record_trace=False,
        core_progress_weight=float(plan["core_progress_weight"]),
        sample_orders=bool(common["sample_orders"]),
        order_shaping_weight=float(plan["order_shaping_weight"]),
        no_order_prob=float(common["no_order_prob"]),
    )
    if kind == "ancestors":
        cfg_kwargs.update(
            opponent_bot="policy",
            opponent_policy_checkpoint=str(pool_path),
            opponent_policy_deterministic=bool(common["opponent_deterministic"]),
            opponent_policy_state_ids={slot: oid for slot, oid in
                                       zip(OPPONENT_SLOTS, plan["opponent_ids"])},
        )
    else:
        cfg_kwargs.update(opponent_bot=kind)  # a scripted variant id

    env = TeamTrainingEnv(TeamTrainingEnvConfig(**cfg_kwargs))
    obs, _ = env.reset(seed=plan["seed"], player_factions=player_factions)
    done = False
    result = None
    steps = 0
    while not done:
        masks = env.action_masks()
        step_outputs = {}
        actions = {}
        for slot in CONTROLLED_SLOTS:
            step = nets[slot_to_net[slot]].act(obs[slot], masks[slot], deterministic=False)
            actions[slot] = step.action
            step_outputs[slot] = step
        result = env.step(actions)
        done = bool(result.terminal or result.truncated)
        for slot in CONTROLLED_SLOTS:
            nid = slot_to_net[slot]
            step = step_outputs[slot]
            buf = buffers[nid]
            buf["observations"].append(obs[slot])
            buf["masks"].append(masks[slot])
            buf["actions"].append(step.action)
            buf["logprobs"].append(step.logprob)
            buf["values"].append(step.value)
            buf["rewards"].append(float(result.rewards[slot]))
            buf["dones"].append(done)
            buf["entropies"].append(step.entropy)
        obs = result.observations
        steps += 1
    info = result.info if result is not None else {}
    winner = info.get("winner") or "DRAW"
    return {"win": 1 if winner == env.controlled_team else 0, "steps": steps,
            "kind": kind, "controlled": list(ctrl)}


def _league_worker(request: dict) -> dict:
    torch.set_num_threads(int(request["torch_threads"]))
    np.random.seed(int(request["worker_seed"]) % (2**32 - 1))
    torch.manual_seed(int(request["worker_seed"]))
    hidden_by_id = request["hidden_by_id"]
    faction_by_id = request["faction_by_id"]
    nets = {}
    for nid, state in request["live_states"].items():
        net = MaskedActorCritic(hidden_size=int(hidden_by_id[nid]))
        net.load_state_dict(state)
        net.eval()
        nets[nid] = net
    buffers = {nid: _empty_buffer() for nid in request["live_states"]}
    episodes = []
    for plan in request["plans"]:
        episodes.append(_run_episode_vs(plan, nets, faction_by_id,
                                        request["frozen_pool_path"], request["config_common"], buffers))
    return {"agents": _buffers_to_agents(buffers), "episodes": episodes}


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class LeagueTrainer:
    def __init__(self, config: LeagueConfig,
                 descendants: Optional[list] = None, ancestors: Optional[list] = None):
        self.config = config
        torch.set_num_threads(int(config.torch_threads))
        self.device = torch.device(
            "cuda" if (config.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
        self.ancestors = ancestors or build_default_population()
        self.descendants = descendants or build_population(self.ancestors)
        self.desc_by_id = {d.net_id: d for d in self.descendants}
        self.anc_by_id = {a.net_id: a for a in self.ancestors}
        self.faction_by_id = {**{d.net_id: d.faction for d in self.descendants},
                              **{a.net_id: a.faction for a in self.ancestors}}
        self.hidden_by_id = {d.net_id: d.hidden_size for d in self.descendants}
        self.run_dir = Path(config.run_dir)
        (self.run_dir / "nets").mkdir(parents=True, exist_ok=True)
        self.total_steps = 0
        self.steps_by_net = {d.net_id: 0 for d in self.descendants}
        self._fork_init()
        self._build_ancestor_pool()

    # --- setup --- #
    def _ancestor_state(self, anc_id: str) -> dict:
        path = Path(self.config.ancestor_run_dir) / "nets" / anc_id / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"ancestor checkpoint missing: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)["policy_state"]

    def _fork_init(self) -> None:
        """Resume from existing descendant checkpoints if present (restart-safe), else
        fork (warm start) from the parent ancestor. Optimizer (Adam) state is not
        persisted — it re-warms over a few updates on resume; negligible for a long run."""
        self.nets: dict[str, MaskedActorCritic] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        resumed = 0
        for d in self.descendants:
            net = MaskedActorCritic(hidden_size=d.hidden_size).to(self.device)
            ckpt = self.run_dir / "nets" / d.net_id / "latest.pt"
            if ckpt.exists():
                blob = torch.load(ckpt, map_location=self.device, weights_only=False)
                net.load_state_dict(blob["policy_state"])
                self.steps_by_net[d.net_id] = int(blob.get("steps_trained", 0))
                resumed += 1
            else:
                net.load_state_dict(self._ancestor_state(d.parent_id))
            self.nets[d.net_id] = net
            self.optimizers[d.net_id] = torch.optim.Adam(net.parameters(), lr=d.learning_rate)
        self.total_steps = sum(self.steps_by_net.values())
        if resumed:
            print(f"resumed {resumed}/{len(self.descendants)} descendants "
                  f"(min steps/net {min(self.steps_by_net.values())})")

    def _build_ancestor_pool(self) -> None:
        states = {a.net_id: self._ancestor_state(a.net_id) for a in self.ancestors}
        self.pool_path = self.run_dir / "ancestor_pool.pt"
        write_frozen_pool(self.pool_path, self.anc_by_id, states, update=0)

    # --- scheduling --- #
    def _triples(self):
        archetypes = sorted({d.archetype for d in self.descendants})
        lines = sorted({int(d.parent_id.split("_")[1]) for d in self.descendants})
        triples = []
        for arc in archetypes:
            for ln in lines:
                ids = tuple(f"{fac}_{ln}__{arc}" for fac in FACTIONS)
                if all(i in self.desc_by_id for i in ids):
                    triples.append(ids)
        return triples

    def _pick_ancestors(self, curriculum, rng):
        weights = dict(curriculum)
        facs = list(weights)
        probs = np.array([weights[f] for f in facs], dtype=float)
        probs = probs / probs.sum()
        chosen = []
        for _ in range(3):
            fac = facs[int(rng.choice(len(facs), p=probs))]
            cands = [a for a in self.anc_by_id if self.faction_by_id[a] == fac and a not in chosen]
            if not cands:
                cands = [a for a in self.anc_by_id if a not in chosen]
            chosen.append(str(rng.choice(cands)))
        return chosen

    def _build_plans(self, update: int) -> list[dict]:
        triples = self._triples()
        if self.config.max_triples_per_update > 0:
            triples = triples[:self.config.max_triples_per_update]
        plans = []
        for ti, triple in enumerate(triples):
            lead = self.desc_by_id[triple[0]]
            for e in range(self.config.episodes_per_triple):
                rng = np.random.default_rng(self.config.master_seed + update * 9176 + ti * 131 + e * 17)
                seed = self.config.master_seed + update * 1_000_003 + ti * 101 + e * 7
                if float(rng.random()) < self.config.scripted_opponent_prob:
                    kind = SCRIPTED_OPPONENTS[int(rng.integers(len(SCRIPTED_OPPONENTS)))]
                    opp_facs = [FACTIONS[int(rng.integers(3))] for _ in range(3)]
                    plans.append({
                        "seed": seed, "controlled": list(triple), "opponent_kind": kind,
                        "opponent_ids": None,
                        "opponent_factions": {s: f for s, f in zip(OPPONENT_SLOTS, opp_facs)},
                        "order_shaping_weight": lead.order_shaping_weight,
                        "core_progress_weight": lead.core_progress_weight,
                    })
                else:
                    opp_ids = self._pick_ancestors(lead.opponent_curriculum, rng)
                    plans.append({
                        "seed": seed, "controlled": list(triple), "opponent_kind": "ancestors",
                        "opponent_ids": opp_ids,
                        "opponent_factions": {s: self.faction_by_id[o]
                                              for s, o in zip(OPPONENT_SLOTS, opp_ids)},
                        "order_shaping_weight": lead.order_shaping_weight,
                        "core_progress_weight": lead.core_progress_weight,
                    })
        return plans

    # --- one update --- #
    def run_update(self, update: int, pool) -> dict:
        plans = self._build_plans(update)
        live_states = {nid: to_cpu_state_dict(net) for nid, net in self.nets.items()}
        common = {"max_ticks": self.config.max_ticks,
                  "opponent_deterministic": self.config.opponent_deterministic,
                  "sample_orders": self.config.sample_orders,
                  "no_order_prob": self.config.no_order_prob}
        wc = max(1, self.config.workers)
        chunks = [plans[i::wc] for i in range(wc)]
        chunks = [c for c in chunks if c]
        requests = []
        for wi, chunk in enumerate(chunks):
            needed = sorted({nid for p in chunk for nid in p["controlled"]})
            requests.append({
                "torch_threads": self.config.torch_threads,
                "worker_seed": self.config.master_seed + update * 7919 + wi * 17,
                "hidden_by_id": self.hidden_by_id, "faction_by_id": self.faction_by_id,
                "frozen_pool_path": str(self.pool_path), "config_common": common,
                "live_states": {nid: live_states[nid] for nid in needed}, "plans": chunk,
            })
        if pool is None or len(requests) == 1:
            results = [_league_worker(r) for r in requests]
        else:
            results = pool.map(_league_worker, requests)

        steps = 0
        losses = {}
        for nid, net in self.nets.items():
            d = self.desc_by_id[nid]
            merged = _merge_net_rollout(results, nid, self.config.gamma, d.gae_lambda)
            if merged is None:
                continue
            n = int(merged["observations"].shape[0])
            steps += n
            self.steps_by_net[nid] += n
            view = _PPOUpdateView(ppo_epochs=self.config.ppo_epochs,
                                  minibatch_size=self.config.minibatch_size,
                                  clip_coef=d.clip_coef, value_coef=self.config.value_coef,
                                  entropy_coef=d.entropy_coef, max_grad_norm=self.config.max_grad_norm)
            losses[nid] = update_policy(net, self.optimizers[nid], merged, view, self.device)
        self.total_steps += steps

        eps = [e for r in results for e in r["episodes"]]
        anc_eps = [e for e in eps if e["kind"] == "ancestors"]
        scr_eps = [e for e in eps if e["kind"] != "ancestors"]
        def _wr(es):
            return round(sum(e["win"] for e in es) / max(1, len(es)), 4)
        return {"kind": "league_update", "update": update, "total_steps": self.total_steps,
                "min_steps_per_net": min(self.steps_by_net.values()),
                "episodes": len(eps), "win_vs_ancestors": _wr(anc_eps),
                "win_vs_scripted": _wr(scr_eps), "scripted_share": round(len(scr_eps) / max(1, len(eps)), 3),
                "mean_episode_steps": round(sum(e["steps"] for e in eps) / max(1, len(eps)), 2),
                "nets_updated": len(losses),
                "sample_loss": next(iter(losses.values()), None)}

    # --- persistence --- #
    def save_checkpoints(self, update: int) -> None:
        for nid, net in self.nets.items():
            d = self.desc_by_id[nid]
            net_dir = self.run_dir / "nets" / nid
            net_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"policy_state": to_cpu_state_dict(net), "update": update,
                        "net_id": nid, "genotype_hash": d.genotype_hash(),
                        "steps_trained": self.steps_by_net[nid]}, net_dir / "latest.pt")

    def write_registry(self, update: int) -> None:
        entries = [{
            "net_id": d.net_id, "faction": d.faction, "generation": d.generation,
            "parent_ids": [d.parent_id], "genotype_hash": d.genotype_hash(),
            "archetype": d.archetype, "strategy_bias": d.strategy_bias,
            "hidden_size": d.hidden_size, "parameter_count": d.parameter_count,
            "checkpoint": f"{self.config.run_dir}/nets/{d.net_id}/latest.pt",
            "status": "training", "updates_trained": update,
            "steps_trained": self.steps_by_net[d.net_id],
            "role": "player", "plays_league": True, "signable": True,
        } for d in self.descendants]
        (self.run_dir / "registry.json").write_text(
            json.dumps({"schema": "league_registry.v1", "updates_trained": update,
                        "total_steps": self.total_steps, "entries": entries},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    def train(self, max_updates: Optional[int] = None) -> None:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=max(1, self.config.workers),
                        initializer=_init_worker, initargs=(self.config.torch_threads,))
        metrics_path = self.run_dir / "metrics.jsonl"
        try:
            with open(metrics_path, "a", encoding="utf-8") as fh:
                u = 0
                while min(self.steps_by_net.values()) < self.config.target_steps_per_net:
                    if max_updates is not None and u >= max_updates:
                        break
                    u += 1
                    t = time.perf_counter()
                    rec = self.run_update(u, pool)
                    rec["elapsed_sec"] = round(time.perf_counter() - t, 3)
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    if u % self.config.checkpoint_interval == 0:
                        self.save_checkpoints(u); self.write_registry(u)
                self.save_checkpoints(u); self.write_registry(u)
        finally:
            pool.close(); pool.join()


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))
