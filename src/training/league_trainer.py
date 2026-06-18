"""League-population trainer — trains the 90 descendants from the 18 ancestors.

Realizes the owner role model (see project_population_roster memory / build_population):
the descendants are forked from the matured ancestors and trained AGAINST them — the
opponents every episode are frozen ancestors (the "train only from ancestors" rule).
The 18 ancestors are NOT modified here; this trainer only writes descendant nets.

Reuses the PROVEN base-18 rollout/PPO machinery unchanged (population._run_episode_collect,
_merge_net_rollout, _PPOUpdateView, ppo_trainer.update_policy) — so base-18 behaviour is
untouched. Each league episode places a same-archetype descendant triple (one per faction)
on team A and three frozen ancestors on team B; only the descendants learn.

Diversity axes enforced here:
  - axis 1 (recipe):      per-descendant lr/entropy/clip/gae in the PPO update.
  - axis 2 (temperament): per-archetype order_shaping_weight + core_progress_weight in env.
  - axis 3 (curriculum):  opponent ancestors sampled by the archetype's faction weights.
  - axis 4 (niche):       recorded in genotype; in-env order-biasing toward strategy_bias
                          is a FOLLOW-UP (env order-sampling is currently unbiased). [TODO]
Fork-init = each descendant starts as a copy of its parent ancestor's latest weights,
then diverges under its own recipe/curriculum (warm start + emergent uniqueness).

NOTE: for a REAL run the ancestors must be frozen (base-18 plateaued + stopped) so the
opponent pool is stable. The --smoke path tolerates still-training ancestors (it only
checks the pipeline spine: fork-init -> rollout vs ancestors -> PPO -> checkpoint I/O).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.training.policy import MaskedActorCritic, to_cpu_state_dict
from src.training.population import (
    NetSpec, EpisodePlan, build_default_population, write_frozen_pool,
    _run_episode_collect, _merge_net_rollout, _PPOUpdateView, _empty_buffer,
)
from src.training.population_roster import DescendantSpec, build_population
from src.training.ppo_trainer import update_policy

FACTIONS = ("humans", "infected", "robots")
ANCESTOR_RUN = "training_runs/population_v1"


@dataclass(frozen=True)
class _FactionSpec:
    """Minimal stand-in: _run_episode_collect only reads spec[id].faction."""
    faction: str


@dataclass
class LeagueConfig:
    run_dir: str
    ancestor_run_dir: str = ANCESTOR_RUN
    master_seed: int = 20260618
    updates: int = 100
    episodes_per_triple: int = 2
    max_ticks: int = 300
    gamma: float = 0.99
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    sample_orders: bool = True
    no_order_prob: float = 0.2
    opponent_deterministic: bool = False
    device: str = "cpu"
    checkpoint_interval: int = 10
    max_triples_per_update: int = 0   # 0 = all 30; >0 caps for smoke/debug
    torch_threads: int = 1


def _buffers_to_agents(buffers: dict[str, dict[str, list]]) -> dict:
    """Same array packing _rollout_worker does, so _merge_net_rollout accepts it."""
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


class LeagueTrainer:
    def __init__(self, config: LeagueConfig,
                 descendants: list[DescendantSpec] | None = None,
                 ancestors: list[NetSpec] | None = None):
        self.config = config
        torch.set_num_threads(int(config.torch_threads))
        self.device = torch.device("cuda" if (config.device == "auto" and torch.cuda.is_available())
                                   else "cpu" if config.device in ("cpu", "auto") else config.device)
        self.ancestors = ancestors or build_default_population()
        self.descendants = descendants or build_population(self.ancestors)
        self.desc_by_id = {d.net_id: d for d in self.descendants}
        self.anc_by_id = {a.net_id: a for a in self.ancestors}
        self.anc_faction = {a.net_id: a.faction for a in self.ancestors}
        self.run_dir = Path(config.run_dir)
        (self.run_dir / "nets").mkdir(parents=True, exist_ok=True)

        self._fork_init()
        self._build_ancestor_pool()

    # --- setup ------------------------------------------------------------- #
    def _ancestor_state(self, anc_id: str) -> dict:
        path = Path(self.config.ancestor_run_dir) / "nets" / anc_id / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"ancestor checkpoint missing: {path} "
                                    f"(base-18 must have trained {anc_id})")
        return torch.load(path, map_location="cpu", weights_only=False)["policy_state"]

    def _fork_init(self) -> None:
        """Each descendant net = copy of its parent ancestor's latest weights."""
        self.nets: dict[str, MaskedActorCritic] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for d in self.descendants:
            parent_state = self._ancestor_state(d.parent_id)
            net = MaskedActorCritic(hidden_size=d.hidden_size).to(self.device)
            net.load_state_dict(parent_state)
            self.nets[d.net_id] = net
            self.optimizers[d.net_id] = torch.optim.Adam(net.parameters(), lr=d.learning_rate)

    def _build_ancestor_pool(self) -> None:
        states = {a.net_id: self._ancestor_state(a.net_id) for a in self.ancestors}
        self.pool_path = self.run_dir / "ancestor_pool.pt"
        write_frozen_pool(self.pool_path, self.anc_by_id, states, update=0)

    # --- scheduling -------------------------------------------------------- #
    def _triples(self) -> list[tuple[str, ...]]:
        """One triple per (archetype, line): same-archetype descendants, one per faction."""
        archetypes = sorted({d.archetype for d in self.descendants})
        lines = sorted({int(d.parent_id.split("_")[1]) for d in self.descendants})
        triples = []
        for arc in archetypes:
            for ln in lines:
                ids = tuple(f"{fac}_{ln}__{arc}" for fac in FACTIONS)
                if all(i in self.desc_by_id for i in ids):
                    triples.append(ids)
        return triples

    def _sample_opponents(self, curriculum: tuple, rng: np.random.Generator) -> tuple[str, str, str]:
        weights = dict(curriculum)
        facs = list(weights)
        probs = np.array([weights[f] for f in facs], dtype=float)
        probs = probs / probs.sum()
        chosen: list[str] = []
        for _ in range(3):
            fac = facs[int(rng.choice(len(facs), p=probs))]
            cands = [a for a in self.anc_by_id if self.anc_faction[a] == fac and a not in chosen]
            if not cands:                                  # fall back to any unused ancestor
                cands = [a for a in self.anc_by_id if a not in chosen]
            chosen.append(str(rng.choice(cands)))
        return tuple(chosen)  # type: ignore[return-value]

    # --- rollout + update -------------------------------------------------- #
    def _collect_triple(self, triple: tuple[str, ...], update: int, idx: int) -> dict:
        lead = self.desc_by_id[triple[0]]               # archetype shared across the triple
        opp_ids = self._sample_opponents(
            lead.opponent_curriculum,
            np.random.default_rng(self.config.master_seed + update * 9176 + idx * 31))
        buffers = {nid: _empty_buffer() for nid in triple}
        spec_by_id = {**{nid: _FactionSpec(self.desc_by_id[nid].faction) for nid in triple},
                      **{oid: _FactionSpec(self.anc_faction[oid]) for oid in opp_ids}}
        nets = {nid: self.nets[nid] for nid in triple}
        config_view = {
            "opponent_deterministic": self.config.opponent_deterministic,
            "max_ticks": self.config.max_ticks,
            "core_progress_weight": lead.core_progress_weight,   # axis 2 (per-archetype)
            "sample_orders": self.config.sample_orders,
            "order_shaping_weight": lead.order_shaping_weight,   # axis 2 (per-archetype)
            "no_order_prob": self.config.no_order_prob,
        }
        summaries = []
        for e in range(self.config.episodes_per_triple):
            plan = EpisodePlan(
                seed=self.config.master_seed + update * 1_000_003 + idx * 101 + e * 7,
                controlled=tuple(triple), opponents=tuple(opp_ids))
            summaries.append(_run_episode_collect(plan, nets, spec_by_id, str(self.pool_path),
                                                  config_view, buffers))
        return {"agents": _buffers_to_agents(buffers), "episodes": summaries, "opponents": opp_ids}

    def run_update(self, update: int) -> dict:
        triples = self._triples()
        if self.config.max_triples_per_update > 0:
            triples = triples[:self.config.max_triples_per_update]
        results = [self._collect_triple(t, update, i) for i, t in enumerate(triples)]

        wins = steps = eps = 0
        losses_by_net: dict[str, dict] = {}
        for r in results:
            for s in r["episodes"]:
                wins += s["win"]; steps += s["steps"]; eps += 1
        for nid, net in self.nets.items():
            d = self.desc_by_id[nid]
            view = _PPOUpdateView(ppo_epochs=self.config.ppo_epochs,
                                  minibatch_size=self.config.minibatch_size,
                                  clip_coef=d.clip_coef, value_coef=self.config.value_coef,
                                  entropy_coef=d.entropy_coef, max_grad_norm=self.config.max_grad_norm)
            merged = _merge_net_rollout(results, nid, self.config.gamma, d.gae_lambda)
            if merged is None:
                continue
            losses_by_net[nid] = update_policy(net, self.optimizers[nid], merged, view, self.device)

        return {"kind": "league_update", "update": update, "triples": len(triples),
                "episodes": eps, "win_rate": round(wins / max(1, eps), 4),
                "mean_episode_steps": round(steps / max(1, eps), 2),
                "nets_updated": len(losses_by_net),
                "sample_loss": next(iter(losses_by_net.values()), None)}

    # --- persistence ------------------------------------------------------- #
    def save_checkpoints(self, update: int) -> None:
        for nid, net in self.nets.items():
            d = self.desc_by_id[nid]
            net_dir = self.run_dir / "nets" / nid
            net_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"policy_state": to_cpu_state_dict(net), "update": update,
                        "net_id": nid, "genotype_hash": d.genotype_hash()},
                       net_dir / "latest.pt")

    def write_registry(self, update: int) -> None:
        entries = []
        for d in self.descendants:
            entries.append({
                "net_id": d.net_id, "faction": d.faction, "generation": d.generation,
                "parent_ids": [d.parent_id], "genotype_hash": d.genotype_hash(),
                "archetype": d.archetype, "strategy_bias": d.strategy_bias,
                "hidden_size": d.hidden_size, "parameter_count": d.parameter_count,
                "checkpoint": f"{self.config.run_dir}/nets/{d.net_id}/latest.pt",
                "status": "training", "updates_trained": update,
                "role": "player", "plays_league": True, "signable": True,
            })
        (self.run_dir / "registry.json").write_text(
            json.dumps({"schema": "league_registry.v1", "updates_trained": update,
                        "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")

    def train(self) -> None:
        metrics_path = self.run_dir / "metrics.jsonl"
        with open(metrics_path, "a", encoding="utf-8") as fh:
            for u in range(1, self.config.updates + 1):
                t = time.perf_counter()
                rec = self.run_update(u)
                rec["elapsed_sec"] = round(time.perf_counter() - t, 3)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                if u % self.config.checkpoint_interval == 0 or u == self.config.updates:
                    self.save_checkpoints(u); self.write_registry(u)
