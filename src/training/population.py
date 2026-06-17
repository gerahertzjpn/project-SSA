"""Population self-play training for a free-agent draft pool.

This module trains a *population* of independent per-player policies. Each net is
its own draft candidate ("1 seed = 1 player"): faction-locked, uniquely seeded
and uniquely configured. There is **no selection/culling** here -- every net in
the population is trained and checkpointed every round.

Core mechanic: frozen-snapshot self-play with mixed-faction teams and randomized
positions.

Per training episode:
  * 3 live nets are sampled from the pool as the controlled team (slot A1/A2/A3),
    their slot assignment is shuffled every episode (position randomization);
  * 3 nets are sampled from a periodically-refreshed *frozen snapshot pool* as
    the opponent team (slot B1/B2/B3);
  * each player's faction follows the net into whatever slot it occupies, so a
    faction-locked net always plays its own faction;
  * the controlled nets learn from the episode via masked PPO; the frozen
    opponents do not learn.

The heavy RL machinery (`MaskedActorCritic`, GAE, `update_policy`) and the
faction-aware `TeamTrainingEnv` / `PolicyBot` paths are reused unchanged, so this
module only owns population bookkeeping, sampling and per-net experience routing.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.core.constants import FACTION_TYPES
from src.training.policy import MaskedActorCritic, to_cpu_state_dict
from src.training.ppo_trainer import _compute_gae, resolve_device, update_policy
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig


CONTROLLED_SLOTS: tuple[str, str, str] = ("A1", "A2", "A3")
OPPONENT_SLOTS: tuple[str, str, str] = ("B1", "B2", "B3")
NETS_PER_FACTION = 6
ARCHITECTURE = "MaskedActorCritic.v1"


# --------------------------------------------------------------------------- #
# Population specification (the 18 unique draft candidates)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recipe:
    """A training recipe. Distinct recipes make same-faction nets genuinely
    unique beyond their seed (CLAUDE.md uniqueness rule)."""

    name: str
    learning_rate: float
    entropy_coef: float
    clip_coef: float = 0.2
    gae_lambda: float = 0.95
    hidden_size: int = 128


# Six interface-compatible recipes (uniform hidden_size keeps the frozen-pool a
# single checkpoint). Reused across the three factions; combined with a unique
# seed and faction this yields 18 distinct genotype signatures.
DEFAULT_RECIPES: tuple[Recipe, ...] = (
    Recipe("baseline", learning_rate=3.0e-4, entropy_coef=0.010, clip_coef=0.20),
    Recipe("low_entropy", learning_rate=3.0e-4, entropy_coef=0.005, clip_coef=0.15),
    Recipe("high_entropy", learning_rate=3.0e-4, entropy_coef=0.020, clip_coef=0.25),
    Recipe("fast_lr", learning_rate=5.0e-4, entropy_coef=0.010, clip_coef=0.20),
    Recipe("slow_explore", learning_rate=1.5e-4, entropy_coef=0.015, clip_coef=0.20),
    Recipe("alt_mix", learning_rate=4.0e-4, entropy_coef=0.007, clip_coef=0.18),
)

FACTION_SEED_BASE = {
    "humans": 20260600,
    "infected": 20260700,
    "robots": 20260800,
}


@dataclass(frozen=True)
class NetSpec:
    net_id: str
    faction: str
    seed: int
    recipe: str
    learning_rate: float
    entropy_coef: float
    clip_coef: float
    gae_lambda: float
    hidden_size: int
    # NOTE: there is deliberately NO per-net "obedience" parameter. The base nets
    # carry NO baked order-following bias and NO special numbers. Order-following is
    # taught UNIFORMLY to all nets during training (a single global order-shaping
    # reward, see PopulationConfig.order_shaping_weight); whether a net ends up more
    # or less obedient EMERGES from training, it is never pre-painted. Uniqueness
    # comes from seed/faction/recipe (CLAUDE.md uniqueness rule).

    def genotype_signature(self) -> dict:
        """Normalized, order-stable description used for the uniqueness hash."""
        return {
            "architecture": ARCHITECTURE,
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "faction": self.faction,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "entropy_coef": self.entropy_coef,
            "clip_coef": self.clip_coef,
            "gae_lambda": self.gae_lambda,
            "recipe": self.recipe,
            "seed": self.seed,
        }

    def genotype_hash(self) -> str:
        payload = json.dumps(self.genotype_signature(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_default_population() -> list[NetSpec]:
    """The canonical 18-net population: 6 unique nets per faction. NO baked obedience
    — every base net is identical in coachability; order-following is learned during
    training, and any spread is emergent (never a special per-net number)."""
    nets: list[NetSpec] = []
    for faction in FACTION_TYPES:
        base = FACTION_SEED_BASE[faction]
        for index in range(NETS_PER_FACTION):
            recipe = DEFAULT_RECIPES[index % len(DEFAULT_RECIPES)]
            seed = base + index + 1
            nets.append(
                NetSpec(
                    net_id=f"{faction}_{index + 1}",
                    faction=faction,
                    seed=seed,
                    recipe=recipe.name,
                    learning_rate=recipe.learning_rate,
                    entropy_coef=recipe.entropy_coef,
                    clip_coef=recipe.clip_coef,
                    gae_lambda=recipe.gae_lambda,
                    hidden_size=recipe.hidden_size,
                )
            )
    return nets


def assert_population_unique(nets: list[NetSpec]) -> None:
    ids = [net.net_id for net in nets]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate net_id in population")
    hashes = [net.genotype_hash() for net in nets]
    if len(set(hashes)) != len(hashes):
        collisions = [net.net_id for net in nets]
        raise ValueError(f"Duplicate genotype_hash in population: {collisions}")


# --------------------------------------------------------------------------- #
# Net construction / checkpoint IO (single-policy format, PolicyBot-compatible)
# --------------------------------------------------------------------------- #
def make_net(spec: NetSpec) -> MaskedActorCritic:
    """Construct a freshly-initialized net with a per-net deterministic seed."""
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(spec.seed)
        np.random.seed(spec.seed % (2**32 - 1))
        net = MaskedActorCritic(hidden_size=spec.hidden_size)
    finally:
        torch.random.set_rng_state(generator_state)
    return net


def parameter_count(net: MaskedActorCritic) -> int:
    return int(sum(param.numel() for param in net.parameters()))


def net_checkpoint_payload(spec: NetSpec, net: MaskedActorCritic, update: int, total_steps: int) -> dict:
    return {
        "kind": "population_net",
        "policy_state": net.state_dict(),
        "config": {"hidden_size": spec.hidden_size},
        "net_spec": asdict(spec),
        "genotype_hash": spec.genotype_hash(),
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "update": int(update),
        "total_steps": int(total_steps),
        "parameter_count": parameter_count(net),
    }


def save_net_checkpoint(path: Path, spec: NetSpec, net: MaskedActorCritic, update: int, total_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net_checkpoint_payload(spec, net, update, total_steps), path)


def load_net_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError(f"Net checkpoint {path} has incompatible observation interface")
    return checkpoint["policy_state"]


def write_frozen_pool(path: Path, specs: dict[str, NetSpec], states: dict[str, dict], update: int) -> None:
    """Write a single checkpoint holding every net's frozen weights, keyed by
    net_id, in a layout `PolicyBot(policy_state_id=net_id)` can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden_sizes = {spec.hidden_size for spec in specs.values()}
    if len(hidden_sizes) != 1:
        raise ValueError(f"Frozen pool requires a uniform hidden_size, got {sorted(hidden_sizes)}")
    payload = {
        "kind": "population_frozen_pool",
        "policy_states": states,
        "config": {"hidden_size": next(iter(hidden_sizes))},
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "update": int(update),
    }
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
@dataclass
class PopulationConfig:
    run_dir: str
    nets: list[NetSpec]
    master_seed: int = 20260609
    updates: int = 200
    episodes_per_net_per_update: int = 2
    workers: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    gamma: float = 0.99
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    max_ticks: int = 300
    snapshot_interval: int = 5
    checkpoint_interval: int = 5
    torch_threads: int = 1
    device: str = "auto"
    opponent_deterministic: bool = False
    mixed_faction: bool = True
    core_progress_weight: float = 0.5
    # LLM-manager order channel. order_shaping_weight is the UNIFORM training signal
    # that teaches order-following to ALL nets equally (0 disables shaping). It is a
    # training hyperparameter, NOT a per-net obedience number — obedience is never
    # baked; any divergence emerges from training.
    sample_orders: bool = True
    order_shaping_weight: float = 0.5
    # Fraction of episodes with NO order (channel zeroed, shaping off) so nets also
    # learn free play when un-ordered ("no order = play as usual").
    no_order_prob: float = 0.2

    def spec_by_id(self) -> dict[str, NetSpec]:
        return {net.net_id: net for net in self.nets}


# --------------------------------------------------------------------------- #
# Episode scheduling
# --------------------------------------------------------------------------- #
@dataclass
class EpisodePlan:
    seed: int
    controlled: tuple[str, str, str]   # net_ids placed at A1, A2, A3 (already shuffled)
    opponents: tuple[str, str, str]    # net_ids placed at B1, B2, B3


def _balanced_controlled_triples(net_ids: list[str], episodes_per_net: int, rng: np.random.Generator) -> list[list[str]]:
    """Each net appears `episodes_per_net` times as a controlled learner, packed
    into triples (the controlled team for one episode)."""
    pool: list[str] = []
    for _ in range(episodes_per_net):
        order = list(net_ids)
        rng.shuffle(order)
        pool.extend(order)
    triples: list[list[str]] = []
    for start in range(0, len(pool) - 2, 3):
        triples.append(pool[start:start + 3])
    return triples


def build_episode_plans(config: PopulationConfig, update: int) -> list[EpisodePlan]:
    rng = np.random.default_rng(config.master_seed + update * 1_000_003)
    net_ids = [net.net_id for net in config.nets]
    triples = _balanced_controlled_triples(net_ids, config.episodes_per_net_per_update, rng)
    plans: list[EpisodePlan] = []
    for episode_index, controlled in enumerate(triples):
        controlled = list(controlled)
        rng.shuffle(controlled)  # position randomization across A1/A2/A3
        opponents = list(rng.choice(net_ids, size=3, replace=False))
        plans.append(
            EpisodePlan(
                seed=config.master_seed + update * 1_000_003 + episode_index * 101 + 7,
                controlled=tuple(controlled),
                opponents=tuple(opponents),
            )
        )
    return plans


# --------------------------------------------------------------------------- #
# Rollout worker
# --------------------------------------------------------------------------- #
def _empty_buffer() -> dict[str, list]:
    return {
        "observations": [],
        "masks": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "dones": [],
        "entropies": [],
    }


def _run_episode_collect(
    plan: EpisodePlan,
    nets: dict[str, MaskedActorCritic],
    spec_by_id: dict[str, NetSpec],
    frozen_pool_path: str,
    config_view: dict,
    buffers: dict[str, dict[str, list]],
) -> dict:
    spec = spec_by_id
    controlled_factions = {
        slot: spec[net_id].faction
        for slot, net_id in zip(CONTROLLED_SLOTS, plan.controlled)
    }
    opponent_factions = {
        slot: spec[net_id].faction
        for slot, net_id in zip(OPPONENT_SLOTS, plan.opponents)
    }
    player_factions = {**controlled_factions, **opponent_factions}
    opponent_state_ids = {
        slot: net_id for slot, net_id in zip(OPPONENT_SLOTS, plan.opponents)
    }
    slot_to_net = {slot: net_id for slot, net_id in zip(CONTROLLED_SLOTS, plan.controlled)}

    # UNIFORM order-shaping weight for every controlled net (no per-net obedience —
    # see NetSpec). All nets get the SAME signal to follow orders; any obedience
    # spread must emerge from training, never from a baked number.
    global_order_weight = float(config_view.get("order_shaping_weight", 0.5))

    env_config = TeamTrainingEnvConfig(
        controlled_players=CONTROLLED_SLOTS,
        seed=plan.seed,
        opponent_bot="policy",
        ally_bot="scripted",  # unused: all of team A is controlled
        opponent_policy_checkpoint=frozen_pool_path,
        opponent_policy_deterministic=bool(config_view["opponent_deterministic"]),
        opponent_policy_state_ids=opponent_state_ids,
        player_factions=player_factions,
        max_ticks=int(config_view["max_ticks"]),
        record_replay=False,
        record_trace=False,
        core_progress_weight=float(config_view.get("core_progress_weight", 0.5)),
        sample_orders=bool(config_view.get("sample_orders", True)),
        order_shaping_weight=global_order_weight,   # uniform for all nets (no baked obedience)
        no_order_prob=float(config_view.get("no_order_prob", 0.0)),
    )
    env = TeamTrainingEnv(env_config)
    observations, _ = env.reset(seed=plan.seed, player_factions=player_factions)
    controlled_team = env.controlled_team

    # Per-net step tracking so we can mark episode-end dones correctly.
    last_index: dict[str, int] = {}
    steps = 0
    done = False
    result = None
    while not done:
        masks = env.action_masks()
        actions = {}
        step_outputs = {}
        for slot in CONTROLLED_SLOTS:
            net_id = slot_to_net[slot]
            step = nets[net_id].act(observations[slot], masks[slot], deterministic=False)
            actions[slot] = step.action
            step_outputs[slot] = step
        result = env.step(actions)
        done = bool(result.terminal or result.truncated)
        for slot in CONTROLLED_SLOTS:
            net_id = slot_to_net[slot]
            step = step_outputs[slot]
            buf = buffers[net_id]
            buf["observations"].append(observations[slot])
            buf["masks"].append(masks[slot])
            buf["actions"].append(step.action)
            buf["logprobs"].append(step.logprob)
            buf["values"].append(step.value)
            buf["rewards"].append(float(result.rewards[slot]))
            buf["dones"].append(done)
            buf["entropies"].append(step.entropy)
            last_index[net_id] = len(buf["dones"]) - 1
        observations = result.observations
        steps += 1

    info = result.info if result is not None else {}
    winner = info.get("winner") or "DRAW"
    winner_type = info.get("winner_type") or "unknown"
    return {
        "steps": steps,
        "winner": winner,
        "winner_type": winner_type,
        "win": 1 if winner == controlled_team else 0,
        "loss": 1 if winner not in {controlled_team, "DRAW", None} else 0,
        "draw": 1 if winner in {"DRAW", None} else 0,
        "controlled": list(plan.controlled),
        "opponents": list(plan.opponents),
    }


def _rollout_worker(request: dict) -> dict:
    torch.set_num_threads(int(request["torch_threads"]))
    np.random.seed(int(request["worker_seed"]) % (2**32 - 1))
    torch.manual_seed(int(request["worker_seed"]))

    spec_by_id = {net_id: NetSpec(**spec) for net_id, spec in request["specs"].items()}
    config_view = request["config_view"]
    frozen_pool_path = request["frozen_pool_path"]

    nets: dict[str, MaskedActorCritic] = {}
    for net_id, state in request["live_states"].items():
        net = MaskedActorCritic(hidden_size=spec_by_id[net_id].hidden_size)
        net.load_state_dict(state)
        net.eval()
        nets[net_id] = net

    buffers: dict[str, dict[str, list]] = {net_id: _empty_buffer() for net_id in request["live_states"]}
    episode_summaries = []
    for plan_dict in request["plans"]:
        plan = EpisodePlan(
            seed=int(plan_dict["seed"]),
            controlled=tuple(plan_dict["controlled"]),
            opponents=tuple(plan_dict["opponents"]),
        )
        summary = _run_episode_collect(plan, nets, spec_by_id, frozen_pool_path, config_view, buffers)
        episode_summaries.append(summary)

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
    return {"agents": agents, "episodes": episode_summaries}


# --------------------------------------------------------------------------- #
# Per-net rollout assembly + PPO update
# --------------------------------------------------------------------------- #
def _merge_net_rollout(worker_results: list[dict], net_id: str, gamma: float, gae_lambda: float) -> Optional[dict]:
    segments = [result["agents"][net_id] for result in worker_results if net_id in result["agents"]]
    if not segments:
        return None
    advantages = []
    returns = []
    for segment in segments:
        # Every worker segment ends on a done step, so the bootstrap value is 0.
        seg_adv, seg_ret = _compute_gae(
            segment["rewards"],
            segment["dones"],
            segment["values"],
            last_value=0.0,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages.append(seg_adv)
        returns.append(seg_ret)
    return {
        "observations": np.concatenate([seg["observations"] for seg in segments]),
        "masks": np.concatenate([seg["masks"] for seg in segments]),
        "actions": np.concatenate([seg["actions"] for seg in segments]),
        "old_logprobs": np.concatenate([seg["logprobs"] for seg in segments]),
        "values": np.concatenate([seg["values"] for seg in segments]),
        "rewards": np.concatenate([seg["rewards"] for seg in segments]),
        "entropies": np.concatenate([seg["entropies"] for seg in segments]),
        "advantages": np.concatenate(advantages),
        "returns": np.concatenate(returns),
    }


@dataclass
class _PPOUpdateView:
    """Minimal config view consumed by `update_policy` for one net."""
    ppo_epochs: int
    minibatch_size: int
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class PopulationTrainer:
    def __init__(self, config: PopulationConfig):
        assert_population_unique(config.nets)
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.device = resolve_device(config.device)
        self.spec_by_id = config.spec_by_id()
        self.nets: dict[str, MaskedActorCritic] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for spec in config.nets:
            net = make_net(spec).to(self.device)
            self.nets[spec.net_id] = net
            self.optimizers[spec.net_id] = torch.optim.Adam(net.parameters(), lr=spec.learning_rate)
        self.start_update = 1
        self.total_steps = 0

    # -- persistence ------------------------------------------------------- #
    def net_dir(self, net_id: str) -> Path:
        return self.run_dir / "nets" / net_id

    def resume(self) -> None:
        """Load each net's latest checkpoint if present (no-op for a fresh run)."""
        resumed = 0
        max_update = 0
        for net_id, net in self.nets.items():
            latest = self.net_dir(net_id) / "latest.pt"
            if latest.exists():
                checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
                net.load_state_dict(checkpoint["policy_state"])
                max_update = max(max_update, int(checkpoint.get("update", 0)))
                self.total_steps = max(self.total_steps, int(checkpoint.get("total_steps", 0)))
                resumed += 1
        if resumed and resumed != len(self.nets):
            raise RuntimeError(
                f"Partial resume: {resumed}/{len(self.nets)} nets have checkpoints. Refusing to continue."
            )
        if resumed:
            self.start_update = max_update + 1

    def save_all(self, update: int, milestone: bool) -> None:
        for net_id, net in self.nets.items():
            spec = self.spec_by_id[net_id]
            net_dir = self.net_dir(net_id)
            save_net_checkpoint(net_dir / "latest.pt", spec, net.to("cpu"), update, self.total_steps)
            net.to(self.device)
            if milestone:
                save_net_checkpoint(net_dir / f"update_{update}.pt", spec, net.to("cpu"), update, self.total_steps)
                net.to(self.device)

    def frozen_pool_path(self, version: int) -> Path:
        return self.run_dir / "frozen_pool" / f"frozen_pool_v{version}.pt"

    def refresh_frozen_pool(self, version: int) -> Path:
        states = {net_id: to_cpu_state_dict(net) for net_id, net in self.nets.items()}
        path = self.frozen_pool_path(version)
        write_frozen_pool(path, self.spec_by_id, states, version)
        return path

    # -- one PPO update over the whole population -------------------------- #
    def run_update(self, update: int, frozen_pool_path: Path, pool: Optional[mp.pool.Pool]) -> dict:
        plans = build_episode_plans(self.config, update)
        live_states = {net_id: to_cpu_state_dict(net) for net_id, net in self.nets.items()}
        specs_serialized = {net_id: asdict(spec) for net_id, spec in self.spec_by_id.items()}
        config_view = {
            "opponent_deterministic": self.config.opponent_deterministic,
            "max_ticks": self.config.max_ticks,
            "core_progress_weight": self.config.core_progress_weight,
            "sample_orders": self.config.sample_orders,
            "order_shaping_weight": self.config.order_shaping_weight,
        }

        worker_count = max(1, self.config.workers)
        plan_dicts = [
            {"seed": plan.seed, "controlled": list(plan.controlled), "opponents": list(plan.opponents)}
            for plan in plans
        ]
        chunks = [plan_dicts[index::worker_count] for index in range(worker_count)]
        chunks = [chunk for chunk in chunks if chunk]

        requests = []
        for worker_index, chunk in enumerate(chunks):
            needed = sorted({net_id for plan in chunk for net_id in plan["controlled"]})
            requests.append({
                "torch_threads": self.config.torch_threads,
                "worker_seed": self.config.master_seed + update * 7919 + worker_index * 17,
                "specs": specs_serialized,
                "config_view": config_view,
                "frozen_pool_path": str(frozen_pool_path),
                "live_states": {net_id: live_states[net_id] for net_id in needed},
                "plans": chunk,
            })

        if pool is None or len(requests) == 1:
            worker_results = [_rollout_worker(request) for request in requests]
        else:
            worker_results = pool.map(_rollout_worker, requests)

        # PPO update per net.
        loss_by_net = {}
        steps_this_update = 0
        for net_id in self.nets:
            spec = self.spec_by_id[net_id]
            merged = _merge_net_rollout(worker_results, net_id, self.config.gamma, spec.gae_lambda)
            if merged is None:
                continue
            steps_this_update += int(merged["observations"].shape[0])
            view = _PPOUpdateView(
                ppo_epochs=self.config.ppo_epochs,
                minibatch_size=self.config.minibatch_size,
                clip_coef=spec.clip_coef,
                value_coef=self.config.value_coef,
                entropy_coef=spec.entropy_coef,
                max_grad_norm=self.config.max_grad_norm,
            )
            loss_by_net[net_id] = update_policy(
                self.nets[net_id], self.optimizers[net_id], merged, view, self.device
            )
        self.total_steps += steps_this_update

        episodes = [summary for result in worker_results for summary in result["episodes"]]
        return self._summarize_update(update, episodes, loss_by_net, steps_this_update)

    def _summarize_update(self, update: int, episodes: list[dict], loss_by_net: dict, steps_this_update: int) -> dict:
        n = max(1, len(episodes))
        coverage_allies: dict[str, set] = {net.net_id: set() for net in self.config.nets}
        coverage_opponents: dict[str, set] = {net.net_id: set() for net in self.config.nets}
        appearances: dict[str, int] = {net.net_id: 0 for net in self.config.nets}
        for ep in episodes:
            controlled = ep["controlled"]
            for net_id in controlled:
                appearances[net_id] += 1
                coverage_allies[net_id].update(other for other in controlled if other != net_id)
                coverage_opponents[net_id].update(ep["opponents"])
        return {
            "kind": "population_update",
            "update": update,
            "episodes": len(episodes),
            "total_steps": self.total_steps,
            "steps_this_update": steps_this_update,
            "controlled_win_rate": round(sum(ep["win"] for ep in episodes) / n, 4),
            "controlled_loss_rate": round(sum(ep["loss"] for ep in episodes) / n, 4),
            "controlled_draw_rate": round(sum(ep["draw"] for ep in episodes) / n, 4),
            "mean_episode_steps": round(sum(ep["steps"] for ep in episodes) / n, 2),
            "min_net_appearances": min(appearances.values()) if appearances else 0,
            "min_distinct_allies_seen": min((len(v) for v in coverage_allies.values()), default=0),
            "loss_by_net": loss_by_net,
        }


def count_total_parameters(trainer: PopulationTrainer) -> int:
    return int(sum(parameter_count(net) for net in trainer.nets.values()))
