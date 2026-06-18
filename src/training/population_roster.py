"""Population roster generator — grows the league population from the 18 ancestors.

This module is ADDITIVE: it never touches the canonical 18 base nets
(`build_default_population` in population.py). The 18 are permanent FREE players
(see role model below); the league population is built as faction-locked
DESCENDANTS forked from them, diversified across four axes:

  1. recipe / hyperparameters  — lr, entropy_coef, clip_coef, gae_lambda
  2. reward-shaping temperament — order_shaping_weight, core_progress_weight
                                   (training PRESSURE, not a baked obedience number;
                                    obedience still EMERGES — CLAUDE.md uniqueness rule)
  3. opponent curriculum        — which factions a net trains against more
  4. strategy niche             — which STRATEGY_PRESET the net is biased toward

A fifth axis, ARCHITECTURE (hidden_size / depth), is supported but OFF by default
(`arch_variants=False`): it needs the frozen-pool loader to accept heterogeneous
hidden_size first (population.py:215). Until enabled, every descendant is 128-wide.

Role model (owner spec, 2026-06-18):
  - The 18 ancestors are ALWAYS free agents: they do NOT play the league and cannot
    be signed; they play SoloQ only. They are the fork-source ("streamers") + elite
    reference players ("esports pros"). New players are trained ONLY from them
    between seasons. (Factions separately fork their own newbies — a distinct
    mechanism, not modelled here.)
  - The descendants generated here are the league population: signable, play the
    league, also in SoloQ. generation=1, parent = one ancestor (faction inherited).

Each descendant = parent-line DNA (inherited recipe) MODULATED by an archetype
flavor, so uniqueness is real (not seed-only): humans_1's rusher and humans_2's
rusher differ in their inherited base hyperparameters, not just the seed.

Dry-run only — this builds specs + registry, it does NOT train or write checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from src.training.population import (
    ARCHITECTURE, NetSpec, build_default_population,
)
from src.training.spaces import (
    ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION, STRATEGY_PRESET_BY_ID,
)

FACTIONS = ("humans", "infected", "robots")

# Parameter count is architecture-dependent; the 128-wide MaskedActorCritic.v1 is
# 740160 params (matches registry.json). Wider variants get recomputed when the
# architecture axis is switched on.
_PARAM_COUNT_BY_HIDDEN = {128: 740160}


@dataclass(frozen=True)
class Archetype:
    """A coherent player flavor applied on top of a parent line. Multipliers act on
    the parent's inherited recipe; the rest are absolute axis values."""

    name: str
    strategy_bias: str          # one of STRATEGY_PRESETS ids (the niche axis)
    entropy_mult: float         # x parent entropy_coef (exploration temperament)
    lr_mult: float              # x parent learning_rate
    clip_delta: float           # + parent clip_coef
    order_shaping_weight: float # reward pressure toward order-following (emergent)
    core_progress_weight: float # reward pressure toward pushing the objective
    opponent_curriculum: tuple[tuple[str, float], ...]  # faction -> train weight


def _curriculum(humans: float, infected: float, robots: float) -> tuple:
    s = humans + infected + robots
    return (("humans", round(humans / s, 3)),
            ("infected", round(infected / s, 3)),
            ("robots", round(robots / s, 3)))


_EVEN = _curriculum(1, 1, 1)

# Five coherent archetypes -> 5 descendants per ancestor line -> 5*6*3 = 90.
# Each is a real "player type", not random knob noise. Scaling to ~400 later = add
# archetypes / curriculum variants here (NOT more seeds — seed alone is not unique).
DEFAULT_ARCHETYPES: tuple[Archetype, ...] = (
    Archetype("skirmisher", "core_rush", entropy_mult=1.30, lr_mult=1.00, clip_delta=0.00,
              order_shaping_weight=0.40, core_progress_weight=0.50, opponent_curriculum=_EVEN),
    Archetype("berserker", "all_in", entropy_mult=1.50, lr_mult=1.20, clip_delta=-0.03,
              order_shaping_weight=0.20, core_progress_weight=0.30,
              opponent_curriculum=_curriculum(1.4, 1.0, 1.0)),   # leans into aggro matchups
    Archetype("sentinel", "turtle", entropy_mult=0.70, lr_mult=0.80, clip_delta=0.00,
              order_shaping_weight=0.70, core_progress_weight=0.60, opponent_curriculum=_EVEN),
    Archetype("architect", "macro", entropy_mult=0.90, lr_mult=0.70, clip_delta=0.00,
              order_shaping_weight=0.60, core_progress_weight=0.80, opponent_curriculum=_EVEN),
    Archetype("warden", "baron", entropy_mult=1.10, lr_mult=1.00, clip_delta=0.00,
              order_shaping_weight=0.50, core_progress_weight=0.70,
              opponent_curriculum=_curriculum(1.0, 1.2, 1.0)),
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class DescendantSpec:
    """A league-population net forked from one ancestor. Superset of NetSpec so it can
    be handed to the trainer later; carries the four diversity axes + role flags."""

    net_id: str
    faction: str
    parent_id: str
    generation: int
    seed: int
    archetype: str
    # axis 1 — inherited+modulated recipe
    learning_rate: float
    entropy_coef: float
    clip_coef: float
    gae_lambda: float
    hidden_size: int
    # axis 2 — reward-shaping temperament (training pressure; obedience still emerges)
    order_shaping_weight: float
    core_progress_weight: float
    # axis 3 — opponent curriculum (faction -> train weight)
    opponent_curriculum: tuple[tuple[str, float], ...]
    # axis 4 — strategy niche
    strategy_bias: str
    # role flags (owner spec) — league players, signable, also SoloQ
    free_agent: bool = False
    plays_league: bool = True
    signable: bool = True
    in_soloq: bool = True
    fork_source: bool = False
    fork_origin: str = "ancestor"   # forked from an ancestor ("streamer")

    def genotype_signature(self) -> dict:
        return {
            "architecture": ARCHITECTURE,
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "faction": self.faction,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "hidden_size": self.hidden_size,
            "learning_rate": round(self.learning_rate, 8),
            "entropy_coef": round(self.entropy_coef, 8),
            "clip_coef": round(self.clip_coef, 8),
            "gae_lambda": round(self.gae_lambda, 8),
            "order_shaping_weight": self.order_shaping_weight,
            "core_progress_weight": self.core_progress_weight,
            "opponent_curriculum": list(self.opponent_curriculum),
            "strategy_bias": self.strategy_bias,
            "archetype": self.archetype,
            "seed": self.seed,
        }

    def genotype_hash(self) -> str:
        payload = json.dumps(self.genotype_signature(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def parameter_count(self) -> int:
        return _PARAM_COUNT_BY_HIDDEN.get(self.hidden_size, -1)


def _ancestor_recipe(a: NetSpec) -> dict:
    return {"learning_rate": a.learning_rate, "entropy_coef": a.entropy_coef,
            "clip_coef": a.clip_coef, "gae_lambda": a.gae_lambda}


def build_population(
    ancestors: Sequence[NetSpec] | None = None,
    archetypes: Sequence[Archetype] = DEFAULT_ARCHETYPES,
    arch_variants: bool = False,
) -> list[DescendantSpec]:
    """Generate the faction-locked league population (one batch of descendants per
    ancestor x archetype). With the default 18 ancestors x 5 archetypes -> 90 nets.

    arch_variants is the architecture axis (hidden_size diversity); kept OFF until the
    frozen-pool loader supports heterogeneous widths (population.py:215)."""
    if ancestors is None:
        ancestors = build_default_population()
    out: list[DescendantSpec] = []
    for a in ancestors:
        base = _ancestor_recipe(a)
        for ai, arc in enumerate(archetypes):
            assert arc.strategy_bias in STRATEGY_PRESET_BY_ID, f"unknown niche {arc.strategy_bias}"
            hidden = a.hidden_size  # arch axis OFF -> inherit 128; ON -> vary here
            seed = a.seed * 31 + (ai + 1)
            out.append(DescendantSpec(
                net_id=f"{a.net_id}__{arc.name}",
                faction=a.faction,
                parent_id=a.net_id,
                generation=1,
                seed=seed,
                archetype=arc.name,
                learning_rate=_clamp(base["learning_rate"] * arc.lr_mult, 5e-5, 1e-3),
                entropy_coef=_clamp(base["entropy_coef"] * arc.entropy_mult, 1e-3, 5e-2),
                clip_coef=_clamp(base["clip_coef"] + arc.clip_delta, 0.05, 0.40),
                gae_lambda=base["gae_lambda"],
                hidden_size=hidden,
                order_shaping_weight=arc.order_shaping_weight,
                core_progress_weight=arc.core_progress_weight,
                opponent_curriculum=arc.opponent_curriculum,
                strategy_bias=arc.strategy_bias,
            ))
    return out


def assert_population_unique(descendants: Sequence[DescendantSpec],
                             ancestors: Sequence[NetSpec] | None = None) -> None:
    """No duplicate ids, no duplicate genotype hashes, and no descendant collides with
    an ancestor's hash (the 18 lines must stay distinct from the population)."""
    ids = [d.net_id for d in descendants]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate descendant net_id")
    dhashes = [d.genotype_hash() for d in descendants]
    if len(set(dhashes)) != len(dhashes):
        dupes = [d.net_id for d, h in zip(descendants, dhashes)
                 if dhashes.count(h) > 1]
        raise ValueError(f"Duplicate descendant genotype_hash: {sorted(set(dupes))}")
    if ancestors is None:
        ancestors = build_default_population()
    ah = {a.genotype_hash() for a in ancestors}
    clash = [d.net_id for d, h in zip(descendants, dhashes) if h in ah]
    if clash:
        raise ValueError(f"Descendant collides with an ancestor genotype: {clash}")


def ancestor_registry_entry(a: NetSpec) -> dict:
    """Registry view of an ancestor: a permanent FREE player (SoloQ-only, fork-source)."""
    return {
        "net_id": a.net_id, "faction": a.faction, "generation": 0, "parent_ids": [],
        "genotype_hash": a.genotype_hash(), "hidden_size": a.hidden_size,
        "parameter_count": _PARAM_COUNT_BY_HIDDEN.get(a.hidden_size, -1),
        "recipe": a.recipe, "seed": a.seed,
        # ALL 18 ancestors are "streamers": they NEVER play the league and are never
        # signable. They only appear in SoloQ and are the fork-source for newbies.
        "role": "ancestor", "persona": "streamer", "free_agent": True,
        "plays_league": False, "signable": False, "in_soloq": True, "fork_source": True,
        "checkpoint": f"training_runs/population_v1/nets/{a.net_id}/latest.pt",
        "status": "free_player",
    }


def descendant_registry_entry(d: DescendantSpec) -> dict:
    return {
        "net_id": d.net_id, "faction": d.faction, "generation": d.generation,
        "parent_ids": [d.parent_id], "genotype_hash": d.genotype_hash(),
        "hidden_size": d.hidden_size, "parameter_count": d.parameter_count,
        "archetype": d.archetype, "strategy_bias": d.strategy_bias,
        "learning_rate": round(d.learning_rate, 8), "entropy_coef": round(d.entropy_coef, 8),
        "clip_coef": round(d.clip_coef, 8), "gae_lambda": d.gae_lambda,
        "order_shaping_weight": d.order_shaping_weight,
        "core_progress_weight": d.core_progress_weight,
        "opponent_curriculum": {k: v for k, v in d.opponent_curriculum},
        "seed": d.seed,
        # descendants are the "classic players / pros": they play the league + SoloQ.
        "role": "player", "persona": "pro", "free_agent": d.free_agent,
        "plays_league": d.plays_league, "signable": d.signable, "in_soloq": d.in_soloq,
        "fork_source": d.fork_source, "fork_origin": d.fork_origin,
        "checkpoint": None,            # not trained yet
        "status": "planned",
    }


def build_roster(ancestors: Sequence[NetSpec] | None = None,
                 archetypes: Sequence[Archetype] = DEFAULT_ARCHETYPES,
                 arch_variants: bool = False) -> dict:
    """The full roster document: ancestors (free) + descendants (league), validated."""
    if ancestors is None:
        ancestors = build_default_population()
    descendants = build_population(ancestors, archetypes, arch_variants)
    assert_population_unique(descendants, ancestors)
    return {
        "schema": "population_roster.v1",
        "architecture": ARCHITECTURE,
        "observation_version": OBSERVATION_VERSION,
        "axes": ["recipe", "reward_shaping", "opponent_curriculum", "strategy_niche"]
                + (["architecture"] if arch_variants else []),
        "archetypes": [a.name for a in archetypes],
        "counts": {"ancestors_free": len(ancestors), "population": len(descendants),
                   "total": len(ancestors) + len(descendants)},
        "ancestors": [ancestor_registry_entry(a) for a in ancestors],
        "population": [descendant_registry_entry(d) for d in descendants],
    }
