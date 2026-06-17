"""Randomness for the SIMULATION layer: entropy by default, determinism on demand.

Owner rule (WORLD_MODE_SPEC §0-ter): real runs must be UNIQUE every time — no
"base seed + 1" derived streams, no repeated tournaments. So a run's public entry
points default to ``seed=None`` and draw ONE fresh seed from OS entropy (which is
RECORDED for history), then thread it down as INDEPENDENT child streams via
``SeedSequence.spawn`` — statistically independent, not ``base + offset``.

Scope is the SIM layer only. Net identity (``genotype_hash`` encodes the seed),
training-run reproducibility and the franchise-derived manager profile stay on
controlled, recorded seeds (see ``training/population.py`` / ``world/manager.py``)
and deliberately do NOT use this module.
"""
from __future__ import annotations

import random
import secrets
from typing import Optional

import numpy as np

SEED_BITS = 64
_MASK = (1 << SEED_BITS) - 1


def fresh_seed() -> int:
    """A genuinely random 64-bit seed from OS entropy (unique per call)."""
    return secrets.randbits(SEED_BITS)


def resolve_seed(seed: Optional[int]) -> int:
    """``seed`` when given (determinism — tests/registry), else a fresh entropy
    seed. This is the one place the entropy-vs-determinism choice is made."""
    return (int(seed) & _MASK) if seed is not None else fresh_seed()


def make_rng(seed: Optional[int]) -> random.Random:
    """A stdlib ``random.Random`` seeded explicitly, or from entropy when None."""
    return random.Random(resolve_seed(seed))


def np_rng(seed: Optional[int]) -> np.random.Generator:
    """A numpy ``Generator`` seeded explicitly, or from entropy when None."""
    return np.random.default_rng(resolve_seed(seed))


class Spawner:
    """Yields INDEPENDENT child seeds from one parent seed — the correct
    replacement for the ``base + index`` pattern. Deterministic given an explicit
    ``seed`` (tests), unique every run when ``seed is None``.

    Each ``seed()`` call spawns a fresh, statistically-independent child; nearby
    children are NOT correlated the way ``base + 1`` would be.
    """

    def __init__(self, seed: Optional[int]):
        self.root_seed = resolve_seed(seed)
        self._seq = np.random.SeedSequence(self.root_seed)

    def seed(self) -> int:
        (child,) = self._seq.spawn(1)
        return int(child.generate_state(1, dtype=np.uint64)[0])

    def seeds(self, n: int) -> list[int]:
        return [int(c.generate_state(1, dtype=np.uint64)[0]) for c in self._seq.spawn(n)]

    def random(self) -> random.Random:
        return random.Random(self.seed())

    def np(self) -> np.random.Generator:
        return np.random.default_rng(self.seed())
