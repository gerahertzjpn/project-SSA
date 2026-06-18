"""Wiring checks for the league trainer that need no ancestor checkpoints (fast).

Covers the axis plumbing (recipe inheritance, archetype scheduling) without building
the 53MB ancestor pool or running episodes — the full end-to-end spine is exercised by
`scripts/train_league_population.py --smoke`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.population import build_default_population, DEFAULT_RECIPES
from src.training.population_roster import (
    DEFAULT_ARCHETYPES, build_population,
)

FACTIONS = ("humans", "infected", "robots")


def test_triple_ids_all_exist_and_cover_population():
    """Every (archetype, line) triple references 3 real descendants, one per faction,
    and the 30 triples cover all 90 nets exactly once."""
    pop = build_population()
    ids = {d.net_id for d in pop}
    lines = sorted({int(d.parent_id.split("_")[1]) for d in pop})
    seen = set()
    triples = 0
    for arc in {d.archetype for d in pop}:
        for ln in lines:
            triple = [f"{fac}_{ln}__{arc}" for fac in FACTIONS]
            assert all(i in ids for i in triple), triple
            seen.update(triple)
            triples += 1
    assert triples == 30
    assert seen == ids and len(seen) == 90


def test_fork_lr_inherits_parent_times_archetype():
    """Axis-1: a descendant's learning_rate == parent ancestor lr * archetype mult
    (clamped). Proves the fork inherits parent DNA, modulated by the archetype."""
    ancestors = {a.net_id: a for a in build_default_population()}
    arc_by_name = {a.name: a for a in DEFAULT_ARCHETYPES}
    for d in build_population():
        parent = ancestors[d.parent_id]
        arc = arc_by_name[d.archetype]
        expected = max(5e-5, min(1e-3, parent.learning_rate * arc.lr_mult))
        assert abs(d.learning_rate - expected) < 1e-9, (d.net_id, d.learning_rate, expected)


def test_same_archetype_shares_temperament_axes():
    """Axis-2 is per-archetype: same-archetype descendants share order/core weights
    (so a same-archetype triple can use one env config_view)."""
    pop = build_population()
    by_arc: dict[str, list] = {}
    for d in pop:
        by_arc.setdefault(d.archetype, []).append(d)
    for arc, members in by_arc.items():
        ows = {m.order_shaping_weight for m in members}
        cws = {m.core_progress_weight for m in members}
        assert len(ows) == 1 and len(cws) == 1, arc


def test_recipe_base_is_a_real_ancestor_recipe():
    recipe_names = {r.name for r in DEFAULT_RECIPES}
    for a in build_default_population():
        assert a.recipe in recipe_names
