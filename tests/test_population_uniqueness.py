"""Uniqueness + role-model invariants for the league population roster.

Guards CLAUDE.md rules: every population net is a genuinely distinct genotype (not a
seed-only clone), distinct from the 18 ancestors, and the role flags match the owner
spec (ancestors = free SoloQ-only fork-sources; descendants = signable league players).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.population import build_default_population
from src.training.population_roster import (
    DEFAULT_ARCHETYPES, assert_population_unique, build_population, build_roster,
)


def test_population_size_and_factions():
    pop = build_population()
    assert len(pop) == 90
    by_fac = {f: sum(1 for d in pop if d.faction == f) for f in ("humans", "infected", "robots")}
    assert by_fac == {"humans": 30, "infected": 30, "robots": 30}


def test_all_genotypes_unique_and_distinct_from_ancestors():
    ancestors = build_default_population()
    pop = build_population(ancestors)
    assert_population_unique(pop, ancestors)  # raises on any collision
    hashes = [d.genotype_hash() for d in pop]
    assert len(set(hashes)) == len(hashes)
    assert set(hashes).isdisjoint({a.genotype_hash() for a in ancestors})


def test_uniqueness_is_not_seed_only():
    """Two same-archetype descendants of different ancestor lines must differ in more
    than their seed (inherited recipe DNA differs) — CLAUDE.md: seed alone != unique."""
    pop = build_population()
    skirms = [d for d in pop if d.archetype == "skirmisher" and d.faction == "humans"]
    assert len(skirms) >= 2
    a, b = skirms[0], skirms[1]
    sig_a = {k: v for k, v in a.genotype_signature().items() if k != "seed"}
    sig_b = {k: v for k, v in b.genotype_signature().items() if k != "seed"}
    assert sig_a != sig_b, "descendants differ only by seed — not genuinely unique"


def test_faction_locked_to_parent():
    pop = build_population()
    anc = {a.net_id: a.faction for a in build_default_population()}
    for d in pop:
        assert d.faction == anc[d.parent_id], f"{d.net_id} broke faction lock"


def test_descendants_bias_only_real_strategy_presets():
    from src.training.spaces import STRATEGY_PRESET_BY_ID
    for d in build_population():
        assert d.strategy_bias in STRATEGY_PRESET_BY_ID


def test_role_flags_match_owner_spec():
    roster = build_roster()
    for a in roster["ancestors"]:
        assert a["free_agent"] and a["in_soloq"] and a["fork_source"]
        assert not a["plays_league"] and not a["signable"]
    for p in roster["population"]:
        assert p["plays_league"] and p["signable"] and p["in_soloq"]
        assert not p["free_agent"] and not p["fork_source"]
        assert p["generation"] == 1 and len(p["parent_ids"]) == 1


def test_archetype_count():
    assert len(DEFAULT_ARCHETYPES) == 5
