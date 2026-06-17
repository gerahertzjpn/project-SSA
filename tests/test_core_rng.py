"""SIM-layer RNG helper: entropy by default, determinism on demand, independent
child streams (no base+offset)."""
from src.core import rng


def test_resolve_seed_explicit_vs_entropy():
    assert rng.resolve_seed(5) == 5
    a, b = rng.resolve_seed(None), rng.resolve_seed(None)
    assert isinstance(a, int) and isinstance(b, int)
    assert a != b                      # fresh entropy each call (overwhelmingly likely)


def test_fresh_seed_unique():
    seeds = {rng.fresh_seed() for _ in range(50)}
    assert len(seeds) == 50            # no collisions in 64-bit entropy


def test_make_rng_explicit_is_reproducible():
    r1 = rng.make_rng(7)
    r2 = rng.make_rng(7)
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]


def test_make_rng_none_differs():
    r1 = rng.make_rng(None)
    r2 = rng.make_rng(None)
    assert [r1.random() for _ in range(5)] != [r2.random() for _ in range(5)]


def test_spawner_deterministic_given_seed():
    s1 = rng.Spawner(11)
    s2 = rng.Spawner(11)
    assert s1.seeds(8) == s2.seeds(8)          # same parent -> same child stream


def test_spawner_children_are_distinct():
    children = rng.Spawner(11).seeds(16)
    assert len(set(children)) == 16            # independent, not base+offset


def test_spawner_none_is_unique_per_instance():
    assert rng.Spawner(None).seeds(4) != rng.Spawner(None).seeds(4)


def test_spawner_random_streams_independent():
    sp = rng.Spawner(3)
    a = sp.random()
    b = sp.random()
    # two children from the same parent are different streams
    assert [a.random() for _ in range(4)] != [b.random() for _ in range(4)]
