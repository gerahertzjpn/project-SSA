"""Training operations A–F: cards, request validation, gated execution, guards."""
import pytest

from src.world import training_ops as ops


class MockBackend(ops.TrainingBackend):
    """A fake training backend so the dispatch is testable without real training."""
    def continue_ppo(self, ctx): return {"trained": "ppo", "steps": ctx["total_steps"]}
    def cotrain(self, ctx): return {"trained": "cotrain", "steps": ctx["total_steps"]}
    def fork(self, ctx): return {"child_net": "fork-1"}

    def evaluate(self, ctx):
        if ctx["op"] == "create_player":
            return {"new_hash": "h-new", "pool_hashes": ["h1", "h2"], "behaviour_distance": 0.2}
        return {"target_before": 0.50, "target_after": 0.60,
                "general_before": 0.50, "general_after": 0.49}


# ----------------------------------------------------------------- cards --- #
def test_six_ops_enabled_g_excluded():
    assert set(ops.ENABLED_OPS) == {"skill_up", "meta_adapt", "teamplay",
                                    "create_player", "style_spec", "obedience_tune"}
    assert "weakness_fix" not in ops.ENABLED_OPS  # G left out for now


def test_every_card_has_manager_facing_fields():
    for op in ops.ENABLED_OPS:
        c = ops.card(op)
        for key in ("title", "what", "efficiency", "requires",
                    "who_can_be_chosen", "min_steps", "recommended_steps", "validation"):
            assert c[key], f"{op} missing {key}"
        assert c["recommended_steps"] >= c["min_steps"]
        assert c["who_can_be_chosen"]  # who can be chosen as target


def test_card_line_is_compact():
    line = ops.card_line("create_player")
    assert "create_player" in line and "4,000,000" in line


def test_validate_request_enforces_steps_and_target():
    with pytest.raises(ValueError):
        ops.validate_request("skill_up", {"kind": "player"}, total_steps=100)   # below min
    with pytest.raises(ValueError):
        ops.validate_request("skill_up", {"kind": "roster"}, total_steps=1_000_000)  # bad target
    ops.validate_request("skill_up", {"kind": "coach"}, total_steps=1_000_000)   # ok (coach allowed)
    ops.validate_request("teamplay", {"kind": "roster"}, total_steps=1_000_000)  # ok


# ------------------------------------------------------------- execution --- #
def test_execute_refused_without_backend():
    with pytest.raises(NotImplementedError):
        ops.execute("skill_up", {"target": {"kind": "player"}}, total_steps=1_000_000)


def test_execute_skill_up_passes_regression_guard():
    res = ops.execute("skill_up", {"target": {"kind": "player"}},
                      total_steps=1_000_000, backend=MockBackend())
    assert res.op == "skill_up" and res.train["trained"] == "ppo"
    assert res.guard["improved_target"] and not res.guard["regressed_general"]
    assert res.passed


def test_execute_create_player_uniqueness_guard():
    res = ops.execute("create_player", {"target": {"kind": "parent", "parent_id": "p1"}},
                      total_steps=4_000_000, backend=MockBackend())
    assert res.train["child_net"] == "fork-1"
    assert res.guard["distinct_hash"] and res.guard["far_enough"] and res.passed


def test_execute_unknown_op_rejected():
    with pytest.raises(ValueError):
        ops.execute("weakness_fix", {"target": {}}, total_steps=10, backend=MockBackend())


# ---------------------------------------------------------------- guards --- #
def test_regression_guard_flags_general_drop():
    g = ops.regression_guard(0.5, 0.6, 0.5, 0.40)  # big general drop
    assert g["improved_target"] and g["regressed_general"] and not g["passed"]


def test_uniqueness_guard_flags_clone():
    g = ops.uniqueness_guard("h1", {"h1", "h2"}, behaviour_distance=0.01)
    assert not g["distinct_hash"] and not g["far_enough"] and not g["passed"]
