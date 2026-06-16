"""Manager game plan — persona defaults, LLM path, fallback, env conversion."""
from src.regionals.franchises import FRANCHISES
from src.regionals.manager import (
    PERSONA_DEFAULT_PLANS,
    match_plan,
    persona_default_plan,
    plan_to_env_order,
)
from src.training.spaces import ORDER_ROLE, ORDER_TEMPO, ORDER_WIN_PATH

PIDS = ["A1", "A2", "A3"]


class _FakeLLM:
    def __init__(self, plan=None, raises=False):
        self._plan = plan
        self._raises = raises

    def plan(self, system, prompt, player_ids, tempo, win_path, role):
        if self._raises:
            raise RuntimeError("api down")
        return self._plan


def test_persona_defaults_are_valid_and_in_character():
    # Every franchise has an in-character default that uses the order vocabulary.
    ids = {f["franchise_id"] for f in FRANCHISES}
    assert set(PERSONA_DEFAULT_PLANS) <= ids
    for fid in ids:
        plan = persona_default_plan(fid, PIDS)
        assert plan["tempo"] in ORDER_TEMPO
        assert plan["win_path"] in ORDER_WIN_PATH
        assert all(plan["roles"][pid] in ORDER_ROLE for pid in PIDS)
    # Spot-check character: the ex-military aggressor pushes; the macro-defender turtles.
    assert persona_default_plan("iron-falcons", PIDS)["tempo"] == "aggressive"
    assert persona_default_plan("northwind", PIDS)["tempo"] == "turtle"


def test_no_llm_falls_back_to_persona_default():
    fr = {"franchise_id": "iron-falcons", "persona": "..."}
    assert match_plan(fr, "situation", PIDS, llm=None) == persona_default_plan("iron-falcons", PIDS)


def test_llm_plan_used_when_valid():
    fr = {"franchise_id": "quantum-owls", "persona": "..."}
    good = {"tempo": "balanced", "win_path": "core_rush",
            "roles": {"A1": "tech_eco", "A2": "carry_push", "A3": "anchor_defend"},
            "rationale": "data says rush"}
    assert match_plan(fr, "s", PIDS, llm=_FakeLLM(plan=good)) == good


def test_invalid_or_failing_llm_falls_back():
    fr = {"franchise_id": "solar-drift", "persona": "..."}
    bad = {"tempo": "nonsense", "win_path": "economy", "roles": {}}
    assert match_plan(fr, "s", PIDS, llm=_FakeLLM(plan=bad)) == persona_default_plan("solar-drift", PIDS)
    assert match_plan(fr, "s", PIDS, llm=_FakeLLM(raises=True)) == persona_default_plan("solar-drift", PIDS)


def test_plan_to_env_order_shape():
    plan = persona_default_plan("crimson-pact", PIDS)
    order = plan_to_env_order(plan, PIDS)
    assert set(order) == set(PIDS)
    for pid in PIDS:
        assert order[pid]["tempo"] == plan["tempo"]
        assert order[pid]["win_path"] == plan["win_path"]
        assert order[pid]["role"] == plan["roles"][pid]
