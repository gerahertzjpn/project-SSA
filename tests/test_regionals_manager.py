"""Manager game plan — persona defaults, LLM path, fallback, env conversion."""
from src.regionals.franchises import FRANCHISES
from src.regionals.manager import (
    match_plan,
    persona_default_plan,
    plan_to_env_order,
)
from src.training.spaces import ORDER_ROLE, ORDER_TEMPO, ORDER_WIN_PATH

PIDS = ["A1", "A2", "A3"]
_BY_ID = {f["franchise_id"]: f for f in FRANCHISES}


class _FakeLLM:
    def __init__(self, plan=None, raises=False):
        self._plan = plan
        self._raises = raises

    def plan(self, system, prompt, player_ids, tempo, win_path, role):
        if self._raises:
            raise RuntimeError("api down")
        return self._plan


def test_persona_defaults_are_valid_and_in_character():
    # Every franchise carries an in-character default in the order vocabulary.
    assert len(FRANCHISES) == 25
    for fr in FRANCHISES:
        plan = persona_default_plan(fr, PIDS)
        assert plan["tempo"] in ORDER_TEMPO
        assert plan["win_path"] in ORDER_WIN_PATH
        assert all(plan["roles"][pid] in ORDER_ROLE for pid in PIDS)
    # Spot-check character: the ex-military aggressor pushes; the macro-defender turtles.
    assert persona_default_plan("iron-falcons", PIDS)["tempo"] == "aggressive"
    assert persona_default_plan("northwind", PIDS)["tempo"] == "turtle"


def test_no_llm_falls_back_to_persona_default():
    fr = _BY_ID["iron-falcons"]
    assert match_plan(fr, "situation", PIDS, llm=None) == persona_default_plan(fr, PIDS)


def test_llm_plan_used_when_valid():
    fr = _BY_ID["quantum-owls"]
    good = {"tempo": "balanced", "win_path": "core_rush",
            "roles": {"A1": "tech_eco", "A2": "carry_push", "A3": "anchor_defend"},
            "rationale": "data says rush"}
    assert match_plan(fr, "s", PIDS, llm=_FakeLLM(plan=good)) == good


def test_invalid_or_failing_llm_falls_back():
    fr = _BY_ID["solar-drift"]
    bad = {"tempo": "nonsense", "win_path": "economy", "roles": {}}
    assert match_plan(fr, "s", PIDS, llm=_FakeLLM(plan=bad)) == persona_default_plan(fr, PIDS)
    assert match_plan(fr, "s", PIDS, llm=_FakeLLM(raises=True)) == persona_default_plan(fr, PIDS)


def test_plan_to_env_order_shape():
    plan = persona_default_plan("crimson-pact", PIDS)
    order = plan_to_env_order(plan, PIDS)
    assert set(order) == set(PIDS)
    for pid in PIDS:
        assert order[pid]["tempo"] == plan["tempo"]
        assert order[pid]["win_path"] == plan["win_path"]
        assert order[pid]["role"] == plan["roles"][pid]


# --- adaptation: manager memory + situation builder ---
from src.regionals.manager import ManagerMemory, build_situation


def test_manager_memory_records_and_renders_recent():
    mem = ManagerMemory(keep=3)
    assert "первый матч" in mem.recent_block("iron-falcons")
    mem.record("iron-falcons", "Quantum Owls", {"tempo": "aggressive", "win_path": "core_rush"}, "W")
    mem.record("iron-falcons", "Northwind", {"tempo": "all_in", "win_path": "baron_lord"}, "L")
    block = mem.recent_block("iron-falcons")
    assert "Quantum Owls" in block and "→W" in block and "→L" in block
    # keep-window trims oldest
    for i in range(5):
        mem.record("iron-falcons", f"T{i}", {"tempo": "balanced", "win_path": "economy"}, "W")
    assert mem.recent_block("iron-falcons").count("→") == 3


def test_build_situation_includes_key_facts():
    s = build_situation("Quantum Owls", "own-cards", "opp-cards", "1-0",
                        ManagerMemory().recent_block("iron-falcons"))
    assert "Quantum Owls" in s and "1-0" in s and "own-cards" in s and "opp-cards" in s


# --- match-loop integration: order overlay reaches the nets ---
from src.core.engine import Action, GameEngine
from src.core.constants import ActionType
from src.core.state import make_initial_state
from src.regionals.manager import team_match_order


class _Probe:
    requires_observation = True

    def __init__(self):
        self.seen = None

    def act(self, obs):
        self.seen = obs.get("order")
        return Action(ActionType.HOLD)


def test_engine_injects_order_overlay_into_obs():
    state = make_initial_state(seed=1)
    probe = _Probe()
    bots = {"A1": probe}
    eng = GameEngine(state, bots, record_trace=False, record_replay=False)
    eng.orders = {"A1": {"tempo": "aggressive", "win_path": "core_rush", "role": "carry_push"}}
    eng.tick()
    assert probe.seen == eng.orders


def test_team_match_order_produces_env_format():
    fr = {"franchise_id": "iron-falcons", "persona": "..."}
    order = team_match_order(fr, "s", ["A1", "A2", "A3"], llm=None)
    assert set(order) == {"A1", "A2", "A3"}
    assert order["A1"]["tempo"] in ORDER_TEMPO and order["A1"]["role"] in ORDER_ROLE
