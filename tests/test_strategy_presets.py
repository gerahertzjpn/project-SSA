"""Coherent strategy presets replace free tempo×win_path (no turtle+core_rush)."""
from src.training.spaces import (
    ORDER_ROLE, ORDER_TEMPO, ORDER_WIN_PATH,
    STRATEGY_PRESETS, STRATEGY_PRESET_BY_ID, strategy_order_for_slot,
)


def test_presets_are_coherent_unique_and_valid():
    assert len(STRATEGY_PRESETS) >= 5
    ids = [p["id"] for p in STRATEGY_PRESETS]
    assert len(set(ids)) == len(ids)  # unique
    for p in STRATEGY_PRESETS:
        assert p["tempo"] in ORDER_TEMPO
        assert p["win_path"] in ORDER_WIN_PATH
        assert len(p["roles"]) == 3 and all(r in ORDER_ROLE for r in p["roles"])
        # the incoherent combo the owner flagged must never be a preset
        assert not (p["tempo"] == "turtle" and p["win_path"] == "core_rush")


def test_strategy_order_for_slot_expands_with_per_position_role():
    o0 = strategy_order_for_slot("core_rush", 0)
    o2 = strategy_order_for_slot("core_rush", 2)
    assert o0["tempo"] == "aggressive" and o0["win_path"] == "core_rush"
    assert o0["role"] == STRATEGY_PRESET_BY_ID["core_rush"]["roles"][0]
    assert o2["role"] == STRATEGY_PRESET_BY_ID["core_rush"]["roles"][2]
    assert o0["strategy"] == "core_rush"
    # unknown strategy -> neutral default (no crash, no 'strategy' tag)
    d = strategy_order_for_slot("does_not_exist", 0)
    assert d["tempo"] in ORDER_TEMPO and "strategy" not in d


def test_training_samples_one_coherent_strategy_for_the_team():
    from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig
    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        seed=11, max_ticks=20, sample_orders=True, no_order_prob=0.0))
    env.reset(seed=11)
    sids = {v.get("strategy") for v in env._order.values()}
    assert len(sids) == 1 and next(iter(sids)) in STRATEGY_PRESET_BY_ID  # one coherent preset, whole team
