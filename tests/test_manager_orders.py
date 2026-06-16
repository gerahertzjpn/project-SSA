"""Order channel (LLM-manager) — obs encoding, mirroring, neutral defaults.

Covers Tier-1 of docs/cloudcode/LLM_MANAGER_ORDERS_SPEC.md: tempo + win_path +
role one-hots, carried through the TEAM_B mirror, with a well-defined neutral
baseline when no order is present. Obedience is NOT in the observation — it is a
per-net training-side shaping weight (see spec), so it is tested in team_env, not
here.
"""
import numpy as np

from src.core.engine import GameEngine
from src.core.state import make_initial_state
from src.training import spaces as sp


def _obs(player_id="A1"):
    state = make_initial_state(seed=1)
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    return engine.observe(player_id)


def _order_block(vec):
    start = sp.GLOBAL_FEATURES - sp.ORDER_FEATURES
    return vec[start:sp.GLOBAL_FEATURES]


def test_order_features_size_and_dim():
    assert sp.ORDER_FEATURES == len(sp.ORDER_TEMPO) + len(sp.ORDER_WIN_PATH) + len(sp.ORDER_ROLE)
    obs = _obs()
    assert sp.encode_observation(obs, "A1").shape[0] == sp.OBSERVATION_DIM


def test_no_order_is_neutral_baseline():
    block = _order_block(sp.encode_observation(_obs(), "A1"))
    assert block[:4][sp.ORDER_TEMPO.index(sp.ORDER_TEMPO_DEFAULT)] == 1.0
    assert block[4:8][sp.ORDER_WIN_PATH.index(sp.ORDER_WIN_PATH_DEFAULT)] == 1.0
    assert block[8:13][sp.ORDER_ROLE.index(sp.ORDER_ROLE_DEFAULT)] == 1.0


def test_order_one_hots_land():
    obs = _obs()
    base = sp.encode_observation(obs, "A1")
    obs["order"] = {"A1": {"tempo": "aggressive", "win_path": "core_rush", "role": "carry_push"}}
    vec = sp.encode_observation(obs, "A1")
    block = _order_block(vec)
    assert block[:4].sum() == 1.0 and block[:4][sp.ORDER_TEMPO.index("aggressive")] == 1.0
    assert block[4:8].sum() == 1.0 and block[4:8][sp.ORDER_WIN_PATH.index("core_rush")] == 1.0
    assert block[8:13].sum() == 1.0 and block[8:13][sp.ORDER_ROLE.index("carry_push")] == 1.0
    # The order must actually change the encoded vector (no-op channel would be dead).
    assert not np.allclose(base, vec)


def test_order_carried_through_team_b_mirror():
    obs = _obs("B1")
    obs["order"] = {"B1": {"tempo": "turtle", "win_path": "baron_lord", "role": "anchor_defend"}}
    block = _order_block(sp.encode_observation(obs, "B1"))
    assert block[:4][sp.ORDER_TEMPO.index("turtle")] == 1.0
    assert block[4:8][sp.ORDER_WIN_PATH.index("baron_lord")] == 1.0
    assert block[8:13][sp.ORDER_ROLE.index("anchor_defend")] == 1.0


# --- env-side: order sampling, injection, and the reward term ---
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig


def test_orders_sampled_and_injected_into_obs():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=3, max_ticks=20, sample_orders=True))
    env.reset()
    assert set(env._order) == set(env.controlled_players)
    for plan in env._order.values():
        assert plan["tempo"] in sp.ORDER_TEMPO
        assert plan["win_path"] in sp.ORDER_WIN_PATH
        assert plan["role"] in sp.ORDER_ROLE
    assert "order" in env.raw_observations()[env.controlled_players[0]]


def test_order_reward_term_present_and_disableable():
    pid = "A1"
    on = TeamTrainingEnv(TeamTrainingEnvConfig(seed=3, max_ticks=20, sample_orders=True, order_shaping_weight=0.5))
    on.reset()
    terms = []
    for _ in range(15):
        res = on.step({p: 0 for p in on.controlled_players})
        terms.append(res.info["reward_components"][pid]["order"])
        if res.terminal or res.truncated:
            break
    assert any(abs(t) > 0 for t in terms)  # channel is live

    off = TeamTrainingEnv(TeamTrainingEnvConfig(seed=3, max_ticks=20, order_shaping_weight=0.0))
    off.reset()
    res = off.step({p: 0 for p in off.controlled_players})
    assert res.info["reward_components"][pid]["order"] == 0.0
