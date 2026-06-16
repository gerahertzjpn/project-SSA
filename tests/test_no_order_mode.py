"""No-order mode: an absent order masks the channel to zeros and kills order shaping,
so the net learns to play freely when un-ordered (distinct from any specific order)."""
from src.training.spaces import ORDER_FEATURES, _order_features
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig


def test_absent_order_encodes_all_zeros_vs_one_hot_for_a_real_order():
    zeros = _order_features({"order": {"A1": {"absent": True}}}, "A1")
    assert len(zeros) == ORDER_FEATURES and sum(zeros) == 0.0
    one_hot = _order_features(
        {"order": {"A1": {"tempo": "aggressive", "win_path": "core_rush", "role": "carry_push"}}}, "A1")
    assert sum(one_hot) == 3.0  # one active per group (tempo/win_path/role)
    # a missing entry keeps the neutral default (back-compat), also one-hot
    assert sum(_order_features({}, "A1")) == 3.0


def test_no_order_prob_one_masks_every_episode():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        seed=5, max_ticks=20, sample_orders=True, no_order_prob=1.0, order_shaping_weight=1.0))
    env.reset(seed=5)
    # whole episode is un-ordered
    assert all(env._order[pid].get("absent") for pid in env.controlled_players)
    # channel reads zeros, and alignment potential is nil -> no order shaping
    pid = env.controlled_players[0]
    assert sum(_order_features({"order": env._order}, pid)) == 0.0
    assert env._order_potential({}, pid) == 0.0


def test_ordered_episode_still_carries_an_order():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        seed=7, max_ticks=20, sample_orders=True, no_order_prob=0.0))
    env.reset(seed=7)
    pid = env.controlled_players[0]
    assert not env._order[pid].get("absent")
    assert sum(_order_features({"order": env._order}, pid)) == 3.0  # a real order
