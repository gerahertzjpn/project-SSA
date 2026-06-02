from scripts.evaluate_checkpoints import (
    aggregate_eval,
    chunks,
    episode_result,
    episode_specs,
    parse_csv_ints,
)


def test_eval_seed_plan_and_chunking_are_deterministic():
    specs = episode_specs([10, 20], 3)

    assert [item["env_seed"] for item in specs] == [10, 11, 12, 20, 21, 22]
    assert [len(chunk) for chunk in chunks(specs, 4)] == [2, 2, 1, 1]
    assert parse_csv_ints("170, 180,400") == [170, 180, 400]


def test_eval_aggregate_exposes_seed_blocks_and_diversity():
    episodes = [
        {
            **episode_result(1.0, 0.0, 1.0, 12, "TEAM_A", "normal", "A1"),
            "seed_block": 10,
            "sequence_hash": "aaa",
            "opening_hash": "open-a",
        },
        {
            **episode_result(-1.0, 0.0, -1.0, 14, "TEAM_B", "death_time", "A1"),
            "seed_block": 20,
            "sequence_hash": "bbb",
            "opening_hash": "open-b",
        },
    ]
    summary = aggregate_eval([{
        "episodes": episodes,
        "actions": [0, 1, 1],
        "invalid_actions": 1,
        "sequence_hashes": ["aaa", "bbb"],
        "opening_hashes": ["open-a", "open-b"],
    }], mode="argmax")

    assert summary["episodes"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["winner_counts"] == {"TEAM_A": 1, "TEAM_B": 1}
    assert summary["action_distribution"]["invalid_action_rate"] == 0.333333
    assert summary["unique_action_sequences"] == 2
    assert set(summary["seed_blocks"]) == {"10", "20"}
