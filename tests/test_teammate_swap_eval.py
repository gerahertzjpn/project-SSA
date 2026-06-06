from scripts.evaluate_teammate_swaps import build_swap_suite


def test_build_swap_suite_keeps_native_controls_before_mixed_teams():
    checkpoints = ["run/checkpoints/checkpoint_update_1.pt", "run/checkpoints/checkpoint_update_2.pt", "run/checkpoints/checkpoint_update_3.pt"]

    teams = build_swap_suite(
        checkpoints,
        suite_kinds={"native", "slot_mixed"},
        max_teams=4,
    )

    assert [team["kind"] for team in teams] == ["native", "native", "native", "slot_mixed"]
    assert teams[0]["slots"]["A1"]["source_player"] == "A1"
    assert teams[0]["slots"]["A2"]["source_player"] == "A2"
    assert teams[0]["slots"]["A3"]["source_player"] == "A3"
    assert teams[3]["slots"]["A1"]["checkpoint"] == checkpoints[0]
    assert teams[3]["slots"]["A2"]["checkpoint"] == checkpoints[1]
    assert teams[3]["slots"]["A3"]["checkpoint"] == checkpoints[2]


def test_build_swap_suite_role_stress_is_explicit_not_primary_swap():
    checkpoint = "run/checkpoints/checkpoint_update_7.pt"

    teams = build_swap_suite(
        [checkpoint],
        suite_kinds={"role_stress"},
    )

    assert len(teams) == 1
    assert teams[0]["kind"] == "role_stress"
    assert teams[0]["slots"]["A1"]["source_player"] == "A2"
    assert teams[0]["slots"]["A2"]["source_player"] == "A3"
    assert teams[0]["slots"]["A3"]["source_player"] == "A1"
