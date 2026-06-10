import sys
from pathlib import Path

import torch

sys.path.insert(0, "c:/project-SSA")

from src.core.constants import PLAYER_IDS
from src.core.replay import load_replay
from src.training.policy import MaskedActorCritic
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION
from viewer import server


def test_ui_generation_replaces_old_replays_and_returns_numbered_list(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "REPLAY_DIR", tmp_path)

    old_replay = tmp_path / "replay_99.json"
    old_replay.write_text("{}", encoding="utf-8")

    payload = {
        "runs": 2,
        "seed": 20260603,
        "players": {player_id: "random" for player_id in PLAYER_IDS},
        "factions": {
            "A1": "infected",
            "A2": "robots",
            "A3": "humans",
            "B1": "robots",
            "B2": "infected",
            "B3": "humans",
        },
    }
    result = server.generate_replays(payload)

    assert not old_replay.exists()
    assert [item["label"] for item in result["replays"]] == ["replay_0.json", "replay_1.json"]
    assert (tmp_path / "replay_0.json").exists()
    assert (tmp_path / "replay_1.json").exists()
    assert result["factions"] == payload["factions"]

    replay = load_replay(str(tmp_path / "replay_0.json"))
    assert replay["meta"]["schema_version"] == "0.7.2"
    assert replay["meta"]["player_factions"] == payload["factions"]
    assert replay["meta"]["rules"]["no_shoot_while_moving"] is False
    assert replay["meta"]["rules"]["ranger_can_attack_after_moving"] is True
    assert replay["meta"]["rules"]["approach_threat_battles"] is True
    assert replay["meta"]["rules"]["node_ranged_approach_battles"] is True
    assert replay["meta"]["rules"]["combat_2d_map"] is True
    assert replay["meta"]["rules"]["map_distance_movement"] is True
    assert result["server"]["schema_version"] == "0.7.2"


def test_replay_files_are_sorted_by_numeric_index(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "REPLAY_DIR", tmp_path)
    for name in ("replay_10.json", "replay_2.json", "replay_1.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert [item["label"] for item in server.replay_files()] == [
        "replay_1.json",
        "replay_2.json",
        "replay_10.json",
    ]


def test_viewer_server_discovers_policy_checkpoints(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "training_runs" / "new_policy" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    policy = MaskedActorCritic(hidden_size=32)
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "config": {"hidden_size": 32},
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
        },
        checkpoint,
    )
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "default_policy_checkpoint", lambda: None)
    monkeypatch.setattr(server, "default_team_policy_checkpoint", lambda: None)
    monkeypatch.setattr(
        server,
        "policy_model_id_for_checkpoint",
        lambda path, policy_state_id=None: (
            f"policy:{Path(path).relative_to(tmp_path).as_posix()}"
            f"{'#' + policy_state_id if policy_state_id else ''}"
        ),
    )

    models = server.available_models()

    discovered = next(
        model for model in models
        if model["id"] == "policy:training_runs/new_policy/checkpoints/latest.pt"
    )
    # discovered training-run checkpoints are demoted to hidden legacy entries
    # (population-18 nets are the first-class policy models now)
    assert discovered["label"] == "NewPol latest (legacy)"
    assert discovered["kind"] == "legacy"
    assert discovered["is_latest"] is True
    assert discovered["show_by_default"] is False
    assert all(len(model["label"].split()) <= 4 for model in models)


def test_viewer_server_exposes_team_policy_states_as_assignable_models(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "training_runs" / "team_policy" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    policy = MaskedActorCritic(hidden_size=32)
    torch.save(
        {
            "policy_states": {
                "A1": policy.state_dict(),
                "A2": policy.state_dict(),
                "A3": policy.state_dict(),
            },
            "config": {"hidden_size": 32},
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
        },
        checkpoint,
    )
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "default_policy_checkpoint", lambda: None)
    monkeypatch.setattr(server, "default_team_policy_checkpoint", lambda: None)
    monkeypatch.setattr(
        server,
        "policy_model_id_for_checkpoint",
        lambda path, policy_state_id=None: (
            f"policy:{Path(path).relative_to(tmp_path).as_posix()}"
            f"{'#' + policy_state_id if policy_state_id else ''}"
        ),
    )

    models = server.available_models()
    policy_models = {
        model["policy_state"]: model
        for model in models
        if model.get("checkpoint") == "training_runs/team_policy/checkpoints/latest.pt"
    }

    assert sorted(policy_models) == ["A1", "A2", "A3"]
    assert policy_models["A1"]["id"].endswith("#A1")
    assert policy_models["A2"]["id"].endswith("#A2")
    assert policy_models["A3"]["id"].endswith("#A3")
    # discovered checkpoints are demoted to hidden legacy entries (see above)
    assert policy_models["A2"]["label"] == "TeaPol latest A2 (legacy)"
    assert policy_models["A2"]["show_by_default"] is False
    assert all(len(model["label"].split()) <= 4 for model in policy_models.values())
