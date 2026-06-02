import sys
from pathlib import Path

sys.path.insert(0, "c:/project-SSA")

from src.core.constants import PLAYER_IDS
from src.core.replay import load_replay
from viewer import server


def test_ui_generation_replaces_old_replays_and_returns_numbered_list(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "REPLAY_DIR", tmp_path)

    old_replay = tmp_path / "replay_99.json"
    old_replay.write_text("{}", encoding="utf-8")

    payload = {
        "runs": 2,
        "seed": 20260603,
        "players": {player_id: "random" for player_id in PLAYER_IDS},
    }
    result = server.generate_replays(payload)

    assert not old_replay.exists()
    assert [item["label"] for item in result["replays"]] == ["replay_0.json", "replay_1.json"]
    assert (tmp_path / "replay_0.json").exists()
    assert (tmp_path / "replay_1.json").exists()

    replay = load_replay(str(tmp_path / "replay_0.json"))
    assert replay["meta"]["schema_version"] == "0.5.1"
    assert replay["meta"]["rules"]["no_shoot_while_moving"] is True
    assert result["server"]["schema_version"] == "0.5.1"


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
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(
        server,
        "policy_model_id_for_checkpoint",
        lambda path: f"policy:{Path(path).relative_to(tmp_path).as_posix()}",
    )

    models = server.available_models()

    assert any(
        model["id"] == "policy:training_runs/new_policy/checkpoints/latest.pt"
        for model in models
    )
