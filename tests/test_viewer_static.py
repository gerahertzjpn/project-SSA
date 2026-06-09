from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"


def test_legacy_viewer_redirects_to_pixi():
    index = (VIEWER / "index.html").read_text(encoding="utf-8")

    assert "Legacy diagnostics UI is intentionally disabled" in index
    assert "../viewer_pixi/index.html" in index
    assert "app.js" not in index
    assert "styles.css" not in index


def test_viewer_generation_server_is_ui_limited_only():
    server = (VIEWER / "server.py").read_text(encoding="utf-8")
    runner = (ROOT / "src" / "core" / "runner.py").read_text(encoding="utf-8")

    assert "MAX_UI_RUNS = 100" in server
    assert "/api/generate-replays" in server
    assert "clear_replays()" in server
    assert "REPLAY_NAME" in server
    assert "server_info()" in server
    assert "source_fingerprint" in server
    assert "discover_policy_checkpoints" in server
    assert "policy_model_id_for_checkpoint" in server
    assert "Access-Control-Allow-Origin" in server
    assert "Cache-Control" in server
    assert "do_OPTIONS" in server
    assert "player_models=player_models" in server
    assert "player_factions=player_factions" in server
    assert '"factions"' in server
    assert "MAX_UI_RUNS" not in runner


def test_viewer_has_one_click_windows_launcher():
    batch = (ROOT / "start_viewer.bat").read_text(encoding="utf-8")
    script = (ROOT / "start_viewer.ps1").read_text(encoding="utf-8")

    assert "ExecutionPolicy Bypass" in batch
    assert "start_viewer.ps1" in batch
    assert "viewer/server.py" in script
    assert "8001..8010" in script
    assert "-WindowStyle Hidden" in script
    assert "viewer_pixi/index.html" in script
