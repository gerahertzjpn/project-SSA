from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER_GAME = ROOT / "viewer_game"


def test_game_viewer_assets_exist_and_are_wired():
    index = (VIEWER_GAME / "index.html").read_text(encoding="utf-8")

    assert '<link rel="stylesheet" href="./styles.css' in index
    assert '<script src="./app.js' in index
    assert 'id="gameCanvas"' in index
    assert 'id="replaySelect"' in index
    assert 'id="generatePanelBtn"' in index
    assert 'id="generatePanel"' in index
    assert 'id="generateBtn"' in index
    assert 'id="playerModelGrid"' in index
    assert 'id="tickSlider"' in index
    assert 'id="inspectPanel"' in index
    assert '../viewer/index.html' in index


def test_game_viewer_has_stable_visual_tracker():
    script = (VIEWER_GAME / "app.js").read_text(encoding="utf-8")

    assert "syncVisuals" in script
    assert "findHandoffVisual" in script
    assert "desiredUnits" in script
    assert "stationedDesired" in script
    assert "movingDesired" in script
    assert "battleDesired" in script
    assert "drawVisualUnits" in script
    assert "pickAt" in script
    assert "zoomAtPointer" in script
    assert "HANDOFF_DISTANCE" in script
    assert "generateReplays" in script
    assert "/api/generate-replays" in script
    assert "populateModelSelects" in script
    assert "startPlaybackTimer" in script
    assert "advancePlayback" in script


def test_game_viewer_layout_is_canvas_first():
    styles = (VIEWER_GAME / "styles.css").read_text(encoding="utf-8")

    assert "#gameCanvas" in styles
    assert ".top-hud" in styles
    assert ".bottom-hud" in styles
    assert ".inspect-panel" in styles
    assert ".generate-panel" in styles
    assert ".model-row" in styles
    assert "position: absolute" in styles
    assert "overflow: hidden" in styles
