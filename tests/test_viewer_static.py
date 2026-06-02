from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"


def test_viewer_assets_exist_and_are_wired():
    index = (VIEWER / "index.html").read_text(encoding="utf-8")

    assert '<link rel="stylesheet" href="./styles.css">' in index
    assert '<script src="./app.js"></script>' in index
    assert 'id="tickSlider"' in index
    assert 'id="playBtn"' in index
    assert 'id="fileInput"' in index
    assert 'id="sampleSelect"' in index
    assert 'id="diagnosticsGrid"' in index
    assert 'preserveAspectRatio="none"' in index
    assert 'role="tablist"' in index
    assert 'data-tab="battles"' in index
    assert 'data-tab="state"' in index
    assert 'data-tab-panel="events"' in index
    assert 'id="stateList"' in index


def test_viewer_loads_compact_replay_schema():
    script = (VIEWER / "app.js").read_text(encoding="utf-8")

    assert "replays_viewer_debug_smoke" in script
    assert "SAMPLE_REPLAYS" in script
    assert "BATTLE_TICK" in script
    assert "summary" in script
    assert "eventCounts" in script
    assert "formationMarkup" in script
    assert "formationStacks" in script
    assert "battleSideCards" in script
    assert "recentResolvedBattles" in script
    assert "renderStatePanel" in script
    assert "stateForceCards" in script
    assert "forceStatus" in script
    assert "movingGroupCard" in script
    assert "movingGroupPosition" in script
    assert "phaseLabel" in script
    assert "attackLinks" in script
    assert "attackSummaryMarkup" in script
    assert "setActiveTab" in script
    assert "displayUnitCode" in script
    assert "requestAnimationFrame" in script
    assert "frame" not in script


def test_viewer_layout_has_arena_layers():
    styles = (VIEWER / "styles.css").read_text(encoding="utf-8")

    assert ".arena" in styles
    assert "#nodeLayer" in styles
    assert "#routeLayer" in styles
    assert "#battleLayer" in styles
    assert ".formation" in styles
    assert ".formation-stack" in styles
    assert ".attack-layer" in styles
    assert ".attack-link.ranged" in styles
    assert ".panel-tab" in styles
    assert ".side-tab.active" in styles
    assert ".state-node-card" in styles
    assert ".state-force-card" in styles
    assert ".state-move-card" in styles
    assert ".state-progress" in styles
    assert ".battle-item.resolved" in styles
    assert ".battle-side" in styles
    assert "@media (max-height: 700px)" in styles
    assert "@media (max-height: 560px)" in styles
    assert "grid-template-rows: minmax(0, 1fr) 54px" in styles
