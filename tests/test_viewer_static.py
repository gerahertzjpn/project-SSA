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
    assert 'id="unitCanvas"' in index
    assert 'id="mapForceLayer"' in index
    assert 'id="lordLayer"' in index
    assert 'id="mapEventLayer"' in index
    assert 'role="tablist"' in index
    assert 'data-tab="battles"' in index
    assert 'data-tab="generate"' in index
    assert 'data-tab="state"' in index
    assert 'data-tab-panel="events"' in index
    assert 'id="stateList"' in index
    assert 'id="stateFilter"' in index
    assert 'id="stateFilterAllBtn"' in index
    assert 'id="stateFilterNoneBtn"' in index
    assert '<option value="2800">0.25x</option>' in index
    assert 'id="matchCountInput"' in index
    assert 'id="seedInput"' in index
    assert 'id="playerModelGrid"' in index
    assert 'id="generateMatchesBtn"' in index
    assert 'id="reconnectApiBtn"' in index
    assert 'id="generationSpinner"' in index


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
    assert "populateStateFilter" in script
    assert "selectedStateNodes" in script
    assert "setStateNodeSelection" in script
    assert "stateForceCards" in script
    assert "forceStatus" in script
    assert "movingGroupCard" in script
    assert "movingGroupPosition" in script
    assert "renderUnitCanvas" in script
    assert "drawBattleFormation" in script
    assert "drawUnitCluster" in script
    assert "drawUnitGlyph" in script
    assert "renderMapForces" in script
    assert "mapForcesAtNode" in script
    assert "renderMapEvents" in script
    assert "TUNNEL_EDGES" in script
    assert "battleCardPosition" in script
    assert "movementSignature" in script
    assert "pathSignature" in script
    assert "started" in script
    assert "alliedLordMarkers" in script
    assert "renderLordOverlay" in script
    assert "stateHpBar" in script
    assert "coreHpMarkup" in script
    assert "LORD_MAX_HP" in script
    assert "phaseLabel" in script
    assert "attackLinks" in script
    assert "attackSummaryMarkup" in script
    assert "setActiveTab" in script
    assert "initGenerator" in script
    assert "populatePlayerModelSelects" in script
    assert "generateMatches" in script
    assert "/api/models" in script
    assert "/api/generate-replays" in script
    assert "API_PORTS" in script
    assert "apiBaseCandidates" in script
    assert "serverStatusLabel" in script
    assert "apiStale" in script
    assert "replayFetchUrl" in script
    assert "DEFAULT_UI_RUNS" in script
    assert "displayUnitCode" in script
    assert "renderPos" in script
    assert "simPos" in script
    assert "requestAnimationFrame" in script
    assert "frame" not in script


def test_viewer_layout_has_arena_layers():
    styles = (VIEWER / "styles.css").read_text(encoding="utf-8")

    assert ".arena" in styles
    assert ".unit-canvas" in styles
    assert "#nodeLayer" in styles
    assert "#mapForceLayer" in styles
    assert "#routeLayer" in styles
    assert "#battleLayer" in styles
    assert "#lordLayer" in styles
    assert "#mapEventLayer" in styles
    assert ".edge-path.tunnel" in styles
    assert ".map-force" in styles
    assert ".map-battle-card" in styles
    assert ".map-event" in styles
    assert ".lord-marker" in styles
    assert ".formation" in styles
    assert ".formation-stack" in styles
    assert ".attack-layer" in styles
    assert ".attack-link.ranged" in styles
    assert ".panel-tab" in styles
    assert ".side-tab.active" in styles
    assert ".generator-panel" in styles
    assert ".model-row" in styles
    assert ".generation-status" in styles
    assert ".spinner.active" in styles
    assert ".mini-command" in styles
    assert ".state-node-card" in styles
    assert ".state-filter" in styles
    assert ".state-filter-grid" in styles
    assert ".state-filter-option" in styles
    assert ".state-force-card" in styles
    assert ".state-move-card" in styles
    assert ".state-progress" in styles
    assert ".state-hp" in styles
    assert ".state-hp-track" in styles
    assert ".battle-item.resolved" in styles
    assert ".battle-side" in styles
    assert "@media (max-height: 700px)" in styles
    assert "@media (max-height: 560px)" in styles
    assert "grid-template-rows: minmax(0, 1fr) 54px" in styles


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
    assert "MAX_UI_RUNS" not in runner


def test_viewer_has_one_click_windows_launcher():
    batch = (ROOT / "start_viewer.bat").read_text(encoding="utf-8")
    script = (ROOT / "start_viewer.ps1").read_text(encoding="utf-8")

    assert "ExecutionPolicy Bypass" in batch
    assert "start_viewer.ps1" in batch
    assert "viewer/server.py" in script
    assert "8001..8010" in script
    assert "-WindowStyle Hidden" in script
    assert "viewer/index.html" in script
