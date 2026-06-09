from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER_PIXI = ROOT / "viewer_pixi"


def test_pixi_viewer_assets_exist_and_are_wired():
    index = (VIEWER_PIXI / "index.html").read_text(encoding="utf-8")

    assert '<link rel="stylesheet" href="./styles.css' in index
    assert '<script type="module" src="./app.js' in index
    assert '"pixi.js"' in index
    assert "pixi.js@8.10.0" in index
    assert 'id="pixiStage"' in index
    assert 'id="replaySelect"' in index
    assert 'id="generateBtn"' in index
    assert 'id="generateCountInput"' in index
    assert 'id="generateSeedInput"' in index
    assert 'id="generateStatus"' in index
    assert 'id="configToggleBtn"' in index
    assert 'id="modelsToggleBtn"' in index
    assert 'id="playerConfigGrid"' in index
    assert 'id="inspectCloseBtn"' in index
    assert 'id="tickSlider"' in index
    assert "../viewer_game/index.html" not in index
    assert "../viewer/index.html" not in index


def test_pixi_viewer_uses_pixi_application_and_replay_positions():
    script = (VIEWER_PIXI / "app.js").read_text(encoding="utf-8")

    assert 'from "pixi.js"' in script
    assert "new Application()" in script
    assert "await app.init" in script
    assert "app.ticker.add" in script
    assert "visualTrackDesired" in script
    assert "tick.visual_units" in script
    assert "normalizeVisualUnits" in script
    assert "SPRITE_EASE" in script
    assert "spriteEaseFor" in script
    assert "battle.formation" in script
    assert "row.length >= 10" in script
    assert "unit.x" in script
    assert "unit.y" in script
    assert "syncSprites" in script
    assert "findHandoffSprite" in script
    assert "drawAttackLines" in script
    assert "battleTickEventForVisual" in script
    assert "mapStateSignature" in script
    assert "lastMapSignature" in script
    assert "HANDOFF_BACKFILL_DISTANCE" in script
    assert "applyVisualHandoffBackfill" in script
    assert "shouldBackfillHandoff" in script
    assert "shouldBridgeTransientStationed" in script
    assert "isBattleAtNode" in script
    assert "deathUnitIds" in script
    assert "smoothstep" in script
    assert "replayInterpolated" in script
    assert "drawLordShape" in script
    assert "isDeadCoreNode" in script
    assert "drawCoreHpBar" in script
    assert "drawNeutralCountLabel" in script
    assert "drawNeutralLordHpBar" in script
    assert "neutralMaxValue" in script
    assert "neutralLordHpValue" in script
    assert "hideSpriteHpBar" in script
    assert "drawNodeBuildings" in script
    assert "drawBuildingMarker" in script
    assert "BUILDING_LABELS" in script
    assert "RAIDER" in script
    assert "BMP" in script
    assert "coreHpValue" in script
    assert "deadCoreBg" in script
    assert "deadUnit" in script
    assert "DEATH_HOLD_SECONDS" in script
    assert "DEATH_FADE_OUT_PER_SECOND" in script
    assert "ATTACK_PREVIEW_ALPHA = 0.5" in script
    assert ".ellipse(" not in script
    assert "/api/models" in script
    assert "/api/generate-replays" in script
    assert "generateReplays" in script
    assert "validateGeneratorControls" in script
    assert "populatePlayerConfigGrid" in script
    assert "toggleConfigMode" in script
    assert "toggleModelScope" in script
    assert "visibleModels" in script
    assert "applyConfigMode" in script
    assert "closeInspector" in script
    assert "playerModelSelection" in script
    assert "playerFactionSelection" in script
    assert "player_factions" in script
    assert "factions" in script
    assert "maxUiRuns" in script
    assert "replays_viewer_debug_smoke" in script


def test_pixi_viewer_layout_is_fullscreen_hud():
    styles = (VIEWER_PIXI / "styles.css").read_text(encoding="utf-8")

    assert "#pixiStage" in styles
    assert "#pixiStage canvas" in styles
    assert ".top-hud" in styles
    assert ".generate-status" in styles
    assert ".mini-field" in styles
    assert ".player-config-grid" in styles
    assert ".top-hud.compact" in styles
    assert ".config-toggle" in styles
    assert ".models-toggle" in styles
    assert ".top-hud.show-all-models" in styles
    assert ".team-config" in styles
    assert ".player-config-row" in styles
    assert ".inspect-panel.is-hidden" in styles
    assert ".inspect-head" in styles
    assert 'input[type="number"]::-webkit-inner-spin-button' in styles
    assert ".bottom-hud" in styles
    assert ".inspect-panel" in styles
    assert "position: absolute" in styles
    assert "overflow: hidden" in styles
