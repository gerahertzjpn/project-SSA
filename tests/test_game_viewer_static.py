from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER_GAME = ROOT / "viewer_game"


def test_legacy_game_viewer_redirects_to_pixi():
    index = (VIEWER_GAME / "index.html").read_text(encoding="utf-8")

    assert "Legacy canvas game UI is intentionally disabled" in index
    assert "../viewer_pixi/index.html" in index
    assert "app.js" not in index
    assert "styles.css" not in index
