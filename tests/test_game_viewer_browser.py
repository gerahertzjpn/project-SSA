import socket
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_http(url: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"HTTP server did not answer: {url}")


def test_game_viewer_loads_replay_and_renders_canvas():
    playwright_module = pytest.importorskip("playwright.sync_api")
    image_module = pytest.importorskip("PIL.Image")
    if not (ROOT / "replays_viewer_debug_smoke" / "replay_0.json").exists():
        pytest.skip("viewer sample replays are not generated")

    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_http(f"http://127.0.0.1:{port}/viewer_game/index.html")
        with playwright_module.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")

            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(f"http://127.0.0.1:{port}/viewer_game/index.html", wait_until="domcontentloaded")
            page.wait_for_function("state.replay && state.visuals.size > 0")
            counts = page.evaluate(
                """() => {
                    const picked = pickAt(70, 90);
                    state.selected = {type: 'node', id: 'A1'};
                    updateInspector();
                    return {
                        visuals: state.visuals.size,
                        replayTicks: state.replay.ticks.length,
                        pickedType: picked?.type || null,
                        selected: state.selected?.id || null,
                        inspect: document.querySelector('#inspectBody').textContent,
                        canvasWidth: document.querySelector('#gameCanvas').width,
                        canvasHeight: document.querySelector('#gameCanvas').height,
                        meta: document.querySelector('#metaText').textContent
                    };
                }"""
            )
            page.click("#loadBtn")
            page.wait_for_function("state.replay && Number(document.querySelector('#tickMax').textContent) > 0")
            page.click("#generatePanelBtn")
            generate_panel = page.evaluate(
                """() => ({
                    hidden: document.querySelector('#generatePanel').hidden,
                    disabled: document.querySelector('#generateBtn').disabled,
                    status: document.querySelector('#generateStatus').textContent,
                    rows: document.querySelectorAll('.model-row').length
                })"""
            )
            play_before = page.evaluate("({tick: state.tickFloat, text: document.querySelector('#playBtn').textContent})")
            page.click("#playBtn")
            page.wait_for_function("state.tickFloat > 0.5")
            play_after = page.evaluate(
                "({tick: state.tickFloat, readout: document.querySelector('#tickNow').textContent, text: document.querySelector('#playBtn').textContent})"
            )
            screenshot = page.screenshot(full_page=False)
            browser.close()

        assert not errors
        assert counts["visuals"] > 0
        assert counts["replayTicks"] > 1
        assert counts["pickedType"] in {"unit", "node"}
        assert counts["selected"] == "A1"
        assert "A1" in counts["inspect"]
        assert counts["canvasWidth"] > 0
        assert counts["canvasHeight"] > 0
        assert "schema" in counts["meta"]
        assert not generate_panel["hidden"]
        assert generate_panel["disabled"]
        assert "API" in generate_panel["status"]
        assert generate_panel["rows"] == 6
        assert play_before["tick"] == 0
        assert play_before["text"] == ">"
        assert play_after["tick"] > 0.5
        assert int(play_after["readout"]) >= 0
        assert play_after["text"] == "||"

        image = image_module.open(BytesIO(screenshot)).convert("RGB")
        pixels = list(image.getdata())
        sample = pixels[::max(1, len(pixels) // 20000)]
        assert len(screenshot) > 50_000
        assert len(set(sample)) > 60
        assert sum(1 for red, green, blue in sample if red + green + blue > 95) > 1000
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()


def test_game_viewer_works_with_generation_api_server():
    playwright_module = pytest.importorskip("playwright.sync_api")
    if not (ROOT / "replays_viewer_debug_smoke" / "replay_0.json").exists():
        pytest.skip("viewer sample replays are not generated")

    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "viewer/server.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_http(f"http://127.0.0.1:{port}/api/models", timeout=20.0)
        with playwright_module.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")

            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(f"http://127.0.0.1:{port}/viewer_game/index.html", wait_until="domcontentloaded")
            page.wait_for_function("state.replay && state.visuals.size > 0")
            page.click("#generatePanelBtn")
            generate_panel = page.evaluate(
                """() => ({
                    hidden: document.querySelector('#generatePanel').hidden,
                    disabled: document.querySelector('#generateBtn').disabled,
                    rows: document.querySelectorAll('.model-row').length,
                    status: document.querySelector('#generateStatus').textContent
                })"""
            )
            page.click("#playBtn")
            page.wait_for_function("state.tickFloat > 0.5")
            play_state = page.evaluate(
                "({tick: state.tickFloat, button: document.querySelector('#playBtn').textContent})"
            )
            browser.close()

        assert not errors
        assert not generate_panel["hidden"]
        assert not generate_panel["disabled"]
        assert generate_panel["rows"] == 6
        assert "Ready" in generate_panel["status"]
        assert play_state["tick"] > 0.5
        assert play_state["button"] == "||"
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()
