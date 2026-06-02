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


def test_viewer_loads_replay_in_browser_with_nonblank_screenshot():
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
        wait_for_http(f"http://127.0.0.1:{port}/viewer/index.html")
        with playwright_module.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")

            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.click("#loadSampleBtn")
            page.wait_for_function("document.querySelector('#replayMeta').textContent.includes('schema')")
            page.click('[data-tab="state"]')
            page.wait_for_selector(".state-node-card")

            counts = page.evaluate(
                """() => ({
                    nodes: document.querySelectorAll('.node').length,
                    routes: document.querySelectorAll('.route').length,
                    stateCards: document.querySelectorAll('.state-node-card').length,
                    forceCards: document.querySelectorAll('.state-force-card').length,
                    activeTab: document.querySelector('.panel-tab.active')?.dataset.tab,
                    meta: document.querySelector('#replayMeta').textContent
                })"""
            )
            screenshot = page.screenshot(full_page=False)
            browser.close()

        image = image_module.open(BytesIO(screenshot)).convert("RGB")
        pixels = list(image.getdata())
        step = max(1, len(pixels) // 20000)
        sample = pixels[::step]

        assert counts["nodes"] == 13
        assert counts["stateCards"] > 0
        assert counts["forceCards"] > 0
        assert counts["activeTab"] == "state"
        assert "schema" in counts["meta"]
        assert len(screenshot) > 50_000
        assert len(set(sample)) > 50
        assert sum(1 for red, green, blue in sample if red + green + blue > 90) > 1000
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()
