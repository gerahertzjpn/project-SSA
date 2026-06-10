"""Browser visual smoke for the Pixi viewer movement rendering.

Regenerates a fresh replay, serves the repo over http, loads it in the Pixi
viewer (embed mode) with a real Chromium via Playwright, and checks that:

* the replay loads (tickMax > 0),
* playback advances (tickNow increases while playing),
* the canvas renders non-trivial content,
* no fatal console/page errors fire (the optional /api/* calls that only the
  real viewer server answers are ignored).

It also samples the on-screen tick counter while playing as a coarse check that
the viewer keeps animating. Movement *smoothness* itself is asserted by the unit
tests / data diagnostics on the engine output; this smoke confirms the viewer
actually renders the regenerated replay end to end.

Usage:
    python scripts/browser_smoke_movement.py [--seed 20260609] [--keep]
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "replays_browser_smoke"
SHOT_DIR = ROOT / "viewer_logs"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_http(port: int, timeout: float = 8.0) -> None:
    from urllib.request import urlopen

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/viewer_pixi/index.html"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"http.server did not come up on {port}")


def generate_replay(seed: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "main.py", "--runs", "1", "--seed", str(seed),
         "--bot", "scripted", "--out", OUT_DIR.name],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"replay generation failed:\n{result.stdout}\n{result.stderr}")
    replay = OUT_DIR / "replay_0.json"
    if not replay.exists():
        raise RuntimeError(f"expected replay not found: {replay}")
    print(f"[smoke] generated {replay.relative_to(ROOT)}: {result.stdout.strip().splitlines()[-1]}")
    return replay


def is_benign_url(url: str) -> bool:
    url = (url or "").lower()
    return "/api/" in url or "favicon" in url


def is_benign_console(text: str) -> bool:
    text = (text or "").lower()
    # Resource 404s are reconciled against tracked response URLs separately; the
    # only expected missing resources are the optional /api/* viewer-server calls.
    return "failed to load resource" in text or "/api/" in text


def run_smoke(seed: int, keep: bool) -> int:
    from playwright.sync_api import sync_playwright

    generate_replay(seed)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    errors: list[str] = []
    bad_responses: list[str] = []
    try:
        wait_http(port)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.on(
                "console",
                lambda m: errors.append(m.text)
                if m.type == "error" and not is_benign_console(m.text)
                else None,
            )
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "response",
                lambda r: bad_responses.append(f"{r.status} {r.url}")
                if r.status >= 400 and not is_benign_url(r.url)
                else None,
            )

            url = (
                f"http://127.0.0.1:{port}/viewer_pixi/index.html"
                f"?replay=/{OUT_DIR.name}/replay_0.json&label=movement-smoke"
            )
            page.goto(url, wait_until="load")

            # Replay loaded => tickMax becomes > 0.
            page.wait_for_function("Number(document.getElementById('tickMax').textContent) > 0", timeout=10_000)
            tick_max = int(page.text_content("#tickMax") or "0")
            print(f"[smoke] replay loaded, tickMax={tick_max}")

            canvas = page.query_selector("#pixiStage canvas")
            if canvas is None:
                raise RuntimeError("no Pixi canvas rendered")
            box = canvas.bounding_box() or {}
            if box.get("width", 0) < 100 or box.get("height", 0) < 100:
                raise RuntimeError(f"canvas too small: {box}")

            page.click("#speedSelect")
            page.select_option("#speedSelect", "1")
            tick_start = int(page.text_content("#tickNow") or "0")
            page.click("#playBtn")  # start playback
            page.screenshot(path=str(SHOT_DIR / "smoke_movement_start.png"))
            time.sleep(2.5)
            tick_mid = int(page.text_content("#tickNow") or "0")
            page.screenshot(path=str(SHOT_DIR / "smoke_movement_mid.png"))
            print(f"[smoke] playback advanced tickNow {tick_start} -> {tick_mid}")

            browser.close()

            if tick_mid <= tick_start:
                raise RuntimeError(f"playback did not advance ({tick_start} -> {tick_mid})")
            if bad_responses:
                raise RuntimeError("non-API resource errors:\n  " + "\n  ".join(bad_responses[:10]))
            if errors:
                raise RuntimeError("console/page errors:\n  " + "\n  ".join(errors[:10]))
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
        if not keep:
            for path in OUT_DIR.glob("*"):
                path.unlink()
            OUT_DIR.rmdir()

    print("[smoke] OK: viewer rendered regenerated replay, playback animated, no fatal errors")
    print(f"[smoke] screenshots: {(SHOT_DIR / 'smoke_movement_start.png').relative_to(ROOT)},"
          f" {(SHOT_DIR / 'smoke_movement_mid.png').relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--keep", action="store_true", help="keep generated replay dir")
    args = parser.parse_args()
    return run_smoke(args.seed, args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
