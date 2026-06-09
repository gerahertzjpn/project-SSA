import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_http(url: str, timeout: float = 5.0) -> bytes:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return response.read()
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"HTTP server did not answer: {url}")


def test_legacy_game_viewer_route_serves_pixi_redirect():
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        body = wait_for_http(f"http://127.0.0.1:{port}/viewer_game/index.html").decode("utf-8")
        assert "Legacy canvas game UI is intentionally disabled" in body
        assert "../viewer_pixi/index.html" in body
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()
