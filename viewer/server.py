from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import PLAYER_IDS
from src.core.runner import run_match
from src.bots.policy_bot import default_policy_checkpoint


REPLAY_DIR = ROOT / "replays_viewer_debug_smoke"
MAX_UI_RUNS = 100
REPLAY_NAME = re.compile(r"^replay_\d+\.json$")


def available_models() -> list[dict]:
    models = [
        {"id": "random", "label": "Random bot"},
        {"id": "scripted", "label": "Scripted bot"},
    ]
    best_checkpoint = default_policy_checkpoint()
    if best_checkpoint is not None:
        models.append({
            "id": "ppo_random_v1",
            "label": f"PPO random v1 ({best_checkpoint.name})",
        })
    latest_checkpoint = ROOT / "training_runs/sprint2_random_first_real_v1/checkpoints/latest.pt"
    if latest_checkpoint.exists():
        models.append({
            "id": "ppo_random_v1_latest",
            "label": "PPO random v1 latest",
        })
    return models


def replay_files() -> list[dict]:
    if not REPLAY_DIR.exists():
        return []
    files = []
    for path in sorted(REPLAY_DIR.iterdir(), key=lambda item: item.name):
        if path.is_file() and REPLAY_NAME.match(path.name):
            files.append({
                "label": path.name,
                "path": f"../{REPLAY_DIR.name}/{path.name}",
            })
    return files


def clear_replays() -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for path in REPLAY_DIR.iterdir():
        if path.is_file() and REPLAY_NAME.match(path.name):
            path.unlink()


def generate_replays(payload: dict) -> dict:
    runs = int(payload.get("runs", 1) or 1)
    seed = int(payload.get("seed", 20260602) or 20260602)
    if runs < 1 or runs > MAX_UI_RUNS:
        raise ValueError(f"runs must be between 1 and {MAX_UI_RUNS}")

    allowed = {model["id"] for model in available_models()}
    players = payload.get("players", {}) or {}
    player_models = {}
    for player_id in PLAYER_IDS:
        model_id = str(players.get(player_id, "random"))
        if model_id not in allowed:
            raise ValueError(f"Unknown model for {player_id}: {model_id}")
        player_models[player_id] = model_id

    clear_replays()
    results = []
    winners: dict[str, int] = {}
    winner_types: dict[str, int] = {}
    for index in range(runs):
        replay_path = REPLAY_DIR / f"replay_{index}.json"
        result = run_match(
            seed=seed + index,
            bot_type="mixed",
            replay_path=str(replay_path),
            mode="replay",
            player_models=player_models,
        )
        winners[result.winner] = winners.get(result.winner, 0) + 1
        winner_types[result.winner_type] = winner_types.get(result.winner_type, 0) + 1
        results.append({
            "index": index,
            "seed": result.seed,
            "winner": result.winner,
            "winner_type": result.winner_type,
            "ticks": result.total_ticks,
            "path": f"../{REPLAY_DIR.name}/replay_{index}.json",
        })

    return {
        "runs": runs,
        "seed": seed,
        "players": player_models,
        "winners": winners,
        "winner_types": winner_types,
        "results": results,
        "replays": replay_files(),
    }


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/models":
            self._send_json({
                "players": PLAYER_IDS,
                "models": available_models(),
                "max_ui_runs": MAX_UI_RUNS,
                "replays": replay_files(),
            })
            return
        if path == "/api/replays":
            self._send_json({"replays": replay_files()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/generate-replays":
            self.send_error(404, "Unknown API endpoint")
            return
        try:
            payload = self._read_json()
            self._send_json(generate_replays(payload))
        except Exception as error:
            self._send_json({"error": str(error)}, status=400)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local replay viewer server with generation API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ViewerRequestHandler)
    print(f"Viewer server: http://{args.host}:{args.port}/viewer/index.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
