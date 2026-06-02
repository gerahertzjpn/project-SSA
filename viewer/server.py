from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import PLAYER_IDS
from src.core.replay import REPLAY_SCHEMA_VERSION
from src.core.runner import current_code_commit, rules_meta, run_match
from src.bots.policy_bot import default_policy_checkpoint, policy_model_id_for_checkpoint


REPLAY_DIR = ROOT / "replays_viewer_debug_smoke"
MAX_UI_RUNS = 100
REPLAY_NAME = re.compile(r"^replay_\d+\.json$")
SERVER_STARTED_AT = int(time.time())
SERVER_COMMIT = current_code_commit()
SERVER_SOURCE_FINGERPRINT = ""


def source_fingerprint() -> str:
    paths = [ROOT / "viewer/server.py"]
    src_root = ROOT / "src"
    if src_root.exists():
        paths.extend(src_root.rglob("*.py"))
    parts = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            parts.append(f"{path.relative_to(ROOT).as_posix()}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            continue
    return "|".join(parts)


SERVER_SOURCE_FINGERPRINT = source_fingerprint()


def available_models() -> list[dict]:
    models = [
        {"id": "random", "label": "Random bot"},
        {"id": "scripted", "label": "Scripted bot"},
    ]
    seen_checkpoints: set[Path] = set()
    best_checkpoint = default_policy_checkpoint()
    if best_checkpoint is not None:
        seen_checkpoints.add(best_checkpoint.resolve())
        models.append({
            "id": "ppo_random_v1",
            "label": f"PPO random v1 ({best_checkpoint.name})",
        })
    latest_checkpoint = ROOT / "training_runs/sprint2_random_first_real_v1/checkpoints/latest.pt"
    if latest_checkpoint.exists():
        seen_checkpoints.add(latest_checkpoint.resolve())
        models.append({
            "id": "ppo_random_v1_latest",
            "label": "PPO random v1 latest",
        })
    for checkpoint in discover_policy_checkpoints():
        resolved = checkpoint.resolve()
        if resolved in seen_checkpoints:
            continue
        seen_checkpoints.add(resolved)
        models.append({
            "id": policy_model_id_for_checkpoint(checkpoint),
            "label": f"Policy {checkpoint.relative_to(ROOT).as_posix()}",
        })
    return models


def discover_policy_checkpoints() -> list[Path]:
    root = ROOT / "training_runs"
    if not root.exists():
        return []
    checkpoints = [
        path
        for path in root.glob("*/checkpoints/*.pt")
        if path.is_file()
    ]
    return sorted(checkpoints, key=lambda path: (path.parent.parent.name, path.name))


def server_info() -> dict:
    current_commit = current_code_commit()
    current_fingerprint = source_fingerprint()
    stale = (
        SERVER_COMMIT != "unknown"
        and current_commit != "unknown"
        and SERVER_COMMIT != current_commit
    ) or SERVER_SOURCE_FINGERPRINT != current_fingerprint
    return {
        "commit": SERVER_COMMIT,
        "current_commit": current_commit,
        "stale": stale,
        "source_fingerprint": SERVER_SOURCE_FINGERPRINT[:24],
        "current_source_fingerprint": current_fingerprint[:24],
        "started_at": SERVER_STARTED_AT,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "rules": rules_meta(),
    }


def replay_files() -> list[dict]:
    if not REPLAY_DIR.exists():
        return []
    files = []
    for path in sorted(REPLAY_DIR.iterdir(), key=replay_sort_key):
        if path.is_file() and REPLAY_NAME.match(path.name):
            files.append({
                "label": path.name,
                "path": f"../{REPLAY_DIR.name}/{path.name}",
            })
    return files


def replay_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    index = int(match.group(1)) if match else 0
    return index, path.name


def clear_replays() -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for path in REPLAY_DIR.iterdir():
        if path.is_file() and REPLAY_NAME.match(path.name):
            path.unlink()


def generate_replays(payload: dict) -> dict:
    info = server_info()
    if info["stale"]:
        raise RuntimeError(
            f"Viewer API is running old code ({info['commit']}); restart viewer for {info['current_commit']}"
        )

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
        "server": info,
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
                "server": server_info(),
            })
            return
        if path == "/api/replays":
            self._send_json({"replays": replay_files(), "server": server_info()})
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
        except RuntimeError as error:
            self._send_json({"error": str(error), "server": server_info()}, status=409)
        except Exception as error:
            self._send_json({"error": str(error), "server": server_info()}, status=400)

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
