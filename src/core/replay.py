import json
from pathlib import Path

REPLAY_SCHEMA_VERSION = "0.4.1"
COMBAT_TRACE_FORMAT = "compact_v3"


def save_replay(
    winner: str,
    total_ticks: int,
    ticks: list,
    path: str,
    extra_meta: dict | None = None,
    pretty: bool = False,
) -> None:
    meta = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "winner": winner,
        "total_ticks": total_ticks,
        "combat_trace_format": COMBAT_TRACE_FORMAT,
    }
    if extra_meta:
        meta.update(extra_meta)

    replay = {
        "meta": meta,
        "ticks": ticks,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(replay, f, ensure_ascii=False, indent=2)
        else:
            json.dump(replay, f, ensure_ascii=False, separators=(",", ":"))


def load_replay(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
