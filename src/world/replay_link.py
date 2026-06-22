"""Link an already-played World match to its REAL on-disk replay files.

Going forward, ``net_match.make_net_play_fn`` records each map's replay filename directly
on the series (``map_results``). But seasons played before that wiring stored only the map
*winners* (``sequence``), not which replay file each map produced — yet the real replays are
still on disk (``data/world/<wid>/replays/league_<seed>.json``). This module recovers the
mapping best-effort, so the broadcast can play the actual match instead of animating numbers.

How: every league replay's ``meta`` records ``player_models`` (A1-3 = home lineup nets,
B1-3 = away) and ``winner`` (TEAM_A/TEAM_B). A revealed series knows its home/away lineup
(from the stored playbook) and each map's winning SIDE (from ``sequence``). So we find the
replay files whose A-side == this series' home nets and B-side == away nets, then hand them
out in winner order. Reading only ``meta`` (not the megabytes of ticks) keeps the index cheap;
it is cached to ``replays/_meta_index.json`` and refreshed incrementally by file size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.world import net_match

_INDEX_NAME = "_meta_index.json"
_MEM_CACHE: dict[str, dict] = {}          # replays_dir -> {"files": {name: size}, "index": {name: rec}}


# --------------------------------------------------------------------------- #
# Cheap meta reader — parse the leading ``"meta"`` object, skip the huge ticks
# --------------------------------------------------------------------------- #
def _read_meta(path: Path) -> Optional[dict]:
    """Return a replay's ``meta`` dict without parsing its (megabytes of) ticks.

    Replays are dumped ``{"meta": {...}, "ticks": [...]}`` with meta first, so we read a
    leading chunk, cut it just before the ``"ticks":`` key and parse only that prefix. Falls
    back to a full parse if the meta is unexpectedly large or the layout differs."""
    try:
        head = path.read_bytes()[:524288].decode("utf-8", "ignore")   # 512 KB is ample for meta
    except OSError:
        return None
    cut = head.find('"ticks":')          # safe: meta's only ticks-ish key is "total_ticks" (no leading quote)
    if cut != -1:
        prefix = head[:cut].rstrip().rstrip(",")    # -> '{"meta": {...}'
        try:
            return json.loads(prefix + "}").get("meta")
        except json.JSONDecodeError:
            pass
    try:                                  # fallback: full parse (rare)
        return json.loads(path.read_text(encoding="utf-8")).get("meta")
    except (OSError, json.JSONDecodeError):
        return None


def _meta_record(meta: dict) -> dict:
    pm = meta.get("player_models") or {}
    return {
        "a": sorted(v for s in net_match.HOME_SLOTS if (v := pm.get(s))),
        "b": sorted(v for s in net_match.AWAY_SLOTS if (v := pm.get(s))),
        "winner": meta.get("winner"),
    }


# --------------------------------------------------------------------------- #
# Index — {filename: {a:[models], b:[models], winner}} cached by file size
# --------------------------------------------------------------------------- #
def build_index(replays_dir: str | Path, *, refresh: bool = False) -> dict:
    """Index every ``league_*.json`` replay by its lineup model-ids + winner. Cached in memory
    and on disk (``replays/_meta_index.json``); only newly-added/resized files are re-read."""
    rdir = Path(replays_dir)
    key = str(rdir.resolve())
    if not refresh and key in _MEM_CACHE:
        return _MEM_CACHE[key]["index"]
    if not rdir.is_dir():
        _MEM_CACHE[key] = {"files": {}, "index": {}}
        return {}

    disk = {}
    cache_path = rdir / _INDEX_NAME
    if cache_path.exists() and not refresh:
        try:
            disk = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            disk = {}
    known_sizes = disk.get("files", {}) if isinstance(disk, dict) else {}
    index = dict(disk.get("index", {})) if isinstance(disk, dict) else {}

    files: dict[str, int] = {}
    changed = False
    for f in rdir.glob("league_*.json"):
        size = f.stat().st_size
        files[f.name] = size
        if known_sizes.get(f.name) == size and f.name in index:
            continue                       # unchanged -> reuse cached record
        meta = _read_meta(f)
        if meta is not None:
            index[f.name] = _meta_record(meta)
            changed = True
    # drop records for files that disappeared
    for stale in [n for n in index if n not in files]:
        index.pop(stale, None)
        changed = True

    if changed or known_sizes != files:
        try:
            cache_path.write_text(json.dumps({"files": files, "index": index}), encoding="utf-8")
        except OSError:
            pass
    _MEM_CACHE[key] = {"files": files, "index": index}
    return index


# --------------------------------------------------------------------------- #
# Resolve a series' home/away lineup -> manifest model-ids
# --------------------------------------------------------------------------- #
def _series_models(world: dict, series: dict, manifest_path: Optional[str | Path]):
    """(home_models, away_models) as sorted ``policy:<ckpt>`` lists for the series' stored lineup,
    or (None, None) if the lineup/nets can't be resolved."""
    pb = series.get("playbook")
    if not isinstance(pb, dict):
        return None, None
    idx = net_match.net_model_index(net_match.load_manifest(manifest_path))
    players = world.get("players", {})

    def models_for(side: str):
        lineup = (pb.get(side) or {}).get("lineup") or {}
        out = []
        for pid in lineup.values():
            net_id = players.get(pid, {}).get("net_id")
            mid = idx.get(net_id) if net_id else None
            if not mid:
                return None
            out.append(mid)
        return sorted(out) if out else None

    return models_for("home"), models_for("away")


# --------------------------------------------------------------------------- #
# Public: replay filename per decisive map of a series
# --------------------------------------------------------------------------- #
def link_series(world: dict, series: dict, replays_dir: str | Path,
                *, manifest_path: Optional[str | Path] = None,
                used: Optional[set] = None) -> list[Optional[str]]:
    """Best-effort replay filename for each decisive map of ``series`` (aligned to its
    ``sequence``). Prefers the link recorded at play time (``map_results``); otherwise matches
    on-disk replays by lineup orientation + winner side. ``used`` (optional) dedupes filenames
    across sibling series so the same replay isn't handed to two matches. Entries are ``None``
    where no confident match exists.

    Recovery limitation: a season played BEFORE ``map_results`` existed stored only the *last*
    game's lineup (``playbook``). If the manager substituted mid-series, the earlier games used a
    different lineup and won't match the strict orientation check — those maps return ``None`` and
    the broadcast falls back to the score for them. This degrades gracefully and never mislinks: a
    returned file is always a real game of THIS exact matchup + lineup + winner. New seasons store
    the per-map link directly, so substitutions are a non-issue going forward."""
    seq = series.get("sequence") or []

    # Fast path: the link was recorded at play time. Only trust it when every decisive map has a
    # real (string) replay filename — a None/garbage entry means it wasn't actually recorded, so
    # fall through to recovery rather than handing the broadcast a broken URL.
    mr = series.get("map_results")
    if isinstance(mr, list) and mr:
        decisive = [m.get("replay") for m in mr if m.get("winner") is not None]
        if len(decisive) == len(seq) and all(isinstance(x, str) and x for x in decisive):
            return decisive

    home, away = series.get("home"), series.get("away")
    home_models, away_models = _series_models(world, series, manifest_path)
    if not home_models or not away_models:
        return [None] * len(seq)

    index = build_index(replays_dir)
    used = used if used is not None else set()
    # candidate files whose A-side == this series' home nets and B-side == away nets
    cands_by_side: dict[str, list[str]] = {"TEAM_A": [], "TEAM_B": []}
    for name, rec in index.items():
        if name in used:
            continue
        if rec.get("a") == home_models and rec.get("b") == away_models:
            cands_by_side.setdefault(rec.get("winner"), []).append(name)

    out: list[Optional[str]] = []
    for winner_team in seq:
        side = "TEAM_A" if winner_team == home else "TEAM_B" if winner_team == away else None
        pool = cands_by_side.get(side, []) if side else []
        pick = pool.pop(0) if pool else None
        if pick:
            used.add(pick)
        out.append(pick)
    return out
