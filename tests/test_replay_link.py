"""Tests for src.world.replay_link — recovering match -> real replay file links.

Pure-Python (no torch): fake replay files on disk + a synthetic world/manifest, exercising the
partial-meta reader, the cached index, lineup+winner matching, dedup, and the stored fast-path.
"""
import json

import pytest

from src.world import replay_link


# --------------------------------------------------------------------------- #
# Fixtures: a tiny manifest, world, and on-disk replays
# --------------------------------------------------------------------------- #
def _manifest():
    # 6 nets, checkpoint per net; net_model_index -> "policy:<checkpoint>"
    return {
        "descendants": [{"net_id": f"n{i}", "checkpoint": f"ck/n{i}.pt"} for i in range(6)],
        "ancestors": [],
    }


def _world():
    # players p0..p5 bound to nets n0..n5
    return {"players": {f"p{i}": {"net_id": f"n{i}", "faction": "humans"} for i in range(6)}}


def _series(home, away, seq, *, hp=("p0", "p1", "p2"), ap=("p3", "p4", "p5")):
    return {
        "home": home, "away": away, "sequence": list(seq),
        "playbook": {"home": {"lineup": {"A1": hp[0], "A2": hp[1], "A3": hp[2]}},
                     "away": {"lineup": {"B1": ap[0], "B2": ap[1], "B3": ap[2]}}},
    }


def _write_replay(path, a_nets, b_nets, winner, *, ticks=3):
    pm = {f"A{i+1}": f"policy:ck/{n}.pt" for i, n in enumerate(a_nets)}
    pm.update({f"B{i+1}": f"policy:ck/{n}.pt" for i, n in enumerate(b_nets)})
    doc = {"meta": {"player_models": pm, "winner": winner, "total_ticks": ticks},
           "ticks": [{"tick": t} for t in range(ticks)]}
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture()
def replays_dir(tmp_path):
    replay_link._MEM_CACHE.clear()                 # tests share the module-level cache
    d = tmp_path / "replays"
    d.mkdir()
    return d


# --------------------------------------------------------------------------- #
# Partial meta reader — must skip the (large) ticks array
# --------------------------------------------------------------------------- #
def test_read_meta_skips_ticks(replays_dir):
    f = replays_dir / "league_1.json"
    big = {"meta": {"player_models": {"A1": "policy:ck/n0.pt"}, "winner": "TEAM_A",
                    "total_ticks": 5000},
           "ticks": [{"tick": t, "blob": "x" * 50} for t in range(5000)]}
    f.write_text(json.dumps(big), encoding="utf-8")
    meta = replay_link._read_meta(f)
    assert meta is not None and meta["winner"] == "TEAM_A"
    assert meta["player_models"]["A1"] == "policy:ck/n0.pt"


# --------------------------------------------------------------------------- #
# Index build + cache
# --------------------------------------------------------------------------- #
def test_build_index_and_disk_cache(replays_dir):
    _write_replay(replays_dir / "league_10.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_A")
    idx = replay_link.build_index(replays_dir)
    assert "league_10.json" in idx
    rec = idx["league_10.json"]
    assert rec["a"] == sorted(["policy:ck/n0.pt", "policy:ck/n1.pt", "policy:ck/n2.pt"])
    assert rec["b"] == sorted(["policy:ck/n3.pt", "policy:ck/n4.pt", "policy:ck/n5.pt"])
    assert rec["winner"] == "TEAM_A"
    assert (replays_dir / replay_link._INDEX_NAME).exists()   # cached to disk


# --------------------------------------------------------------------------- #
# Linking by lineup orientation + per-map winner
# --------------------------------------------------------------------------- #
def test_link_series_matches_winner_order(replays_dir, tmp_path):
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps(_manifest()), encoding="utf-8")
    # series HOME vs AWAY, Bo3 sequence: home, away, home
    s = _series("HOME", "AWAY", ["HOME", "AWAY", "HOME"])
    # 2 home-wins (TEAM_A) + 1 away-win (TEAM_B), correctly oriented A=home nets, B=away nets
    _write_replay(replays_dir / "league_a1.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_A")
    _write_replay(replays_dir / "league_a2.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_A")
    _write_replay(replays_dir / "league_b1.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_B")
    # decoys that must NOT be picked: different nets, and reversed orientation
    _write_replay(replays_dir / "league_dec.json", ["n0", "n1", "n5"], ["n3", "n4", "n2"], "TEAM_A")
    _write_replay(replays_dir / "league_rev.json", ["n3", "n4", "n5"], ["n0", "n1", "n2"], "TEAM_A")

    links = replay_link.link_series(_world(), s, replays_dir, manifest_path=str(mani))
    assert len(links) == 3
    # map1 home-win -> a TEAM_A file; map2 away-win -> the TEAM_B file; map3 home-win -> the other TEAM_A
    assert links[1] == "league_b1.json"
    assert set([links[0], links[2]]) == {"league_a1.json", "league_a2.json"}
    assert "league_dec.json" not in links and "league_rev.json" not in links


def test_link_series_dedup_across_siblings(replays_dir, tmp_path):
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps(_manifest()), encoding="utf-8")
    _write_replay(replays_dir / "league_x1.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_A")
    _write_replay(replays_dir / "league_x2.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_A")
    used = set()
    s1 = _series("HOME", "AWAY", ["HOME"])
    s2 = _series("HOME", "AWAY", ["HOME"])
    l1 = replay_link.link_series(_world(), s1, replays_dir, manifest_path=str(mani), used=used)
    l2 = replay_link.link_series(_world(), s2, replays_dir, manifest_path=str(mani), used=used)
    assert l1[0] and l2[0] and l1[0] != l2[0]      # same lineup, but no file handed out twice


def test_link_series_prefers_stored_map_results(replays_dir, tmp_path):
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps(_manifest()), encoding="utf-8")
    s = _series("HOME", "AWAY", ["HOME", "AWAY"])
    s["map_results"] = [{"winner": "HOME", "replay": "league_stored1.json"},
                        {"winner": "AWAY", "replay": "league_stored2.json"}]
    # no matching files on disk at all -> must still use the stored link
    links = replay_link.link_series(_world(), s, replays_dir, manifest_path=str(mani))
    assert links == ["league_stored1.json", "league_stored2.json"]


def test_link_series_falls_through_on_unrecorded_map_results(replays_dir, tmp_path):
    """A map_results entry whose replay is None (not actually recorded) must NOT short-circuit the
    fast path — fall through to on-disk recovery instead of returning a broken None link."""
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps(_manifest()), encoding="utf-8")
    s = _series("HOME", "AWAY", ["HOME", "AWAY"])
    s["map_results"] = [{"winner": "HOME", "replay": None},          # not recorded
                        {"winner": "AWAY", "replay": None}]
    _write_replay(replays_dir / "league_h.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_A")
    _write_replay(replays_dir / "league_a.json", ["n0", "n1", "n2"], ["n3", "n4", "n5"], "TEAM_B")
    links = replay_link.link_series(_world(), s, replays_dir, manifest_path=str(mani))
    assert links == ["league_h.json", "league_a.json"]   # recovered from disk, not the None fast-path


def test_link_series_no_lineup_returns_nones(replays_dir, tmp_path):
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps(_manifest()), encoding="utf-8")
    s = {"home": "HOME", "away": "AWAY", "sequence": ["HOME", "AWAY"]}   # no playbook lineup
    links = replay_link.link_series(_world(), s, replays_dir, manifest_path=str(mani))
    assert links == [None, None]
