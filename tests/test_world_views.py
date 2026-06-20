"""Read-model view builders for the World UI (pure, no HTTP)."""
import json
import pathlib

import pytest

from src.world import season, views
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i:02d}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _manifest():
    p = pathlib.Path("population/frozen_population_v1.json")
    if not p.exists():
        pytest.skip("frozen population manifest not present")
    return json.loads(p.read_text(encoding="utf-8"))


def _world_after_a_season():
    w = new_world("w-view", FRANCHISES, seed=3, population=_manifest())
    season.bootstrap_world(w, seed=3)
    season.run_regular_season(w)
    return w


def test_ranked_players_all_ranked_with_delta():
    w = _world_after_a_season()
    ranked = views.ranked_players(w)
    assert len(ranked) == len(w["players"])           # EVERY player, not just rostered
    assert [v["rank"] for v in ranked] == list(range(1, len(ranked) + 1))
    assert all(ranked[i]["rating"] >= ranked[i + 1]["rating"] for i in range(len(ranked) - 1))
    # rich detail beyond "2 words + 1 number"
    top = ranked[0]
    for key in ("nickname", "faction", "archetype", "role", "team", "rating", "delta",
                "rating_history", "net_id", "games"):
        assert key in top
    # role filter works
    pros = views.ranked_players(w, role="pro")
    assert pros and all(v["role"] == "pro" for v in pros)


def test_league_tables_split_by_group_with_places():
    w = _world_after_a_season()
    tables = views.league_tables(w)
    assert {"upper", "lower_A", "lower_Z"} <= set(tables)
    assert len(tables["upper"]) == 10
    assert len(tables["lower_A"]) == 6 and len(tables["lower_Z"]) == 6
    # groups are NOT mixed: every lower_A row is group A, lower_Z is group Z
    assert all(r["group"] == "A" for r in tables["lower_A"])
    assert all(r["group"] == "Z" for r in tables["lower_Z"])
    # explicit, contiguous places per table
    assert [r["place"] for r in tables["upper"]] == list(range(1, 11))


def test_team_power_is_global_upper_generally_above_lower():
    import statistics
    w = _world_after_a_season()
    tables = views.league_tables(w)
    upper_pow = [r["power"] for r in tables["upper"]]
    lower_pow = [r["power"] for r in (tables["lower_A"] + tables["lower_Z"])]
    # GLOBAL power: upper teams are generally stronger (median), since they earned promotion;
    # this is NOT absolute (a strong lower team can out-power a weak upper one).
    assert statistics.median(upper_pow) > statistics.median(lower_pow)
    # power is the global number; raw intra-league elo is exposed separately
    assert all("elo" in r and "power" in r for r in tables["upper"])


def test_team_view_has_record_and_roster():
    w = _world_after_a_season()
    tid = next(t["team_id"] for t in w["teams"].values() if t.get("league") == "upper")
    tv = views.team_view(w, tid)
    assert tv["roster"] and all("rating" in p for p in tv["roster"])
    rec = tv["record"]
    assert rec["series_wins"] + rec["series_losses"] > 0     # played in the season
    assert "elo_history" in tv and "power" in tv


def test_schedule_view_regular_plus_bracket():
    w = _world_after_a_season()
    sv = views.schedule_view(w, season=w["history"][-1]["season"])
    assert sv["regular"], "regular-season fixtures present"
    assert all(m["day"] >= 1 and m["home_abbrev"] and m["away_abbrev"] for m in sv["regular"])
    # a playoff bracket exists for at least one league, ordered by stage
    assert sv["bracket"], "playoff bracket present"
    some = next(iter(sv["bracket"].values()))
    assert some and some[-1]["stage"] in views.PLAYOFF_STAGE_ORDER
