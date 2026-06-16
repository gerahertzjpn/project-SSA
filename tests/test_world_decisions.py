"""Player valuation + manager decisions (buy/sell/rookie) + rookie generation."""
import pytest

from src.world import rookies, transfers, valuation
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _roster(world, tid, pids, soloqs):
    for pid, r in zip(pids, soloqs):
        p = world["players"][pid]
        p["status"] = "rostered"; p["team_id"] = tid; p["soloq_rating"] = float(r)
        world["teams"][tid]["roster"].append(pid)
        if pid in world["free_agents"]:
            world["free_agents"].remove(pid)


# --------------------------------------------------------------- valuation- #
def test_skill_score_tracks_rating():
    hi = valuation.skill_score({"soloq_rating": 1400, "attributes": {}})
    lo = valuation.skill_score({"soloq_rating": 800, "attributes": {}})
    assert hi > lo


def test_should_buy_upgrade_affordable_true():
    w = new_world("w-buy", FRANCHISES, seed=1)
    tid = list(w["teams"])[0]
    pids = list(w["players"])[:3]
    _roster(w, tid, pids, [950, 940, 930])           # weak starters
    w["teams"][tid]["treasury"] = 100_000
    target = w["players"][list(w["players"])[10]]
    target["soloq_rating"] = 1400                    # clear upgrade
    price = transfers.fair_value(target)
    res = valuation.should_buy(w, w["teams"][tid], target, price)
    assert res["decision"] is True and "upgrade" in res["reason"]


def test_should_buy_overpriced_and_unaffordable():
    w = new_world("w-buy2", FRANCHISES, seed=2)
    tid = list(w["teams"])[0]
    _roster(w, tid, list(w["players"])[:3], [950, 940, 930])
    target = w["players"][list(w["players"])[10]]
    target["soloq_rating"] = 1400
    fair = transfers.fair_value(target)
    w["teams"][tid]["treasury"] = 100_000
    assert valuation.should_buy(w, w["teams"][tid], target, fair * 3)["decision"] is False  # overpriced
    w["teams"][tid]["treasury"] = 1
    assert valuation.should_buy(w, w["teams"][tid], target, fair)["decision"] is False       # unaffordable


def test_should_sell_surplus_yes_starter_no():
    w = new_world("w-sell", FRANCHISES, seed=3)
    tid = list(w["teams"])[0]
    pids = list(w["players"])[:4]
    _roster(w, tid, pids, [1400, 1380, 1360, 900])   # 3 strong starters + 1 weak bench
    w["teams"][tid]["treasury"] = 100_000
    bench = w["players"][pids[3]]
    star = w["players"][pids[0]]
    sell = valuation.should_sell(w, w["teams"][tid], bench, transfers.fair_value(bench))
    keep = valuation.should_sell(w, w["teams"][tid], star, transfers.fair_value(star))
    assert sell["decision"] is True and keep["decision"] is False


def test_should_draft_rookie_rules():
    w = new_world("w-rk", FRANCHISES, seed=4)
    tid = list(w["teams"])[0]
    team = w["teams"][tid]
    team["league"] = "upper"
    rookie = w["players"][list(w["players"])[0]]
    assert valuation.should_draft_rookie(w, team, rookie)["decision"] is False  # upper can't
    team["league"] = "lower"
    assert valuation.should_draft_rookie(w, team, rookie)["decision"] is True   # squad short


# ----------------------------------------------------------------- rookies- #
def test_eligible_parents_excludes_top():
    w = new_world("w-par", FRANCHISES, seed=5)
    # make one obvious top player
    top = list(w["players"])[0]
    w["players"][top]["soloq_rating"] = 5000
    parents = rookies.eligible_parents(w)
    assert top not in parents
    assert len(parents) < len(w["players"])  # top fraction removed


def test_generate_rookies_creates_raw_high_upside_cards():
    w = new_world("w-gen", FRANCHISES, seed=6)
    budget = rookies.RookieBudget(n_rookies=4, steps_per_rookie=1_000_000, extra_minutes=30)
    rep = rookies.generate_rookies(w, budget, seed=6)
    assert rep["n"] == 4 and rep["total_steps"] == 4_000_000 and rep["extra_minutes"] == 30
    elig = set(rookies.eligible_parents(w))
    seeds = set()
    for rid in rep["created"]:
        p = w["players"][rid]
        assert p["status"] == "rookie"
        assert p["lineage"]["origin"] == "rookie_fork"
        assert p["lineage"]["parent_id"] in elig          # never forked from a top player
        assert p["soloq_rating"] == 930.0                 # raw, must prove itself
        assert p["attributes"]["potential"] >= 60         # high upside
        seeds.add(p["lineage"]["seed"])
    assert len(seeds) == 4                                 # each rookie a unique seed
    assert set(rep["created"]).issubset(set(w["rookies_pool"]))


def test_rookie_budget_enforces_floor():
    with pytest.raises(ValueError):
        rookies.RookieBudget(n_rookies=1, steps_per_rookie=500_000)  # below 800k floor


def test_top_prospects_by_upside():
    w = new_world("w-top", FRANCHISES, seed=7)
    rookies.generate_rookies(w, rookies.RookieBudget(n_rookies=6), seed=7)
    top = rookies.top_prospects(w, k=3)
    assert len(top) == 3
    pots = [w["players"][r]["attributes"]["potential"] for r in top]
    assert pots == sorted(pots, reverse=True)
