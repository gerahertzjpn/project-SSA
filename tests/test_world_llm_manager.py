"""Gated LLM manager layer: gating, heuristic fallbacks + notifications, the 🔴 L2
live-auction guard, and the LLM-on path (with a fake client)."""
import contextlib

import pytest

from src.world import events, feature_flags, llm_manager, playbook, season
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


@contextlib.contextmanager
def _llm_on():
    feature_flags.FLAGS["llm_manager"]["enabled"] = True
    try:
        yield
    finally:
        feature_flags.FLAGS["llm_manager"]["enabled"] = False


class FakeLLM:
    """A stand-in client: returns a canned answer or raises."""
    model = "fake"

    def __init__(self, answer=None, boom=False):
        self.answer, self.boom = answer, boom

    def structured(self, system, prompt, schema, max_tokens=900):
        if self.boom:
            raise RuntimeError("LLM exploded")
        return self.answer


def _bootstrapped():
    w = new_world("w-llm", FRANCHISES, seed=3)
    season.bootstrap_world(w, seed=3)
    return w


# ----------------------------------------------------------------- gating -- #
def test_draft_pick_fn_is_none_when_gated():
    w = _bootstrapped()
    # flag locked by default -> no LLM draft, heuristic is used (None pick_fn)
    assert llm_manager.draft_pick_fn(w, llm=FakeLLM()) is None


def test_author_playbook_heuristic_when_gated_emits_event():
    w = _bootstrapped()
    feed = events.chronicle()
    team = next(t for t in w["teams"].values() if len(t["roster"]) >= 3)
    plan = llm_manager.author_playbook(w, team, llm=FakeLLM(), feed=feed)
    assert plan["base"]["strategy"] in playbook.STRATEGY_IDS
    assert w["teams"][team["team_id"]]["tournament_plan"] is plan      # stored
    pb = [e for e in feed if e["kind"] == "playbook"]
    assert pb and pb[0]["source"] == "эвристика"                       # heuristic source noted


# ------------------------------------------------------- fallback + notify -- #
def test_author_playbook_llm_failure_falls_back_and_notifies():
    w = _bootstrapped()
    feed = events.chronicle()
    team = next(t for t in w["teams"].values() if len(t["roster"]) >= 3)
    with _llm_on():
        plan = llm_manager.author_playbook(w, team, llm=FakeLLM(boom=True), feed=feed)
    assert plan["base"]["strategy"] in playbook.STRATEGY_IDS           # heuristic plan
    notes = events.fallbacks(feed)
    assert notes and "playbook" in notes[0]["where"]                  # never silent


def test_author_playbook_uses_llm_when_active():
    w = _bootstrapped()
    team = next(t for t in w["teams"].values() if len(t["roster"]) >= 3)
    # LLM authors base strategy + an opponent-conditional SUBSTITUTION (player at A1)
    sub_pid = team["roster"][-1]
    answer = {"base": {"strategy": "turtle"},
              "overrides": [{"condition": "opp_robots", "strategy": "core_rush",
                             "lineup": {"A1": sub_pid}, "note": "vs bots"}]}
    with _llm_on():
        plan = llm_manager.author_playbook(w, team, llm=FakeLLM(answer=answer))
    assert plan["base"]["strategy"] == "turtle"
    assert plan["overrides"][0]["when"] == {"opp_faction_majority": "robots"}
    assert plan["overrides"][0]["strategy"] == "core_rush"
    assert plan["overrides"][0]["lineup"] == {"A1": sub_pid}      # substitution honored


def test_llm_bad_strategy_is_rejected_and_falls_back():
    w = _bootstrapped()
    feed = events.chronicle()
    team = next(t for t in w["teams"].values() if len(t["roster"]) >= 3)
    bad = {"base": {"strategy": "not_a_strategy"}, "overrides": []}
    with _llm_on():
        plan = llm_manager.author_playbook(w, team, llm=FakeLLM(answer=bad), feed=feed)
    assert plan["base"]["strategy"] in playbook.STRATEGY_IDS           # validated -> fallback
    assert events.fallbacks(feed)


def test_draft_pick_fn_validates_and_falls_back_on_unknown_id():
    w = _bootstrapped()
    feed = events.chronicle()
    with _llm_on():
        pick = llm_manager.draft_pick_fn(w, llm=FakeLLM(answer={"player_id": "ghost", "reason": "x"}),
                                         feed=feed)
        team = next(t for t in w["teams"].values())
        available = list(w["players"].values())[:5]
        assert pick(team, available) is None                          # unknown id -> fallback
    assert events.fallbacks(feed)


# ------------------------------------------------------------ market (L1) -- #
def test_heuristic_market_lists_weakest_for_churny_manager():
    from src.world import manager
    w = _bootstrapped()
    team = next(t for t in w["teams"].values() if len(t["roster"]) >= 3)
    # force a bench + low loyalty -> lists weakest player
    extra = w["free_agents"][0]
    w["players"][extra].update(status="rostered", team_id=team["team_id"])
    team["roster"].append(extra)
    prof = manager.ManagerProfile(aggression=.5, frugality=.5, youth_focus=.5, risk=.5,
                                  patience=.5, loyalty=0.1)
    plan = llm_manager._heuristic_market(w, team, prof)
    assert plan["spend"] in ("skill_up", "meta_adapt", "teamplay", "create_player")
    assert len(plan["sell"]) == 1


def test_auction_max_bids_reflect_character_and_budget():
    w = _bootstrapped()
    tids = [t["team_id"] for t in list(w["teams"].values())[:3]]
    for tid in tids:
        w["teams"][tid]["treasury"] = 1000
    bids = llm_manager.auction_max_bids(w, tids, item_value=500)
    assert set(bids) == set(tids)
    assert all(0 <= b <= 1000 for b in bids.values())


# ------------------------------------------------- 🔴 L2 live-auction guard -- #
def test_live_auction_locked_by_default():
    assert llm_manager.live_auction_available() is False
    with pytest.raises(feature_flags.FeatureLocked):
        llm_manager.assert_live_auction(confirmed=True)


def test_live_auction_requires_explicit_confirmation():
    feature_flags.FLAGS["live_auction_l2"]["enabled"] = True
    try:
        with pytest.raises(PermissionError):
            llm_manager.assert_live_auction(confirmed=False)       # must press YES
        msg = llm_manager.assert_live_auction(confirmed=True)
        assert "L2" in msg
    finally:
        feature_flags.FLAGS["live_auction_l2"]["enabled"] = False
