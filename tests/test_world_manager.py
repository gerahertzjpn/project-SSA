"""World manager (front office): profiles, reworked character, decisions, coach pick."""
from src.world import coach, manager, valuation
from src.world.manager import ManagerProfile
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}",
               "manager": f"Mgr {i}"} for i in range(22)]


def _profile(**over):
    base = dict(aggression=0.5, frugality=0.5, youth_focus=0.5, risk=0.5,
                patience=0.5, coach_priority=0.5, loyalty=0.5)
    base.update(over)
    return ManagerProfile(**base)


# -------------------------------------------------------------- character -- #
def test_rules_are_front_office_not_tactics():
    r = manager.MANAGER_SYSTEM_RULES
    assert "ТРЕНЕР" in r and "тренеров" in r          # picks coaches, coach owns tactics
    assert "верхней лиг" in r                          # always aims for the upper title
    # no in-game order vocabulary leaks into the manager's job
    for tactic in ("tempo", "win_path", "core_rush", "turtle"):
        assert tactic not in r


def test_natural_profile_deterministic_and_diverse():
    p1 = manager.natural_profile("iron-falcons")
    assert manager.natural_profile("iron-falcons").as_dict() == p1.as_dict()  # stable
    aggressions = {round(manager.natural_profile(f["franchise_id"]).aggression, 2)
                   for f in FRANCHISES}
    assert len(aggressions) > 8                         # spreads on its own


def test_persona_describes_management_not_playstyle():
    fr = {"name": "Team X", "manager": "Иван"}
    blurb = manager.manager_persona(fr, _profile(aggression=0.9, youth_focus=0.9))
    assert "Иван" in blurb and "не вмешиваешься" in blurb
    for tactic in ("tempo", "win_path", "core_rush"):
        assert tactic not in blurb


# --------------------------------------------------------------- decisions- #
def test_aggressive_pays_and_bids_more_than_frugal():
    aggro, frugal = _profile(aggression=0.95, frugality=0.05), _profile(aggression=0.05, frugality=0.95)
    assert manager.max_offer(aggro, 1000, 10_000) > manager.max_offer(frugal, 1000, 10_000)
    assert manager.auction_bid(aggro, 1000, 10_000) > manager.auction_bid(frugal, 1000, 10_000)


def test_offers_never_absurd():
    # even a maximally aggressive manager stays within a sane multiple of fair value
    assert manager.max_offer(_profile(aggression=1, risk=1, frugality=0), 1000, 10**9) <= 1700


def test_loyal_manager_holds_out_for_more():
    w = new_world("w-sell", FRANCHISES, seed=1)
    tid = list(w["teams"])[0]
    pids = list(w["players"])[:4]
    for pid, r in zip(pids, [1400, 1380, 1360, 900]):
        w["players"][pid].update(status="rostered", team_id=tid, soloq_rating=float(r))
        w["teams"][tid]["roster"].append(pid)
    bench = w["players"][pids[3]]
    import src.world.transfers as transfers
    fair = transfers.fair_value(bench)
    loyal = manager.wants_to_sell(w, w["teams"][tid], bench, fair, _profile(loyalty=0.9))
    disloyal = manager.wants_to_sell(w, w["teams"][tid], bench, fair, _profile(loyalty=0.1))
    assert loyal["decision"] is False and disloyal["decision"] is True


def test_certificate_intent_reflects_character():
    assert manager.certificate_intent(_profile(youth_focus=0.9), has_fork_budget=True) == "create_player"
    assert manager.certificate_intent(_profile(youth_focus=0.1, aggression=0.9), True) == "meta_adapt"
    assert manager.certificate_intent(_profile(aggression=0.1, patience=0.9), False) == "teamplay"


# ------------------------------------------------------------- coach pick -- #
def test_pick_coach_prefers_best_and_respects_priority():
    w = new_world("w-coach", FRANCHISES, seed=2)
    tid = list(w["teams"])[0]
    cands = list(w["free_coaches"])[:3]
    w["coaches"][cands[0]]["rating"] = 1100
    w["coaches"][cands[1]]["rating"] = 1000
    w["coaches"][cands[2]]["rating"] = 900
    # no current coach -> takes the strongest
    assert manager.pick_coach(w, w["teams"][tid], cands, _profile(coach_priority=0.9)) == cands[0]
    # with a decent current coach, a low-priority manager won't bother for a small gain
    coach.hire_coach(w, tid, cands[1])               # current rating 1000
    w["coaches"][cands[0]]["rating"] = 1010          # tiny upgrade
    assert manager.pick_coach(w, w["teams"][tid], [cands[0]], _profile(coach_priority=0.05)) is None


def test_run_coaching_stage_assigns_coaches():
    w = new_world("w-cs", FRANCHISES, seed=3)
    moves = manager.run_coaching_stage(w)
    assert len(moves) > 0
    assert all(w["teams"][m["team_id"]]["coach"] == m["hired_coach"] for m in moves)
