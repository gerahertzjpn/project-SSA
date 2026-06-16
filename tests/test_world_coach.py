"""Coach (RL tactical brain): pool, hiring, lineup, subs, position swaps, bonus."""
import pytest

from src.world import coach
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _world_with_team(n_players=4):
    w = new_world("w-coach", FRANCHISES, seed=1)
    tid = list(w["teams"])[0]
    pids = list(w["players"])[:n_players]
    for pid in pids:
        w["players"][pid]["status"] = "rostered"
        w["players"][pid]["team_id"] = tid
        w["teams"][tid]["roster"].append(pid)
        if pid in w["free_agents"]:
            w["free_agents"].remove(pid)
    return w, tid, pids


# ------------------------------------------------------------------ pool --- #
def test_world_creates_coach_pool():
    w = new_world("w-cp", FRANCHISES, seed=2)
    assert len(w["coaches"]) >= len(w["teams"])          # one per team + surplus
    assert set(w["free_coaches"]) == set(w["coaches"])   # all start as free agents
    assert all(c["status"] == "free_agent" for c in w["coaches"].values())


def test_assign_default_coaches_gives_each_team_one():
    w = new_world("w-cd", FRANCHISES, seed=3)
    coach.assign_default_coaches(w)
    assert all(t["coach"] for t in w["teams"].values())
    for t in w["teams"].values():
        assert w["coaches"][t["coach"]]["team_id"] == t["team_id"]


def test_hire_and_release_round_trips():
    w, tid, _ = _world_with_team()
    cid = w["free_coaches"][0]
    coach.hire_coach(w, tid, cid)
    assert w["teams"][tid]["coach"] == cid
    assert cid not in w["free_coaches"]
    released = coach.release_coach(w, tid)
    assert released == cid and w["teams"][tid]["coach"] is None
    assert cid in w["free_coaches"]


# ---------------------------------------------------------------- lineup --- #
def test_select_lineup_picks_three_with_positions_and_order():
    w, tid, pids = _world_with_team(n_players=4)
    coach.hire_coach(w, tid, w["free_coaches"][0])
    lineup = coach.select_lineup(w, tid)
    assert len(lineup["starters"]) == 3
    assert len(lineup["bench"]) == 1                      # 4 rostered -> 1 benched
    assert set(lineup["positions"].values()) == {"A1", "A2", "A3"}
    for pid in lineup["starters"]:
        o = lineup["order"][pid]
        assert o["tempo"] in coach.ORDER_TEMPO and o["role"] in coach.ORDER_ROLE


def test_select_lineup_requires_three():
    w, tid, _ = _world_with_team(n_players=2)
    with pytest.raises(coach.CoachError):
        coach.select_lineup(w, tid)


# ------------------------------------------------------------ subs/swaps --- #
def test_can_substitute_only_with_bench():
    w, tid, _ = _world_with_team(n_players=4)
    assert coach.can_substitute(w, tid) is True
    w2, tid2, _ = _world_with_team(n_players=3)
    assert coach.can_substitute(w2, tid2) is False


def test_apply_substitution_swaps_starter_for_bench():
    w, tid, _ = _world_with_team(n_players=4)
    coach.hire_coach(w, tid, w["free_coaches"][0])
    lineup = coach.select_lineup(w, tid)
    out_pid = lineup["starters"][2]
    in_pid = lineup["bench"][0]
    coach.apply_substitution(lineup, out_pid, in_pid)
    assert in_pid in lineup["starters"] and out_pid in lineup["bench"]
    assert in_pid in lineup["positions"] and in_pid in lineup["order"]
    assert out_pid not in lineup["positions"]


def test_swap_positions_exchanges_roles():
    w, tid, _ = _world_with_team(n_players=3)
    coach.hire_coach(w, tid, w["free_coaches"][0])
    lineup = coach.select_lineup(w, tid)
    a, b = lineup["starters"][0], lineup["starters"][1]
    pos_a, pos_b = lineup["positions"][a], lineup["positions"][b]
    coach.swap_positions(lineup, a, b)
    assert lineup["positions"][a] == pos_b and lineup["positions"][b] == pos_a


# ----------------------------------------------------------------- bonus --- #
def test_coach_bonus_scales_with_rating():
    w, tid, _ = _world_with_team(n_players=3)
    cid = w["free_coaches"][0]
    coach.hire_coach(w, tid, cid)
    w["coaches"][cid]["rating"] = 1000.0
    assert coach.coach_bonus(w, tid) == 0.0
    w["coaches"][cid]["rating"] = 1500.0
    assert coach.coach_bonus(w, tid) == pytest.approx(coach.COACH_BONUS_SCALE)
