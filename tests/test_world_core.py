"""World foundation: data model, persistence, SoloQ, team Elo, economy."""
import random

from src.world import economy, soloq, team_elo
from src.world.world import (
    list_worlds, load_world, new_world, save_world,
)

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


# ---------------------------------------------------------------- model ---- #
def test_new_world_shape_and_diversity():
    w = new_world("w-test", FRANCHISES, seed=1)
    assert len(w["teams"]) == 22
    assert w["season"] == 0 and w["phase"] == "created"
    # every player starts as a free agent, no money paid yet
    assert all(p["status"] == "free_agent" and p["team_id"] is None
               for p in w["players"].values())
    assert all(t["treasury"] == 0 for t in w["teams"].values())
    # diversity: more than a handful of distinct attribute totals
    totals = {sum(p["attributes"].values()) for p in w["players"].values()}
    assert len(totals) > 10
    # net link is optional in the foundation
    assert all(p["net_id"] is None for p in w["players"].values())


def test_persistence_roundtrip(tmp_path):
    w = new_world("w-rt", FRANCHISES, seed=2)
    save_world(w, world_root=tmp_path)
    assert "w-rt" in list_worlds(world_root=tmp_path)
    again = load_world("w-rt", world_root=tmp_path)
    assert again["world_id"] == "w-rt"
    assert len(again["players"]) == len(w["players"])


# ---------------------------------------------------------------- soloq ---- #
def test_soloq_bands_never_cross():
    # two clear bands; matches must keep all six players in one 100-band
    ratings = {f"lo{i}": 950.0 for i in range(6)}
    ratings.update({f"hi{i}": 1050.0 for i in range(6)})
    matches = soloq._round_matches(ratings, random.Random(0))
    assert len(matches) == 2
    for a, b in matches:
        bands = {soloq.band_of(ratings[p]) for p in a + b}
        assert len(bands) == 1


def test_soloq_sub1000_share_one_pool():
    # everyone below 1000 matchmakes together (different 100-floors, ONE pool); >=1000 still 100-banded
    ratings = {f"lo{i}": 720.0 + i * 40 for i in range(6)}    # 720..920, all < 1000
    ratings.update({f"hi{i}": 1050.0 for i in range(6)})      # a separate 1000-band division
    matches = soloq._round_matches(ratings, random.Random(0))
    assert len(matches) == 2                                  # the entry pool + the 1000 band
    lo, hi = {f"lo{i}" for i in range(6)}, {f"hi{i}" for i in range(6)}
    for a, b in matches:                                      # pools never cross
        ids = set(a + b)
        assert ids <= lo or ids <= hi
    assert soloq.band_of(720) == soloq.band_of(990) == 0      # different sub-1000 floors -> same pool
    assert soloq.band_of(1000) == 1000 and soloq.band_of(1150) == 1100   # >=1000 unchanged


def test_soloq_zero_sum_but_still_rewards_impact():
    # anti-inflation: an equal-rated pool must CONSERVE total rating (was +400+/10 rounds before the
    # mean-subtraction). Only sub-cent rounding residual is allowed.
    ratings = {f"z{i}": 1000.0 for i in range(60)}
    new, _ = soloq.simulate_soloq(ratings, rounds=10, seed=3)
    assert abs(sum(new.values()) - sum(ratings.values())) < 5.0
    # but playing well STILL earns more: a higher-impact winner gains more than a lower-impact one
    assert (soloq.soloq_rating_delta(1000, 1000, True, 0.9)
            > soloq.soloq_rating_delta(1000, 1000, True, 0.3))


def test_soloq_skips_when_band_too_small():
    ratings = {f"p{i}": 1000.0 for i in range(5)}  # only 5 -> can't fill a 6-man match
    matches = soloq._round_matches(ratings, random.Random(0))
    assert matches == []


def test_soloq_team_balance():
    group = ["a", "b", "c", "d", "e", "f"]
    ratings = {"a": 1200, "b": 1150, "c": 1000, "d": 980, "e": 850, "f": 820}
    ta, tb = soloq._balanced_two_teams(group, ratings)
    avg_a = sum(ratings[p] for p in ta) / 3
    avg_b = sum(ratings[p] for p in tb) / 3
    assert abs(avg_a - avg_b) < 60  # close averages


def test_soloq_impact_modulation():
    # impact modulates the Elo delta by ±IMPACT_SPREAD (the win/loss term stays dominant)
    spread = soloq.IMPACT_SPREAD
    base = soloq.soloq_rating_delta(1000, 1000, won=True, impact=0.5)
    hi = soloq.soloq_rating_delta(1000, 1000, won=True, impact=1.0)
    lo = soloq.soloq_rating_delta(1000, 1000, won=True, impact=0.0)
    assert abs(hi - base * (1 + spread)) < 1e-9
    assert abs(lo - base * (1 - spread)) < 1e-9


def test_soloq_deterministic():
    ratings = {f"p{i}": 1000.0 + (i % 3) * 100 for i in range(24)}
    r1, _ = soloq.simulate_soloq(ratings, seed=7)
    r2, _ = soloq.simulate_soloq(ratings, seed=7)
    assert r1 == r2


def test_apply_soloq_writes_history():
    w = new_world("w-sq", FRANCHISES, seed=3)
    soloq.apply_soloq(w, rounds=10, seed=3)
    moved = [p for p in w["players"].values() if p["soloq_history"]]
    assert len(moved) > 0
    # most players play and move off the 1000 start
    assert any(p["soloq_rating"] != 1000.0 for p in w["players"].values())


# ------------------------------------------------------------------ elo ---- #
def test_elo_margin_factor():
    assert team_elo.margin_factor(2, 0, 3) == 1.0
    assert team_elo.margin_factor(2, 1, 3) == 0.5
    assert team_elo.margin_factor(3, 0, 5) == 1.0
    assert abs(team_elo.margin_factor(3, 2, 5) - 1 / 3) < 1e-9


def test_elo_sweep_moves_more_than_close():
    sweep = team_elo.series_elo_delta(1000, 1000, 3, 0, 5)
    close = team_elo.series_elo_delta(1000, 1000, 3, 2, 5)
    assert sweep > close > 0
    assert abs(sweep - 8.0) < 1e-9  # K=16, margin 1.0, (1-0.5)


def test_elo_zero_sum():
    w = new_world("w-elo", FRANCHISES, seed=4)
    a, b = list(w["teams"])[:2]
    before = w["teams"][b]["elo"]
    delta = team_elo.apply_series_result(w, a, b, 2, 0, 3)
    assert abs((w["teams"][a]["elo"] - 1000.0) - delta) < 1e-6
    assert abs((before - w["teams"][b]["elo"]) - delta) < 1e-6


# -------------------------------------------------------------- economy ---- #
def test_prizes_upper_beats_lower():
    assert economy.placement_prize("upper", 1) == 1000
    assert economy.placement_prize("lower", 1) == 300        # 1st WITHIN a group (A and Z identical)
    assert economy.placement_prize("upper", 1) > economy.placement_prize("lower", 1)
    assert economy.placement_prize("upper", 99) == 0
    # lower groups have 6 places; A's place-N prize == Z's place-N prize (same table)
    assert economy.placement_prize("lower", 6) > 0 and economy.placement_prize("lower", 7) == 0


def test_prize_schedule_structure():
    sched = economy.prize_schedule()
    assert len(sched["upper"]["places"]) == 10 and sched["upper"]["champion_bonus"] > 0
    assert len(sched["lower"]["places"]) == 6 and sched["lower"]["per_group"] is True
    # places are strictly non-increasing (1st pays the most)
    for lg in ("upper", "lower"):
        pl = sched[lg]["places"]
        assert all(pl[i] >= pl[i + 1] for i in range(len(pl) - 1))


def test_award_prizes_credits_treasury_and_splits_player_earnings():
    w = new_world("w-prize", FRANCHISES, seed=4)
    tid = next(iter(w["teams"]))
    team = w["teams"][tid]
    roster = list(w["players"])[:3]
    team["roster"] = roster
    for pid in roster:
        w["players"][pid]["earnings"] = 0
    before = team["treasury"]
    awarded = economy.award_prizes(w, [{"team_id": tid, "league": "upper", "place": 1,
                                         "grand_final_winner": True}])
    b = awarded[tid]
    assert b["prize"] == 1000 and b["bonus"] == 500 and b["total"] == 1500
    assert team["treasury"] == before + 1500                       # purse credited to the club
    share = 1500 // 3
    assert all(w["players"][pid]["earnings"] == share for pid in roster)   # split among the roster
    assert sum(w["players"][pid]["earnings"] for pid in roster) <= 1500    # never over-pays


def test_salary_first_contract_jumps_then_moves_slowly():
    # first contract jumps to target
    assert economy.smoothed_salary(0.0, 120.0) == 120.0
    # one bad season: small drop (~25% of the gap)
    one = economy.smoothed_salary(200.0, 50.0)
    assert abs((200.0 - one) / 200.0 - 0.1875) < 1e-6  # ~19% of salary, 25% of gap
    # three bad seasons: closes ~58% of the gap ("loses a lot")
    s = 200.0
    for _ in range(3):
        s = economy.smoothed_salary(s, 50.0)
    closed = (200.0 - s) / (200.0 - 50.0)
    assert abs(closed - (1 - 0.75 ** 3)) < 1e-2  # ~58% of the gap (2-dp rounding)
    assert closed > 0.5


def test_top_player_stays_valuable():
    star = {"soloq_rating": 1500, "attributes": {k: 90 for k in economy.ATTR_KEYS}}
    weak = {"soloq_rating": 800, "attributes": {k: 40 for k in economy.ATTR_KEYS}}
    assert economy.salary_target(star) > 3 * economy.salary_target(weak)


def test_pay_salaries_skips_unplayed_rookie():
    w = new_world("w-pay", FRANCHISES, seed=5)
    tid = list(w["teams"])[0]
    pid = list(w["players"])[0]
    w["players"][pid]["status"] = "rookie"
    w["players"][pid]["seasons_played"] = 0
    w["teams"][tid]["roster"] = [pid]
    w["teams"][tid]["treasury"] = 1000
    economy.pay_salaries(w)
    assert w["teams"][tid]["treasury"] == 1000  # rookie not charged yet
    assert w["players"][pid]["salary"] == 0.0
