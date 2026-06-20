"""World foundation: transfer window rules + training auction structure."""
import pytest

from src.world import training_auction as ta
from src.world import transfers
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _world_with_two_rostered():
    w = new_world("w-mkt", FRANCHISES, seed=1)
    t1, t2 = list(w["teams"])[:2]
    p1, p2 = list(w["players"])[:2]
    for pid, tid in ((p1, t1), (p2, t2)):
        w["players"][pid]["status"] = "rostered"
        w["players"][pid]["team_id"] = tid
        w["teams"][tid]["roster"].append(pid)
        if pid in w["free_agents"]:
            w["free_agents"].remove(pid)
    return w, t1, t2, p1, p2


# -------------------------------------------------------------- transfers -- #
def test_min_price_blocks_lowball():
    w, t1, t2, p1, p2 = _world_with_two_rostered()
    w["teams"][t1]["treasury"] = 100_000
    win = transfers.open_window()
    res = transfers.propose_transfer(w, win, t1, p2, offer=1)
    assert res["status"] == "rejected" and "min price" in res["reason"]


def test_transfer_completes_at_fair_offer():
    w, t1, t2, p1, p2 = _world_with_two_rostered()
    w["teams"][t1]["treasury"] = 100_000
    offer = transfers.fair_value(w["players"][p2]) + 5
    win = transfers.open_window()
    res = transfers.propose_transfer(w, win, t1, p2, offer=offer)
    assert res["status"] == "completed"
    assert p2 in w["teams"][t1]["roster"]
    assert p2 not in w["teams"][t2]["roster"]
    assert w["teams"][t2]["treasury"] > 0  # seller got paid


def test_request_budget_capped_at_three():
    # budget counts requests; 3 distinct targets exhaust it, the 4th is blocked
    w = new_world("w-bud", FRANCHISES, seed=7)
    buyer = list(w["teams"])[0]
    w["teams"][buyer]["treasury"] = 100_000
    targets = list(w["players"])[1:5]
    for i, pid in enumerate(targets):
        owner = list(w["teams"])[i + 1]
        w["players"][pid]["status"] = "rostered"
        w["players"][pid]["team_id"] = owner
        w["teams"][owner]["roster"].append(pid)
    win = transfers.open_window()
    for pid in targets[:3]:                       # 3 requests (rejected on price, still counted)
        transfers.propose_transfer(w, win, buyer, pid, offer=1)
    fourth = transfers.propose_transfer(w, win, buyer, targets[3], offer=1)
    assert fourth["status"] == "rejected" and "budget" in fourth["reason"]


def test_same_player_twice_in_one_round_blocked():
    w, t1, t2, p1, p2 = _world_with_two_rostered()
    win = transfers.open_window()
    transfers.propose_transfer(w, win, t1, p2, offer=1)
    again = transfers.propose_transfer(w, win, t1, p2, offer=1)
    assert "already requested this player this round" in again["reason"]


def test_free_agent_auction_highest_bidder_wins():
    w = new_world("w-auc", FRANCHISES, seed=2)
    t1, t2 = list(w["teams"])[:2]
    fa = w["free_agents"][0]
    w["teams"][t1]["treasury"] = 100_000
    w["teams"][t2]["treasury"] = 100_000
    floor = transfers.min_price(w["players"][fa])
    bids = [{"team_id": t1, "player_id": fa, "amount": floor + 10},
            {"team_id": t2, "player_id": fa, "amount": floor + 50}]
    signings = transfers.resolve_free_agent_auction(w, bids, seed=0)
    assert len(signings) == 1
    s = signings[0]
    assert s["player_id"] == fa and s["team_id"] == t2 and s["price"] == floor + 50
    assert s["signing_bonus"] == floor + 50          # the fee BURNS (signing bonus, no seller)
    assert fa in w["teams"][t2]["roster"]


def test_rostered_player_auction_multiple_clubs_contend():
    # several clubs fight over the SAME signed player; highest accepted bid wins, owner is PAID.
    w = new_world("w-rauc", FRANCHISES, seed=4)
    teams = list(w["teams"])
    owner, b1, b2 = teams[0], teams[1], teams[2]
    pid = w["free_agents"][0]
    transfers._move_player_to_team(w, pid, owner)        # now rostered on `owner`
    w["teams"][b1]["treasury"] = w["teams"][b2]["treasury"] = 100_000
    fv = transfers.fair_value(w["players"][pid])
    bids = [{"team_id": b1, "player_id": pid, "amount": fv + 10},
            {"team_id": b2, "player_id": pid, "amount": fv + 80}]    # b2 outbids b1
    before_owner = w["teams"][owner]["treasury"]
    done = transfers.resolve_transfer_auction(w, bids, seed=0)
    assert len(done) == 1 and done[0]["to"] == b2 and done[0]["from"] == owner
    assert pid in w["teams"][b2]["roster"] and pid not in w["teams"][owner]["roster"]
    assert w["teams"][owner]["treasury"] == before_owner + fv + 80    # selling club is PAID


def test_rookie_draft_lower_only_weakest_first():
    w = new_world("w-rk", FRANCHISES, seed=3)
    # two lower teams with different Elo; one rookie -> weakest must get it
    teams = list(w["teams"].values())
    strong, weak = teams[0], teams[1]
    strong["league"], weak["league"] = "lower", "lower"
    strong["elo"], weak["elo"] = 1200.0, 800.0
    strong["treasury"] = weak["treasury"] = 5000      # clubs can afford the (cheap) signing bonus
    rook = w["free_agents"][0]
    w["free_agents"].remove(rook)
    w["players"][rook]["status"] = "rookie"
    w["rookies_pool"] = [rook]
    before = weak["treasury"]
    picks = transfers.rookie_draft(w)
    assert picks and picks[0]["team_id"] == weak["team_id"]
    assert rook in weak["roster"]
    # rookies are NOT free: a cheap signing bonus was charged (and it burns)
    assert picks[0]["signing_bonus"] > 0
    assert weak["treasury"] == before - picks[0]["signing_bonus"]


# --------------------------------------------------------- training auction- #
def test_training_auction_awards_a_certificate():
    w = new_world("w-tr", FRANCHISES, seed=4)
    t1, t2 = list(w["teams"])[:2]
    w["teams"][t1]["treasury"] = 1000
    w["teams"][t2]["treasury"] = 1000
    state = ta.TrainingAuctionState(ta.TrainingConfig(steps_per_slot=1_000_000), n_trainings=3)
    award = ta.auction_one_training(w, state, {t1: 100, t2: 250}, seed=0)
    assert award["team_id"] == t2 and award["price"] == 250 and award["steps"] == 1_000_000
    assert w["teams"][t2]["treasury"] == 750
    assert state.remaining == 2
    assert ta.total_banked_steps(w, t2) == 1_000_000


def test_create_player_requires_four_million_steps():
    w = new_world("w-tr2", FRANCHISES, seed=5)
    t1 = list(w["teams"])[0]
    w["teams"][t1]["treasury"] = 10_000
    state = ta.TrainingAuctionState(ta.TrainingConfig(steps_per_slot=1_000_000), n_trainings=5)
    for _ in range(3):
        ta.auction_one_training(w, state, {t1: 10}, seed=0)
    certs = [c["cert_id"] for c in ta.team_certificates(w, t1)]
    with pytest.raises(ValueError):                       # 3M < 4M
        ta.request_spend(w, t1, "create_player", certs)
    ta.auction_one_training(w, state, {t1: 10}, seed=0)   # 4th cert -> 4M
    certs = [c["cert_id"] for c in ta.team_certificates(w, t1)]
    job = ta.request_spend(w, t1, "create_player", certs)
    assert job["total_steps"] == 4_000_000 and job["status"] == "pending_execution"
    assert ta.team_certificates(w, t1) == []             # certs consumed


def test_certificates_are_bankable_and_combinable_for_smaller_slots():
    # owner sets a smaller slot this season; team needs 8 certs to fork
    w = new_world("w-tr4", FRANCHISES, seed=8)
    t1 = list(w["teams"])[0]
    w["teams"][t1]["treasury"] = 10_000
    state = ta.TrainingAuctionState(ta.TrainingConfig(steps_per_slot=500_000), n_trainings=10)
    for _ in range(8):
        ta.auction_one_training(w, state, {t1: 5}, seed=0)
    assert ta.total_banked_steps(w, t1) == 4_000_000
    # spend only 2 certs on a small skill_up; rest stay banked (deferral)
    certs = [c["cert_id"] for c in ta.team_certificates(w, t1)]
    job = ta.request_spend(w, t1, "skill_up", certs[:2])
    assert job["total_steps"] == 1_000_000
    assert ta.total_banked_steps(w, t1) == 3_000_000     # 6 certs deferred


def test_eta_shown_for_slot():
    cfg = ta.TrainingConfig(steps_per_slot=1_000_000, throughput_steps_per_min=50_000)
    assert cfg.est_minutes_per_slot() == 20.0


def test_execute_training_refused_without_backend():
    # valid job, but no real backend -> NullBackend refuses (no training runs)
    job = {"team_id": "t", "spend_type": "skill_up", "total_steps": 1_000_000,
           "target": {"kind": "player", "player_id": "p"}}
    with pytest.raises(NotImplementedError):
        ta.execute_training({"teams": {}}, job, ta.TrainingConfig())


def test_public_state_shows_awards_for_llm():
    w = new_world("w-tr3", FRANCHISES, seed=6)
    t1 = list(w["teams"])[0]
    w["teams"][t1]["treasury"] = 1000
    state = ta.TrainingAuctionState(ta.TrainingConfig(steps_per_slot=1_000_000), n_trainings=2)
    ta.auction_one_training(w, state, {t1: 100}, seed=0)
    pub = state.public_state()
    assert pub["trainings_left"] == 1
    assert pub["awards"] == [{"team_id": t1, "price": 100}]
    assert pub["steps_per_slot"] == 1_000_000
