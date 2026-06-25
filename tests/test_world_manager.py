"""World manager — the single brain (front office + playbook): profiles, decisions,
planning. No coach (WORLD_MODE_SPEC §0-bis)."""
from src.world import manager, playbook, season
from src.world.manager import ManagerProfile
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}",
               "manager": f"Mgr {i}"} for i in range(22)]


def test_manager_comments_llm_does_not_clobber_accumulator():
    # regression: the first-person LLM rewrite must NOT reassign the `out` list (was `out = llm.say(...)`),
    # which crashed on the 2nd team's out.append with an ACTIVE llm — and silently failed the off-season.
    from src.world import commentary

    class StubLLM:
        def say(self, persona, prompt, max_tokens=None, **k):
            return "Мы идём за титулом — верю в свою тройку."

    world = {"season": 1, "teams": {}, "players": {}}
    for i in range(3):
        tid = f"t{i}"
        world["teams"][tid] = {"team_id": tid, "name": f"T{i}", "franchise_id": f"fr{i:02d}",
                               "league": "upper", "elo": 1000, "roster": []}
        for j in range(3):
            pid = f"{tid}-p{j}"
            world["players"][pid] = {"player_id": pid, "nickname": f"N{i}{j}", "faction": "humans"}
            world["teams"][tid]["roster"].append(pid)
    out = commentary.manager_comments(world, llm=StubLLM(), lang="ru")
    assert len(out) == 3 and all(o["text"] for o in out)        # all 3 teams comment, no clobber crash


def _profile(**over):
    base = dict(aggression=0.5, frugality=0.5, youth_focus=0.5, risk=0.5,
                patience=0.5, loyalty=0.5)
    base.update(over)
    return ManagerProfile(**base)


# -------------------------------------------------------------- character -- #
def test_rules_are_manager_only_with_playbook():
    r = manager.MANAGER_SYSTEM_RULES
    assert "МЕНЕДЖЕР" in r and "ПЛЕЙБУК" in r          # single brain authors the playbook
    assert "тренера НЕТ" in r                           # explicitly no coach
    assert "верхней лиг" in r                           # always aims for the upper title
    # no in-game order vocabulary leaks into the manager's job
    for tactic in ("tempo", "win_path", "core_rush", "turtle"):
        assert tactic not in r


def test_natural_profile_deterministic_and_diverse():
    p1 = manager.natural_profile("iron-falcons")
    assert manager.natural_profile("iron-falcons").as_dict() == p1.as_dict()  # stable
    aggressions = {round(manager.natural_profile(f["franchise_id"]).aggression, 2)
                   for f in FRANCHISES}
    assert len(aggressions) > 8                         # spreads on its own


def test_persona_describes_management_and_playbook():
    fr = {"name": "Team X", "manager": "Иван"}
    blurb = manager.manager_persona(fr, _profile(aggression=0.9, youth_focus=0.9))
    assert "Иван" in blurb and "плейбук" in blurb.lower()
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


# --------------------------------------------------------------- planning -- #
def test_run_planning_stage_sets_playbooks():
    w = new_world("w-plan", FRANCHISES, seed=2)
    season.bootstrap_world(w, seed=2)
    plans = manager.run_planning_stage(w)
    assert len(plans) == 22                             # every team gets a playbook
    for pl in plans:
        plan = w["teams"][pl["team_id"]]["tournament_plan"]
        assert plan is not None
        assert plan["base"]["strategy"] in playbook.STRATEGY_IDS
        assert plan["overrides"]                        # has contingencies


def test_aggressive_manager_tilts_base_aggressive():
    w = new_world("w-tilt", FRANCHISES, seed=2)
    season.bootstrap_world(w, seed=2)
    team = list(w["teams"].values())[0]
    plan = manager.set_tournament_playbook(w, team, _profile(aggression=0.95, patience=0.1))
    assert plan["base"]["strategy"] == playbook.aggressive_comeback(w, team["team_id"])
