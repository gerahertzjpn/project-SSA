"""Tests for the league layer: scheduling, simulation, standings, persistence.

Most tests inject a fast fake match runner so they exercise league logic without
the cost of full simulations. One smoke test runs a single real encounter to
verify integration with ``run_match`` and that replay files are written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.league import (
    create_league,
    advance_day,
    league_public_view,
    load_league,
    list_leagues,
    encounter_detail,
)
from src.league.league import _round_robin_pairings
from src.league.templates import get_template, list_templates


# --------------------------------------------------------------------------- #
# Fake match runner
# --------------------------------------------------------------------------- #

@dataclass
class _FakePlayer:
    core_hp: float = 80.0
    alive: bool = True
    resources: float = 5.0


@dataclass
class _FakeState:
    players: dict = field(default_factory=dict)
    core_damage_by_team: dict = field(default_factory=lambda: {"TEAM_A": 0.0, "TEAM_B": 0.0})


@dataclass
class _FakeResult:
    seed: int
    winner: str
    winner_type: str
    total_ticks: int
    state: _FakeState


def make_fake_runner(winner_for_seed):
    """Return a run_match-compatible fn whose winner is decided by ``winner_for_seed``."""

    def fake_run_match(seed, bot_type, replay_path=None, mode="replay",
                       player_models=None, player_factions=None):
        winner = winner_for_seed(seed)
        players = {slot: _FakePlayer() for slot in ("A1", "A2", "A3", "B1", "B2", "B3")}
        state = _FakeState(players=players)
        if replay_path:
            Path(replay_path).parent.mkdir(parents=True, exist_ok=True)
            Path(replay_path).write_text(json.dumps({"meta": {"winner": winner}, "ticks": []}),
                                         encoding="utf-8")
        return _FakeResult(seed=seed, winner=winner, winner_type="normal",
                           total_ticks=42, state=state)

    return fake_run_match


def make_teams(count=6, players_per_team=3):
    return [
        {
            "name": f"Team {i + 1}",
            "players": [
                {"name": f"P{i + 1}-{j + 1}", "model": "random", "faction": "humans"}
                for j in range(players_per_team)
            ],
        }
        for i in range(count)
    ]


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

def test_template_classic_6x3_present():
    template = get_template("classic_6x3")
    assert template["team_count"] == 6
    assert template["players_per_team"] == 3
    assert template["best_of"] == 3
    assert any(t["id"] == "classic_6x3" for t in list_templates())


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        get_template("does_not_exist")


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #

def test_round_robin_covers_all_pairs_once():
    ids = [f"t{i}" for i in range(6)]
    pairs = _round_robin_pairings(ids)
    assert len(pairs) == 15  # C(6, 2)
    seen = {frozenset(pair) for pair in pairs}
    assert len(seen) == 15
    for pair in pairs:
        assert pair[0] != pair[1]


# --------------------------------------------------------------------------- #
# Creation + simulation (fake runner)
# --------------------------------------------------------------------------- #

def test_create_league_simulates_all_encounters(tmp_path):
    # Home (TEAM_A) always wins -> every encounter is a 2-0 home win.
    runner = make_fake_runner(lambda seed: "TEAM_A")
    league = create_league(
        name="Test League",
        template_id="classic_6x3",
        teams=make_teams(),
        seed=123,
        allowed_models={"random"},
        run_match_fn=runner,
        leagues_root=tmp_path,
    )
    assert league["total_days"] == 15
    assert len(league["schedule"]) == 15
    for enc in league["schedule"]:
        assert enc["result"] in ("home", "away", "draw")
        # Bo3, home always wins -> clinches at 2-0, third game not played.
        assert len(enc["games"]) == 2
        assert enc["home_game_wins"] == 2
        # replay files exist (stored path is absolute when leagues_root is a tmp dir)
        for game in enc["games"]:
            assert Path(game["replay"]).exists()


def test_results_hidden_until_day_advances(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_A")
    league = create_league("Hidden", "classic_6x3", make_teams(), seed=7,
                           allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)
    view = league_public_view(league)
    assert view["current_day"] == 0
    assert all(not enc["revealed"] for enc in view["schedule"])
    assert all("result" not in enc for enc in view["schedule"])
    assert view["history"] == []
    # standings exist but everyone has 0 played
    assert all(row["played"] == 0 for row in view["standings"])


def test_advance_day_reveals_one_encounter(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_A")
    league = create_league("Reveal", "classic_6x3", make_teams(), seed=7,
                           allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)
    view = advance_day(league["id"], leagues_root=tmp_path)
    assert view["current_day"] == 1
    revealed = [enc for enc in view["schedule"] if enc["revealed"]]
    assert len(revealed) == 1
    assert revealed[0]["result"] == "home"
    assert len(view["history"]) == 1
    # one encounter played -> two teams have played == 1
    played = [row for row in view["standings"] if row["played"] == 1]
    assert len(played) == 2
    winner_row = max(view["standings"], key=lambda r: r["points"])
    assert winner_row["points"] == 3


def test_standings_points_after_full_reveal(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_A")
    league = create_league("Full", "classic_6x3", make_teams(), seed=7,
                           allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)
    for _ in range(league["total_days"]):
        view = advance_day(league["id"], leagues_root=tmp_path)
    assert view["current_day"] == 15
    assert len(view["history"]) == 15
    # every encounter played -> each team played 5 matches
    assert all(row["played"] == 5 for row in view["standings"])
    total_points = sum(row["points"] for row in view["standings"])
    assert total_points == 15 * 3  # all decisive (no draws)


def test_encounter_detail_gated_and_has_player_stats(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_A")
    league = create_league("Detail", "classic_6x3", make_teams(), seed=7,
                           allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)
    # before reveal -> error
    with pytest.raises(Exception):
        encounter_detail(league["id"], 0, leagues_root=tmp_path)
    advance_day(league["id"], leagues_root=tmp_path)
    detail = encounter_detail(league["id"], 0, leagues_root=tmp_path)
    assert detail["revealed"] is True
    assert len(detail["games"]) == 2
    stats = detail["games"][0]["players"]
    assert set(stats) == {"A1", "A2", "A3", "B1", "B2", "B3"}
    assert stats["A1"]["side"] == "home"
    assert stats["B1"]["side"] == "away"
    assert "core_hp" in stats["A1"]


def test_persistence_roundtrip_and_listing(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_B")
    league = create_league("Persist", "classic_6x3", make_teams(), seed=99,
                           allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)
    reloaded = load_league(league["id"], leagues_root=tmp_path)
    assert reloaded["id"] == league["id"]
    assert len(reloaded["schedule"]) == 15
    listed = list_leagues(leagues_root=tmp_path)
    assert any(item["id"] == league["id"] for item in listed)


def test_invalid_team_count_rejected(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_A")
    with pytest.raises(Exception):
        create_league("Bad", "classic_6x3", make_teams(count=4), seed=1,
                      allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)


def test_unknown_model_rejected(tmp_path):
    runner = make_fake_runner(lambda seed: "TEAM_A")
    teams = make_teams()
    teams[0]["players"][0]["model"] = "totally_unknown_model"
    with pytest.raises(Exception):
        create_league("BadModel", "classic_6x3", teams, seed=1,
                      allowed_models={"random"}, run_match_fn=runner, leagues_root=tmp_path)


# --------------------------------------------------------------------------- #
# Real integration smoke (single encounter)
# --------------------------------------------------------------------------- #

def test_real_single_encounter_smoke(tmp_path):
    """One real Bo3 encounter via run_match; verifies replays + stats are real."""
    from src.core.runner import run_match

    teams = make_teams(count=6)
    league = create_league("RealSmoke", "classic_6x3", teams, seed=20260609,
                           allowed_models={"random"}, run_match_fn=run_match,
                           leagues_root=tmp_path)
    advance_day(league["id"], leagues_root=tmp_path)
    detail = encounter_detail(league["id"], 0, leagues_root=tmp_path)
    assert 2 <= len(detail["games"]) <= 3
    game = detail["games"][0]
    assert game["ticks"] > 0
    assert Path(game["replay"]).exists()
