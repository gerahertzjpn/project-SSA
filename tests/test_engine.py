import pytest
import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.state import make_initial_state
from src.core.engine import GameEngine, Action
from src.core.constants import ActionType, UnitType, PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot


def make_bots(bot_class, seed_offset=0):
    bots = {}
    for i, pid in enumerate(PLAYER_IDS):
        team = "TEAM_A" if pid in TEAM_A_PLAYERS else "TEAM_B"
        if bot_class == RandomBot:
            bots[pid] = bot_class(pid, team, seed=i + seed_offset)
        else:
            bots[pid] = bot_class(pid, team)
    return bots


def test_match_completes_random_bots():
    state = make_initial_state(seed=42)
    bots = make_bots(RandomBot)
    engine = GameEngine(state, bots)
    winner = engine.run()
    assert winner in ("TEAM_A", "TEAM_B", "DRAW")
    assert state.tick > 0


def test_match_produces_replay():
    state = make_initial_state(seed=1)
    bots = make_bots(RandomBot)
    engine = GameEngine(state, bots)
    engine.run()
    assert len(engine.replay_ticks) > 0
    first_tick = engine.replay_ticks[0]
    assert "tick" in first_tick
    assert "nodes" in first_tick
    assert "players" in first_tick


def test_match_completes_scripted_bots():
    state = make_initial_state(seed=7)
    bots = make_bots(ScriptedBot)
    engine = GameEngine(state, bots)
    winner = engine.run()
    assert winner in ("TEAM_A", "TEAM_B", "DRAW")


def test_players_start_alive():
    state = make_initial_state(seed=0)
    for pid in PLAYER_IDS:
        assert state.players[pid].alive


def test_neutral_nodes_start_with_guards():
    state = make_initial_state(seed=0)
    for cp in ["CP1", "CP2", "CP3", "CP4", "CP5", "CP6"]:
        assert state.nodes[cp].neutral_hp > 0
        assert state.nodes[cp].owner == "NEUTRAL"


def test_mirror_neutrals():
    # CP1 should equal CP6, CP2 should equal CP5
    state = make_initial_state(seed=42)
    assert state.nodes["CP1"].neutral_hp == state.nodes["CP6"].neutral_hp
    assert state.nodes["CP2"].neutral_hp == state.nodes["CP5"].neutral_hp


def test_winner_not_set_at_start():
    state = make_initial_state(seed=0)
    assert state.winner is None


def test_replay_nodes_have_owner():
    state = make_initial_state(seed=5)
    bots = make_bots(RandomBot)
    engine = GameEngine(state, bots)
    engine.run()
    for tick_snap in engine.replay_ticks[:5]:
        for node_id, node_data in tick_snap["nodes"].items():
            assert "owner" in node_data


def test_20_matches_complete_without_error():
    """Smoke test: 20 matches of random bots, all must complete."""
    for seed in range(20):
        state = make_initial_state(seed=seed)
        bots = make_bots(RandomBot, seed_offset=seed * 10)
        engine = GameEngine(state, bots)
        winner = engine.run()
        assert winner in ("TEAM_A", "TEAM_B", "DRAW"), f"Match {seed} failed: winner={winner}"
        assert state.tick > 0, f"Match {seed} never advanced"
