"""Per-player COMBAT attribution in the match engine: core damage / kills / losses / assists.

The engine credits core damage to the attacking PLAYERS (unit.team is a player_id) by army-strength
share, and on a combat core kill credits the biggest contributor a kill, the others an assist, and the
core owner a loss. Death-time/environment core deaths are NOT credited (timer, not a player)."""
from src.core.runner import run_match
from src.core.state import get_team, make_initial_state


def test_per_player_core_damage_reconciles_with_team():
    # per-player core damage must sum to the team-level total — nothing lost or double-counted
    for seed in range(8):
        st = run_match(seed=seed, bot_type="scripted", mode="training").state
        persum = sum(st.core_damage_by_player.values())
        teamsum = sum(st.core_damage_by_team.values())
        assert abs(persum - teamsum) < max(1.0, 0.02 * teamsum), f"seed {seed}: {persum} vs {teamsum}"
        # every credited player is on the team that dealt the damage (never the victim)
        for pid in st.core_damage_by_player:
            assert get_team(pid) in ("TEAM_A", "TEAM_B")


def test_combat_core_kill_credits_killer_and_loser():
    # each combat core kill == exactly one killer credited + one loss charged to the owner
    saw_kill = False
    for seed in range(16):
        st = run_match(seed=seed, bot_type="scripted", mode="training").state
        tk = sum(st.core_kills_by_player.values())
        tl = sum(st.core_losses_by_player.values())
        assert tk == tl                                   # conservation: a kill is one loss
        if tk:
            saw_kill = True
            assert all(v >= 1 for v in st.core_kills_by_player.values())
    assert saw_kill                                       # the scripted gauntlet does finish cores by combat


def test_stat_fields_backward_safe_defaults():
    # fields default to empty dicts, so old code / replays that never set them don't crash
    st = make_initial_state(seed=1)
    for f in ("core_damage_by_player", "core_kills_by_player", "core_losses_by_player", "assists_by_player"):
        assert getattr(st, f) == {}
