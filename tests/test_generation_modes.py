from pathlib import Path

from src.core.replay import load_replay
from src.core.runner import GENERATION_PROFILES, run_match


def _unit_key(unit_type) -> str:
    return unit_type.value if hasattr(unit_type, "value") else str(unit_type)


def _units_digest(units: dict) -> tuple:
    return tuple(
        sorted(
            (_unit_key(unit_type), int(count))
            for unit_type, count in (units or {}).items()
            if int(count or 0) > 0
        )
    )


def _float(value: float) -> float:
    return round(float(value), 4)


def _moving_group_digest(group) -> tuple:
    return (
        group.player_id,
        group.team,
        _units_digest(group.units),
        group.src,
        group.dst,
        group.arrival_tick,
        group.started_tick,
        _float(group.travel_ticks),
        _float(group.progress),
        group.intent,
        tuple(group.path),
        group.segment_index,
        group.phase,
        group.segment_node,
        _float(group.speed),
        group.final_dst,
    )


def _battle_unit_digest(unit) -> tuple:
    return (
        unit.unit_type,
        unit.team,
        _float(unit.hp),
        _float(unit.max_hp),
        _float(unit.atk),
        getattr(unit, "attack_rate", 1),
        _float(unit.range),
        _float(unit.speed),
        _float(unit.pos),
    )


def _battle_digest(battle) -> tuple:
    return (
        battle.battle_id,
        battle.location_kind,
        battle.location_id,
        battle.started_at_tick,
        battle.last_tick,
        tuple(sorted(battle.participant_routes.items())),
        tuple(sorted(battle.participant_intents.items())),
        tuple(sorted(battle.participant_entry_sides.items())),
        tuple(sorted((pid, tuple(path)) for pid, path in battle.participant_paths.items())),
        tuple(sorted(battle.participant_segment_indices.items())),
        tuple(sorted((pid, _float(speed)) for pid, speed in battle.participant_speeds.items())),
        tuple(sorted(battle.participant_final_dsts.items())),
        battle.edge_endpoints,
        _float(battle.edge_position),
        tuple(sorted(battle.neutral_aggro_teams)),
        battle.objective,
        battle.core_owner,
        tuple(sorted(_battle_unit_digest(unit) for unit in battle.units)),
    )


def _state_digest(state) -> dict:
    return {
        "tick": state.tick,
        "winner": state.winner,
        "winner_type": state.winner_type,
        "core_combat_damage": _float(state.core_combat_damage),
        "core_damage_by_team": tuple(
            sorted((team, _float(damage)) for team, damage in state.core_damage_by_team.items())
        ),
        "death_time_final_core_deaths": tuple(sorted(state.death_time_final_core_deaths.items())),
        "players": {
            player_id: (
                player.team,
                player.core_node,
                _float(player.resources),
                player.alive,
                _float(player.core_hp),
            )
            for player_id, player in sorted(state.players.items())
        },
        "baron": (
            state.baron.hp,
            state.baron.max_hp,
            state.baron.controlled_by,
            state.baron.ticks_remaining,
            state.baron.allied_lord_owner,
            _float(state.baron.allied_lord_hp),
            state.baron.allied_lord_target,
            state.baron.allied_lord_alive,
            state.baron.allied_lord_expended,
        ),
        "nodes": {
            node_id: (
                node.owner,
                _units_digest(node.units),
                node.neutral_hp,
                node.neutral_income,
                tuple(sorted(node.neutral_aggro_teams)),
                tuple(
                    sorted(
                        (player_id, _units_digest(units))
                        for player_id, units in node.visiting_units.items()
                    )
                ),
                tuple(sorted(node.unit_ready_tick.items())),
                tuple(sorted(node.unit_entry_side.items())),
            )
            for node_id, node in sorted(state.nodes.items())
        },
        "moving_groups": tuple(
            sorted(_moving_group_digest(group) for group in state.moving_groups)
        ),
        "active_battles": tuple(
            sorted(
                (battle_key, _battle_digest(battle))
                for battle_key, battle in state.active_battles.items()
            )
        ),
    }


def test_generation_profiles_are_behavior_equivalent_for_random_bots(tmp_path: Path):
    replay_path = tmp_path / "replay_random.json"

    replay = run_match(seed=20260602, bot_type="random", replay_path=str(replay_path), mode="replay")
    training = run_match(seed=20260602, bot_type="random", mode="training")

    assert replay.winner == training.winner
    assert replay.total_ticks == training.total_ticks
    assert _state_digest(replay.state) == _state_digest(training.state)

    assert replay.record_trace is True
    assert replay.record_replay is True
    assert replay.replay_ticks_recorded == replay.total_ticks
    assert replay_path.exists()

    assert training.record_trace is False
    assert training.record_replay is False
    assert training.replay_path is None
    assert training.replay_ticks_recorded == 0


def test_generation_profiles_are_behavior_equivalent_for_scripted_bots(tmp_path: Path):
    replay_path = tmp_path / "replay_scripted.json"

    replay = run_match(seed=77, bot_type="scripted", replay_path=str(replay_path), mode="replay")
    training = run_match(seed=77, bot_type="scripted", mode="training")

    assert replay.winner == training.winner
    assert replay.total_ticks == training.total_ticks
    assert _state_digest(replay.state) == _state_digest(training.state)


def test_replay_profile_writes_replay_metadata(tmp_path: Path):
    replay_path = tmp_path / "classic_replay.json"

    result = run_match(seed=5, bot_type="random", replay_path=str(replay_path), mode="replay")
    replay = load_replay(str(replay_path))

    assert result.mode == "replay"
    assert replay["meta"]["generation_mode"] == "replay"
    assert replay["meta"]["winner"] == result.winner
    assert replay["meta"]["total_ticks"] == result.total_ticks
    assert len(replay["ticks"]) == result.total_ticks


def test_training_profile_requires_no_replay_path(tmp_path: Path):
    result = run_match(
        seed=5,
        bot_type="random",
        replay_path=str(tmp_path / "ignored.json"),
        mode="training",
    )

    assert result.mode == "training"
    assert result.replay_path is None
    assert not (tmp_path / "ignored.json").exists()
    assert result.replay_ticks_recorded == 0


def test_replay_mode_requires_path():
    try:
        run_match(seed=5, bot_type="random", mode="replay")
    except ValueError as error:
        assert "replay_path is required" in str(error)
    else:
        raise AssertionError("replay mode must require a replay_path")


def test_generation_profile_contract_is_explicit():
    replay = GENERATION_PROFILES["replay"]
    training = GENERATION_PROFILES["training"]

    assert replay.record_trace is True
    assert replay.record_replay is True
    assert replay.save_replay is True

    assert training.record_trace is False
    assert training.record_replay is False
    assert training.save_replay is False
