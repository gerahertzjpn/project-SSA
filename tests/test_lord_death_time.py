from src.core.combat import build_core_unit, build_neutral_units, build_player_units
from src.core.constants import (
    BARON_HP,
    CORE_HP,
    DEATH_TIME_CORE_DAMAGE_PER_TICK,
    DEATH_TIME_START_TICK,
    TICK_LIMIT,
    UnitType,
)
from src.core.engine import Action, GameEngine
from src.core.state import Battle, make_initial_state
from src.core.constants import ActionType
from src.training.env import TrainingEnv, TrainingEnvConfig


class HoldBot:
    def act(self, obs):
        return Action(ActionType.HOLD)


def hold_bots(state):
    return {player_id: HoldBot() for player_id in state.players}


def test_death_time_damage_leaves_full_core_at_one_hp_before_final():
    state = make_initial_state(seed=101)
    engine = GameEngine(state, hold_bots(state), record_trace=False, record_replay=False)

    for _ in range(TICK_LIMIT):
        engine.tick()

    assert state.winner is None
    assert all(round(player.core_hp, 6) == 1.0 for player in state.players.values())
    assert DEATH_TIME_CORE_DAMAGE_PER_TICK == (CORE_HP - 1) / (TICK_LIMIT - DEATH_TIME_START_TICK)


def test_tick_limit_draws_only_when_no_core_combat_damage():
    state = make_initial_state(seed=102)
    engine = GameEngine(state, hold_bots(state), record_trace=False, record_replay=False)

    winner = engine.run()

    assert winner == "DRAW"
    assert state.winner_type == "draw_no_core_damage"
    assert state.death_time_final_core_deaths == {"TEAM_A": 3, "TEAM_B": 3}


def test_death_time_awards_team_with_more_cores_alive_at_final_hit():
    state = make_initial_state(seed=103)
    state.players["B1"].core_hp = CORE_HP - 5
    state.core_combat_damage = 5.0
    state.core_damage_by_team["TEAM_A"] = 5.0
    engine = GameEngine(state, hold_bots(state), record_trace=False, record_replay=False)

    winner = engine.run()

    assert winner == "TEAM_A"
    assert state.winner_type == "death_time"
    assert state.death_time_final_core_deaths["TEAM_A"] == 3
    assert state.death_time_final_core_deaths["TEAM_B"] == 2


def test_core_battle_persists_core_hp_and_records_combat_damage():
    state = make_initial_state(seed=104)
    state.players["B1"].core_hp = 5.0
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    battle = Battle(
        battle_id="node:B1",
        location_kind="node",
        location_id="B1",
        units=[
            *build_player_units({UnitType.VANGUARD.value: 1}, "A1", "attacker"),
            build_core_unit("B1", hp=5.0),
        ],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="core",
        core_owner="B1",
    )
    for unit in battle.units:
        if unit.unit_type == "CORE":
            unit.hp = 2.0

    event = engine._finish_node_battle(battle)

    assert event["core_destroyed"] is False
    assert state.players["B1"].core_hp == 2.0
    assert state.core_combat_damage == 3.0
    assert state.core_damage_by_team["TEAM_A"] == 3.0


def test_neutral_guards_keep_survivor_count_after_repelling_attack():
    state = make_initial_state(seed=110)
    state.nodes["CP1"].neutral_hp = 5
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    neutral_units = build_neutral_units(5)
    for unit in neutral_units[2:]:
        unit.hp = 0.0
    battle = Battle(
        battle_id="node:CP1",
        location_kind="node",
        location_id="CP1",
        units=neutral_units,
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="neutral",
    )

    event = engine._finish_node_battle(battle)

    assert event["winner"] == "NEUTRAL"
    assert event["remaining_neutral_hp"] == 2
    assert state.nodes["CP1"].neutral_hp == 2


def test_killing_neutral_lord_spawns_allied_lord_push():
    state = make_initial_state(seed=105)
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    battle = Battle(
        battle_id="node:Baron",
        location_kind="node",
        location_id="Baron",
        units=build_player_units({UnitType.VANGUARD.value: 1}, "A1", "attacker"),
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="baron",
    )

    event = engine._finish_node_battle(battle)

    assert event["baron_killed"] is True
    assert event["lord_spawned"]["player"] == "A1"
    assert state.baron.allied_lord_owner == "A1"
    assert state.baron.allied_lord_hp == state.baron.max_hp
    assert state.baron.allied_lord_alive is True
    assert state.moving_groups[-1].intent == "LORD_PUSH"
    assert state.moving_groups[-1].units == {UnitType.LORD.value: 1}
    assert state.moving_groups[-1].path[0] == "Baron"
    assert state.moving_groups[-1].final_dst in {"B1", "B2", "B3"}


def test_lord_push_continues_after_winning_intermediate_node_battle():
    state = make_initial_state(seed=106)
    state.nodes["CP5"].owner = "B1"
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = 77.0
    state.baron.allied_lord_target = "B2"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    battle = Battle(
        battle_id="node:CP5",
        location_kind="node",
        location_id="CP5",
        units=build_player_units({UnitType.LORD.value: 1}, "A1", "attacker"),
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
    )
    for unit in battle.units:
        if unit.unit_type == UnitType.LORD.value:
            unit.hp = 77.0
    battle.participant_intents["A1"] = "LORD_PUSH"
    battle.participant_final_dsts["A1"] = "B2"

    event = engine._finish_node_battle(battle)

    assert event["lord_continued"]["target"] == "B2"
    assert state.nodes["CP5"].owner == "A1"
    assert state.nodes["CP5"].units == {}
    assert state.moving_groups[-1].intent == "LORD_PUSH"
    assert state.moving_groups[-1].path == ["CP5", "B2"]
    assert state.baron.allied_lord_hp == 77.0


def test_allied_lord_hp_persists_between_battles():
    state = make_initial_state(seed=108)
    state.nodes["CP5"].owner = "B1"
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = float(BARON_HP)
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    battle = Battle(
        battle_id="node:CP5",
        location_kind="node",
        location_id="CP5",
        units=[
            *build_player_units({UnitType.LORD.value: 1}, "A1", "attacker"),
            *build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender"),
        ],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
    )
    battle.participant_intents["A1"] = "LORD_PUSH"
    battle.participant_final_dsts["A1"] = "B1"
    for unit in battle.units:
        if unit.unit_type == UnitType.LORD.value:
            unit.hp = 42.0

    engine._finish_node_battle(battle)

    assert state.baron.allied_lord_hp == 42.0
    next_battle = Battle(
        battle_id="node:B1",
        location_kind="node",
        location_id="B1",
        units=[],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="core",
        core_owner="B1",
    )
    engine._add_player_units_to_battle(
        next_battle,
        player_id="A1",
        units={UnitType.LORD.value: 1},
        src="CP5",
        dst="B1",
        intent="LORD_PUSH",
    )
    lord = next(unit for unit in next_battle.units if unit.unit_type == UnitType.LORD.value)
    assert lord.hp == 42.0


def test_allied_lord_dies_after_destroying_first_core():
    state = make_initial_state(seed=109)
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = 30.0
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    battle = Battle(
        battle_id="node:B1",
        location_kind="node",
        location_id="B1",
        units=build_player_units({UnitType.LORD.value: 1}, "A1", "attacker"),
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="core",
        core_owner="B1",
    )
    battle.units.append(build_core_unit("B1", hp=1.0))
    for unit in battle.units:
        if unit.unit_type == "CORE":
            unit.hp = 0.0
    battle.participant_intents["A1"] = "LORD_PUSH"

    event = engine._finish_node_battle(battle)

    assert event["core_destroyed"] is True
    assert event["lord_expended"] is True
    assert state.baron.allied_lord_alive is False
    assert state.baron.allied_lord_expended is True
    assert state.baron.allied_lord_hp == 0.0
    assert UnitType.LORD.value not in state.nodes["B1"].units


def test_death_time_win_reward_is_halved_for_training():
    env = TrainingEnv(TrainingEnvConfig(seed=107))
    env.reset()
    env.state.winner = "TEAM_A"
    env.state.winner_type = "death_time"

    assert env._terminal_reward() == 0.5
