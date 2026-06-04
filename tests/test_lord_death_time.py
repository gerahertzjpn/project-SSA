from src.core.combat import build_baron_unit, build_core_unit, build_neutral_units, build_player_units
from src.core.constants import (
    BARON_HP,
    CORE_HP,
    DEATH_TIME_CORE_DAMAGE_TOTAL,
    DEATH_TIME_DAMAGE_TICKS,
    DEATH_TIME_START_TICK,
    TICK_LIMIT,
    SPECIAL_DEST_ENEMY_LORD,
    UnitType,
    death_time_core_damage_for_tick,
)
from src.core.engine import Action, GameEngine
from src.core.state import Battle, MovingGroup, make_initial_state
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
    damages = [
        death_time_core_damage_for_tick(tick)
        for tick in range(DEATH_TIME_START_TICK, TICK_LIMIT)
    ]
    assert DEATH_TIME_DAMAGE_TICKS == TICK_LIMIT - DEATH_TIME_START_TICK
    assert DEATH_TIME_CORE_DAMAGE_TOTAL == CORE_HP - 1
    assert sum(damages) == CORE_HP - 1
    assert all(isinstance(damage, int) for damage in damages)
    assert max(damages) - min(damages) <= 1


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
    assert state.baron.allied_lord_ready_tick == state.tick + 2
    assert state.baron.allied_lord_path[0] == "Baron"
    assert state.baron.allied_lord_target in {"B1", "B2", "B3"}
    assert state.moving_groups == []

    state.tick = state.baron.allied_lord_ready_tick
    launch_event = engine._launch_ready_allied_lord_push()

    assert launch_event["type"] == "LORD_PUSH_STARTED"
    assert state.moving_groups[-1].intent == "LORD_PUSH"
    assert state.moving_groups[-1].units == {UnitType.LORD.value: 1}
    assert state.moving_groups[-1].path[0] == "Baron"
    assert state.moving_groups[-1].final_dst in {"B1", "B2", "B3"}


def test_ready_lord_push_takes_support_units_waiting_at_baron():
    state = make_initial_state(seed=117)
    state.tick = 10
    state.nodes["Baron"].owner = "A1"
    state.nodes["Baron"].units = {UnitType.SKIRMISHER.value: 2}
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = float(BARON_HP)
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_path = ["Baron", "CP5", "B1"]
    state.baron.allied_lord_ready_tick = state.tick
    state.baron.allied_lord_arrival_tick = state.tick + 10
    state.baron.allied_lord_alive = True
    state.baron.allied_lord_support_units = {UnitType.VANGUARD.value: 3}
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)

    event = engine._launch_ready_allied_lord_push()

    assert event["type"] == "LORD_PUSH_STARTED"
    assert state.moving_groups[-1].units == {
        UnitType.LORD.value: 1,
        UnitType.VANGUARD.value: 3,
    }
    assert state.nodes["Baron"].units == {UnitType.SKIRMISHER.value: 2}
    assert state.baron.allied_lord_support_units == {}


def test_join_lord_arrival_waits_for_delayed_lord_push():
    state = make_initial_state(seed=120)
    state.tick = 10
    state.nodes["Baron"].owner = "A1"
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = float(BARON_HP)
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_path = ["Baron", "CP5", "B1"]
    state.baron.allied_lord_ready_tick = state.tick + 2
    state.baron.allied_lord_arrival_tick = state.tick + 12
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    group = MovingGroup(
        player_id="A1",
        team="TEAM_A",
        units={UnitType.RANGER.value: 2},
        src="CP1",
        dst="Baron",
        arrival_tick=state.tick,
        intent="JOIN_LORD",
        path=["CP1", "Baron"],
    )

    event = engine._handle_lord_special_arrival(group)

    assert event["type"] == "LORD_SUPPORT_WAITING"
    assert state.baron.allied_lord_support_units == {UnitType.RANGER.value: 2}


def test_killed_neutral_lord_visual_identity_becomes_allied_lord_push():
    state = make_initial_state(seed=111)
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    attackers = build_player_units({UnitType.VANGUARD.value: 1}, "A1", "attacker")
    attackers[0].visual_id = "vu:test:a1:vanguard"
    baron = build_baron_unit(BARON_HP)
    baron.visual_id = "vu:test:baron:lord"
    baron.hp = 0.0
    battle = Battle(
        battle_id="node:Baron",
        location_kind="node",
        location_id="Baron",
        units=[*attackers, baron],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="baron",
    )

    event = engine._finish_node_battle(battle)

    assert event["baron_killed"] is True
    assert event["lord_spawned"]["player"] == "A1"
    assert state.baron.allied_lord_visual_units == {
        UnitType.LORD.value: ["vu:test:baron:lord"]
    }


def test_dead_neutral_lord_stays_visible_until_baron_claim_resolves():
    state = make_initial_state(seed=112)
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    attackers = build_player_units({UnitType.VANGUARD.value: 1}, "A1", "attacker")
    defenders = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")
    baron = build_baron_unit(BARON_HP)
    baron.visual_id = "vu:test:baron:fallen"
    baron.hp = 0.0
    baron.x = 340.0
    baron.y = 210.0
    battle = Battle(
        battle_id="node:Baron",
        location_kind="approach",
        location_id="Baron",
        units=[*attackers, *defenders, baron],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="baron",
        approach_id="CP1->Baron",
    )
    state.active_battles[battle.battle_id] = battle

    rows = engine._visual_unit_snapshots(state)

    fallen = next(row for row in rows if row["id"] == "vu:test:baron:fallen")
    assert fallen["team"] == "BARON"
    assert fallen["type"] == "L"
    assert fallen["hp"] == 0.0
    assert fallen["hp_ratio"] == 0.0
    assert fallen["state"] == "fallen"
    assert fallen["layer"] == "battle"


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


def test_lord_push_support_continues_after_intermediate_battle():
    state = make_initial_state(seed=118)
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
        units=build_player_units(
            {UnitType.LORD.value: 1, UnitType.VANGUARD.value: 2},
            "A1",
            "attacker",
        ),
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

    assert event["lord_continued"]["units"] == {
        UnitType.LORD.value: 1,
        UnitType.VANGUARD.value: 2,
    }
    assert state.nodes["CP5"].units == {}
    assert state.moving_groups[-1].units == {
        UnitType.LORD.value: 1,
        UnitType.VANGUARD.value: 2,
    }


def test_enemy_can_hunt_delayed_lord_at_baron():
    state = make_initial_state(seed=119)
    state.tick = 10
    state.nodes["Baron"].owner = "B1"
    state.baron.allied_lord_owner = "B1"
    state.baron.allied_lord_hp = float(BARON_HP)
    state.baron.allied_lord_target = "A1"
    state.baron.allied_lord_path = ["Baron", "CP1", "A1"]
    state.baron.allied_lord_ready_tick = state.tick + 2
    state.baron.allied_lord_arrival_tick = state.tick + 12
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    group = MovingGroup(
        player_id="A1",
        team="TEAM_A",
        units={UnitType.RANGER.value: 2},
        src="CP1",
        dst="Baron",
        arrival_tick=state.tick,
        intent="HUNT_LORD",
        path=["CP1", "Baron"],
        final_dst=SPECIAL_DEST_ENEMY_LORD,
    )

    event = engine._handle_baron_arrival(group)

    assert event["type"] == "BATTLE_STARTED"
    battle = state.active_battles["node:Baron"]
    assert any(unit.team == "B1" and unit.unit_type == UnitType.LORD.value for unit in battle.units)
    assert any(unit.team == "A1" and unit.unit_type == UnitType.RANGER.value for unit in battle.units)


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


def test_allied_lord_retargets_if_original_core_was_destroyed_before_arrival():
    state = make_initial_state(seed=113)
    state.players["B1"].alive = False
    state.players["B1"].core_hp = 0.0
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = 88.0
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    group = MovingGroup(
        player_id="A1",
        team="TEAM_A",
        units={UnitType.LORD.value: 1},
        src="CP5",
        dst="B1",
        arrival_tick=state.tick,
        intent="LORD_PUSH",
        path=["CP5", "B1"],
        final_dst="B1",
        visual_units={UnitType.LORD.value: ["vu:test:lord:retarget"]},
    )

    event = engine._handle_core_arrival(group)

    assert event["type"] == "LORD_RETARGETED"
    assert event["target"] in {"B2", "B3"}
    assert state.baron.allied_lord_target == event["target"]
    assert state.baron.allied_lord_alive is True
    assert state.moving_groups[-1].intent == "LORD_PUSH"
    assert state.moving_groups[-1].visual_units == {
        UnitType.LORD.value: ["vu:test:lord:retarget"]
    }


def test_allied_lord_retargets_after_dead_core_pvp_resolution():
    state = make_initial_state(seed=114)
    state.players["B1"].alive = False
    state.players["B1"].core_hp = 0.0
    state.baron.allied_lord_owner = "A1"
    state.baron.allied_lord_hp = 75.0
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    lord.visual_id = "vu:test:lord:dead-core"
    battle = Battle(
        battle_id="node:B1",
        location_kind="node",
        location_id="B1",
        units=[lord],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
    )
    battle.participant_intents["A1"] = "LORD_PUSH"

    event = engine._finish_node_battle(battle)

    assert event["result"] == "DEAD_CORE_PVP_RESOLVED"
    assert event["lord_continued"]["type"] == "LORD_RETARGETED"
    assert event["lord_continued"]["target"] in {"B2", "B3"}
    assert state.nodes["B1"].visiting_units.get("A1", {}) == {}
    assert state.moving_groups[-1].visual_units == {
        UnitType.LORD.value: ["vu:test:lord:dead-core"]
    }


def test_active_lord_edge_battle_retargets_when_current_core_dies():
    state = make_initial_state(seed=115)
    state.players["B3"].alive = False
    state.players["B3"].core_hp = 0.0
    state.baron.allied_lord_owner = "A2"
    state.baron.allied_lord_hp = 113.0
    state.baron.allied_lord_target = "B3"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    lord = build_player_units({UnitType.LORD.value: 1}, "A2", "attacker")[0]
    battle = Battle(
        battle_id="edge:Baron|CP5",
        location_kind="edge",
        location_id="Baron|CP5",
        units=[lord],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
        edge_position=0.5,
    )
    battle.participant_routes["A2"] = ("Baron", "CP5")
    battle.participant_intents["A2"] = "LORD_PUSH"
    battle.participant_paths["A2"] = ["Baron", "CP5", "B3"]
    battle.participant_segment_indices["A2"] = 1
    battle.participant_final_dsts["A2"] = "B3"

    event = engine._refresh_active_lord_target(battle)

    assert event["type"] == "LORD_RETARGETED"
    assert event["previous_target"] == "B3"
    assert event["target"] in {"B1", "B2"}
    assert state.baron.allied_lord_target == event["target"]
    assert battle.participant_final_dsts["A2"] == event["target"]
    assert battle.participant_paths["A2"][:2] == ["Baron", "CP5"]
    assert battle.participant_paths["A2"][-1] == event["target"]


def test_active_lord_hp_syncs_during_battle_tick():
    state = make_initial_state(seed=116)
    state.baron.allied_lord_owner = "A1"
    starting_lord_hp = float(BARON_HP)
    state.baron.allied_lord_hp = starting_lord_hp
    state.baron.allied_lord_target = "B1"
    state.baron.allied_lord_alive = True
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    ranger = build_player_units({UnitType.RANGER.value: 1}, "B1", "defender")[0]
    lord.x = 100.0
    lord.y = 50.0
    lord.forward_x = 1.0
    lord.forward_y = 0.0
    ranger.x = 90.0
    ranger.y = 50.0
    battle = Battle(
        battle_id="edge:A1|CP1",
        location_kind="edge",
        location_id="A1|CP1",
        units=[lord, ranger],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
    )
    state.active_battles[battle.battle_id] = battle

    engine._tick_active_battles([])

    assert state.baron.allied_lord_hp == starting_lord_hp - ranger.atk


def test_death_time_win_reward_is_halved_for_training():
    env = TrainingEnv(TrainingEnvConfig(seed=107))
    env.reset()
    env.state.winner = "TEAM_A"
    env.state.winner_type = "death_time"

    assert env._terminal_reward() == 0.5
