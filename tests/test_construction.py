from src.core.constants import (
    BUILDING_BUILD_TICKS,
    BUILDING_COST,
    BUILDING_REGEN_DELAY_TICKS,
    BUILDING_REGEN_HP_PER_TICK,
    BUILDING_ZERO_HP_REGEN_DELAY_MULTIPLIER,
    BUILDING_STATS,
    FACTION_TYPES,
    RESOURCE_PER_TICK,
    ActionType,
    BuildingType,
    UnitType,
)
from src.core.combat import build_building_unit, build_core_unit
from src.core.combat2d import tick_battle_units_2d
from src.core.engine import Action, GameEngine
from src.core.production import apply_build, apply_production, collect_income
from src.core.state import Battle, MovingGroup, make_initial_state, get_team
from src.core.units import UnitInstance


class HoldBot:
    requires_observation = False

    def act(self, obs):
        from src.core.constants import ActionType
        from src.core.engine import Action

        return Action(ActionType.HOLD)


def make_engine(state):
    bots = {
        player_id: HoldBot()
        for player_id in state.players
    }
    return GameEngine(state, bots, record_trace=False, record_replay=False)


def test_initial_faction_schema_uses_humans_with_future_slots():
    state = make_initial_state(seed=0)

    assert FACTION_TYPES == ("humans", "infected", "robots")
    assert {player.faction for player in state.players.values()} == {"humans"}


def test_initial_state_can_seed_faction_starting_units():
    state = make_initial_state(seed=0, player_factions={"A1": "infected", "B1": "robots"})

    assert state.players["A1"].faction == "infected"
    assert state.nodes["A1"].units == {UnitType.SWARMLING.value: 6}
    assert state.players["B1"].faction == "robots"
    assert state.nodes["B1"].units == {UnitType.WALKER.value: 2}


def test_tech_buildings_unlock_human_tech_units_only_after_completion():
    state = make_initial_state(seed=1)
    player = state.players["A1"]
    player.resources = 200

    assert apply_production(state, "A1", UnitType.RAIDER) is False
    assert apply_build(state, "A1", "A1", BuildingType.TECH_2) is False

    assert apply_build(state, "A1", "A1", BuildingType.TECH_1) is True
    assert apply_production(state, "A1", UnitType.RAIDER) is False

    state.tick += BUILDING_BUILD_TICKS[BuildingType.TECH_1]
    assert apply_production(state, "A1", UnitType.RAIDER) is True
    assert state.nodes["A1"].units[UnitType.RAIDER.value] == 1

    assert apply_build(state, "A1", "A1", BuildingType.TECH_2) is True
    assert apply_production(state, "A1", UnitType.BMP) is False
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TECH_2]
    assert apply_production(state, "A1", UnitType.BMP) is True
    assert state.nodes["A1"].units[UnitType.BMP.value] == 1


def test_units_are_gated_by_player_faction():
    state = make_initial_state(seed=30)
    for player in state.players.values():
        player.resources = 300

    assert apply_production(state, "A1", UnitType.VANGUARD) is True
    assert apply_production(state, "A1", UnitType.SWARMLING) is False
    assert apply_production(state, "A1", UnitType.WALKER) is False

    state.players["A2"].faction = "infected"
    assert apply_production(state, "A2", UnitType.SWARMLING) is True
    assert apply_production(state, "A2", UnitType.VANGUARD) is False
    assert apply_production(state, "A2", UnitType.WALKER) is False

    state.players["A3"].faction = "robots"
    assert apply_production(state, "A3", UnitType.WALKER) is True
    assert apply_production(state, "A3", UnitType.VANGUARD) is False
    assert apply_production(state, "A3", UnitType.SWARMLING) is False


def test_non_human_tech_units_require_matching_tech_chain():
    state = make_initial_state(seed=31)
    state.players["A1"].faction = "infected"
    state.players["A1"].resources = 400

    assert apply_production(state, "A1", UnitType.RAVAGER) is False
    assert apply_build(state, "A1", "A1", BuildingType.TECH_1) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TECH_1]
    assert apply_production(state, "A1", UnitType.RAVAGER) is True
    assert apply_production(state, "A1", UnitType.BEHEMOTH) is False

    assert apply_build(state, "A1", "A1", BuildingType.TECH_2) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TECH_2]
    assert apply_production(state, "A1", UnitType.BEHEMOTH) is True
    assert apply_production(state, "A1", UnitType.GUARDIAN_MECH) is False


def test_tech_units_require_alive_tech_chain():
    state = make_initial_state(seed=28)
    player = state.players["A1"]
    player.resources = 300

    assert apply_build(state, "A1", "A1", BuildingType.TECH_1) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TECH_1]
    assert apply_build(state, "A1", "A1", BuildingType.TECH_2) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TECH_2]

    tech1, tech2 = state.nodes["A1"].buildings
    tech1.hp = 0
    assert apply_production(state, "A1", UnitType.RAIDER) is False
    assert apply_production(state, "A1", UnitType.BMP) is False

    tech1.hp = 1
    tech2.hp = 0
    assert apply_production(state, "A1", UnitType.RAIDER) is True
    assert apply_production(state, "A1", UnitType.BMP) is False


def test_economy_collector_adds_income_after_completion():
    state = make_initial_state(seed=2)
    state.players["A1"].resources = 100

    assert apply_build(state, "A1", "A1", BuildingType.ECONOMY_COLLECTOR) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.ECONOMY_COLLECTOR]

    before = state.players["A1"].resources
    collect_income(state)
    expected_income = RESOURCE_PER_TICK + BUILDING_STATS[BuildingType.ECONOMY_COLLECTOR]["income"]
    assert state.players["A1"].resources == before + expected_income


def test_dead_economy_collector_keeps_slot_but_does_not_add_income():
    state = make_initial_state(seed=24)
    state.players["A1"].resources = 100

    assert apply_build(state, "A1", "A1", BuildingType.ECONOMY_COLLECTOR) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.ECONOMY_COLLECTOR]
    state.nodes["A1"].buildings[0].hp = 0
    state.nodes["A1"].buildings[0].last_damage_tick = state.tick

    before = state.players["A1"].resources
    collect_income(state)

    assert state.players["A1"].resources == before + RESOURCE_PER_TICK
    assert apply_build(state, "A1", "A1", BuildingType.ECONOMY_COLLECTOR) is False


def test_faction_unique_buildings_are_faction_gated():
    state = make_initial_state(seed=20)
    state.players["A1"].resources = 200

    assert apply_build(state, "A1", "A1", BuildingType.BOOSTER) is True
    assert apply_build(state, "A1", "A1", BuildingType.POWER) is False
    assert apply_build(state, "A1", "A1", BuildingType.LIFE) is False

    state.players["B1"].faction = "robots"
    state.players["B1"].resources = 200
    assert apply_build(state, "B1", "B1", BuildingType.POWER) is True
    assert apply_build(state, "B1", "B1", BuildingType.BOOSTER) is False


def test_human_booster_increases_departing_core_group_speed():
    state = make_initial_state(seed=21)
    state.players["A1"].resources = 200
    assert apply_build(state, "A1", "A1", BuildingType.BOOSTER) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.BOOSTER]
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 7}
    engine = make_engine(state)

    event = engine._apply_move(
        "A1",
        Action(ActionType.MOVE, source="A1", destination="CP1"),
    )

    assert event["base_speed"] == 1.5
    assert event["speed"] == 2.5
    assert state.moving_groups[-1].speed == 2.5


def test_infected_life_building_spawns_free_unit_on_core():
    state = make_initial_state(seed=22)
    state.players["A1"].faction = "infected"
    state.players["A1"].resources = 200
    assert apply_build(state, "A1", "A1", BuildingType.LIFE) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.LIFE]
    engine = make_engine(state)
    before = int(state.nodes["A1"].units.get(UnitType.SWARMLING.value, 0) or 0)

    engine._apply_life_building_spawns([])

    assert state.nodes["A1"].units[UnitType.SWARMLING.value] == before + 1


def test_robot_power_building_adds_hp_to_materialized_units():
    state = make_initial_state(seed=23)
    state.players["A1"].faction = "robots"
    state.players["A1"].resources = 200
    assert apply_build(state, "A1", "A1", BuildingType.POWER) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.POWER]
    engine = make_engine(state)
    battle = engine._start_node_battle(
        group=MovingGroup(
            player_id="A1",
            team=get_team("A1"),
            units={UnitType.WALKER.value: 1},
            src="A1",
            dst="CP1",
            arrival_tick=state.tick,
        ),
        node_id="CP1",
        objective="neutral",
        include_neutral=True,
    )
    assert battle["type"] == "BATTLE_STARTED"
    unit = next(unit for unit in state.active_battles["node:CP1"].units if unit.team == "A1")
    assert unit.max_hp == 36
    assert unit.hp == 36


def test_control_point_turret_joins_defensive_battle():
    state = make_initial_state(seed=3)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {}
    state.players["A1"].resources = 100
    assert apply_build(state, "A1", "CP1", BuildingType.TURRET) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TURRET]

    engine = make_engine(state)
    group = MovingGroup(
        player_id="B1",
        team=get_team("B1"),
        units={UnitType.VANGUARD.value: 3},
        src="CP5",
        dst="CP1",
        arrival_tick=state.tick,
    )
    engine._start_node_battle(
        group=group,
        node_id="CP1",
        objective="pvp",
        include_owner_units=True,
        include_buildings=True,
    )

    battle = state.active_battles["node:CP1"]
    assert any(unit.unit_type == BuildingType.TURRET.value and unit.team == "A1" for unit in battle.units)


def test_destroyed_building_stays_in_slot_and_regenerates_after_delay():
    state = make_initial_state(seed=25)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.players["A1"].resources = 100
    assert apply_build(state, "A1", "CP1", BuildingType.TURRET) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TURRET]

    turret = state.nodes["CP1"].buildings[0]
    turret.hp = 0
    turret.last_damage_tick = state.tick
    engine = make_engine(state)

    assert apply_build(state, "A1", "CP1", BuildingType.TURRET) is False

    events = []
    engine._regenerate_buildings(events)
    assert turret.hp == 0
    assert events == []

    zero_hp_delay = BUILDING_REGEN_DELAY_TICKS * BUILDING_ZERO_HP_REGEN_DELAY_MULTIPLIER
    state.tick += zero_hp_delay - 1
    engine._regenerate_buildings(events)
    assert turret.hp == 0
    assert events == []

    state.tick += 1
    engine._regenerate_buildings(events)
    assert turret.hp == BUILDING_REGEN_HP_PER_TICK
    assert events[-1]["type"] == "BUILDING_REGEN"


def test_damaged_building_regenerates_after_normal_delay_before_zero_hp():
    state = make_initial_state(seed=27)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.players["A1"].resources = 100
    assert apply_build(state, "A1", "CP1", BuildingType.TURRET) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TURRET]

    turret = state.nodes["CP1"].buildings[0]
    turret.hp = turret.max_hp - 10
    turret.last_damage_tick = state.tick
    engine = make_engine(state)

    events = []
    state.tick += BUILDING_REGEN_DELAY_TICKS - 1
    engine._regenerate_buildings(events)
    assert turret.hp == turret.max_hp - 10
    assert events == []

    state.tick += 1
    engine._regenerate_buildings(events)
    assert turret.hp == turret.max_hp - 5
    assert events[-1]["type"] == "BUILDING_REGEN"


def test_battle_sync_keeps_destroyed_building_for_auto_regen():
    state = make_initial_state(seed=26)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.players["A1"].resources = 100
    assert apply_build(state, "A1", "CP1", BuildingType.TURRET) is True
    state.tick += BUILDING_BUILD_TICKS[BuildingType.TURRET]
    engine = make_engine(state)

    dead_turret = build_building_unit(BuildingType.TURRET, "A1", hp=0)
    battle = Battle(
        battle_id="node:CP1:test",
        location_kind="node",
        location_id="CP1",
        units=[dead_turret],
        started_at_tick=state.tick,
        last_tick=state.tick,
    )

    engine._sync_buildings_after_node_battle(battle)

    assert len(state.nodes["CP1"].buildings) == 1
    assert state.nodes["CP1"].buildings[0].hp == 0
    assert state.nodes["CP1"].buildings[0].last_damage_tick == state.tick


def test_core_assault_can_destroy_only_one_building_layer_per_tick():
    attacker = UnitInstance(
        uid=900001,
        unit_type=UnitType.BMP.value,
        team="B1",
        hp=1000,
        max_hp=1000,
        atk=500,
        range=100,
        speed=0,
        pos=0,
        attack_rate=20,
        x=0,
        y=0,
    )
    slot_1 = build_building_unit(BuildingType.ECONOMY_COLLECTOR, "A1", hp=12)
    slot_1.visual_id = "building:A1:0:ECONOMY_COLLECTOR"
    slot_1.x = slot_1.y = 0
    slot_2 = build_building_unit(BuildingType.TECH_1, "A1", hp=12)
    slot_2.visual_id = "building:A1:1:TECH_1"
    slot_2.x = slot_2.y = 0
    core = build_core_unit("A1", hp=80)
    core.x = core.y = 0
    battle = Battle(
        battle_id="node:A1:test",
        location_kind="node",
        location_id="A1",
        units=[attacker, slot_1, slot_2, core],
        started_at_tick=0,
        last_tick=0,
        objective="core",
        core_owner="A1",
    )

    tick_battle_units_2d(battle.units, record_trace=False, battle=battle)
    assert slot_2.hp <= 0
    assert slot_1.hp == 12
    assert core.hp == 80

    tick_battle_units_2d(battle.units, record_trace=False, battle=battle)
    assert slot_1.hp <= 0
    assert core.hp == 80

    tick_battle_units_2d(battle.units, record_trace=False, battle=battle)
    assert core.hp < 80


def test_reverse_group_cannot_pass_through_active_approach_battle():
    state = make_initial_state(seed=29)
    engine = make_engine(state)
    battle = Battle(
        battle_id="node:CP5",
        location_kind="approach",
        location_id="CP5",
        units=[],
        started_at_tick=10,
        last_tick=10,
        objective="pvp",
        approach_id="CP3->CP5",
        edge_position=0.1,
        threat_zone_ids=["ranged:CP5"],
    )
    state.active_battles["node:CP5"] = battle

    far_group = MovingGroup(
        player_id="B3",
        team=get_team("B3"),
        units={UnitType.VANGUARD.value: 1},
        src="CP5",
        dst="CP3",
        arrival_tick=state.tick + 1,
        phase="edge",
        progress=0.6,
    )
    events = []
    remaining = engine._join_reached_approach_battles([far_group], events)
    assert remaining == [far_group]
    assert events == []
    assert battle.units == []

    reached_group = MovingGroup(
        player_id="B3",
        team=get_team("B3"),
        units={UnitType.VANGUARD.value: 1},
        src="CP5",
        dst="CP3",
        arrival_tick=state.tick + 1,
        phase="edge",
        progress=0.95,
    )
    events = []
    remaining = engine._join_reached_approach_battles([reached_group], events)
    assert remaining == []
    assert events[0]["type"] == "BATTLE_REINFORCEMENT"
    assert events[0]["location_kind"] == "approach"
    assert any(unit.team == "B3" for unit in battle.units)


def test_empty_capture_removes_previous_owner_unfinished_building():
    state = make_initial_state(seed=4)
    state.nodes["CP1"].owner = "A2"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {}
    state.players["A2"].resources = 100
    assert apply_build(state, "A2", "CP1", BuildingType.TURRET) is True
    assert state.nodes["CP1"].buildings

    engine = make_engine(state)
    group = MovingGroup(
        player_id="B1",
        team=get_team("B1"),
        units={UnitType.VANGUARD.value: 1},
        src="CP5",
        dst="CP1",
        arrival_tick=state.tick,
    )

    event = engine._handle_control_point_arrival(group)

    assert event["type"] == "CAPTURE"
    assert state.nodes["CP1"].owner == "B1"
    assert state.nodes["CP1"].buildings == []
