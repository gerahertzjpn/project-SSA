import pytest
import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.map import ADJACENCY, bfs_distance, get_legal_moves, NODE_TYPES
from src.core.constants import NodeType


def test_adjacency_symmetric():
    for node, neighbors in ADJACENCY.items():
        for neighbor in neighbors:
            assert node in ADJACENCY[neighbor], f"{neighbor} should have {node} as neighbor"


def test_all_nodes_present():
    expected = {"A1", "A2", "A3", "B1", "B2", "B3",
                "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "Baron"}
    assert set(ADJACENCY.keys()) == expected


def test_cores_connect_to_correct_cps():
    for core in ["A1", "A2", "A3"]:
        assert set(ADJACENCY[core]) == {"CP1", "CP2"}
    for core in ["B1", "B2", "B3"]:
        assert set(ADJACENCY[core]) == {"CP5", "CP6"}


def test_baron_neighbors():
    assert set(ADJACENCY["Baron"]) == {"CP1", "CP2", "CP5", "CP6"}


def test_cp3_only_reachable_via_cp1_and_cp5():
    assert set(ADJACENCY["CP3"]) == {"CP1", "CP5"}


def test_cp4_only_reachable_via_cp2_and_cp6():
    assert set(ADJACENCY["CP4"]) == {"CP2", "CP6"}


def test_bfs_distance_self():
    assert bfs_distance("A1", "A1") == 0


def test_bfs_distance_direct_neighbor():
    assert bfs_distance("A1", "CP1") == 1
    assert bfs_distance("A1", "CP2") == 1


def test_bfs_distance_baron():
    # A1 -> CP1 -> Baron = 2
    assert bfs_distance("A1", "Baron") == 2


def test_bfs_distance_enemy_core():
    # A1 -> CP1 -> Baron -> CP5 -> B1 = 4
    assert bfs_distance("A1", "B1") == 4


def test_bfs_distance_cp3():
    # A1 -> CP1 -> CP3 = 2
    assert bfs_distance("A1", "CP3") == 2


def test_node_types():
    for core in ["A1", "A2", "A3", "B1", "B2", "B3"]:
        assert NODE_TYPES[core] == NodeType.CORE
    for cp in ["CP1", "CP2", "CP3", "CP4", "CP5", "CP6"]:
        assert NODE_TYPES[cp] == NodeType.CONTROL_POINT
    assert NODE_TYPES["Baron"] == NodeType.BARON


def test_legal_moves_returns_neighbors():
    node_owners = {n: "NEUTRAL" for n in ADJACENCY}
    node_owners["A1"] = "A1"
    moves = get_legal_moves("A1", "A1", node_owners)
    assert set(moves) == {"CP1", "CP2"}


def test_legal_moves_all_neighbors_allowed():
    # All neighbors are legal targets (attack enemy = valid move)
    node_owners = {n: "NEUTRAL" for n in ADJACENCY}
    node_owners["CP1"] = "B1"  # enemy holds CP1
    moves = get_legal_moves("A1", "A1", node_owners)
    # CP1 is enemy but still legal to move to (to attack)
    assert "CP1" in moves
    assert "CP2" in moves
