"""Tests for the regionals mode: cards, Elo, draft, playoff, economy, gating,
commentary. All tests use a fake match runner and a fake LLM, so they exercise
the tournament logic without simulations or network calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.regionals import (
    PlayersError,
    RegionalsError,
    add_commentary,
    advance_day,
    create_regionals,
    encounter_detail,
    ensure_player_store,
    list_regionals,
    load_regionals,
    player_cards,
    regionals_public_view,
)
from src.regionals.draft import run_draft
from src.regionals.elo import rating_delta
from src.regionals.franchises import ensure_franchise_store, franchise_specs
from src.regionals.templates import get_template

import random


# --------------------------------------------------------------------------- #
# Fixtures: fake registry, fake match runner, fake LLM
# --------------------------------------------------------------------------- #

RECIPES = ["baseline", "low_entropy", "high_entropy", "fast_lr", "slow_explore", "alt_mix"]
FACTIONS = ["humans", "infected", "robots"]


def write_fake_registry(path: Path) -> None:
    entries = []
    for f_index, faction in enumerate(FACTIONS):
        for n in range(1, 7):
            net_id = f"{faction}_{n}"
            entries.append({
                "net_id": net_id,
                "faction": faction,
                "recipe": RECIPES[(n - 1) % len(RECIPES)],
                "checkpoint": f"training_runs/pop/nets/{net_id}/latest.pt",
                "seed": 1000 + f_index * 10 + n,
                "genotype_hash": f"hash_{net_id}",
                "status": "trained",
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


@dataclass
class _FakePlayer:
    core_hp: float = 80.0
    alive: bool = True
    resources: float = 5.0


@dataclass
class _FakeState:
    players: dict = field(default_factory=dict)
    core_damage_by_team: dict = field(default_factory=lambda: {"TEAM_A": 10.0, "TEAM_B": 5.0})


@dataclass
class _FakeResult:
    seed: int
    winner: str
    winner_type: str
    total_ticks: int
    state: _FakeState


def make_fake_runner(winner_for_seed):
    def fake_run_match(seed, bot_type, replay_path=None, mode="replay",
                       player_models=None, player_factions=None):
        winner = winner_for_seed(seed)
        players = {slot: _FakePlayer() for slot in ("A1", "A2", "A3", "B1", "B2", "B3")}
        if replay_path:
            Path(replay_path).parent.mkdir(parents=True, exist_ok=True)
            Path(replay_path).write_text(json.dumps({"meta": {"winner": winner}}), encoding="utf-8")
        return _FakeResult(seed=seed, winner=winner, winner_type="normal",
                           total_ticks=42, state=_FakeState(players=players))
    return fake_run_match


class FakeLLM:
    """Deterministic LLM stub; records calls and mimics the usage counter."""

    def __init__(self, pick_player=None, fail_picks=False):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.pick_calls = []
        self.say_calls = []
        self._pick_player = pick_player
        self._fail_picks = fail_picks

    def pick(self, system, prompt, allowed_ids):
        self.pick_calls.append({"system": system, "prompt": prompt, "allowed": list(allowed_ids)})
        self.usage["calls"] += 1
        self.usage["input_tokens"] += 100
        self.usage["output_tokens"] += 10
        if self._fail_picks:
            return {"player_id": "nonexistent_player", "reason": "bad"}
        player = self._pick_player(allowed_ids) if self._pick_player else sorted(allowed_ids)[0]
        return {"player_id": player, "reason": f"беру {player}"}

    def say(self, system, prompt, max_tokens=400):
        self.say_calls.append({"system": system, "prompt": prompt})
        self.usage["calls"] += 1
        self.usage["input_tokens"] += 200
        self.usage["output_tokens"] += 50
        return f"Комментарий #{len(self.say_calls)}"


@pytest.fixture
def paths(tmp_path):
    registry = tmp_path / "registry.json"
    write_fake_registry(registry)
    return {
        "registry_path": registry,
        "analysis_path": tmp_path / "missing_analysis.json",
        "players_store_path": tmp_path / "players.json",
        "franchises_store_path": tmp_path / "franchises.json",
        "regionals_root": tmp_path / "regionals",
    }


def make_regionals(paths, llm=None, seed=7, winner_for_seed=None, name="Test Cup"):
    runner = make_fake_runner(winner_for_seed or (lambda s: "TEAM_A"))
    return create_regionals(
        name=name, template_id="regionals_v1", seed=seed, llm=llm,
        run_match_fn=runner, **paths)


# --------------------------------------------------------------------------- #
# Player cards
# --------------------------------------------------------------------------- #

def test_player_store_creates_18_unique_cards(paths):
    store = ensure_player_store(paths["players_store_path"], paths["registry_path"],
                                paths["analysis_path"])
    cards = player_cards(store)
    assert len(cards) == 18
    ids = {c["player_id"] for c in cards}
    models = {c["model"] for c in cards}
    nicknames = {c["nickname"] for c in cards}
    assert len(ids) == 18 and len(models) == 18 and len(nicknames) == 18
    for card in cards:
        assert card["rating"] == 1000
        assert card["model"] == f"policy:training_runs/pop/nets/{card['net_id']}/latest.pt"
        assert set(card["attributes"]) == {"attack", "defense", "economy", "tempo",
                                           "consistency", "potential"}
    # store persists and reloads without duplication
    store2 = ensure_player_store(paths["players_store_path"], paths["registry_path"],
                                 paths["analysis_path"])
    assert len(store2["players"]) == 18


def test_player_store_rejects_duplicate_nets(tmp_path):
    registry = tmp_path / "registry.json"
    entries = [{"net_id": "humans_1", "faction": "humans", "recipe": "baseline",
                "checkpoint": "x.pt", "seed": 1, "genotype_hash": "h"}] * 2
    registry.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    with pytest.raises(PlayersError):
        ensure_player_store(tmp_path / "players.json", registry, tmp_path / "none.json")


def test_attributes_grounded_in_analysis(paths, tmp_path):
    analysis = tmp_path / "analysis.json"
    winrates = {f"{f}_{n}": 0.2 + 0.04 * (i * 6 + n)
                for i, f in enumerate(FACTIONS) for n in range(1, 7)}
    mechanics = {}
    for i, f in enumerate(FACTIONS):
        for n in range(1, 7):
            k = i * 6 + n
            mechanics[f"{f}_{n}"] = {
                "decisions": 1000, "hold": 10 * k, "produce": 500 + 10 * k,
                "multi": 300, "resource_on_units": 1000 * k,
                "resource_on_buildings": 200 * (19 - k),
                "buildings_built": {"TURRET": 5 * k, "BOOSTER": 19 - k},
            }
    analysis.write_text(json.dumps({"experiment_b_per_net": {
        "net_winrate": winrates, "net_mechanics": mechanics}}), encoding="utf-8")
    store = ensure_player_store(tmp_path / "p.json", paths["registry_path"], analysis)
    cards = player_cards(store)
    potentials = {c["attributes"]["potential"] for c in cards}
    assert len(potentials) > 5  # derived from distinct winrates, not a constant


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #

def test_elo_win_vs_strong_beats_win_vs_weak():
    strong = rating_delta(1000, 1200, won=True)
    weak = rating_delta(1000, 800, won=True)
    assert strong > weak > 0


def test_elo_loss_to_weak_costs_more():
    to_weak = rating_delta(1000, 800, won=False)
    to_strong = rating_delta(1000, 1200, won=False)
    assert to_weak < to_strong < 0


def test_elo_impact_is_bounded_and_directional():
    base = rating_delta(1000, 1000, won=True, impact=0.5)
    carry = rating_delta(1000, 1000, won=True, impact=1.0)
    passenger = rating_delta(1000, 1000, won=True, impact=0.0)
    assert passenger < base < carry
    assert carry <= base * 1.2 + 1e-9
    # a good performance in a loss softens the penalty but stays negative
    bad_loss = rating_delta(1000, 1000, won=False, impact=0.0)
    good_loss = rating_delta(1000, 1000, won=False, impact=1.0)
    assert bad_loss < good_loss < 0


# --------------------------------------------------------------------------- #
# Draft
# --------------------------------------------------------------------------- #

def _draft_setup(paths):
    store = ensure_player_store(paths["players_store_path"], paths["registry_path"],
                                paths["analysis_path"])
    return get_template("regionals_v1"), franchise_specs(), player_cards(store)


def test_draft_structure_and_balance(paths):
    template, specs, cards = _draft_setup(paths)
    llm = FakeLLM()
    rosters, log = run_draft(template, specs, cards, random.Random(42), llm)
    assert len(log["rounds"]) == 3
    picked = []
    for rnd in log["rounds"]:
        assert len(rnd["picks"]) == 6
        picked.extend(p["player_id"] for p in rnd["picks"])
    assert len(picked) == 18 and len(set(picked)) == 18
    for roster in rosters.values():
        assert len(roster) == 3
    # balance: no franchise drafts from the same queue position twice
    orders = [rnd["order"] for rnd in log["rounds"]]
    for pos in range(6):
        assert len({order[pos] for order in orders}) == 3
    # every pick came from the LLM with its persona in the system prompt
    assert all(p["source"] == "llm" for rnd in log["rounds"] for p in rnd["picks"])
    assert len(llm.pick_calls) == 18
    assert log["usage"]["calls"] == 18


def test_draft_random_order_differs_between_seeds(paths):
    template, specs, cards = _draft_setup(paths)
    _, log_a = run_draft(template, specs, cards, random.Random(1), FakeLLM())
    _, log_b = run_draft(template, specs, cards, random.Random(2), FakeLLM())
    assert [r["order"] for r in log_a["rounds"]] != [r["order"] for r in log_b["rounds"]]


def test_draft_falls_back_on_invalid_llm_answer(paths):
    template, specs, cards = _draft_setup(paths)
    rosters, log = run_draft(template, specs, cards, random.Random(42),
                             FakeLLM(fail_picks=True))
    picks = [p for rnd in log["rounds"] for p in rnd["picks"]]
    assert all(p["source"] == "auto" for p in picks)
    assert len({p["player_id"] for p in picks}) == 18


def test_draft_without_llm_completes(paths):
    template, specs, cards = _draft_setup(paths)
    rosters, log = run_draft(template, specs, cards, random.Random(42), llm=None)
    assert log["llm"] is False
    assert sum(len(r) for r in rosters.values()) == 18


def test_draft_manager_sees_only_public_data(paths):
    template, specs, cards = _draft_setup(paths)
    llm = FakeLLM()
    run_draft(template, specs, cards, random.Random(42), llm)
    sample = llm.pick_calls[5]["prompt"]
    assert "Доступные игроки" in sample
    assert "genotype" not in sample and "checkpoint" not in sample


# --------------------------------------------------------------------------- #
# Full tournament: playoff, economy, persistence
# --------------------------------------------------------------------------- #

def test_create_regionals_full_flow(paths):
    llm = FakeLLM()
    data = make_regionals(paths, llm=llm)
    league_encs = [e for e in data["schedule"] if e["stage"] == "league"]
    assert len(league_encs) == 15
    assert data["league_days"] == 15 and data["total_days"] == 17

    semifinal = next(e for e in data["schedule"] if e["stage"] == "semifinal")
    final = next(e for e in data["schedule"] if e["stage"] == "final")
    # rebuild the league table to check seeding
    from src.league.league import _compute_standings
    table = _compute_standings({"template": data["template"]["league"],
                                "teams": data["teams"], "current_day": 15,
                                "schedule": league_encs})
    seed_order = [row["team_id"] for row in table]
    assert semifinal["home"] == seed_order[1] and semifinal["away"] == seed_order[2]
    assert final["home"] == seed_order[0]
    # home always wins -> SF winner is seed 2, champion is seed 1
    assert final["away"] == seed_order[1]
    assert data["champion"]["team_id"] == seed_order[0]

    # placements + prizes
    places = {p["place"]: p for p in data["placements"]}
    assert places[1]["team_id"] == seed_order[0]
    assert places[1]["prize"] == 500 and places[1]["champion_bonus"] == 250
    assert places[2]["team_id"] == seed_order[1]
    assert places[3]["team_id"] == seed_order[2]
    assert [places[i]["prize"] for i in range(1, 7)] == [500, 300, 200, 120, 80, 50]

    # opening commentary generated by the LLM
    assert data["commentary"][0]["kind"] == "opening"
    assert "Комментарий" in data["commentary"][0]["text"]

    # persistent stores updated
    store = ensure_player_store(paths["players_store_path"], paths["registry_path"],
                                paths["analysis_path"])
    drafted = [p["player_id"] for t in data["teams"] for p in t["players"]]
    assert any(store["players"][pid]["rating"] != 1000 for pid in drafted)
    assert all(store["players"][pid]["games"] > 0 for pid in drafted)
    assert all(store["players"][pid]["tournaments"] == 1 for pid in drafted)
    champions = {p["player_id"] for p in
                 next(t for t in data["teams"] if t["id"] == data["champion"]["team_id"])["players"]}
    assert all(store["players"][pid]["titles"] == 1 for pid in champions)
    assert all(store["players"][pid]["money"] == 250 for pid in champions)  # (500+250)//3

    franchises = ensure_franchise_store(paths["franchises_store_path"])["franchises"]
    champ = franchises[data["champion"]["team_id"]]
    assert champ["money"] == 750 and champ["titles"] == 1 and champ["seasons"] == 1
    h2h_total = sum(sum(rec.values()) for f in franchises.values()
                    for rec in f["head_to_head"].values())
    assert h2h_total == 17 * 2  # every encounter recorded on both sides


def test_playoff_draw_goes_to_higher_seed(paths):
    data = make_regionals(paths, winner_for_seed=lambda s: "draw")
    semifinal = next(e for e in data["schedule"] if e["stage"] == "semifinal")
    final = next(e for e in data["schedule"] if e["stage"] == "final")
    assert semifinal["result"] == "draw" and final["result"] == "draw"
    # drawn series go to the higher league seed: SF -> rank 2, final -> rank 1
    assert final["away"] == semifinal["home"]
    assert data["champion"]["team_id"] == final["home"]


def test_elo_changes_recorded_per_encounter(paths):
    data = make_regionals(paths)
    enc = data["schedule"][0]
    changes = enc["rating_changes"]
    assert len(changes) == 6
    for change in changes.values():
        assert change["after"] == pytest.approx(change["before"] + change["delta"], abs=0.2)
        assert change["games"]
    # winners gained, losers lost (home always wins)
    teams_by_id = {t["id"]: t for t in data["teams"]}
    home_ids = {p["player_id"] for p in teams_by_id[enc["home"]]["players"]}
    for pid, change in changes.items():
        if pid in home_ids:
            assert change["delta"] > 0
        else:
            assert change["delta"] < 0


# --------------------------------------------------------------------------- #
# Day gating
# --------------------------------------------------------------------------- #

def test_day_gating_and_bracket_masking(paths):
    data = make_regionals(paths)
    rid = data["id"]
    root = paths["regionals_root"]

    view = regionals_public_view(load_regionals(rid, root))
    assert view["current_day"] == 0
    assert all(not item["revealed"] for item in view["schedule"])
    assert all(row["played"] == 0 for row in view["standings"])
    assert all(node["home_name"] == "?" and node["away_name"] == "?"
               for node in view["bracket"])
    assert "placements" not in view and "champion" not in view
    assert all(p["rating_now"] == p["rating_start"] for p in view["players"])

    for _ in range(15):  # reveal the whole league
        view = advance_day(rid, root)
    assert view["league_finished"] is True and view["finished"] is False
    sf = next(n for n in view["bracket"] if n["stage"] == "semifinal")
    fin = next(n for n in view["bracket"] if n["stage"] == "final")
    assert sf["home_name"] != "?" and sf["away_name"] != "?"
    assert fin["home_name"] != "?" and fin["away_name"] == "?"  # SF not revealed yet
    assert "placements" not in view

    view = advance_day(rid, root)  # semifinal day
    fin = next(n for n in view["bracket"] if n["stage"] == "final")
    assert fin["away_name"] != "?"

    view = advance_day(rid, root)  # final day
    assert view["finished"] is True
    assert view["champion"]["team_id"] == data["champion"]["team_id"]
    assert len(view["placements"]) == 6
    # ratings now reflect the whole tournament
    finals = {p["player_id"]: p["rating_now"] for p in view["players"]}
    assert finals == {pid: pytest.approx(v, abs=0.5)
                      for pid, v in data["final_ratings"].items()}


def test_encounter_detail_gated(paths):
    data = make_regionals(paths)
    rid, root = data["id"], paths["regionals_root"]
    with pytest.raises(RegionalsError):
        encounter_detail(rid, 0, root)
    advance_day(rid, root)
    detail = encounter_detail(rid, 0, root)
    assert detail["revealed"] and detail["games"]
    assert detail["rating_changes"] and detail["top_performers"]


# --------------------------------------------------------------------------- #
# Commentary
# --------------------------------------------------------------------------- #

def test_commentary_flow_with_memory_and_cache(paths):
    creation_llm = FakeLLM()
    data = make_regionals(paths, llm=creation_llm)
    rid, root = data["id"], paths["regionals_root"]

    llm = FakeLLM()
    with pytest.raises(RegionalsError):
        add_commentary(rid, {"type": "encounter", "index": 0}, llm, root)  # not revealed

    advance_day(rid, root)
    entry = add_commentary(rid, {"type": "encounter", "index": 0}, llm, root)
    assert entry["kind"] == "match" and entry["encounter_index"] == 0
    # the commentator saw his own opening speech in memory
    assert "прошлые реплики" in llm.say_calls[0]["system"]
    # cached: a second click does not call the LLM again
    calls_before = len(llm.say_calls)
    cached = add_commentary(rid, {"type": "encounter", "index": 0}, llm, root)
    assert cached["text"] == entry["text"] and len(llm.say_calls) == calls_before

    with pytest.raises(RegionalsError):
        add_commentary(rid, {"type": "final"}, llm, root)  # tournament not finished
    for _ in range(16):
        advance_day(rid, root)
    final_entry = add_commentary(rid, {"type": "final"}, llm, root)
    assert final_entry["kind"] == "final"
    # memory grew: final call context includes the match comment
    assert entry["text"] in llm.say_calls[-1]["system"]

    saved = load_regionals(rid, root)
    assert len(saved["commentary"]) == 3  # opening + match + final


def test_commentary_requires_llm(paths):
    data = make_regionals(paths)
    rid, root = data["id"], paths["regionals_root"]
    advance_day(rid, root)
    with pytest.raises(RegionalsError, match="LLM"):
        add_commentary(rid, {"type": "encounter", "index": 0}, None, root)


# --------------------------------------------------------------------------- #
# Persistence / listing
# --------------------------------------------------------------------------- #

def test_persistence_and_listing(paths):
    data = make_regionals(paths, name="Кубок Севера")
    listed = list_regionals(paths["regionals_root"])
    assert len(listed) == 1
    assert listed[0]["id"] == data["id"]
    assert listed[0]["team_count"] == 6
    loaded = load_regionals(data["id"], paths["regionals_root"])
    assert loaded["name"] == "Кубок Севера"
    assert len(loaded["schedule"]) == 17
    # replays written next to the tournament
    replays = list((paths["regionals_root"] / data["id"] / "replays").glob("*.json"))
    assert replays


def test_second_tournament_accumulates_career(paths):
    make_regionals(paths, seed=7)
    make_regionals(paths, seed=8, name="Second Cup")
    store = ensure_player_store(paths["players_store_path"], paths["registry_path"],
                                paths["analysis_path"])
    assert all(p["tournaments"] == 2 for p in store["players"].values())
    franchises = ensure_franchise_store(paths["franchises_store_path"])["franchises"]
    assert all(f["seasons"] == 2 for f in franchises.values())
