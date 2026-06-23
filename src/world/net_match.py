"""Connect the frozen population's nets to World matches.

World's SoloQ ladder and league season both take a pluggable match function. By default
those are rating-based stochastic models (no nets). This module supplies the REAL ones: a
3v3 GameEngine match via ``src.core.runner.run_match`` — the same runner regionals and the
viewer Match tab use — driven by each player's bound net checkpoint. So with a world that was
seeded from the frozen population, ratings (SoloQ) and league results come from actual net play.

Resolution: each World player carries a ``net_id``; the frozen manifest
(``population/frozen_population_v1.json``) maps that to a ``policy:<checkpoint>`` model id, which
run_match turns into a PolicyBot (LRU-cached checkpoint loads). Per-player SoloQ impact comes
from own core survival + economy in the final state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from src.core.constants import CORE_HP
from src.core.runner import run_match as _default_run_match

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "population" / "frozen_population_v1.json"
HOME_SLOTS = ("A1", "A2", "A3")
AWAY_SLOTS = ("B1", "B2", "B3")

_MANIFEST_CACHE: dict[str, dict] = {}


def load_manifest(path: Optional[str | Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_MANIFEST
    key = str(p.resolve())
    if key not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
    return _MANIFEST_CACHE[key]


def net_model_index(manifest: dict) -> dict[str, str]:
    """net_id -> ``policy:<checkpoint>`` model id consumable by run_match."""
    return {
        e["net_id"]: f"policy:{e['checkpoint']}"
        for group in ("descendants", "ancestors")
        for e in manifest.get(group, [])
    }


def register_forked_net(manifest_path, net_id: str, faction: str, checkpoint: str,
                        archetype: Optional[str] = None) -> None:
    """Add a freshly-forked rookie/child net to the population manifest so net_match (SoloQ + league)
    can RESOLVE it. Without this a forked net trains into the registry but is unusable in play. Appends
    to descendants, de-duped by net_id; clears the cached manifest so the next load sees it."""
    import json
    p = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    man = json.loads(p.read_text(encoding="utf-8"))
    desc = man.setdefault("descendants", [])
    if any(e.get("net_id") == net_id for e in desc):
        for e in desc:                                  # already present -> just refresh the checkpoint
            if e.get("net_id") == net_id:
                e["checkpoint"] = str(checkpoint)
    else:
        desc.append({"net_id": net_id, "faction": faction, "checkpoint": str(checkpoint),
                     "archetype": archetype, "role": "player", "persona": "pro",
                     "origin": "rookie_fork"})
    p.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    _MANIFEST_CACHE.pop(str(p.resolve()), None)         # force a fresh load next time


def _slot_impacts(state, slot_pids: list[tuple[str, str]]) -> dict[str, float]:
    """Per-player SoloQ impact in [0,1] (0.5 neutral) from own core survival + economy,
    scaled relative to the strongest economy in THIS match so teammates differ."""
    raw: dict[str, tuple[float, float]] = {}
    for slot, pid in slot_pids:
        p = state.players.get(slot)
        core = max(0.0, float(getattr(p, "core_hp", 0.0) or 0.0)) / float(CORE_HP)
        res = max(0.0, float(getattr(p, "resources", 0.0) or 0.0))
        raw[pid] = (min(1.0, core), res)
    max_res = max((r for _, r in raw.values()), default=0.0) or 1.0
    return {
        pid: max(0.0, min(1.0, 0.6 * core + 0.4 * (res / max_res)))
        for pid, (core, res) in raw.items()
    }


def _models_factions(world: dict, idx: dict[str, str],
                     team_a: list[str], team_b: list[str]):
    players = world["players"]
    models: dict[str, str] = {}
    factions: dict[str, str] = {}
    slot_pids: list[tuple[str, str]] = []
    for slots, team in ((HOME_SLOTS, team_a), (AWAY_SLOTS, team_b)):
        for slot, pid in zip(slots, team):
            p = players[pid]
            net_id = p.get("net_id")
            if not net_id or net_id not in idx:
                raise KeyError(f"player {pid} has no manifest net (net_id={net_id!r})")
            models[slot] = idx[net_id]
            factions[slot] = p["faction"]
            slot_pids.append((slot, pid))
    return models, factions, slot_pids


def make_soloq_match_fn(
    world: dict,
    manifest_path: Optional[str | Path] = None,
    *,
    run_match_fn: Callable = _default_run_match,
    orders_provider: Optional[Callable[[list, list], Optional[dict]]] = None,
    replay_dir: Optional[str | Path] = None,
) -> Callable:
    """Return a SoloQ ``match_fn(team_a, team_b, ratings, rng) -> (winner, impacts)`` that plays
    a real 3v3 net match. ``team_a``/``team_b`` are 3 World player ids each.

    ``orders_provider(team_a, team_b) -> orders|None`` lets a caller overlay manager game-plans
    (SoloQ normally has none — the nets play free). ``replay_dir`` set -> EVERY match is saved as
    a replay there (owner: record all World matches); None -> headless (tests)."""
    idx = net_model_index(load_manifest(manifest_path))
    rdir = Path(replay_dir) if replay_dir is not None else None
    if rdir is not None:
        rdir.mkdir(parents=True, exist_ok=True)

    def match_fn(team_a, team_b, ratings, rng):
        models, factions, slot_pids = _models_factions(world, idx, list(team_a), list(team_b))
        seed = rng.randrange(1, 2 ** 31 - 1)
        rec = rdir is not None
        kwargs = dict(seed=seed, bot_type="league",
                      mode="replay" if rec else "training",
                      replay_path=str(rdir / f"soloq_{seed}.json") if rec else None,
                      player_models=models, player_factions=factions)
        if orders_provider is not None:
            orders = orders_provider(list(team_a), list(team_b))
            if orders:
                kwargs["orders"] = orders
        result = run_match_fn(**kwargs)
        if result.winner == "TEAM_A":
            winner = "A"
        elif result.winner == "TEAM_B":
            winner = "B"
        else:  # genuine draw — no rating-flip bias, pick a side by the seeded rng
            winner = rng.choice(["A", "B"])
        return winner, _slot_impacts(result.state, slot_pids)

    return match_fn


class SoloQSeriesPlayer:
    """Picklable, STATELESS SoloQ 3v3 net player keyed by a PRE-ASSIGNED ``match_seed`` — so a round's
    independent matches can run over a process pool, bit-identical to sequential (each match re-seeds
    torch from its own seed). ``__call__(team_a, team_b, match_seed) -> (winner 'A'|'B', impacts)``.
    Genuine draws break deterministically off the seed (no shared rng). SoloQ has no manager orders."""

    def __init__(self, world: dict, manifest_path: Optional[str | Path] = None, *,
                 run_match_fn: Callable = _default_run_match, replay_dir: Optional[str | Path] = None):
        self.world = world
        self.manifest_path = str(manifest_path) if manifest_path else None
        self.run_match_fn = run_match_fn
        self.replay_dir = str(replay_dir) if replay_dir is not None else None
        self._idx: Optional[dict] = None

    def __call__(self, team_a, team_b, match_seed: int):
        if self._idx is None:
            self._idx = net_model_index(load_manifest(self.manifest_path))
        rdir = Path(self.replay_dir) if self.replay_dir is not None else None
        if rdir is not None:
            rdir.mkdir(parents=True, exist_ok=True)
        models, factions, slot_pids = _models_factions(self.world, self._idx, list(team_a), list(team_b))
        rec = rdir is not None
        result = self.run_match_fn(seed=match_seed, bot_type="league",
                                   mode="replay" if rec else "training",
                                   replay_path=str(rdir / f"soloq_{match_seed}.json") if rec else None,
                                   player_models=models, player_factions=factions)
        if result.winner == "TEAM_A":
            winner = "A"
        elif result.winner == "TEAM_B":
            winner = "B"
        else:                                            # deterministic draw break off the match seed
            winner = "A" if (match_seed % 2 == 0) else "B"
        return winner, _slot_impacts(result.state, slot_pids)


class NetSeriesPlayer:
    """Picklable, STATELESS Bo-N net-series player: plays ONE series for a PRE-ASSIGNED ``series_seed``
    and returns the result WITHOUT mutating the world. Position-stat credit is returned as ``map_lineups``
    (per map: home lineup, away lineup, winner side) for the CALLER to apply — so a worker process never
    writes to the (read-only) world snapshot. The sequential ``play()`` below and the parallel pool
    (``world.parallel``) both go through this one implementation, so their results are identical.

    Picklable: holds only the world dict (plain JSON), a manifest path, flags + a run-match callable
    (defaults to the module ``run_match`` which is picklable). The net-id index is rebuilt lazily inside
    the worker. Determinism: every per-map seed is derived from ``series_seed`` via ``Spawner``, and
    ``run_match`` re-seeds torch per map — so the outcome depends only on (home, away, best_of, seed)."""

    def __init__(self, world: dict, manifest_path: Optional[str | Path] = None, *,
                 run_match_fn: Callable = _default_run_match, use_playbook: bool = True,
                 replay_dir: Optional[str | Path] = None):
        self.world = world
        self.manifest_path = str(manifest_path) if manifest_path else None
        self.run_match_fn = run_match_fn
        self.use_playbook = use_playbook
        self.replay_dir = str(replay_dir) if replay_dir is not None else None
        self._idx: Optional[dict] = None

    def __call__(self, home: str, away: str, best_of: int, series_seed: int) -> dict:
        from src.core.rng import Spawner
        from src.training.spaces import strategy_order_for_slot
        from src.world import playbook
        from src.world.world import POSITIONS

        if self._idx is None:
            self._idx = net_model_index(load_manifest(self.manifest_path))
        idx, world = self._idx, self.world
        rdir = Path(self.replay_dir) if self.replay_dir is not None else None
        if rdir is not None:
            rdir.mkdir(parents=True, exist_ok=True)

        def _lineup_pids(lineup_map: dict) -> list[str]:
            return [lineup_map[pos] for pos in POSITIONS]

        def _orders(home_strat, away_strat) -> Optional[dict]:
            o = {}
            for i, slot in enumerate(HOME_SLOTS):
                o[slot] = strategy_order_for_slot(home_strat, i)
            for i, slot in enumerate(AWAY_SLOTS):
                o[slot] = strategy_order_for_slot(away_strat, i)
            return o

        needed = best_of // 2 + 1
        game_spawner = Spawner(series_seed)              # per-game seeds derived from THIS series seed
        mh = ma = 0
        streak_h = streak_a = 0          # consecutive-loss streaks for the comeback IF-THEN rules
        sequence: list[str] = []         # winner of each decisive map (for the no-scrub broadcast)
        map_results: list[dict] = []     # per map played: winner + the REAL replay file (for "watch the match")
        map_lineups: list[tuple] = []    # per map: (home_lineup, away_lineup, winner_side) for position credit
        last_meta = None
        for _ in range(best_of):  # hard cap: a Bo-N can't exceed N games even with draws
            if mh >= needed or ma >= needed:
                break
            # RE-RESOLVE the playbook with the live series score -> conditional strategy switch +
            # substitution fire here (the club can change its three nets between games).
            if self.use_playbook:
                sh = {"series_won": mh, "series_lost": ma, "playoff_losing_streak": streak_h}
                sa = {"series_won": ma, "series_lost": mh, "playoff_losing_streak": streak_a}
                _dh, _da, meta = playbook.matchup_modifiers(world, home, away, best_of,
                                                            state_home=sh, state_away=sa)
                home_lu, away_lu = meta["home"]["lineup"], meta["away"]["lineup"]
                orders = _orders(meta["home"]["strategy"], meta["away"]["strategy"])
                last_meta = meta
            else:
                home_lu = playbook.optimal_lineup(world, home)
                away_lu = playbook.optimal_lineup(world, away)
                orders = None
            models, factions, slot_pids = _models_factions(
                world, idx, _lineup_pids(home_lu), _lineup_pids(away_lu))
            # slot -> World player NAME (+ SoloQ) so the replay can label each player at their core
            player_names = {slot: world["players"].get(pid, {}).get("nickname", pid)
                            for slot, pid in slot_pids}

            gseed = game_spawner.seed()
            rec = rdir is not None
            replay_name = f"league_{gseed}.json" if rec else None
            result = self.run_match_fn(seed=gseed, bot_type="league",
                                       mode="replay" if rec else "training",
                                       replay_path=str(rdir / replay_name) if rec else None,
                                       player_models=models, player_factions=factions, orders=orders,
                                       player_names=player_names)
            if result.winner == "TEAM_A":
                mh += 1; streak_h = 0; streak_a += 1; sequence.append(home)
                map_winner = home
            elif result.winner == "TEAM_B":
                ma += 1; streak_a = 0; streak_h += 1; sequence.append(away)
                map_winner = away
            else:
                map_winner = None                        # draw: played but clinches nothing
            # remember the REAL replay file for this map so the broadcast can play the actual match
            map_results.append({"winner": map_winner, "winner_side": result.winner, "replay": replay_name})
            map_lineups.append((home_lu, away_lu, result.winner))   # caller credits position stats
        winner = home if mh > ma else (away if ma > mh else home)  # tie on the cap -> home seed
        out = {"home": home, "away": away, "maps_home": mh, "maps_away": ma, "winner": winner,
               "best_of": best_of, "sequence": sequence, "map_results": map_results,
               "map_lineups": map_lineups}
        if last_meta:
            out["playbook"] = last_meta
        return out


def apply_position_credit(world: dict, map_lineups: list) -> None:
    """Credit per-position winrate from a series' ``map_lineups`` (in map order) — the side effect
    NetSeriesPlayer deferred so workers stay world-read-only. Commutative, so order-safe."""
    from src.world import playbook
    for home_lu, away_lu, side in map_lineups or []:
        playbook.record_position_results(world, home_lu, side == "TEAM_A")
        playbook.record_position_results(world, away_lu, side == "TEAM_B")


def make_net_play_fn(
    world: dict,
    manifest_path: Optional[str | Path] = None,
    *,
    seed: Optional[int] = None,
    run_match_fn: Callable = _default_run_match,
    use_playbook: bool = True,
    record_positions: bool = True,
    replay_dir: Optional[str | Path] = None,
) -> Callable:
    """A season ``play(home, away, best_of) -> result`` that plays a REAL net Bo-N series
    between each team's playbook LINEUP. Same result contract as ``season.make_play_fn`` (home,
    away, maps_home, maps_away, winner, playbook-meta) so it drops in as the season's ``play=``,
    but the winner comes from actual GameEngine matches instead of roster-rating coin flips.

    The manager-LLM still chooses strategy + lineup (via ``playbook.matchup_modifiers``); here
    that lineup decides WHICH three nets play, and its strategy is sent to the engine as a
    per-side ORDER overlay so the nets actually play the called game plan. The playbook is
    RE-RESOLVED before every game with the live series score, so conditional strategy switches AND
    substitutions (a club swapping a player mid-series) really happen — the club, not its current
    three nets, is what persists.

    Sequential default: this wrapper draws each series' seed from a single ``Spawner`` and applies
    position credit immediately. The PARALLEL path (``live_season.play_and_store(workers=N)``) instead
    pre-assigns all series seeds, runs ``NetSeriesPlayer`` over a process pool, and applies the credit
    afterwards in canonical order — identical results either way."""
    from src.core.rng import Spawner

    player = NetSeriesPlayer(world, manifest_path, run_match_fn=run_match_fn,
                             use_playbook=use_playbook, replay_dir=replay_dir)
    series_spawner = Spawner(seed)

    def play(home: str, away: str, best_of: int) -> dict:
        out = player(home, away, best_of, series_spawner.seed())
        lineups = out.pop("map_lineups", [])
        if use_playbook and record_positions:
            apply_position_credit(world, lineups)
        return out

    return play


def world_match_fns(world: dict, manifest_path: Optional[str | Path] = None,
                    replay_dir: Optional[str | Path] = None) -> tuple:
    """Convenience: (soloq_match_fn, season_play_fn) for a population-backed world, or
    ``(None, None)`` for an abstract world (callers then fall back to the rating defaults).
    ``replay_dir`` set -> every World match is recorded as a replay there."""
    if not world.get("streamers") and not any(
            p.get("net_id") for p in world.get("players", {}).values()):
        return None, None
    return (make_soloq_match_fn(world, manifest_path, replay_dir=replay_dir),
            make_net_play_fn(world, manifest_path, replay_dir=replay_dir))
