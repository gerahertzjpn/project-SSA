"""
Round-robin strength test for the current v3/v4 neural network population.
Tests all models against each other (and vs scripted baseline), reports win-rates.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.constants import PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.core.engine import GameEngine
from src.core.runner import current_code_commit, make_bots_from_models, rules_meta, run_match
from src.core.state import make_initial_state


# ── Candidates ──────────────────────────────────────────────────────────────
# Format: (label, relative_path_to_checkpoint)
CANDIDATES = [
    ("spv3_01",    "training_runs/fresh_population_v31_selfplay_v3_tmix10_vs_tmix11_s20264201/checkpoints/checkpoint_update_200.pt"),
    ("spv3_02",    "training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix08_s20264202/checkpoints/checkpoint_update_240.pt"),
    ("spv3_03R",   "training_runs/fresh_population_v31_selfplay_v3_recovery_spv303_vs_peer10_s20264303/checkpoints/checkpoint_update_420.pt"),
    ("spv3_04",    "training_runs/fresh_population_v31_selfplay_v3_tmix09_vs_tmix10_s20264204/checkpoints/checkpoint_update_240.pt"),
    ("spv3_05R",   "training_runs/fresh_population_v31_selfplay_v3_recovery_spv305_vs_peer10_s20264305/checkpoints/checkpoint_update_340.pt"),
    ("spv3_06",    "training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix09_s20264206/checkpoints/checkpoint_update_200.pt"),
    ("v4_tmix12",  "training_runs/fresh_population_v31_teammate_mix_v4_tmix12_465_s20264412/checkpoints/latest.pt"),
    ("tmix13_init","training_runs/fresh_population_v31_teammate_mix_inits/tmix13_spv4_243.pt"),
    ("tmix14_init","training_runs/fresh_population_v31_teammate_mix_inits/tmix14_spv4_345.pt"),
]

SEEDS_PER_MATCHUP = 20
BASELINE_SEEDS = 20
BASE_SEED = 20260901


def model_id(rel_path: str) -> str:
    return f"policy:{rel_path}"


def team_player_models(a_path: str, b_path: str) -> dict[str, str]:
    """Map each player_id to the correct team checkpoint.
    Team B naturally mirrors Team A slot by slot via PolicyBot fallback."""
    return {
        "A1": model_id(a_path),
        "A2": model_id(a_path),
        "A3": model_id(a_path),
        "B1": model_id(b_path),
        "B2": model_id(b_path),
        "B3": model_id(b_path),
    }


def run_series(a_label: str, a_path: str, b_label: str, b_path: str,
               n_seeds: int, seed_offset: int) -> dict:
    winners: Counter = Counter()
    winner_types: Counter = Counter()
    ticks: list[int] = []
    for i in range(n_seeds):
        seed = seed_offset + i
        try:
            player_models = team_player_models(a_path, b_path)
            state = make_initial_state(seed=seed)
            bots = make_bots_from_models(player_models, seed_offset=seed * 10)
            engine = GameEngine(state, bots, record_trace=False, record_replay=False)
            winner = engine.run()
            winners[winner] += 1
            winner_types[state.winner_type or "unknown"] += 1
            ticks.append(state.tick)
        except Exception as exc:
            winners["ERROR"] += 1
            print(f"  ERROR seed={seed}: {exc}", flush=True)
    games = n_seeds
    a_wins = winners.get("TEAM_A", 0)
    b_wins = winners.get("TEAM_B", 0)
    draws = winners.get("DRAW", 0)
    errors = winners.get("ERROR", 0)
    return {
        "a": a_label,
        "b": b_label,
        "games": games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "errors": errors,
        "a_win_rate": round(a_wins / max(1, games - errors), 3),
        "b_win_rate": round(b_wins / max(1, games - errors), 3),
        "winner_types": dict(winner_types),
        "mean_ticks": round(sum(ticks) / max(1, len(ticks)), 1) if ticks else 0,
    }


def run_baseline_series(label: str, path: str, opponent: str,
                        n_seeds: int, seed_offset: int) -> dict:
    """Test a NN team as Team A against a fixed bot opponent (scripted/random)."""
    winners: Counter = Counter()
    winner_types: Counter = Counter()
    ticks: list[int] = []
    for i in range(n_seeds):
        seed = seed_offset + i
        try:
            player_models = {
                "A1": model_id(path),
                "A2": model_id(path),
                "A3": model_id(path),
                "B1": opponent,
                "B2": opponent,
                "B3": opponent,
            }
            state = make_initial_state(seed=seed)
            bots = make_bots_from_models(player_models, seed_offset=seed * 10)
            engine = GameEngine(state, bots, record_trace=False, record_replay=False)
            winner = engine.run()
            winners[winner] += 1
            winner_types[state.winner_type or "unknown"] += 1
            ticks.append(state.tick)
        except Exception as exc:
            winners["ERROR"] += 1
    a_wins = winners.get("TEAM_A", 0)
    b_wins = winners.get("TEAM_B", 0)
    errors = winners.get("ERROR", 0)
    return {
        "model": label,
        "opponent": opponent,
        "games": n_seeds,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "errors": errors,
        "win_rate": round(a_wins / max(1, n_seeds - errors), 3),
        "winner_types": dict(winner_types),
        "mean_ticks": round(sum(ticks) / max(1, len(ticks)), 1) if ticks else 0,
    }


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}", flush=True)


def main() -> None:
    t0 = time.time()
    print(f"NN Strength Test | commit={current_code_commit()} | {len(CANDIDATES)} models")
    print(f"Seeds per matchup: {SEEDS_PER_MATCHUP} | Base seed: {BASE_SEED}")

    # ── Baseline: each model vs scripted ────────────────────────────────────
    print_header("Baseline: NN vs scripted")
    baselines: list[dict] = []
    for i, (label, path) in enumerate(CANDIDATES):
        r = run_baseline_series(
            label, path, opponent="scripted",
            n_seeds=BASELINE_SEEDS,
            seed_offset=BASE_SEED + i * 1000
        )
        baselines.append(r)
        print(f"  {label:20s} vs scripted : win={r['win_rate']:.3f} ({r['a_wins']}/{r['games']})  "
              f"types={r['winner_types']}  avg_ticks={r['mean_ticks']}", flush=True)

    # ── Round-robin: all pairs ───────────────────────────────────────────────
    print_header("Round-robin matchups")
    matchups: list[dict] = []
    pair_offset = BASE_SEED + 100_000
    for i, (la, pa) in enumerate(CANDIDATES):
        for j, (lb, pb) in enumerate(CANDIDATES):
            if j <= i:
                continue
            seed_off = pair_offset + (i * len(CANDIDATES) + j) * SEEDS_PER_MATCHUP
            r = run_series(la, pa, lb, pb, SEEDS_PER_MATCHUP, seed_off)
            matchups.append(r)
            print(f"  {la:20s} vs {lb:20s} : "
                  f"A_win={r['a_win_rate']:.3f} ({r['a_wins']}/{r['games']})  "
                  f"types={r['winner_types']}  avg_ticks={r['mean_ticks']}", flush=True)

    # ── Aggregate ELO-style win table ────────────────────────────────────────
    print_header("Win-rate summary table (rows=Team A, cols=Team B)")
    labels = [c[0] for c in CANDIDATES]
    win_table: dict[str, dict[str, str]] = {la: {} for la, _ in CANDIDATES}
    for r in matchups:
        win_table[r["a"]][r["b"]] = f"{r['a_win_rate']:.2f}"
        win_table[r["b"]][r["a"]] = f"{r['b_win_rate']:.2f}"
    header = f"{'':20s}" + "".join(f"{lb:12s}" for lb in labels)
    print(header)
    for la in labels:
        row = f"{la:20s}"
        for lb in labels:
            if la == lb:
                row += f"{'---':12s}"
            else:
                row += f"{win_table[la].get(lb, '?'):12s}"
        print(row)

    # ── Total wins per model ─────────────────────────────────────────────────
    print_header("Aggregate wins (round-robin)")
    total_wins: Counter = Counter()
    total_played: Counter = Counter()
    for r in matchups:
        total_wins[r["a"]] += r["a_wins"]
        total_wins[r["b"]] += r["b_wins"]
        played = r["games"] - r["errors"]
        total_played[r["a"]] += played
        total_played[r["b"]] += played
    print(f"{'Model':20s} {'Wins':>6} {'Played':>7} {'Win%':>7}")
    for label, _ in sorted(CANDIDATES, key=lambda c: total_wins[c[0]], reverse=True):
        wins = total_wins[label]
        played = total_played[label]
        pct = wins / played if played > 0 else 0.0
        print(f"  {label:20s} {wins:>6}  {played:>6}  {pct:.3f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    report = {
        "meta": {"commit": current_code_commit(), "elapsed_s": elapsed, "base_seed": BASE_SEED},
        "candidates": [{"label": la, "path": pa} for la, pa in CANDIDATES],
        "baselines": baselines,
        "matchups": matchups,
    }
    out = Path("analysis") / "nn_strength_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}  (elapsed {elapsed}s)")


if __name__ == "__main__":
    main()
