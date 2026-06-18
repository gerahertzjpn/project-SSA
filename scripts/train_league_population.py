"""Train the 90-net league population from the 18 ancestors (fork + train-vs-ancestors).

    python scripts/train_league_population.py --init-only        # fork-init + pool, no training
    python scripts/train_league_population.py --smoke            # tiny end-to-end pipeline check
    python scripts/train_league_population.py --updates 100      # a real chunk (needs frozen ancestors)

A REAL run should wait until base-18 has plateaued and stopped (ancestors frozen). --smoke
tolerates still-training ancestors: it only verifies the spine (fork-init -> rollout vs
ancestor pool -> PPO update -> checkpoint save/load).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from src.training.league_trainer import LeagueConfig, LeagueTrainer  # noqa: E402
from src.training.policy import MaskedActorCritic  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="training_runs/league_population_v1")
    ap.add_argument("--ancestor-run-dir", default="training_runs/population_v1")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--episodes-per-triple", type=int, default=2)
    ap.add_argument("--max-triples-per-update", type=int, default=0)
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init-only", action="store_true", help="fork-init + ancestor pool, then exit")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end pipeline check")
    args = ap.parse_args()

    run_dir = args.run_dir
    if args.smoke:
        run_dir = "training_runs/_league_smoke"

    cfg = LeagueConfig(
        run_dir=run_dir, ancestor_run_dir=args.ancestor_run_dir,
        updates=2 if args.smoke else args.updates,
        episodes_per_triple=1 if args.smoke else args.episodes_per_triple,
        max_triples_per_update=2 if args.smoke else args.max_triples_per_update,
        checkpoint_interval=1 if args.smoke else args.checkpoint_interval,
        device=args.device,
    )
    trainer = LeagueTrainer(cfg)
    print(f"forked {len(trainer.nets)} descendants from {len(trainer.ancestors)} ancestors "
          f"-> {cfg.run_dir}")
    print(f"ancestor opponent pool: {trainer.pool_path.name}  "
          f"({trainer.pool_path.stat().st_size} bytes)")

    if args.init_only:
        # prove a forked net is a real, distinct, runnable policy
        any_id = next(iter(trainer.nets))
        n_params = sum(p.numel() for p in trainer.nets[any_id].parameters())
        print(f"sample net {any_id}: {n_params} params, lr={trainer.desc_by_id[any_id].learning_rate}")
        print("init-only: OK (no training)")
        return 0

    if args.smoke:
        print(f"triples this update (capped): {min(cfg.max_triples_per_update, len(trainer._triples()))}")

    for u in range(1, cfg.updates + 1):
        rec = trainer.run_update(u)
        print(f"  update {u}: triples={rec['triples']} eps={rec['episodes']} "
              f"win={rec['win_rate']} mean_steps={rec['mean_episode_steps']} "
              f"nets_updated={rec['nets_updated']}")
        if rec["sample_loss"]:
            sl = rec["sample_loss"]
            print(f"    sample loss keys: {sorted(sl)[:6]}")

    trainer.save_checkpoints(cfg.updates)
    trainer.write_registry(cfg.updates)

    # checkpoint save/load round-trip check
    any_id = next(iter(trainer.nets))
    ck = Path(cfg.run_dir) / "nets" / any_id / "latest.pt"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    reloaded = MaskedActorCritic(hidden_size=trainer.desc_by_id[any_id].hidden_size)
    reloaded.load_state_dict(blob["policy_state"])
    print(f"checkpoint round-trip OK: {any_id} ({ck.stat().st_size} bytes, "
          f"genotype {blob['genotype_hash']})")
    print("SMOKE OK" if args.smoke else "DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
