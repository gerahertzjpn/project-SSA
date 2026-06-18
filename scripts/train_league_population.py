"""Train the 90-net league population (parallel) from the 18 ancestors.

    python scripts/train_league_population.py --init-only        # fork-init + pool, no training
    python scripts/train_league_population.py --smoke            # tiny parallel end-to-end check
    python scripts/train_league_population.py                    # real run: ~1.4M steps/net

Each descendant forks (warm start) from its parent ancestor and trains against a DIVERSE
opponent pool (frozen ancestors + scripted rush/tech/turtle) to break the rush monoculture.
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
    ap.add_argument("--target-steps-per-net", type=int, default=1_400_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scripted-prob", type=float, default=0.45)
    ap.add_argument("--episodes-per-triple", type=int, default=2)
    ap.add_argument("--checkpoint-interval", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-updates", type=int, default=None,
                    help="cap updates (calibration/debug); real run omits this")
    args = ap.parse_args()

    run_dir = "training_runs/_league_smoke" if args.smoke else args.run_dir
    cfg = LeagueConfig(
        run_dir=run_dir, ancestor_run_dir=args.ancestor_run_dir,
        target_steps_per_net=1 if args.smoke else args.target_steps_per_net,
        workers=2 if args.smoke else args.workers,
        scripted_opponent_prob=args.scripted_prob,
        episodes_per_triple=1 if args.smoke else args.episodes_per_triple,
        max_triples_per_update=2 if args.smoke else 0,
        checkpoint_interval=1 if args.smoke else args.checkpoint_interval,
        device=args.device,
    )
    trainer = LeagueTrainer(cfg)
    print(f"forked {len(trainer.nets)} descendants from {len(trainer.ancestors)} ancestors "
          f"-> {cfg.run_dir} (device {trainer.device})")
    print(f"ancestor pool: {trainer.pool_path.name} ({trainer.pool_path.stat().st_size} bytes); "
          f"scripted_opponent_prob={cfg.scripted_opponent_prob}")

    if args.init_only:
        any_id = next(iter(trainer.nets))
        print(f"sample {any_id}: {sum(p.numel() for p in trainer.nets[any_id].parameters())} params, "
              f"lr={trainer.desc_by_id[any_id].learning_rate}")
        print("init-only: OK")
        return 0

    if args.smoke:
        trainer.train(max_updates=2)
        # verify a checkpoint round-trips
        any_id = next(iter(trainer.nets))
        ck = Path(cfg.run_dir) / "nets" / any_id / "latest.pt"
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        MaskedActorCritic(hidden_size=trainer.desc_by_id[any_id].hidden_size).load_state_dict(blob["policy_state"])
        print(f"checkpoint round-trip OK: {any_id} (genotype {blob['genotype_hash']}, "
              f"steps {blob['steps_trained']})")
        print("SMOKE OK")
        return 0

    print(f"training to {cfg.target_steps_per_net:,} steps/net on {cfg.workers} workers"
          f"{' (max %d updates)' % args.max_updates if args.max_updates else ''}...")
    trainer.train(max_updates=args.max_updates)
    print(f"DONE: total_steps {trainer.total_steps:,}, min steps/net {min(trainer.steps_by_net.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
