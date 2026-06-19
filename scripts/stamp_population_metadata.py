"""Stamp policy-interface metadata into checkpoints that lack it.

The league-population trainer saved each net as {policy_state, update, net_id, genotype_hash,
steps_trained} but WITHOUT the observation_version / observation_dim / action_dim keys that
PolicyBot.validate_checkpoint_compatible requires (the ancestors, saved by the standard path,
do have them). The WEIGHTS are compatible — body.0.weight is (hidden, 5201) = the current obs
dim — they just aren't self-describing, so PolicyBot refuses to load them for real matches.

This migration loads each checkpoint, reads the REAL (obs_dim, action_dim) straight from the
weights via policy_state_interface, verifies they match the current interface, and only then
adds the three metadata keys + re-saves. The policy_state (the actual net) is byte-identical —
nothing about the trained weights changes. Idempotent: already-stamped checkpoints are skipped.

    python scripts/stamp_population_metadata.py --run-dir training_runs/league_population_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.bots.policy_bot import (  # noqa: E402
    OBSERVATION_VERSION, OBSERVATION_DIM, ACTION_DIM, policy_state_interface,
)


def stamp_checkpoint(path: Path) -> str:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "policy_state" not in ckpt:
        return "skip (no policy_state)"
    if ckpt.get("observation_version") is not None:
        return "already stamped"
    obs_dim, act_dim = policy_state_interface(ckpt["policy_state"])
    if obs_dim != OBSERVATION_DIM or act_dim != ACTION_DIM:
        return f"INCOMPATIBLE weights obs/action=({obs_dim},{act_dim}) — NOT stamped"
    ckpt["observation_version"] = OBSERVATION_VERSION
    ckpt["observation_dim"] = obs_dim
    ckpt["action_dim"] = act_dim
    torch.save(ckpt, path)
    return f"stamped ({OBSERVATION_VERSION}, {obs_dim}, {act_dim})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="training_runs/league_population_v1")
    args = ap.parse_args()
    nets = ROOT / args.run_dir / "nets"
    if not nets.exists():
        print(f"no nets dir at {nets}")
        sys.exit(1)

    ckpts = sorted(nets.glob("*/latest.pt"))
    print(f"stamping {len(ckpts)} checkpoints in {args.run_dir}/nets ...")
    counts: dict[str, int] = {}
    problems = []
    for ck in ckpts:
        result = stamp_checkpoint(ck)
        key = result.split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
        if "INCOMPATIBLE" in result:
            problems.append(f"{ck.parent.name}: {result}")
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n}")
    for p in problems:
        print("  !", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
