"""Freeze the trained population into a single canonical manifest that World consumes.

After the league-population run completes (see training_runs/league_population_v1), the owner
"freezes" the population: this writes ONE immutable file, population/frozen_population_v1.json,
that is the authoritative trained roster for World mode —

  - 90 DESCENDANTS (the league "pros"): signable, play league + SoloQ. Pulled from the run
    registry (net_id, faction, archetype, generation, parent, genotype_hash, params, steps,
    strategy_bias) + their checkpoint at <run>/nets/<net_id>/latest.pt.
  - 18 ANCESTORS (the "streamers"): permanent free agents, SoloQ only, NOT signable, the
    fork-source for rookies. Pulled from build_default_population() + their checkpoint at
    training_runs/population_v1/nets/<net_id>/latest.pt.

The manifest records a content hash of every checkpoint so the freeze is verifiable and
tamper-evident. World's bootstrap reads this file to seed its player pool (net-backed players)
instead of synthesising abstract ones. Re-running is safe and idempotent (same inputs -> same
manifest, minus the timestamp).

    python scripts/freeze_population.py            # write + verify
    python scripts/freeze_population.py --verify    # verify an existing manifest only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.training.population import build_default_population  # noqa: E402

RUN_DIR = ROOT / "training_runs" / "league_population_v1"
ANCESTOR_NETS = ROOT / "training_runs" / "population_v1" / "nets"
OUT = ROOT / "population" / "frozen_population_v1.json"
SCHEMA = "frozen_population/v1"


def _ckpt_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _latest_diversity() -> dict:
    p = RUN_DIR / "diversity_progress.jsonl"
    if not p.exists():
        return {}
    last = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not last:
        return {}
    rec = json.loads(last[-1])
    return {
        "diversity_score": rec.get("diversity_score"),
        "brother_pairs_total": rec.get("brother_pairs_total"),
        "by_faction": {f: v.get("brother_pairs") for f, v in (rec.get("by_faction") or {}).items()},
    }


def build_manifest(hash_checkpoints: bool = True) -> dict:
    reg = json.loads((RUN_DIR / "registry.json").read_text(encoding="utf-8"))

    descendants = []
    for n in reg["entries"]:
        ckpt = RUN_DIR / "nets" / n["net_id"] / "latest.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"descendant checkpoint missing: {ckpt}")
        descendants.append({
            "net_id": n["net_id"],
            "role": "pro",                       # league player
            "faction": n["faction"],
            "archetype": n.get("archetype"),
            "generation": n["generation"],
            "parent_ids": n.get("parent_ids", []),
            "genotype_hash": n.get("genotype_hash"),
            "hidden_size": n.get("hidden_size"),
            "parameter_count": n.get("parameter_count"),
            "strategy_bias": n.get("strategy_bias"),
            "steps_trained": n.get("steps_trained"),
            "signable": True,
            "plays_league": True,
            "free_agent": False,
            "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256_16": _ckpt_hash(ckpt) if hash_checkpoints else None,
        })

    ancestors = []
    for a in build_default_population():
        ckpt = ANCESTOR_NETS / a.net_id / "latest.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"ancestor checkpoint missing: {ckpt}")
        ancestors.append({
            "net_id": a.net_id,
            "role": "streamer",                  # free agent, SoloQ-only, fork-source
            "faction": a.faction,
            "hidden_size": a.hidden_size,
            "genotype_hash": a.genotype_hash(),
            "signable": False,
            "plays_league": False,
            "free_agent": True,
            "soloq_only": True,
            "fork_source": True,
            "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256_16": _ckpt_hash(ckpt) if hash_checkpoints else None,
        })

    return {
        "schema": SCHEMA,
        "frozen": True,
        "frozen_at": int(time.time()),
        "source_run": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "total_steps": reg.get("total_steps"),
        "updates_trained": reg.get("updates_trained"),
        "counts": {"descendants": len(descendants), "ancestors": len(ancestors),
                   "total": len(descendants) + len(ancestors)},
        "factions": {f: sum(1 for d in descendants if d["faction"] == f)
                     for f in ("humans", "infected", "robots")},
        "diversity": _latest_diversity(),
        "descendants": descendants,
        "ancestors": ancestors,
    }


def verify(manifest: dict) -> list[str]:
    """Return a list of problems (empty = OK): checkpoints present + hashes match."""
    problems = []
    for group in ("descendants", "ancestors"):
        for e in manifest.get(group, []):
            ckpt = ROOT / e["checkpoint"]
            if not ckpt.exists():
                problems.append(f"missing checkpoint: {e['net_id']} -> {ckpt}")
                continue
            want = e.get("checkpoint_sha256_16")
            if want and _ckpt_hash(ckpt) != want:
                problems.append(f"checkpoint hash mismatch (modified): {e['net_id']}")
    c = manifest.get("counts", {})
    if c.get("descendants") != 90:
        problems.append(f"expected 90 descendants, got {c.get('descendants')}")
    if c.get("ancestors") != 18:
        problems.append(f"expected 18 ancestors, got {c.get('ancestors')}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify the existing manifest only")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    if args.verify:
        if not out.exists():
            print(f"no manifest at {out}")
            sys.exit(1)
        manifest = json.loads(out.read_text(encoding="utf-8"))
        problems = verify(manifest)
        print(f"verify {out.name}: {'OK' if not problems else str(len(problems)) + ' PROBLEMS'}")
        for p in problems:
            print("  -", p)
        sys.exit(0 if not problems else 1)

    print("building frozen population manifest (hashing 108 checkpoints)...")
    manifest = build_manifest()
    problems = verify(manifest)
    if problems:
        print("REFUSING to write — verification failed:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    c = manifest["counts"]
    print(f"FROZEN -> {out.relative_to(ROOT)}")
    print(f"  {c['descendants']} descendants + {c['ancestors']} ancestors = {c['total']} nets")
    print(f"  factions: {manifest['factions']}  | total_steps {manifest['total_steps']:,}")
    print(f"  diversity: {manifest['diversity']}")
    print("  all 108 checkpoints present + hashed (tamper-evident).")


if __name__ == "__main__":
    main()
