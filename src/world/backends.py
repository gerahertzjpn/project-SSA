"""Real training backend — wired to the PPO/lineage infra, GATED in development.

This adapter connects the World training dispatch (training_ops.execute) to the
real machinery:
  - fork (create_player / rookies) -> src/training/lineage.run_branch_select_cycle
  - skill_up / meta_adapt / style_spec / obedience_tune -> PopulationTrainer PPO
  - teamplay -> joint roster training in the team env
  - evaluate -> held-out gauntlet winrate / net-Elo

It is intentionally LOCKED: every method asserts the ``real_training_backend``
feature flag, which is OFF (``in_development``). So the backend is *wired and
selectable in code* but *refuses to run* until the owner enables it — matching
"вшей backend, но временно заблокируй для выбора, пометь как в разработке".

The launch path for the nets themselves already exists:
``python scripts/train_population.py`` (18 unique-seed, faction-locked nets).
"""
from __future__ import annotations

from src.world import feature_flags, training_ops

DEFAULT_RUN_DIR = "training_runs/league_population_v1"   # the frozen league population (90 nets)


def _ctx_net_id(ctx: dict) -> str:
    """The net to (re)train from a training-op context (player card -> net_id)."""
    if ctx.get("net_id"):
        return str(ctx["net_id"])
    tgt = ctx.get("target") or {}
    nid = tgt.get("net_id") or tgt.get("player_id")
    if not nid:
        raise ValueError(f"training ctx has no net_id/target: {ctx!r}")
    return str(nid)


def _merge_registry_steps(run_dir: str, net_id: str, steps: int, update: int) -> None:
    """Update ONLY this net's row in the league registry (never clobber the other 89)."""
    import json
    from pathlib import Path
    reg_path = Path(run_dir) / "registry.json"
    if not reg_path.exists():
        return
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for e in reg.get("entries", []):
        if e.get("net_id") == net_id:
            e["steps_trained"] = int(steps)
            e["updates_trained"] = int(update)
            break
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _rederive_card(net_id: str) -> None:
    """Re-measure this net's card after training (owner: cards change only after training)."""
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / "scripts" / "derive_player_cards.py"),
                    "--nets", net_id, "--games", "12"], cwd=str(root), check=False)


def train_one_net(run_dir: str, net_id: str, steps_goal: int, progress_cb=None) -> dict:
    """REAL single-net PPO — continue training ONE population net for ~steps_goal steps vs the
    league opponent mix (frozen ancestors + co-evo snapshot + scripted), then save ONLY this net's
    checkpoint, merge its registry row, and re-derive its card.

    The LeagueTrainer schedules controlled play in faction-balanced TRIPLES (one net per faction;
    `_triples()` needs >=1 descendant per faction or it yields zero plans). So to continue-train a
    single target we add ONE sparring partner from each OTHER faction — that forms exactly one valid
    triple. The target trains faction-locked in its slot; the two partners co-train in-memory only and
    are DISCARDED (never saved), so their on-disk checkpoints are untouched. Faithful to the original
    league recipe, and it isolates the write to the target alone."""
    import torch
    from pathlib import Path
    from src.training.league_trainer import LeagueTrainer, LeagueConfig, FACTIONS
    from src.training.population_roster import build_population
    from src.training.policy import to_cpu_state_dict

    pop = build_population()
    target = next((d for d in pop if d.net_id == net_id), None)
    if target is None:
        raise ValueError(f"net {net_id!r} is not in the population roster")

    # one sparring partner per OTHER faction -> exactly one faction-balanced triple
    partners = []
    for fac in FACTIONS:
        if fac == target.faction:
            continue
        cand = next((d for d in pop if d.faction == fac and d.net_id != net_id), None)
        if cand is None:
            raise RuntimeError(f"no sparring partner available for faction {fac!r}")
        partners.append(cand)
    descendants = [target, *partners]

    # target_steps_per_net is effectively unbounded here; we stop on the target's own step delta.
    # No co-evo descendant opponents: the only descendants loaded are the controlled triple itself,
    # so the "descendants" opponent branch would have nobody to draw -> train vs frozen ancestors
    # (the rush pressure) + scripted baselines, which is the faithful single-net opponent mix.
    cfg = LeagueConfig(run_dir=run_dir, workers=1, episodes_per_triple=2,
                       target_steps_per_net=1_000_000_000_000,
                       ancestor_opponent_prob=0.7, descendant_opponent_prob=0.0)
    trainer = LeagueTrainer(cfg, descendants=descendants)   # auto-resumes the 3 nets + builds a pool
    fresh_pool = getattr(trainer, "pool_path", None)        # the pool_v0 we just created (cleanup later)

    start = int(trainer.steps_by_net[net_id])
    update = 0
    try:
        while int(trainer.steps_by_net[net_id]) - start < steps_goal:
            update += 1
            trainer.run_update(update, pool=None)
            if progress_cb:
                progress_cb(min(steps_goal, int(trainer.steps_by_net[net_id]) - start), steps_goal)

        # save ONLY the target (mirror LeagueTrainer.save_checkpoints for one net) — never the partners
        net = trainer.nets[net_id]
        net_dir = Path(run_dir) / "nets" / net_id
        net_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"policy_state": to_cpu_state_dict(net), "update": update, "net_id": net_id,
                    "genotype_hash": target.genotype_hash(),
                    "steps_trained": int(trainer.steps_by_net[net_id])}, net_dir / "latest.pt")
        _merge_registry_steps(run_dir, net_id, int(trainer.steps_by_net[net_id]), update)
        _rederive_card(net_id)
    finally:
        # drop the transient pool we wrote in __init__ so the canonical run dir stays clean
        if fresh_pool and Path(fresh_pool).name.startswith("pool_v0") and Path(fresh_pool).exists():
            try:
                Path(fresh_pool).unlink()
            except OSError:
                pass

    return {"net_id": net_id, "updates": update,
            "steps_added": int(trainer.steps_by_net[net_id]) - start,
            "steps_trained": int(trainer.steps_by_net[net_id])}


class RealTrainingBackend(training_ops.TrainingBackend):
    def __init__(self, run_dir: str = DEFAULT_RUN_DIR, *, require_flag: bool = True):
        self.run_dir = run_dir
        self.require_flag = require_flag

    def _gate(self) -> None:
        if self.require_flag:
            feature_flags.assert_enabled("real_training_backend")  # raises FeatureLocked

    # -- PPO-style ops ----------------------------------------------------- #
    def continue_ppo(self, ctx: dict, progress_cb=None) -> dict:
        """skill_up / meta_adapt / style_spec / obedience_tune — REAL PPO on the target net."""
        self._gate()
        return train_one_net(self.run_dir, _ctx_net_id(ctx),
                             max(1, int(ctx.get("total_steps", 1_000_000))), progress_cb)

    def cotrain(self, ctx: dict) -> dict:
        self._gate()
        # TODO(owner 'go'): joint training of the roster's 3 nets in TeamTrainingEnv.
        raise NotImplementedError("cotrain: enable the flag and wire team co-training")

    def fork(self, ctx: dict) -> dict:
        self._gate()
        # TODO(owner 'go'): lineage.run_branch_select_cycle from ctx['target'] parent
        # with ctx['total_steps']; register the unique child (genotype_hash) + lineage.
        raise NotImplementedError("fork: enable the flag and wire lineage.run_branch_select_cycle")

    def evaluate(self, ctx: dict) -> dict:
        self._gate()
        # TODO(owner 'go'): held-out gauntlet winrate / net-Elo before vs after.
        raise NotImplementedError("evaluate: enable the flag and wire the gauntlet")


def get_backend(*, require_flag: bool = True) -> RealTrainingBackend:
    """Factory used by the engine/UI. The returned backend is locked until the
    feature flag is enabled."""
    return RealTrainingBackend(require_flag=require_flag)
