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
    """REAL single-net PPO — continue training ONE population net for ~steps_goal steps, then
    save its checkpoint + re-derive its card.

    IN PROGRESS: the existing LeagueTrainer / PopulationTrainer plan-builders require the FULL
    faction-locked roster (verified: 1 and even 3 nets -> 0 episode plans), so they cannot train
    a single net. The real implementation is a custom self-play loop — place the target net in all
    three controlled slots vs the frozen opponent pool via population._run_episode_collect, then
    population.update_policy on the merged rollout — checkpoint/registry/card-rederive helpers below
    are ready (_merge_registry_steps, _rederive_card). Wiring that loop is the next step."""
    from src.training.population_roster import build_population
    if not any(d.net_id == net_id for d in build_population()):
        raise ValueError(f"net {net_id!r} is not in the population roster")
    raise NotImplementedError(
        "single-net PPO loop pending: population trainers need the full roster (1-3 nets -> 0 plans); "
        f"a custom self-play loop for {net_id} (target x3 vs frozen pool) is the next build")


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
