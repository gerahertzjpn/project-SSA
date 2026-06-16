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

DEFAULT_RUN_DIR = "training_runs/population_18_v1"


class RealTrainingBackend(training_ops.TrainingBackend):
    def __init__(self, run_dir: str = DEFAULT_RUN_DIR, *, require_flag: bool = True):
        self.run_dir = run_dir
        self.require_flag = require_flag

    def _gate(self) -> None:
        if self.require_flag:
            feature_flags.assert_enabled("real_training_backend")  # raises FeatureLocked

    # -- PPO-style ops ----------------------------------------------------- #
    def continue_ppo(self, ctx: dict) -> dict:
        self._gate()
        # TODO(owner 'go'): load PopulationTrainer(run_dir), pick ctx['target'] net,
        # run ctx['total_steps'] PPO updates (for meta_adapt weight the opponent pool
        # toward the season's top teams; for style_spec/obedience_tune bias the
        # reward/order shaping), checkpoint + update registry.
        raise NotImplementedError("continue_ppo: enable the flag and wire PopulationTrainer")

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
