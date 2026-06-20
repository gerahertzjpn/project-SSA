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

from pathlib import Path
from typing import Optional

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


def train_one_net(run_dir: str, net_id: str, steps_goal: int, progress_cb=None,
                  *, workers: int = 26, sparring: Optional[list] = None,
                  order_weight: Optional[float] = None, forced_strategy: Optional[str] = None,
                  no_order_prob: Optional[float] = None, meta_opponents: Optional[list] = None) -> dict:
    """REAL single-net PPO — continue training ONE population net for ~steps_goal steps vs the
    league opponent mix (frozen ancestors + scripted), then save ONLY this net's checkpoint, merge
    its registry row, and re-derive its card.

    The LeagueTrainer schedules controlled play in faction-balanced TRIPLES (one net per faction;
    `_triples()` needs >=1 descendant per faction or it yields zero plans). So to continue-train a
    single target we add ONE sparring partner from each OTHER faction — that forms exactly one valid
    triple. The target trains faction-locked in its slot; the two partners co-train in-memory only and
    are DISCARDED (never saved), so their on-disk checkpoints are untouched.

    ``sparring`` optionally pins the partner net_ids (must be one per OTHER faction) — used to honour
    the "no nets from other teams as allies" rule (the caller passes own-team / free-agent nets).
    Default picks the first roster-agnostic candidate per faction.

    ``order_weight`` overrides the target's order-shaping pressure (obedience_tune); ``forced_strategy``
    pins every ordered episode to one preset id (style_spec); ``no_order_prob`` overrides the unordered
    fraction. The net's saved identity (genotype_hash) is the ORIGINAL — these only change the training
    pressure, not who the net is.

    ``workers`` fans episodes across a spawn pool; episodes_per_triple is matched to it so every
    worker gets an episode."""
    import multiprocessing as mp
    import torch
    from dataclasses import replace
    from pathlib import Path
    from src.training.league_trainer import LeagueTrainer, LeagueConfig, FACTIONS, _init_worker
    from src.training.population_roster import build_population
    from src.training.policy import to_cpu_state_dict

    pop = build_population()
    by_id = {d.net_id: d for d in pop}
    target = by_id.get(net_id)
    if target is None:
        raise ValueError(f"net {net_id!r} is not in the population roster")

    # one sparring partner per OTHER faction -> exactly one faction-balanced triple
    other_facs = [f for f in FACTIONS if f != target.faction]
    if sparring is not None:
        partners = [by_id[s] for s in sparring]
        have = {p.faction for p in partners}
        if have != set(other_facs):
            raise ValueError(f"sparring must be one net per other faction {other_facs}, got {sorted(have)}")
    else:
        partners = []
        for fac in other_facs:
            cand = next((d for d in pop if d.faction == fac and d.net_id != net_id), None)
            if cand is None:
                raise RuntimeError(f"no sparring partner available for faction {fac!r}")
            partners.append(cand)
    # the target spec used for TRAINING may carry an overridden order weight (obedience_tune);
    # identity (net_id, genotype_hash on save) stays the ORIGINAL target.
    train_target = target if order_weight is None else replace(target, order_shaping_weight=float(order_weight))
    descendants = [train_target, *partners]

    # target_steps_per_net is effectively unbounded here; we stop on the target's own step delta.
    # No co-evo descendant opponents: the only descendants loaded are the controlled triple itself,
    # so the "descendants" opponent branch would have nobody to draw -> train vs frozen ancestors
    # (the rush pressure) + scripted baselines, which is the faithful single-net opponent mix.
    workers = max(1, int(workers))
    cfg = LeagueConfig(run_dir=run_dir, workers=workers, episodes_per_triple=max(2, workers),
                       target_steps_per_net=1_000_000_000_000,
                       ancestor_opponent_prob=0.7, descendant_opponent_prob=0.0,
                       forced_strategy=forced_strategy,
                       no_order_prob=0.2 if no_order_prob is None else float(no_order_prob))
    trainer = LeagueTrainer(cfg, descendants=descendants)   # auto-resumes the 3 nets + builds a pool
    fresh_pool = getattr(trainer, "pool_path", None)        # the pool_v0 we just created (cleanup later)
    meta_pool = None
    if meta_opponents:
        # meta_adapt: train the target AGAINST the current top nets (the meta), not the ancestors.
        meta_pool = _install_meta_pool(trainer, run_dir, [m for m in meta_opponents if m != net_id])
        trainer.config.ancestor_opponent_prob = 0.0
        trainer.config.descendant_opponent_prob = 1.0

    pool = None
    if workers > 1:
        pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker,
                                            initargs=(cfg.torch_threads,))
    start = int(trainer.steps_by_net[net_id])
    update = 0
    try:
        while int(trainer.steps_by_net[net_id]) - start < steps_goal:
            update += 1
            trainer.run_update(update, pool=pool)
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
        if pool is not None:
            pool.close()
            pool.join()
        # drop the transient pools we wrote so the canonical run dir stays clean
        for stray in (fresh_pool, meta_pool):
            if stray and Path(stray).name.startswith("pool_") and Path(stray).exists():
                try:
                    Path(stray).unlink()
                except OSError:
                    pass

    return {"net_id": net_id, "updates": update,
            "steps_added": int(trainer.steps_by_net[net_id]) - start,
            "steps_trained": int(trainer.steps_by_net[net_id])}


_MANIFEST = Path(__file__).resolve().parents[2] / "population" / "frozen_population_v1.json"


def _manifest_index() -> dict:
    """net_id -> ('policy:<checkpoint>', faction) for every frozen net (pros + streamers)."""
    import json
    m = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return {e["net_id"]: ("policy:" + e["checkpoint"], e["faction"])
            for g in ("descendants", "ancestors") for e in m[g]}


def _one_per_faction(idx: dict, exclude=()) -> list:
    """One net_id per faction (humans/infected/robots) from the manifest, skipping ``exclude``."""
    from src.training.population_roster import FACTIONS
    out = []
    for fac in FACTIONS:
        pick = next((n for n, (_m, f) in idx.items() if f == fac and n not in exclude and n not in out), None)
        if pick is not None:
            out.append(pick)
    return out


def evaluate_net(net_id: str, opponents, *, run_dir: Optional[str] = None, n_games: int = 8,
                 teammates: Optional[list] = None, seed_base: int = 7000) -> float:
    """Headless gauntlet winrate of ``net_id`` (+2 teammates) vs ``opponents`` (3 nets), via
    run_match. The target plays its LIVE checkpoint under ``run_dir`` if present, else the manifest
    one — so before/after training is measured on the right weights."""
    from src.core.runner import run_match
    idx = _manifest_index()
    if net_id not in idx:
        raise ValueError(f"net {net_id!r} not in the frozen manifest")
    tgt_model, tgt_fac = idx[net_id]
    if run_dir:
        live = Path(run_dir) / "nets" / net_id / "latest.pt"
        if live.exists():
            tgt_model = "policy:" + str(live)
    opponents = [o for o in list(opponents) if o != net_id][:3]   # never face a copy of yourself
    if teammates is None:
        teammates = _one_per_faction(idx, exclude={net_id, *opponents})[:2]
    teammates = list(teammates)[:2]
    HOME, AWAY = ("A1", "A2", "A3"), ("B1", "B2", "B3")
    wins = played = 0
    for g in range(int(n_games)):
        models, factions = {}, {}
        for s, n in zip(HOME, [net_id, *teammates]):
            models[s], factions[s] = (tgt_model, tgt_fac) if n == net_id else idx[n]
        for s, n in zip(AWAY, opponents):
            models[s], factions[s] = idx[n]
        r = run_match(seed=seed_base + g, bot_type="league", mode="training",
                      player_models=models, player_factions=factions)
        wins += 1 if r.winner == "TEAM_A" else 0
        played += 1
    return round(wins / max(1, played), 4)


def _default_meta(run_dir: str, exclude: str, k: int = 6) -> list:
    """Top-``k`` nets by trained steps from the league registry (the de-facto 'meta'), minus
    ``exclude`` — the fallback opponent set for meta_adapt when the world doesn't pass one."""
    import json
    reg = Path(run_dir) / "registry.json"
    if not reg.exists():
        return []
    entries = json.loads(reg.read_text(encoding="utf-8")).get("entries", [])
    ranked = sorted((e for e in entries if e.get("net_id") != exclude),
                    key=lambda e: e.get("steps_trained", 0), reverse=True)
    return [e["net_id"] for e in ranked[:k]]


def _install_meta_pool(trainer, run_dir: str, meta_ids: list):
    """Point the trainer's opponent pool at the given 'meta' nets (loaded from their live
    checkpoints) and make _pick_descendants draw from them, so the target trains vs the top nets."""
    import torch
    from src.training.population import write_frozen_pool
    from src.training.population_roster import build_population
    by_id = {d.net_id: d for d in build_population()}
    states, specs = {}, {}
    for mid in meta_ids:
        ckpt = Path(run_dir) / "nets" / mid / "latest.pt"
        if not ckpt.exists() or mid not in by_id:
            continue
        states[mid] = torch.load(ckpt, map_location="cpu", weights_only=False)["policy_state"]
        specs[mid] = by_id[mid]
        trainer.faction_by_id[mid] = by_id[mid].faction
        trainer.hidden_by_id[mid] = by_id[mid].hidden_size
        trainer.spec_by_id[mid] = by_id[mid]
    if not states:
        raise RuntimeError("meta_adapt: none of the meta opponents had a checkpoint")
    path = Path(run_dir) / "pool_meta.pt"
    write_frozen_pool(path, specs, states, update=0)
    trainer.pool_path = path
    trainer._pool_sample = list(states.keys())
    ids = list(states.keys())

    def _pick_meta(exclude, rng):
        cand = [m for m in ids if m not in exclude] or ids
        return [str(rng.choice(cand)) for _ in range(3)]

    trainer._pick_descendants = _pick_meta
    return path


def cotrain_team(run_dir: str, net_ids: list, steps_goal: int, progress_cb=None,
                 *, workers: int = 26) -> dict:
    """teamplay — train a team's 3 starter nets JOINTLY as one controlled team (any factions),
    vs ancestors+scripted. The faction-balanced _triples scheduler is overridden to pin the one
    real roster triple; all three nets are saved (they all belong to the team)."""
    import multiprocessing as mp
    import torch
    from src.training.league_trainer import LeagueTrainer, LeagueConfig, _init_worker
    from src.training.population_roster import build_population
    from src.training.policy import to_cpu_state_dict

    by_id = {d.net_id: d for d in build_population()}
    ids = list(net_ids)[:3]
    if len(ids) < 3 or any(i not in by_id for i in ids):
        raise ValueError(f"cotrain needs 3 roster nets present in the population, got {net_ids!r}")
    descendants = [by_id[i] for i in ids]
    workers = max(1, int(workers))
    cfg = LeagueConfig(run_dir=run_dir, workers=workers, episodes_per_triple=max(2, workers),
                       target_steps_per_net=1_000_000_000_000,
                       ancestor_opponent_prob=0.7, descendant_opponent_prob=0.0)
    trainer = LeagueTrainer(cfg, descendants=descendants)
    trainer._triples = lambda: [tuple(ids)]   # the roster plays as ONE fixed triple (any factions)
    fresh_pool = getattr(trainer, "pool_path", None)

    pool = None
    if workers > 1:
        pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker,
                                            initargs=(cfg.torch_threads,))
    start = {i: int(trainer.steps_by_net[i]) for i in ids}
    update = 0
    try:
        while min(int(trainer.steps_by_net[i]) - start[i] for i in ids) < steps_goal:
            update += 1
            trainer.run_update(update, pool=pool)
            if progress_cb:
                done = min(int(trainer.steps_by_net[i]) - start[i] for i in ids)
                progress_cb(min(steps_goal, done), steps_goal)
        for i in ids:
            net = trainer.nets[i]
            net_dir = Path(run_dir) / "nets" / i
            net_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"policy_state": to_cpu_state_dict(net), "update": update, "net_id": i,
                        "genotype_hash": by_id[i].genotype_hash(),
                        "steps_trained": int(trainer.steps_by_net[i])}, net_dir / "latest.pt")
            _merge_registry_steps(run_dir, i, int(trainer.steps_by_net[i]), update)
            _rederive_card(i)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if fresh_pool and Path(fresh_pool).name.startswith("pool_") and Path(fresh_pool).exists():
            try:
                Path(fresh_pool).unlink()
            except OSError:
                pass
    return {"net_ids": ids, "updates": update,
            "steps_added": {i: int(trainer.steps_by_net[i]) - start[i] for i in ids}}


def fork_net(run_dir: str, parent_net_id: str, steps_goal: int, progress_cb=None,
             *, workers: int = 26) -> dict:
    """create_player — spawn a NEW unique child from a parent: clone the parent's weights, perturb
    them (Gaussian, for a unique behaviour + warm start), assign a fresh spec (unique genotype_hash),
    mini-train it vs ancestors+scripted, save under the child net_id, and append it to the registry."""
    import multiprocessing as mp
    import numpy as np
    import torch
    from src.training.league_trainer import LeagueTrainer, LeagueConfig, FACTIONS, _init_worker
    from src.training.population_roster import build_population, respawn_descendant, descendant_registry_entry
    from src.training.policy import MaskedActorCritic, to_cpu_state_dict

    pop = build_population()
    by_id = {d.net_id: d for d in pop}
    parent = by_id.get(parent_net_id)
    if parent is None:
        raise ValueError(f"parent {parent_net_id!r} is not in the population roster")
    parent_ckpt = Path(run_dir) / "nets" / parent_net_id / "latest.pt"
    if not parent_ckpt.exists():
        raise FileNotFoundError(f"parent checkpoint missing: {parent_ckpt}")

    existing = {d.genotype_hash() for d in pop}
    recipe = {"learning_rate": parent.learning_rate, "entropy_coef": parent.entropy_coef,
              "clip_coef": parent.clip_coef, "gae_lambda": parent.gae_lambda}
    seed = (abs(hash(parent_net_id)) % (2**31 - 1)) or 1
    rng = np.random.default_rng(seed)
    gen = parent.generation + 1
    child = None
    for k in range(1, 1000):                       # find a fresh net_id + unique genotype
        cand = respawn_descendant(parent.faction, parent_net_id, parent.archetype, gen, k,
                                  rng, existing, recipe)
        if cand.net_id not in by_id:
            child = cand
            break
    if child is None:
        raise RuntimeError(f"could not allocate a unique child net_id for parent {parent_net_id!r}")

    # clone parent weights + Gaussian perturb -> the child's warm-start, pre-saved so the trainer resumes it
    blob = torch.load(parent_ckpt, map_location="cpu", weights_only=False)
    net = MaskedActorCritic(hidden_size=child.hidden_size)
    net.load_state_dict(blob["policy_state"])
    gen_t = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.dtype.is_floating_point:
                p.add_(torch.randn(p.shape, generator=gen_t) * 0.01)
    child_dir = Path(run_dir) / "nets" / child.net_id
    child_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": to_cpu_state_dict(net), "update": 0, "net_id": child.net_id,
                "genotype_hash": child.genotype_hash(), "steps_trained": 0}, child_dir / "latest.pt")

    # mini-train the child vs ancestors+scripted (one partner per OTHER faction -> valid triple)
    partners = []
    for fac in FACTIONS:
        if fac == child.faction:
            continue
        partners.append(next(d for d in pop if d.faction == fac and d.net_id != parent_net_id))
    descendants = [child, *partners]
    workers = max(1, int(workers))
    cfg = LeagueConfig(run_dir=run_dir, workers=workers, episodes_per_triple=max(2, workers),
                       target_steps_per_net=1_000_000_000_000,
                       ancestor_opponent_prob=0.7, descendant_opponent_prob=0.0)
    trainer = LeagueTrainer(cfg, descendants=descendants)
    fresh_pool = getattr(trainer, "pool_path", None)

    pool = None
    if workers > 1:
        pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker,
                                            initargs=(cfg.torch_threads,))
    start = int(trainer.steps_by_net[child.net_id])
    update = 0
    try:
        while int(trainer.steps_by_net[child.net_id]) - start < steps_goal:
            update += 1
            trainer.run_update(update, pool=pool)
            if progress_cb:
                progress_cb(min(steps_goal, int(trainer.steps_by_net[child.net_id]) - start), steps_goal)
        net = trainer.nets[child.net_id]
        torch.save({"policy_state": to_cpu_state_dict(net), "update": update, "net_id": child.net_id,
                    "genotype_hash": child.genotype_hash(),
                    "steps_trained": int(trainer.steps_by_net[child.net_id])}, child_dir / "latest.pt")
        _append_registry_entry(run_dir, descendant_registry_entry(child),
                               int(trainer.steps_by_net[child.net_id]), update)
        _rederive_card(child.net_id)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if fresh_pool and Path(fresh_pool).name.startswith("pool_") and Path(fresh_pool).exists():
            try:
                Path(fresh_pool).unlink()
            except OSError:
                pass
    return {"net_id": child.net_id, "parent_id": parent_net_id, "genotype_hash": child.genotype_hash(),
            "generation": child.generation, "updates": update,
            "steps_trained": int(trainer.steps_by_net[child.net_id])}


def _append_registry_entry(run_dir: str, entry: dict, steps: int, update: int) -> None:
    """Append a new net's row to the league registry (forked children), de-duping by net_id."""
    import json
    reg_path = Path(run_dir) / "registry.json"
    reg = (json.loads(reg_path.read_text(encoding="utf-8"))
           if reg_path.exists() else {"schema": "league_registry.v1", "entries": []})
    entry = {**entry, "steps_trained": int(steps), "updates_trained": int(update),
             "checkpoint": f"{run_dir}/nets/{entry['net_id']}/latest.pt"}
    entries = [e for e in reg.get("entries", []) if e.get("net_id") != entry["net_id"]]
    entries.append(entry)
    reg["entries"] = entries
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


# default broad opponent cast for the evaluate guard (resolved per-faction from the manifest)
def _general_cast(exclude=()) -> list:
    return _one_per_faction(_manifest_index(), exclude=set(exclude))


class RealTrainingBackend(training_ops.TrainingBackend):
    def __init__(self, run_dir: str = DEFAULT_RUN_DIR, *, require_flag: bool = True):
        self.run_dir = run_dir
        self.require_flag = require_flag

    def _gate(self) -> None:
        if self.require_flag:
            feature_flags.assert_enabled("real_training_backend")  # raises FeatureLocked

    # -- PPO-style ops: skill_up / meta_adapt / style_spec / obedience_tune ---- #
    def continue_ppo(self, ctx: dict, progress_cb=None) -> dict:
        """REAL PPO on the target net; the op picks the recipe (all four differ now)."""
        self._gate()
        op = ctx.get("op", "skill_up")
        net_id = _ctx_net_id(ctx)
        steps = max(1, int(ctx.get("total_steps", 1_000_000)))
        kw: dict = {}
        if op == "obedience_tune":                 # crank order-following, always under orders
            kw = {"order_weight": 0.9, "no_order_prob": 0.0}
        elif op == "style_spec":                   # specialize in the net's own strategy niche
            from src.training.population_roster import build_population
            spec = next((d for d in build_population() if d.net_id == net_id), None)
            kw = {"order_weight": 0.85, "no_order_prob": 0.0,
                  "forced_strategy": spec.strategy_bias if spec else None}
        elif op == "meta_adapt":                   # train against the current top nets (the meta)
            kw = {"meta_opponents": ctx.get("meta_opponents") or _default_meta(self.run_dir, net_id)}
        # measure the pre-training winrate so execute()'s regression guard has a real baseline
        if ctx.get("op"):
            n = int(ctx.get("eval_games", 8))
            tgt_opps = ctx.get("target_opponents") or (kw.get("meta_opponents") or _general_cast(exclude=net_id))
            ctx.setdefault("target_before", evaluate_net(net_id, tgt_opps, run_dir=self.run_dir, n_games=n))
            ctx.setdefault("general_before",
                           evaluate_net(net_id, _general_cast(exclude=net_id), run_dir=self.run_dir,
                                        n_games=n, seed_base=9100))
        return train_one_net(self.run_dir, net_id, steps, progress_cb, **kw)

    def cotrain(self, ctx: dict, progress_cb=None) -> dict:
        """teamplay — joint training of a roster's 3 nets."""
        self._gate()
        ids = ctx.get("net_ids") or (ctx.get("target") or {}).get("net_ids")
        if not ids:
            raise ValueError("cotrain needs ctx['net_ids'] — the roster's 3 nets")
        return cotrain_team(self.run_dir, ids, max(1, int(ctx.get("total_steps", 1_000_000))), progress_cb)

    def fork(self, ctx: dict, progress_cb=None) -> dict:
        """create_player — spawn + mini-train a unique child from the parent net."""
        self._gate()
        return fork_net(self.run_dir, _ctx_net_id(ctx),
                        max(1, int(ctx.get("total_steps", 1_000_000))), progress_cb)

    def evaluate(self, ctx: dict) -> dict:
        """Held-out gauntlet winrate AFTER training; before-metrics come from ctx (set pre-train)."""
        self._gate()
        net_id = _ctx_net_id(ctx)
        n = int(ctx.get("eval_games", 8))
        general_after = evaluate_net(net_id, _general_cast(exclude=net_id), run_dir=self.run_dir,
                                     n_games=n, seed_base=9100)
        tgt_opps = ctx.get("target_opponents") or ctx.get("meta_opponents") or _general_cast(exclude=net_id)
        target_after = evaluate_net(net_id, tgt_opps, run_dir=self.run_dir, n_games=n)
        return {"target_before": ctx.get("target_before", target_after), "target_after": target_after,
                "general_before": ctx.get("general_before", general_after), "general_after": general_after,
                "n_games": n}


def get_backend(*, require_flag: bool = True) -> RealTrainingBackend:
    """Factory used by the engine/UI. The returned backend is locked until the
    feature flag is enabled."""
    return RealTrainingBackend(require_flag=require_flag)
