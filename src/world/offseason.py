"""Off-season — the stage sequence between seasons, each with a clear report.

Owner requirements:
- every window produces a readable report (SoloQ, rookies, free agents, auctions,
  transfers, drafts);
- TRAINING is one COMBINED launch — team-requested trainings + rookie forks +
  free-agent upkeep go into a single queue, fully previewed BEFORE it runs (who
  requested what, what happens, how long), so the owner starts everything once and
  walks away. Real training is gated (feature flag): by default the off-season
  PREVIEWS the launch but does not execute it.

Stage order (rationale in docs/cloudcode/WORLD_OFFSEASON.md):
  1. soloq        — refresh player ratings (drives valuation/salary/market)
  2. rookies      — generate this off-season's prospects (owner budget)
  3. free_agents  — pick free agents to keep in form (owner budget, < rookie budget)
  4. training     — COMBINED launch (team + rookies + free agents): preview, (exec gated)
  5. transfers    — 2-round window (buy/sell, free-agent auctions)
  6. rookie_draft — lower league drafts prospects, weakest-first
  7. coaching     — managers hire/release coaches
  8. advance      — close the off-season: skip a calendar month, season += 1,
                    reset transfer-window limits, set phase back to 'regular'
"""
from __future__ import annotations

from src.world import (
    calendar as cal, feature_flags, freeagents, rookies, soloq,
    training_auction, training_ops, transfers,
)

OFFSEASON_STAGES = ("soloq", "rookies", "free_agents", "training", "transfers",
                    "rookie_draft", "coaching", "advance")

# Labels for system-driven training jobs that are not team-auction ops.
SYSTEM_OP_LABELS = {
    "rookie_fork": "Создание новичка (форк)",
    "fa_maintain": "Поддержание формы (фри-агент)",
}


def _op_title(op: str) -> str:
    if op in training_ops.TRAINING_CARDS:
        return training_ops.TRAINING_CARDS[op]["title"]
    return SYSTEM_OP_LABELS.get(op, op)


def _job_op(job: dict) -> str:
    return job.get("op") or job.get("spend_type") or "?"


# --------------------------------------------------------------------------- #
# Combined training queue + preview
# --------------------------------------------------------------------------- #
def build_training_queue(team_jobs: list[dict] | None,
                         rookie_jobs: list[dict] | None,
                         fa_jobs: list[dict] | None) -> list[dict]:
    """Merge the three training sources into ONE queue (team trainings + rookie
    forks + free-agent upkeep), tagging each job's ``kind``."""
    queue = []
    for j in (team_jobs or []):
        queue.append({**j, "kind": j.get("kind", "team")})
    queue.extend(rookie_jobs or [])
    queue.extend(fa_jobs or [])
    return queue


def training_preview(jobs: list[dict], config: training_auction.TrainingConfig,
                     extra_minutes: float = 0.0) -> dict:
    """Render the combined training launch BEFORE it runs: who requested what,
    target, steps, ETA each, totals per kind, and the total wall-clock — so the
    owner sees exactly what will happen and for how long, then starts it once."""
    rows = []
    by_kind: dict[str, dict] = {}
    for j in jobs:
        op = _job_op(j)
        steps = int(j.get("total_steps", 0))
        kind = j.get("kind", "team")
        rows.append({
            "kind": kind,
            "team_id": j.get("team_id"),
            "op": op,
            "title": _op_title(op),
            "target": j.get("target", {}),
            "steps": steps,
            "est_minutes": round(config.est_minutes(steps), 1),
            "below_floor": steps < training_ops.MIN_TRAINING_STEPS,
        })
        agg = by_kind.setdefault(kind, {"n": 0, "steps": 0})
        agg["n"] += 1
        agg["steps"] += steps
    total_steps = sum(r["steps"] for r in rows)
    total_minutes = round(config.est_minutes(total_steps) + extra_minutes, 1)
    return {
        "n_requests": len(rows),
        "requests": rows,
        "by_kind": by_kind,
        "slot_steps": config.steps_per_slot,
        "total_steps": total_steps,
        "extra_minutes": extra_minutes,
        "total_minutes": total_minutes,
        "min_steps_floor": training_ops.MIN_TRAINING_STEPS,
    }


def render_training_preview(preview: dict) -> list[str]:
    """Human-readable preview lines for the owner's confirmation screen."""
    kinds = preview.get("by_kind", {})
    head = ", ".join(f"{k}:{v['n']}" for k, v in kinds.items()) or "—"
    lines = [f"ТРЕНИРОВКА (один запуск) — заявок: {preview['n_requests']} ({head}); "
             f"слот={preview['slot_steps']:,} шагов",
             f"Итого: {preview['total_steps']:,} шагов ≈ {preview['total_minutes']} мин"]
    for r in preview["requests"]:
        flag = "  ⚠️<мин!" if r["below_floor"] else ""
        tgt = (r["target"].get("player_id") or r["target"].get("rookie_id")
               or r["target"].get("parent_id") or r["target"].get("kind") or "—")
        who = r["team_id"] or r["kind"]
        lines.append(f"  [{r['kind']}] {who}: {r['op']} «{r['title']}» цель={tgt} "
                     f"{r['steps']:,}ш ≈{r['est_minutes']}мин{flag}")
    return lines


# --------------------------------------------------------------------------- #
# Off-season run
# --------------------------------------------------------------------------- #
def run_offseason(world: dict, *, training_jobs: list[dict] | None = None,
                  rookie_budget: rookies.RookieBudget | None = None,
                  fa_budget: freeagents.FreeAgentBudget | None = None,
                  training_config: training_auction.TrainingConfig | None = None,
                  backend=None, seed: int = 0, soloq_rounds: int = soloq.DEFAULT_ROUNDS) -> dict:
    """Run the off-season stages and return a structured report. The training
    stage previews the COMBINED launch (team + rookies + free agents); it only
    executes if a real (enabled) ``backend`` is supplied, otherwise it is recorded
    as locked/in-development."""
    cfg = training_config or training_auction.TrainingConfig()
    report = {"season_just_ended": world["season"], "stages": {}}

    # 1. SoloQ
    log = soloq.apply_soloq(world, rounds=soloq_rounds, seed=seed)
    report["stages"]["soloq"] = {"rounds": soloq_rounds,
                                 "last_round_matches": log[-1]["matches"] if log else 0}

    # 2. Rookies (cards now; their forks go into the combined launch)
    if rookie_budget is not None:
        rk = rookies.generate_rookies(world, rookie_budget, seed=seed)
    else:
        rk = {"n": 0, "fork_plan": [], "note": "no rookie budget set this off-season"}
    report["stages"]["rookies"] = rk
    rookie_jobs = rookies.rookie_training_jobs(rk)

    # 3. Free agents (keep form)
    if fa_budget is not None:
        fa_jobs = freeagents.maintenance_jobs(world, fa_budget)
    else:
        fa_jobs = []
    report["stages"]["free_agents"] = {"n": len(fa_jobs),
                                       "total_steps": freeagents.total_steps(fa_jobs),
                                       "note": "form upkeep" if fa_jobs else "no free-agent budget"}

    # 4. Combined training launch — ONE preview, (exec gated)
    queue = build_training_queue(training_jobs, rookie_jobs, fa_jobs)
    preview = training_preview(queue, cfg)
    execution = {"executed": False, "reason": "preview only — no training backend supplied",
                 "results": []}
    if backend is not None and queue:
        execution = _execute_training(world, queue, cfg, backend)
    report["stages"]["training"] = {"preview": preview, "execution": execution}

    # 5. Transfers (rules ready; LLM decisions gated)
    report["stages"]["transfers"] = {
        "window_rounds": transfers.ROUNDS,
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "правила окна готовы; решения LLM-менеджера в разработке",
    }

    # 6. Rookie draft (lower league, weakest-first)
    picks = transfers.rookie_draft(world)
    report["stages"]["rookie_draft"] = {"picks": picks, "n": len(picks)}

    # 7. Coaching
    report["stages"]["coaching"] = {
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "наём/смена тренера готов; решения LLM в разработке",
    }

    # 8. Advance: close the off-season -> next season
    cal.skip_month(world, 1)
    world["season"] += 1
    world["phase"] = "regular"
    for t in world["teams"].values():
        t["transfer_requests_used"] = 0
    report["stages"]["advance"] = {"month": world["month"], "next_season": world["season"]}
    report["next_season"] = world["season"]
    return report


def _execute_training(world: dict, jobs: list[dict], cfg, backend) -> dict:
    """Attempt to run the combined queue via the backend, per kind. A locked/in-dev
    backend refuses (FeatureLocked) — recorded honestly, not silently skipped."""
    results = []
    executed_any = False
    for job in jobs:
        kind, op = job.get("kind", "team"), _job_op(job)
        try:
            if kind == "team":
                res = training_auction.execute_training(world, {**job, "spend_type": op},
                                                        cfg, backend=backend)
                passed = getattr(res, "passed", None)
            elif kind == "rookie":
                backend.fork({"world": world, "target": job["target"],
                              "op": "create_player", "total_steps": job["total_steps"]})
                passed = None
            else:  # free_agent
                backend.continue_ppo({"world": world, "target": job["target"],
                                      "op": "skill_up", "total_steps": job["total_steps"]})
                passed = None
            results.append({"kind": kind, "team_id": job.get("team_id"), "op": op,
                            "status": "executed", "passed": passed})
            executed_any = True
        except feature_flags.FeatureLocked as exc:
            results.append({"kind": kind, "op": op, "status": "locked", "reason": str(exc)})
        except NotImplementedError as exc:
            results.append({"kind": kind, "op": op, "status": "not_wired", "reason": str(exc)})
    reason = "executed" if executed_any else "backend locked/in-development — nothing ran"
    return {"executed": executed_any, "reason": reason, "results": results}


def render_report(report: dict) -> list[str]:
    """Compact human-readable summary of an off-season."""
    lines = [f"=== МЕЖСЕЗОНЬЕ после сезона {report['season_just_ended']} -> "
             f"сезон {report.get('next_season')} ==="]
    st = report["stages"]
    lines.append(f"SoloQ: {st['soloq']['rounds']} раундов")
    lines.append(f"Новички: создано {st['rookies'].get('n', 0)}; "
                 f"фри-агенты на тренировке: {st['free_agents']['n']}")
    lines += render_training_preview(st["training"]["preview"])
    lines.append(f"Исполнение тренировок: {st['training']['execution']['reason']}")
    lines.append(f"Драфт новичков: {st['rookie_draft']['n']} подписаний")
    return lines
