"""Off-season — the stage sequence between seasons, each with a clear report.

Owner requirements: every window produces a readable report (SoloQ, rookies,
auctions, transfers, drafts) and — most importantly — TRAINING is fully previewed
BEFORE it runs: who requested what, what will happen, what to expect and how long
it takes. Real training is gated (feature flag), so by default the off-season
PREVIEWS training but does not execute it.

Proposed stage order (rationale in docs/cloudcode/WORLD_OFFSEASON.md):
  1. soloq            — refresh player ratings (drives valuation/salary/market)
  2. rookies          — generate this off-season's prospects (owner budget)
  3. training_auction — auction gradeslots, PREVIEW the requested trainings, (exec gated)
  4. transfers        — 2-round window (buy/sell, free-agent auctions)
  5. rookie_draft     — lower league drafts prospects, weakest-first
  6. coaching         — managers hire/release coaches
  7. advance          — skip the off-season month, start the next season
"""
from __future__ import annotations

from src.world import (
    calendar as cal, feature_flags, rookies, soloq, training_auction,
    training_ops, transfers,
)

OFFSEASON_STAGES = ("soloq", "rookies", "training_auction", "transfers",
                    "rookie_draft", "coaching", "advance")


# --------------------------------------------------------------------------- #
# Training preview — the owner sees exactly what will happen before any run
# --------------------------------------------------------------------------- #
def training_preview(jobs: list[dict], config: training_auction.TrainingConfig,
                     extra_minutes: float = 0.0) -> dict:
    """Render the pending training jobs: who requested what, target, steps, ETA,
    and the total wall-clock — BEFORE anything runs. ``jobs`` items:
    ``{team_id, spend_type, target, total_steps}``. ``extra_minutes`` (e.g. rookie
    compute) ADDS to the total time."""
    rows = []
    for j in jobs:
        op = j["spend_type"]
        c = training_ops.card(op)
        steps = int(j.get("total_steps", 0))
        rows.append({
            "team_id": j.get("team_id"),
            "op": op,
            "title": c["title"],
            "does": c["what"],
            "target": j.get("target", {}),
            "steps": steps,
            "est_minutes": round(config.est_minutes(steps), 1),
            "below_floor": steps < training_ops.MIN_TRAINING_STEPS,
        })
    total_steps = sum(r["steps"] for r in rows)
    total_minutes = round(config.est_minutes(total_steps) + extra_minutes, 1)
    return {
        "n_requests": len(rows),
        "requests": rows,
        "slot_steps": config.steps_per_slot,
        "total_steps": total_steps,
        "extra_minutes": extra_minutes,
        "total_minutes": total_minutes,
        "min_steps_floor": training_ops.MIN_TRAINING_STEPS,
    }


def render_training_preview(preview: dict) -> list[str]:
    """Human-readable preview lines for the owner's confirmation screen."""
    lines = [f"ТРЕНИРОВКА — заявок: {preview['n_requests']}, слот={preview['slot_steps']:,} шагов",
             f"Итого: {preview['total_steps']:,} шагов ≈ {preview['total_minutes']} мин "
             f"(+{preview['extra_minutes']} мин новички)"]
    for r in preview["requests"]:
        flag = "  ⚠️<мин!" if r["below_floor"] else ""
        tgt = r["target"].get("player_id") or r["target"].get("kind") or "—"
        lines.append(f"  {r['team_id']}: {r['op']} «{r['title']}» цель={tgt} "
                     f"{r['steps']:,}ш ≈{r['est_minutes']}мин{flag}")
    return lines


# --------------------------------------------------------------------------- #
# Off-season run
# --------------------------------------------------------------------------- #
def run_offseason(world: dict, *, training_jobs: list[dict] | None = None,
                  rookie_budget: rookies.RookieBudget | None = None,
                  training_config: training_auction.TrainingConfig | None = None,
                  backend=None, seed: int = 0, soloq_rounds: int = soloq.DEFAULT_ROUNDS) -> dict:
    """Run the off-season stages and return a structured report. Training is
    previewed; it only executes if a real (enabled) ``backend`` is supplied —
    otherwise it is recorded as locked/in-development."""
    cfg = training_config or training_auction.TrainingConfig()
    report = {"season_just_ended": world["season"], "stages": {}}

    # 1. SoloQ
    log = soloq.apply_soloq(world, rounds=soloq_rounds, seed=seed)
    report["stages"]["soloq"] = {"rounds": soloq_rounds,
                                 "last_round_matches": log[-1]["matches"] if log else 0}

    # 2. Rookies
    if rookie_budget is not None:
        report["stages"]["rookies"] = rookies.generate_rookies(world, rookie_budget, seed=seed)
    else:
        report["stages"]["rookies"] = {"n": 0, "note": "no rookie budget set this off-season"}
    extra_minutes = report["stages"]["rookies"].get("extra_minutes", 0.0)

    # 3. Training auction — PREVIEW first (always), execute only if enabled backend
    jobs = training_jobs or []
    preview = training_preview(jobs, cfg, extra_minutes=extra_minutes)
    execution = {"executed": False, "reason": "preview only — no training backend supplied",
                 "results": []}
    if backend is not None and jobs:
        execution = _execute_training(world, jobs, cfg, backend)
    report["stages"]["training_auction"] = {"preview": preview, "execution": execution}

    # 4. Transfers (rules ready; LLM decisions gated)
    report["stages"]["transfers"] = {
        "window_rounds": transfers.ROUNDS,
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "правила окна готовы; решения LLM-менеджера в разработке",
    }

    # 5. Rookie draft (lower league, weakest-first)
    picks = transfers.rookie_draft(world)
    report["stages"]["rookie_draft"] = {"picks": picks, "n": len(picks)}

    # 6. Coaching
    report["stages"]["coaching"] = {
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "наём/смена тренера готов; решения LLM в разработке",
    }

    # 7. Advance: off-season month skip, next season
    cal.skip_month(world, 1)
    world["season"] += 1
    world["phase"] = "regular"
    # reset per-window transfer budgets
    for t in world["teams"].values():
        t["transfer_requests_used"] = 0
    report["stages"]["advance"] = {"month": world["month"], "next_season": world["season"]}
    report["next_season"] = world["season"]
    return report


def _execute_training(world: dict, jobs: list[dict], cfg, backend) -> dict:
    """Attempt to run the requested trainings via the backend. A locked/in-dev
    backend refuses (FeatureLocked) — recorded honestly, not silently skipped."""
    results = []
    executed_any = False
    for job in jobs:
        try:
            res = training_auction.execute_training(world, job, cfg, backend=backend)
            results.append({"team_id": job.get("team_id"), "op": job["spend_type"],
                            "status": "executed", "passed": getattr(res, "passed", None)})
            executed_any = True
        except feature_flags.FeatureLocked as exc:
            results.append({"team_id": job.get("team_id"), "op": job["spend_type"],
                            "status": "locked", "reason": str(exc)})
        except NotImplementedError as exc:
            results.append({"team_id": job.get("team_id"), "op": job["spend_type"],
                            "status": "not_wired", "reason": str(exc)})
    reason = "executed" if executed_any else "backend locked/in-development — nothing ran"
    return {"executed": executed_any, "reason": reason, "results": results}


def render_report(report: dict) -> list[str]:
    """Compact human-readable summary of an off-season."""
    lines = [f"=== МЕЖСЕЗОНЬЕ после сезона {report['season_just_ended']} -> "
             f"сезон {report.get('next_season')} ==="]
    st = report["stages"]
    lines.append(f"SoloQ: {st['soloq']['rounds']} раундов")
    lines.append(f"Новички: создано {st['rookies'].get('n', 0)}")
    tp = st["training_auction"]["preview"]
    lines += render_training_preview(tp)
    lines.append(f"Исполнение тренировок: {st['training_auction']['execution']['reason']}")
    lines.append(f"Драфт новичков: {st['rookie_draft']['n']} подписаний")
    return lines
