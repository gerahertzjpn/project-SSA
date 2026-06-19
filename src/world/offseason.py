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
  7. planning     — each manager authors its tournament PLAYBOOK (no coach; §0-bis)
  8. advance      — close the off-season: skip a calendar month, season += 1,
                    reset transfer-window limits, set phase back to 'regular'

Every stage appends to an EVENT FEED (events.py) — a step-by-step chronicle the UI
plays back; fallbacks emit notifications into the same feed.
"""
from __future__ import annotations

from typing import Optional

from src.core import rng as seeds
from src.world import (
    calendar as cal, events, feature_flags, freeagents, llm_manager,
    rookies, soloq, training_auction, training_ops, transfers,
)

OFFSEASON_STAGES = ("soloq", "rookies", "free_agents", "training", "transfers",
                    "rookie_draft", "planning", "advance")

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
                  backend=None, llm=None, seed: Optional[int] = None,
                  soloq_match_fn=None,
                  soloq_rounds: int = soloq.DEFAULT_ROUNDS) -> dict:
    """Run the off-season stages and return a structured report (incl. a step-by-step
    event feed). The training stage previews the COMBINED launch (team + rookies +
    free agents); it only executes if a real (enabled) ``backend`` is supplied. ``llm``
    (when the flag is on) authors playbooks; otherwise the heuristic runs + notifies.
    ``seed`` optional: None → fresh entropy (unique off-season); explicit → reproducible."""
    cfg = training_config or training_auction.TrainingConfig()
    sp = seeds.Spawner(seed)              # independent child streams (no base+offset)
    feed: list = events.chronicle()
    report = {"season_just_ended": world["season"], "seed": sp.root_seed,  # recorded for history
              "stages": {}, "events": feed}

    # 1. SoloQ (net-backed when soloq_match_fn supplied; default None -> rating model)
    log = soloq.apply_soloq(world, rounds=soloq_rounds, seed=sp.seed(), match_fn=soloq_match_fn)
    report["stages"]["soloq"] = {"rounds": soloq_rounds,
                                 "last_round_matches": log[-1]["matches"] if log else 0}
    events.emit(feed, "stage", f"SoloQ: сыграно {soloq_rounds} раундов, рейтинги обновлены")

    # 2. Rookies (cards now; their forks go into the combined launch)
    if rookie_budget is not None:
        rk = rookies.generate_rookies(world, rookie_budget, seed=sp.seed())
    else:
        rk = {"n": 0, "fork_plan": [], "note": "no rookie budget set this off-season"}
    report["stages"]["rookies"] = rk
    events.emit(feed, "stage", f"Новички: создано {rk.get('n', 0)} проспектов")
    rookie_jobs = rookies.rookie_training_jobs(rk)

    # 3. Free agents (keep form)
    if fa_budget is not None:
        fa_jobs = freeagents.maintenance_jobs(world, fa_budget)
    else:
        fa_jobs = []
    report["stages"]["free_agents"] = {"n": len(fa_jobs),
                                       "total_steps": freeagents.total_steps(fa_jobs),
                                       "note": "form upkeep" if fa_jobs else "no free-agent budget"}
    if fa_jobs:
        events.emit(feed, "train", f"Фри-агенты: {len(fa_jobs)} на поддержании формы")

    # 4. Combined training launch — ONE preview, (exec gated)
    queue = build_training_queue(training_jobs, rookie_jobs, fa_jobs)
    preview = training_preview(queue, cfg)
    execution = {"executed": False, "reason": "preview only — no training backend supplied",
                 "results": []}
    if backend is not None and queue:
        execution = _execute_training(world, queue, cfg, backend, feed=feed)
    report["stages"]["training"] = {"preview": preview, "execution": execution}
    events.emit(feed, "train", f"Тренировка (один запуск): {preview['n_requests']} заявок, "
                f"{preview['total_steps']:,} шагов; исполнение — {execution['reason']}")

    # 5. Transfers (rules ready; LLM decisions gated). When the LLM market runs it
    #    appends bid/buy/sell/decline events to the feed (see llm_market.py).
    report["stages"]["transfers"] = {
        "window_rounds": transfers.ROUNDS,
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "правила окна готовы; решения LLM-менеджера в разработке",
    }

    # 6. Rookie draft (lower league, weakest-first) — chronicle each signing
    picks = transfers.rookie_draft(world)
    for pk in picks:
        nick = world["players"][pk["player_id"]]["nickname"]
        tname = world["teams"][pk["team_id"]]["name"]
        events.emit(feed, "draft", f"Драфт новичков: {tname} подписал {nick}", **pk)
    report["stages"]["rookie_draft"] = {"picks": picks, "n": len(picks)}

    # 7. Planning — each manager authors its tournament PLAYBOOK (LLM if active, else
    #    heuristic; author_playbook emits the chronicle + any fallback notification).
    plans = []
    for team in world["teams"].values():
        if len(team.get("roster", [])) < 3:
            continue
        plan = llm_manager.author_playbook(world, team, llm=llm, feed=feed)
        plans.append({"team_id": team["team_id"], "base": plan["base"]["strategy"]})
    report["stages"]["planning"] = {
        "plans": plans, "n": len(plans),
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "менеджеры задают плейбук (LLM если включён, иначе эвристика)",
    }

    # 8. Advance: close the off-season -> next season
    cal.skip_month(world, 1)
    world["season"] += 1
    world["phase"] = "regular"
    for t in world["teams"].values():
        t["transfer_requests_used"] = 0
    report["stages"]["advance"] = {"month": world["month"], "next_season": world["season"]}
    events.emit(feed, "stage", f"Межсезонье закрыто → сезон {world['season']}")
    report["next_season"] = world["season"]
    return report


def _execute_training(world: dict, jobs: list[dict], cfg, backend,
                      feed: Optional[list] = None) -> dict:
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
            events.emit(feed, "train", f"Тренировка {op} ({kind}): заблокировано — {exc}")
        except NotImplementedError as exc:
            results.append({"kind": kind, "op": op, "status": "not_wired", "reason": str(exc)})
            events.emit(feed, "train", f"Тренировка {op} ({kind}): backend не подключён")
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
    lines.append(f"Плейбуки на турнир: {st.get('planning', {}).get('n', 0)} команд")
    feed = report.get("events") or []
    if feed:
        lines.append("--- ЛЕНТА СОБЫТИЙ ---")
        lines += events.render(feed)
    return lines
