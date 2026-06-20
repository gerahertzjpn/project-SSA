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


# A bench player is "clearly surplus" (release candidate) only if his market value is well
# below the club's WEAKEST starter — so clubs cut dead weight without ever gutting the trio.
RELEASE_SURPLUS_FRACTION = 0.55


def _surplus_bench(world: dict, team: dict) -> list[str]:
    """Release candidates: only fires when a club is at the roster cap and a bench player
    is markedly weaker than its weakest starter. Conservative by design — the population
    is persistent (released players go to free agency, never deleted; no culling)."""
    from src.world import economy
    from src.world.world import MAX_ROSTER, get_player
    roster = team.get("roster", [])
    if len(roster) < MAX_ROSTER:
        return []
    starters = roster[:transfers.MIN_STARTERS]
    bench = roster[transfers.MIN_STARTERS:]
    if not starters or not bench:
        return []
    weakest_starter = min(economy.player_value(get_player(world, pid)) for pid in starters)
    return [pid for pid in bench
            if economy.player_value(get_player(world, pid)) < weakest_starter * RELEASE_SURPLUS_FRACTION]


def _job_op(job: dict) -> str:
    return job.get("op") or job.get("spend_type") or "?"


# --------------------------------------------------------------------------- #
# Training MARKET — the auction + spend that PRODUCES the team training jobs
# --------------------------------------------------------------------------- #
DEFAULT_TRAINING_SLOTS = 6        # gradeslots auctioned per off-season (owner-tunable)


def _slot_anchor(world: dict, team_ids: list[str]) -> int:
    """A nominal 'reference value' for one gradeslot (the LLM/heuristic bids around it)."""
    treas = sorted(world["teams"][tid]["treasury"] for tid in team_ids)
    median = treas[len(treas) // 2] if treas else 100
    return max(10, int(median * 0.15))


def run_training_market(world: dict, cfg: training_auction.TrainingConfig, *,
                        llm=None, feed: Optional[list] = None, seed: int = 0,
                        n_slots: Optional[int] = None) -> tuple[list[dict], dict]:
    """The off-season training economy: auction N gradeslots (managers bid via the LLM, heuristic
    fallback) → winners bank step-valued CERTIFICATES → each holder decides how to SPEND them
    (op + target) or bank them. Returns (pending team jobs, auction summary). Mutates the world
    (treasury + certificates), but does NOT run training — that's the backend's job."""
    state = training_auction.TrainingAuctionState(cfg, season=world["season"],
                                                  n_trainings=n_slots or DEFAULT_TRAINING_SLOTS)
    team_ids = [t["team_id"] for t in world["teams"].values() if len(t.get("roster", [])) >= 3]
    if not team_ids:
        return [], {"slots": state.n_trainings, "awarded": 0, "spent": 0}
    item_value = _slot_anchor(world, team_ids)
    bids = llm_manager.auction_max_bids(world, team_ids, item_value, llm=llm, feed=feed)
    awarded = 0
    awards: list[dict] = []                       # who won a gradeslot and for how much (for the UI)
    while state.remaining > 0:                    # sealed max-bid; winners' interest halves -> spreads
        award = training_auction.auction_one_training(world, state, bids, seed=seed)
        if award:
            awarded += 1
            tname = world["teams"][award["team_id"]]["name"]
            awards.append({"team_id": award["team_id"], "team": tname, "price": award["price"]})
            events.emit(feed, "train", f"Аукцион тренировок: {tname} выиграл слот за {award['price']}")
            bids[award["team_id"]] = bids.get(award["team_id"], 0) // 2
    jobs = [job for team in world["teams"].values()
            if (job := llm_manager.spend_certificates(world, team, llm=llm, feed=feed,
                                                      config=cfg)) is not None]
    return jobs, {"slots": state.n_trainings, "awarded": awarded, "spent": len(jobs),
                  "awards": awards}


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
# Transactions digest (the off-season window the UI shows)
# --------------------------------------------------------------------------- #
def _build_transactions(world: dict, *, auction, team_jobs, fa_bids, signings,
                        picks, club_transfers, releases) -> dict:
    """Assemble the structured off-season transactions the UI renders: auctions,
    training requests, signings, club-to-club buys, departures, releases and bids —
    every entry with resolved team/player names."""
    def pl(pid):
        return (world["players"].get(pid) or {}).get("nickname", pid)

    def tm(tid):
        return (world["teams"].get(tid) or {}).get("name", tid)

    def job_target(job):
        t = job.get("target") or {}
        if t.get("player_id"):
            return t["player_id"], pl(t["player_id"])
        if t.get("parent_id"):
            return t["parent_id"], f"новичок (форк {pl(t['parent_id'])})"
        return None, _op_title(_job_op(job))

    won = {(s["team_id"], s["player_id"]) for s in signings}
    auctions = list((auction or {}).get("awards", []))
    training_requests = []
    for j in team_jobs or []:
        op = _job_op(j)
        pid, label = job_target(j)
        training_requests.append({"team_id": j.get("team_id"), "team": tm(j.get("team_id")),
                                  "op": op, "op_title": _op_title(op), "player_id": pid,
                                  "player": label, "steps": j.get("total_steps")})
    sign_rows = [{"team_id": s["team_id"], "team": tm(s["team_id"]), "player_id": s["player_id"],
                  "player": pl(s["player_id"]), "bonus": s.get("signing_bonus", s.get("price")),
                  "kind": "free_agent"} for s in signings]
    rookie_rows = [{"team_id": p["team_id"], "team": tm(p["team_id"]), "player_id": p["player_id"],
                    "player": pl(p["player_id"]), "bonus": p.get("signing_bonus"),
                    "kind": "rookie"} for p in picks]
    buys = [{"player_id": d["player_id"], "player": pl(d["player_id"]),
             "from_team_id": d["from"], "from_team": tm(d["from"]),
             "to_team_id": d["to"], "to_team": tm(d["to"]), "price": d["price"]}
            for d in club_transfers]
    release_rows = [{"player_id": r["player_id"], "player": r.get("player") or pl(r["player_id"]),
                     "team_id": r["from"], "team": tm(r["from"])} for r in releases]
    requests = [{"team_id": b["team_id"], "team": tm(b["team_id"]), "player_id": b["player_id"],
                 "player": pl(b["player_id"]), "amount": b.get("amount"),
                 "won": (b["team_id"], b["player_id"]) in won} for b in (fa_bids or [])]
    departures = ([{"player_id": d["player_id"], "player": pl(d["player_id"]),
                    "from_team": tm(d["from"]), "to_team": tm(d["to"]), "kind": "transfer"}
                   for d in club_transfers]
                  + [{"player_id": r["player_id"], "player": r.get("player") or pl(r["player_id"]),
                      "from_team": tm(r["from"]), "to_team": "Свободный агент", "kind": "release"}
                     for r in releases])
    return {"auctions": auctions, "training_requests": training_requests,
            "signings": sign_rows + rookie_rows, "buys": buys, "departures": departures,
            "releases": release_rows, "requests": requests,
            "counts": {"auctions": len(auctions), "training_requests": len(training_requests),
                       "signings": len(sign_rows) + len(rookie_rows), "buys": len(buys),
                       "releases": len(release_rows), "requests": len(requests)}}


# --------------------------------------------------------------------------- #
# Off-season run
# --------------------------------------------------------------------------- #
def run_offseason(world: dict, *, training_jobs: list[dict] | None = None,
                  rookie_budget: rookies.RookieBudget | None = None,
                  fa_budget: freeagents.FreeAgentBudget | None = None,
                  training_config: training_auction.TrainingConfig | None = None,
                  backend=None, llm=None, seed: Optional[int] = None,
                  soloq_match_fn=None, auction_slots: int | None = None,
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

    # 4. Training — AUCTION (managers bid via LLM) -> certificates -> SPEND (LLM op+target) -> jobs.
    #    team_jobs are produced by the market unless the caller injected them (tests/owner). Then the
    #    combined queue (team + rookie + free-agent) is previewed; execution is gated on a backend.
    if training_jobs is None:
        team_jobs, auction = run_training_market(world, cfg, llm=llm, feed=feed, seed=sp.seed(),
                                                 n_slots=auction_slots)
    else:
        team_jobs, auction = list(training_jobs), None
    queue = build_training_queue(team_jobs, rookie_jobs, fa_jobs)
    preview = training_preview(queue, cfg)
    execution = {"executed": False, "reason": "preview only — no training backend supplied",
                 "results": []}
    if backend is not None and queue:
        execution = _execute_training(world, queue, cfg, backend, feed=feed)
    report["stages"]["training"] = {"auction": auction, "preview": preview,
                                    "execution": execution, "jobs": team_jobs}
    events.emit(feed, "train", f"Тренировка: аукцион {(auction or {}).get('awarded', 0)} слотов, "
                f"{preview['n_requests']} заявок, {preview['total_steps']:,} шагов; "
                f"исполнение — {execution['reason']}")

    # 5. Transfers — clubs below the roster cap sign DEPTH (free agents) for the bench, so mid-
    #    series substitutions have someone to bring in. The manager (LLM) decides bids; the sealed
    #    FA auction resolves multi-club contention (several clubs can want the same player).
    fa_bids = llm_manager.transfer_bids(world, llm=llm, feed=feed)
    signings = transfers.resolve_free_agent_auction(world, fa_bids, seed=sp.seed())
    for s in signings:
        nick = world["players"][s["player_id"]]["nickname"]
        tname = world["teams"][s["team_id"]]["name"]
        events.emit(feed, "transfer", f"Трансфер: {tname} подписал {nick} "
                    f"(бонус {s['signing_bonus']}, сгорает)", **s)
    report["stages"]["transfers"] = {
        "window_rounds": transfers.ROUNDS, "signings": len(signings),
        "llm_decisions": feature_flags.ui_state("llm_manager"),
        "note": "клубы подписывают глубину состава (свободные агенты); аукцион решает спор",
    }

    # 6. Rookie draft (lower league, weakest-first) — chronicle each signing
    picks = transfers.rookie_draft(world)
    for pk in picks:
        nick = world["players"][pk["player_id"]]["nickname"]
        tname = world["teams"][pk["team_id"]]["name"]
        events.emit(feed, "draft", f"Драфт новичков: {tname} подписал {nick}", **pk)
    report["stages"]["rookie_draft"] = {"picks": picks, "n": len(picks)}

    # 6.5 Club market — clubs cut clearly-surplus bench (kicks → free agency, never deleted),
    #     then bid on OTHER clubs' bench players. Real club-to-club moves: the buyer pays the
    #     SELLING club (resolve_transfer_auction), so money + a player change hands. Bench-only
    #     on both sides, so no club ever loses its starting trio.
    releases = []
    for team in list(world["teams"].values()):
        for pid in _surplus_bench(world, team):
            ev = transfers.release_to_free_agency(world, team["team_id"], pid)
            if ev:
                releases.append(ev)
                events.emit(feed, "transfer",
                            f"{team['name']} отпустил {ev['player']} в свободные агенты", **ev)
    club_bids = transfers.heuristic_club_transfer_bids(world)
    club_transfers = transfers.resolve_transfer_auction(world, club_bids, seed=sp.seed())
    for d in club_transfers:
        nick = world["players"][d["player_id"]]["nickname"]
        frm = world["teams"][d["from"]]["name"]
        to = world["teams"][d["to"]]["name"]
        events.emit(feed, "transfer", f"Трансфер: {to} купил {nick} у {frm} за {d['price']}", **d)

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

    # Structured TRANSACTIONS digest the off-season window UI reads (separate from the
    # event chronicle): who bid, who signed/bought/left/got cut, auction wins + prices,
    # training requests. Kept at the report top level + persisted on the world.
    report["transactions"] = _build_transactions(
        world, auction=auction, team_jobs=team_jobs, fa_bids=fa_bids, signings=signings,
        picks=picks, club_transfers=club_transfers, releases=releases)
    world["last_offseason"] = {"season_ended": report["season_just_ended"],
                               "transactions": report["transactions"]}

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
