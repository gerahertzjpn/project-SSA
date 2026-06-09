"""Server-level tests for the league background-job plumbing.

``create_league`` itself is covered in test_league.py; here we monkeypatch it so
we can exercise the job registry / status flow in viewer.server quickly without
running real simulations.
"""

from __future__ import annotations

import time

import pytest

import viewer.server as server


def _wait_for_job(job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = server.league_job_status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.02)
    return server.league_job_status(job_id)


def test_start_league_job_completes(monkeypatch):
    monkeypatch.setattr(server, "available_models", lambda: [{"id": "random"}])

    def fake_create_league(name, template_id, teams, seed, allowed_models, progress_cb):
        progress_cb(0, 3)
        progress_cb(3, 3)
        return {"id": "fake-league-xyz"}

    monkeypatch.setattr(server, "create_league", fake_create_league)

    response = server.start_league_job({"name": "T", "template": "classic_6x3", "teams": [], "seed": 1})
    assert response["status"] == "running"
    status = _wait_for_job(response["job_id"])
    assert status["status"] == "done"
    assert status["league_id"] == "fake-league-xyz"
    assert status["done"] == 3 and status["total"] == 3


def test_start_league_job_reports_error(monkeypatch):
    monkeypatch.setattr(server, "available_models", lambda: [{"id": "random"}])

    def boom(*args, **kwargs):
        raise ValueError("bad teams")

    monkeypatch.setattr(server, "create_league", boom)

    response = server.start_league_job({"name": "T", "template": "classic_6x3", "teams": []})
    status = _wait_for_job(response["job_id"])
    assert status["status"] == "error"
    assert "bad teams" in (status["error"] or "")


def test_unknown_job_raises():
    with pytest.raises(server.LeagueError):
        server.league_job_status("no-such-job")
