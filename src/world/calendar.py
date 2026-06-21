"""Calendar — schedule, recorded results, and the month-skip between seasons.

The world keeps a flat ``calendar`` list of events. A season fills it with
league/playoff series (with results once played); ``skip_month`` advances the
clock between seasons (the off-season month).
"""
from __future__ import annotations

MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def month_label(month_index: int) -> str:
    year = month_index // 12
    return f"{MONTHS[month_index % 12]} Y{year}"


def add_match(world: dict, *, season: int, stage: str, league: str,
              home: str, away: str, best_of: int,
              maps_home: int | None = None, maps_away: int | None = None,
              winner: str | None = None,
              home_strategy: str | None = None, away_strategy: str | None = None,
              sequence: list | None = None) -> dict:
    """Append a (scheduled or played) match to the calendar. ``home_strategy`` /
    ``away_strategy`` record the CALLS each manager made (the playbook's active
    strategy) so the scouting layer can learn from past games. ``sequence`` is the
    per-map winner list (for the match-detail page) — public; the hidden cores stat is NOT stored."""
    event = {
        "season": season,
        "month": world.get("month", 0),
        "league": league,
        "stage": stage,
        "home": home,
        "away": away,
        "best_of": best_of,
        "maps_home": maps_home,
        "maps_away": maps_away,
        "winner": winner,
        "home_strategy": home_strategy,      # the manager's CALL (None if unknown)
        "away_strategy": away_strategy,
        "sequence": sequence,                # per-map winner team-ids (None if unknown)
        "played": winner is not None,
    }
    world.setdefault("calendar", []).append(event)
    return event


def record_series(world: dict, season: int, league: str, series: list[dict]) -> None:
    """Record played series (from brackets/group play) onto the calendar, including
    each side's CALL (strategy) from the playbook meta when present."""
    for r in series:
        pb = r.get("playbook") or {}
        add_match(
            world, season=season, stage=r.get("stage", "league"), league=league,
            home=r["home"], away=r["away"], best_of=r.get("best_of", 3),
            maps_home=r["maps_home"], maps_away=r["maps_away"], winner=r["winner"],
            home_strategy=(pb.get("home") or {}).get("strategy"),
            away_strategy=(pb.get("away") or {}).get("strategy"),
            sequence=r.get("sequence"),
        )


def skip_month(world: dict, n: int = 1) -> int:
    """Advance the off-season clock by ``n`` months; returns the new month index."""
    world["month"] = world.get("month", 0) + int(n)
    return world["month"]


def season_calendar(world: dict, season: int) -> list[dict]:
    return [e for e in world.get("calendar", []) if e["season"] == season]


def results_table(world: dict, season: int | None = None) -> list[str]:
    """Human-readable one-liners of played matches (for a results view)."""
    out = []
    for e in world.get("calendar", []):
        if season is not None and e["season"] != season:
            continue
        if not e["played"]:
            continue
        out.append(
            f"S{e['season']} {month_label(e['month'])} [{e['league']}/{e['stage']}] "
            f"{e['home']} {e['maps_home']}-{e['maps_away']} {e['away']} -> {e['winner']}"
        )
    return out
