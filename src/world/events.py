"""Off-season EVENT FEED — a step-by-step chronicle of who did what.

Owner requirement (§0-bis): the off-season should read as a sequence of concrete
events — who requests whom, who accepts, who buys, who trains, who drafts — emitted
one at a time (a small interval in the UI). Example:
    "менеджер X хочет купить A5 → менеджер Y хочет купить A5 → владелец A5 продал Y".

The feed is plain data (mechanical display, no LLM). It ALSO carries fallback
NOTIFICATIONS: every time an LLM decision fails and the heuristic takes over, an
event of kind ``fallback`` is emitted so it is never silent (owner requirement).
"""
from __future__ import annotations

from typing import Optional

# Event kinds (for filtering / styling in the UI).
KINDS = ("stage", "bid", "buy", "sell", "decline", "draft", "train", "playbook",
         "auction", "fallback")


def chronicle() -> list:
    """A fresh, empty feed (a list of ordered event dicts)."""
    return []


def emit(feed: Optional[list], kind: str, msg: str, **data) -> Optional[dict]:
    """Append one event. ``feed`` may be None (events simply discarded — so callers
    can stay feed-agnostic). Returns the event (or None)."""
    if feed is None:
        return None
    ev = {"seq": len(feed) + 1, "kind": kind, "msg": msg, **data}
    feed.append(ev)
    return ev


def fallback(feed: Optional[list], where: str, reason: str) -> Optional[dict]:
    """Notify that an LLM decision failed and the heuristic took over (never silent)."""
    return emit(feed, "fallback", f"⚠️ фолбэк [{where}]: ЛЛМ не сработал — решает эвристика ({reason})",
                where=where, reason=reason)


def render(feed: list, kinds: Optional[tuple] = None) -> list[str]:
    """Human-readable, numbered lines for step-by-step display."""
    return [f"{e['seq']:>3}. {e['msg']}"
            for e in feed if kinds is None or e["kind"] in kinds]


def fallbacks(feed: list) -> list[dict]:
    """Just the fallback notifications."""
    return [e for e in feed if e["kind"] == "fallback"]
