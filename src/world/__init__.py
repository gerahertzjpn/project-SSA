"""World mode — persistent multi-season league ecosystem.

A self-contained, persistent world: two leagues (upper 10 / lower 2x6), an
economy (prize money + slow-moving salaries), player SoloQ ratings, team Elo,
a transfer window, a training auction and a calendar.

Status: deterministic mechanics, the LLM manager (gated by ``feature_flags``),
and the real PPO/lineage training backend (``backends.RealTrainingBackend``) are
all wired and live. Heuristics remain as the fallback layer behind clean
interfaces. See docs/cloudcode/WORLD_MODE_SPEC.md.
"""
