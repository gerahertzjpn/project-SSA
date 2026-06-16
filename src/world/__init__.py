"""World mode — persistent multi-season league ecosystem.

A self-contained, persistent world: two leagues (upper 10 / lower 2x6), an
economy (prize money + slow-moving salaries), player SoloQ ratings, team Elo,
a transfer window, a (beta) training auction and a calendar.

Foundation status: deterministic mechanics are implemented and tested; LLM
manager decisions and real net training are stubbed behind clean interfaces.
See docs/cloudcode/WORLD_MODE_SPEC.md.
"""
