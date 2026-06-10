"""Chess-style Elo with a small per-player impact modifier.

Core rules (per the product spec):

- Everyone starts at 1000 (see ``players.START_RATING``).
- A win adds points, a loss subtracts points.
- Beating a *stronger* opponent gives more than beating a weaker one, and
  losing to a *weaker* opponent costs more than losing to a stronger one —
  classic Elo expected-score behavior.
- A player's in-game impact nudges the delta by at most ±20%: playing well in
  a loss softens the hit, carrying a win earns a bit extra. The win/loss
  outcome stays the dominant term.
"""

from __future__ import annotations

K_FACTOR = 32.0
IMPACT_SPREAD = 0.2  # max ±20% modulation of the base delta


def expected_score(rating: float, opponent_rating: float) -> float:
    """Probability-like expected score of `rating` vs `opponent_rating`."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def rating_delta(
    rating: float,
    opponent_rating: float,
    won: bool,
    impact: float = 0.5,
    k: float = K_FACTOR,
) -> float:
    """Rating change for one game.

    ``opponent_rating`` is the average rating of the enemy team. ``impact`` is
    the player's normalized contribution in this game (0..1, 0.5 = neutral):
    high impact amplifies gains and softens losses, low impact does the
    opposite — bounded by ``IMPACT_SPREAD`` so the result always stays close
    to the pure Elo delta.
    """
    impact = min(1.0, max(0.0, float(impact)))
    score = 1.0 if won else 0.0
    base = k * (score - expected_score(rating, opponent_rating))
    if won:
        modifier = 1.0 + (impact - 0.5) * 2.0 * IMPACT_SPREAD
    else:
        # base is negative on a loss: high impact shrinks the penalty.
        modifier = 1.0 - (impact - 0.5) * 2.0 * IMPACT_SPREAD
    return base * modifier


def team_average(ratings: list[float]) -> float:
    return sum(ratings) / len(ratings) if ratings else 0.0
