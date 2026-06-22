"""Deterministic parallel match execution.

The regular season and each SoloQ round are made of INDEPENDENT matches (no match's outcome
feeds another's), so they can run concurrently. The catch is determinism + torch:

  * ``runner.run_match`` calls ``torch.manual_seed(seed)`` on the PROCESS-GLOBAL torch RNG and the
    policy then samples from it. Two threads in one interpreter would race that global generator and
    break per-match reproducibility — so we use PROCESSES (each owns its own torch RNG), never threads.
  * ``run_match`` RE-SEEDS torch at the start of every match, so a match's outcome depends ONLY on its
    own pre-assigned seed — never on which other matches ran before it. Therefore, as long as every
    match's seed is assigned UP FRONT (not drawn lazily in play order), a parallel run is BIT-IDENTICAL
    to a sequential one, regardless of how work is distributed across workers.

This module is intentionally tiny and SAFE: ``workers <= 1`` (or a single task) runs inline, and ANY
pool/pickle/spawn failure falls back to sequential execution — the results are identical either way,
so the parallel path can only ever change SPEED, never correctness. Playoffs (a dependency DAG),
bootstrap pre-seasons and qualification runs must stay sequential — this helper is only ever handed
independent batches.
"""
from __future__ import annotations

import os
from typing import Callable, Sequence

_WORKER_FN: Callable | None = None       # per-process: the picklable player set by the pool initializer


def default_workers(cap: int = 8) -> int:
    """A safe worker count: physical-ish cores minus one, capped (avoid CPU thrash / oversubscription)."""
    try:
        n = os.cpu_count() or 1
    except Exception:
        n = 1
    return max(1, min(int(cap), n - 1))


def _init(player: Callable) -> None:
    global _WORKER_FN
    _WORKER_FN = player
    try:                                  # one match per worker at a time -> avoid BLAS oversubscription
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run(args: tuple):
    return _WORKER_FN(*args)              # type: ignore[misc]


def run_batch(player: Callable, tasks: Sequence[tuple], workers: int) -> list:
    """Run ``player(*task)`` for every task and return results IN INPUT ORDER.

    ``player`` must be picklable (a top-level function or a class instance with picklable state — NOT a
    closure) when ``workers > 1``. ``workers <= 1`` or a trivial batch runs inline (no pool). Any pool
    error falls back to sequential — identical results, so this is always safe.
    """
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        return [player(*a) for a in tasks]
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks)),
                                  initializer=_init, initargs=(player,)) as ex:
            return list(ex.map(_run, tasks))     # ex.map preserves input order
    except Exception:
        return [player(*a) for a in tasks]       # safe fallback: bit-identical sequential run
