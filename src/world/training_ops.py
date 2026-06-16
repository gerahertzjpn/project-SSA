"""Training operations A–F: cards (manager-facing) + correct execution dispatch.

This is the "make it really work" layer. Each operation has:
  - a CARD: what it does, efficiency, what it requires, valid targets, min /
    recommended steps, how the effect is validated — so a manager (LLM) or the UI
    knows exactly what a certificate buys.
  - an EXECUTOR: the correct plumbing that performs the operation against a
    pluggable ``backend`` (the real PPO/lineage training infra). The backend is
    injected so the dispatch is unit-tested now with a mock, and the real GPU
    backend is wired on the owner's "go" — NO real training runs by default
    (``NullBackend`` refuses).

Validation is uniform: held-out gauntlet winrate / net-Elo before vs after, a
regression guard (don't trade target gains for general-pool losses), and (forks)
a uniqueness guard. See docs/cloudcode/WORLD_TRAINING_OPTIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

FORK_MIN_STEPS = 4_000_000
# Absolute floor for ANY single training run, regardless of op (owner rule).
MIN_TRAINING_STEPS = 800_000

# --------------------------------------------------------------------------- #
# Cards — the manager-facing definition of each training operation
# --------------------------------------------------------------------------- #
# target kinds: "player" (one net), "roster" (the trio), "parent" (fork source),
# "coach" (the team's RL coach net). who_can_be_chosen lists allowed targets.
TRAINING_CARDS: dict[str, dict] = {
    "skill_up": {
        "op": "skill_up",
        "title": "Прокачка скилла",
        "what": "Продолженный self-play (PPO) против пула снапшотов — общая острота "
                "политики: микро, макро, тайминги.",
        "efficiency": "Высокая на первых шагах, затухает при насыщении.",
        "requires": "1 цель + шаги.",
        "who_can_be_chosen": ["player", "coach"],
        "min_steps": 800_000,
        "recommended_steps": 1_000_000,
        "validation": "Δ винрейт/netElo против фикс. гонтлета; нет переобучения под одного.",
    },
    "meta_adapt": {
        "op": "meta_adapt",
        "title": "Адаптация под мету",
        "what": "PPO с пулом оппонентов, взвешенным в сторону текущих топ-команд сезона.",
        "efficiency": "Средняя; точечно поднимает матчапы против меты.",
        "requires": "1 цель + список мета-оппонентов (берём топ сезона автоматически).",
        "who_can_be_chosen": ["player", "coach"],
        "min_steps": 800_000,
        "recommended_steps": 1_500_000,
        "validation": "Винрейт против меты ДО/ПОСЛЕ + guard: нет регресса против общего пула.",
    },
    "teamplay": {
        "op": "teamplay",
        "title": "Сыгранность тройки",
        "what": "Совместная тренировка трёх нейронок ростера в team_env — синергия ролей.",
        "efficiency": "Высокая для слаженности; не растит соло-скилл.",
        "requires": "Ростер из 3 (стартовая тройка).",
        "who_can_be_chosen": ["roster"],
        "min_steps": 800_000,
        "recommended_steps": 2_500_000,
        "validation": "Командный винрейт именно этой тройки против эталонных троек.",
    },
    "create_player": {
        "op": "create_player",
        "title": "Создать игрока (форк)",
        "what": "Lineage-форк: клон весов родителя + пертурбация + мини-обучение → "
                "новый уникальный игрок-потомок (родословная трекается).",
        "efficiency": "Разовая; даёт новую карту, а не апгрейд существующей.",
        "requires": f"Родитель (любой игрок) + суммарно ≥ {FORK_MIN_STEPS:,} шагов.",
        "who_can_be_chosen": ["parent"],
        "min_steps": FORK_MIN_STEPS,
        "recommended_steps": FORK_MIN_STEPS,
        "validation": "Уникальность (genotype_hash + поведенческая дистанция) + базовый винрейт.",
    },
    "style_spec": {
        "op": "style_spec",
        "title": "Специализация стиля",
        "what": "Смещение reward/order-веса в сторону паттерна (агро core-rush / "
                "economy-outlast и т.п.) — углубляет один стиль, поддерживает разнообразие.",
        "efficiency": "Средняя; меняет характер игры, не обязательно винрейт.",
        "requires": "1 цель + выбранный стиль (tempo/win_path).",
        "who_can_be_chosen": ["player", "coach"],
        "min_steps": 800_000,
        "recommended_steps": 1_000_000,
        "validation": "Поведенческие метрики сдвинулись как задумано И винрейт не упал.",
    },
    "obedience_tune": {
        "op": "obedience_tune",
        "title": "Настройка послушания",
        "what": "Тренировка с order-каналом/шейпингом — нейронка лучше (или осознанно "
                "хуже) следует плану тренера; это per-net характер.",
        "efficiency": "Средняя; меняет управляемость под тактику.",
        "requires": "1 цель + желаемое направление послушания.",
        "who_can_be_chosen": ["player"],
        "min_steps": 800_000,
        "recommended_steps": 800_000,
        "validation": "order-alignment Φ растёт в нужную сторону, винрейт стабилен.",
    },
}

ENABLED_OPS = tuple(TRAINING_CARDS.keys())  # A–F enabled; G (weakness_fix) left out for now


def card(op: str) -> dict:
    if op not in TRAINING_CARDS:
        raise ValueError(f"unknown training op {op!r} (G/weakness_fix not enabled yet)")
    return TRAINING_CARDS[op]


def card_line(op: str) -> str:
    """Compact one-liner for token-lean LLM prompts."""
    c = card(op)
    return (f"{c['op']}|{c['title']}|цель:{'/'.join(c['who_can_be_chosen'])}"
            f"|реком:{c['recommended_steps']:,}ш|мин:{c['min_steps']:,}ш — {c['what']}")


def validate_request(op: str, target: dict, total_steps: int) -> None:
    """Check an op request against its card before execution."""
    c = card(op)
    if total_steps < MIN_TRAINING_STEPS:
        raise ValueError(f"absolute minimum is {MIN_TRAINING_STEPS} steps per training, "
                         f"got {total_steps}")
    if total_steps < c["min_steps"]:
        raise ValueError(f"{op}: need >= {c['min_steps']} steps, got {total_steps}")
    kind = (target or {}).get("kind")
    if kind and kind not in c["who_can_be_chosen"]:
        raise ValueError(f"{op}: target kind {kind!r} not allowed; "
                         f"choose {c['who_can_be_chosen']}")


# --------------------------------------------------------------------------- #
# Backends — real training is injected; default refuses (no training now)
# --------------------------------------------------------------------------- #
class TrainingBackend:
    """Interface the executors call. Implement against the real PPO/lineage infra
    when the owner enables training. All methods return a small result dict."""

    def continue_ppo(self, ctx: dict) -> dict: raise NotImplementedError
    def cotrain(self, ctx: dict) -> dict: raise NotImplementedError
    def fork(self, ctx: dict) -> dict: raise NotImplementedError
    def evaluate(self, ctx: dict) -> dict: raise NotImplementedError


class NullBackend(TrainingBackend):
    """Default: training is OFF. Any execution attempt is refused loudly."""

    def _refuse(self, *_a, **_k):
        raise NotImplementedError(
            "training backend is disabled — wire the real PPO/lineage backend and "
            "pass it explicitly to enable training (owner 'go' required)."
        )
    continue_ppo = cotrain = fork = evaluate = _refuse


# op -> backend method name
_DISPATCH = {
    "skill_up": "continue_ppo",
    "meta_adapt": "continue_ppo",
    "style_spec": "continue_ppo",
    "obedience_tune": "continue_ppo",
    "teamplay": "cotrain",
    "create_player": "fork",
}


# --------------------------------------------------------------------------- #
# Validation guards (uniform across ops)
# --------------------------------------------------------------------------- #
def regression_guard(target_before: float, target_after: float,
                     general_before: float, general_after: float,
                     max_general_drop: float = 0.03) -> dict:
    """Improvement on target must not cost more than ``max_general_drop`` winrate
    against the general pool (prevents overfit hard-counters)."""
    improved = target_after > target_before
    regressed = (general_before - general_after) > max_general_drop
    return {"improved_target": improved, "regressed_general": regressed,
            "passed": improved and not regressed,
            "target_delta": round(target_after - target_before, 4),
            "general_delta": round(general_after - general_before, 4)}


def uniqueness_guard(new_hash: str, pool_hashes: set, behaviour_distance: float,
                     min_distance: float = 0.05) -> dict:
    """A fork must differ from every existing net (hash) and be behaviourally far
    enough from the population (no clones)."""
    distinct = new_hash not in pool_hashes
    far = behaviour_distance >= min_distance
    return {"distinct_hash": distinct, "far_enough": far,
            "passed": distinct and far, "behaviour_distance": behaviour_distance}


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    op: str
    total_steps: int
    target: dict
    train: dict          # backend training result
    evaluation: dict     # before/after metrics
    guard: dict          # guard verdict
    passed: bool


def execute(op: str, ctx: dict, *, total_steps: int,
            backend: Optional[TrainingBackend] = None) -> TrainingResult:
    """Run one training operation correctly: validate -> train (backend) ->
    evaluate on the held-out gauntlet -> guard. ``ctx`` carries world/target/etc.
    ``backend`` defaults to NullBackend (refuses) so nothing trains without an
    explicit, owner-enabled backend.
    """
    if op not in ENABLED_OPS:
        raise ValueError(f"op {op!r} is not enabled (A–F only; G left out)")
    target = ctx.get("target", {})
    validate_request(op, target, total_steps)
    backend = backend or NullBackend()

    run_ctx = {**ctx, "op": op, "total_steps": total_steps}
    method = getattr(backend, _DISPATCH[op])
    train = method(run_ctx)                       # NullBackend raises here

    evaluation = backend.evaluate(run_ctx)
    if op == "create_player":
        guard = uniqueness_guard(
            evaluation.get("new_hash", ""), set(evaluation.get("pool_hashes", [])),
            evaluation.get("behaviour_distance", 0.0))
    else:
        guard = regression_guard(
            evaluation.get("target_before", 0.0), evaluation.get("target_after", 0.0),
            evaluation.get("general_before", 0.0), evaluation.get("general_after", 0.0))
    return TrainingResult(op, total_steps, target, train, evaluation, guard, guard["passed"])
