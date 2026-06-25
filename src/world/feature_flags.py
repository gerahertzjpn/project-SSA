"""Feature flags — gate systems that are wired but not yet open for selection.

Per owner: the real training backend, the LLM manager and the World UI tab are
WIRED but must be temporarily LOCKED in the UI and marked "in development".
Off-season coach training is a deferred future feature. These flags are the
single source the UI and the engine read to decide what is selectable.
"""
from __future__ import annotations

FLAGS: dict[str, dict] = {
    "real_training_backend": {
        "enabled": True, "status": "live",
        "label": "Реальное обучение нейронок",
        "note": "Разблокирован. ВНИМАНИЕ: реальный PPO — это часы; в живом цикле тренировка "
                "идёт фоновым джобом, межсезонье запускает её и пересчитывает карточку.",
    },
    "llm_manager": {
        "enabled": True, "status": "live",
        "label": "LLM-менеджер (трансферы/аукционы/сертификаты)",
        "note": "Разблокирован. Реальные решения LLM (драфт/плейбук/рынок); фолбэк на эвристику при сбое.",
    },
    "ui_world_tab": {
        "enabled": True, "status": "live",
        "label": "Вкладка World (UI)",
        "note": "Создание мира, лиги/таблицы/составы и прогон сезонов на реальных нейронках.",
    },
    "live_auction_l2": {
        "enabled": True, "status": "live",
        "label": "🔴 Живые торги аукциона (L2)",
        "note": "ВКЛ. Тренировочный аукцион идёт как ЖИВЫЕ восходящие торги: клубы поднимают ставку по "
                "очереди (LLM — полная свобода), победитель платит свою финальную ставку (~чуть выше "
                "второго), проигравшие не платят. Много LLM-вызовов на межсезонье.",
    },
    "post_group_replan": {
        "enabled": False, "status": "beta",
        "label": "Пере-план базовой стратегии после групп-этапа",
        "note": "Если клуб провалил группу — менеджер-LLM переписывает БАЗОВУЮ стратегию плейбука перед "
                "плей-офф (состав НЕ меняется). По умолчанию выкл (+1 LLM-вызов на отстающий клуб).",
    },
}


class FeatureLocked(Exception):
    pass


def flag(name: str) -> dict:
    if name not in FLAGS:
        raise KeyError(f"unknown feature flag {name!r}")
    return FLAGS[name]


def is_enabled(name: str) -> bool:
    return bool(flag(name)["enabled"])


def assert_enabled(name: str) -> None:
    f = flag(name)
    if not f["enabled"]:
        raise FeatureLocked(f"{name} is {f['status']}: {f['note']}")


def ui_state(name: str) -> dict:
    """What the UI should render for this feature: locked + status + labels."""
    f = flag(name)
    return {"name": name, "selectable": f["enabled"], "locked": not f["enabled"],
            "status": f["status"], "label": f["label"], "note": f["note"]}


def all_ui_states() -> list[dict]:
    return [ui_state(name) for name in FLAGS]
