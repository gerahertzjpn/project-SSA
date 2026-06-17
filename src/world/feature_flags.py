"""Feature flags — gate systems that are wired but not yet open for selection.

Per owner: the real training backend, the LLM manager and the World UI tab are
WIRED but must be temporarily LOCKED in the UI and marked "in development".
Off-season coach training is a deferred future feature. These flags are the
single source the UI and the engine read to decide what is selectable.
"""
from __future__ import annotations

FLAGS: dict[str, dict] = {
    "real_training_backend": {
        "enabled": False, "status": "in_development",
        "label": "Реальное обучение нейронок",
        "note": "Backend вшит, но временно заблокирован для выбора (в разработке).",
    },
    "llm_manager": {
        "enabled": False, "status": "in_development",
        "label": "LLM-менеджер (трансферы/аукционы/сертификаты)",
        "note": "Интерфейс вшит; решения пока эвристика. В разработке.",
    },
    "ui_world_tab": {
        "enabled": False, "status": "in_development",
        "label": "Вкладка World (UI)",
        "note": "Движок готов; UI-вкладка в разработке.",
    },
    "live_auction_l2": {
        "enabled": False, "status": "secondary",
        "label": "🔴 Живые торги аукциона (L2)",
        "note": "Второстепенная фича: полные живые торги ЛЛМ за каждый лот — много "
                "времени и денег. Требует явного подтверждения «ДА». По умолчанию L1.",
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
