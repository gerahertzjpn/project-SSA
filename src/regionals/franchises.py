"""The six fixed franchises with distinct LLM manager personas.

Franchises persist across tournaments in ``data/regionals/franchises.json``:
treasury (credits), titles, seasons played and the team-vs-team head-to-head
history. Rosters do NOT persist — every regionals starts with a fresh draft.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = ROOT / "data" / "regionals" / "franchises.json"

# Personas are deliberately different: each weighs the draft differently, so
# six managers given the same board make different picks.
FRANCHISES: list[dict] = [
    {
        "franchise_id": "iron-falcons",
        "name": "Iron Falcons",
        "manager": "Виктор «Железняк» Орлов",
        "persona": (
            "Ты Виктор Орлов, менеджер Iron Falcons. Бывший военный, говоришь коротко и жёстко. "
            "Драфт-философия: агрессия и давление решают всё — цени atk и tmp, презирай пассивную игру. "
            "Слабый потолок простишь, трусость — нет."
        ),
    },
    {
        "franchise_id": "quantum-owls",
        "name": "Quantum Owls",
        "manager": "Доктор София Линдквист",
        "persona": (
            "Ты София Линдквист, менеджер Quantum Owls, в прошлом учёный-статистик. "
            "Драфт-философия: только данные — максимизируй ожидаемую силу состава, цени pot и con, "
            "избегай рискованных нестабильных игроков. Решения объясняешь сухо, через цифры."
        ),
    },
    {
        "franchise_id": "crimson-pact",
        "name": "Crimson Pact",
        "manager": "Марко Девич",
        "persona": (
            "Ты Марко Девич, менеджер Crimson Pact, бывший киберспортсмен-хайроллер. "
            "Драфт-философия: бери игроков с высоким потолком и хаосом — высокий pot важнее con, "
            "любишь импровизаторов и джокеров. Скучные стабильные пики — не твоё."
        ),
    },
    {
        "franchise_id": "northwind",
        "name": "Northwind",
        "manager": "Ингрид Хальворсен",
        "persona": (
            "Ты Ингрид Хальворсен, менеджер Northwind из северной школы макро-игры. "
            "Драфт-философия: матчи выигрывает оборона и экономика — цени def и eco, "
            "строй состав, который не проигрывает, а дожимает поздней игрой."
        ),
    },
    {
        "franchise_id": "solar-drift",
        "name": "Solar Drift",
        "manager": "Кэндзи Ватанабэ",
        "persona": (
            "Ты Кэндзи Ватанабэ, менеджер Solar Drift, тренер-гармонизатор. "
            "Драфт-философия: синергия важнее звёзд — собирай сбалансированную тройку, "
            "закрывай слабые места уже взятых игроков, смотри на сочетание фракций и ролей."
        ),
    },
    {
        "franchise_id": "black-lotus",
        "name": "Black Lotus",
        "manager": "Амара Окафор",
        "persona": (
            "Ты Амара Окафор, менеджер Black Lotus, агент звёзд. "
            "Драфт-философия: бери лучшего доступного игрока по рейтингу и репутации, "
            "имена и трофеи продают клуб. Карьерная статистика и rating — твои главные аргументы."
        ),
    },
]


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save_franchise_store(store: dict, store_path: Optional[Path] = None) -> None:
    path = Path(store_path) if store_path else DEFAULT_STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=1)


def ensure_franchise_store(store_path: Optional[Path] = None) -> dict:
    store_path = Path(store_path) if store_path else DEFAULT_STORE_PATH
    store = _load_json(store_path) or {"version": 1, "franchises": {}}
    franchises = store.setdefault("franchises", {})
    changed = False
    for spec in FRANCHISES:
        if spec["franchise_id"] not in franchises:
            franchises[spec["franchise_id"]] = {
                "franchise_id": spec["franchise_id"],
                "name": spec["name"],
                "manager": spec["manager"],
                "money": 0,
                "titles": 0,
                "seasons": 0,
                "head_to_head": {},   # opponent_id -> {"wins": n, "losses": n, "draws": n}
            }
            changed = True
    if changed:
        save_franchise_store(store, store_path)
    return store


def franchise_specs() -> list[dict]:
    return [dict(spec) for spec in FRANCHISES]


def record_head_to_head(store: dict, home_id: str, away_id: str, result: str) -> None:
    """result: 'home' | 'away' | 'draw' (encounter-level)."""
    home = store["franchises"][home_id]["head_to_head"].setdefault(
        away_id, {"wins": 0, "losses": 0, "draws": 0})
    away = store["franchises"][away_id]["head_to_head"].setdefault(
        home_id, {"wins": 0, "losses": 0, "draws": 0})
    if result == "home":
        home["wins"] += 1
        away["losses"] += 1
    elif result == "away":
        home["losses"] += 1
        away["wins"] += 1
    else:
        home["draws"] += 1
        away["draws"] += 1
