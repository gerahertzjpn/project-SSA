"""The franchise pool (25) with distinct LLM-manager personas.

A regionals tournament draws ``team_count`` franchises from this pool, so the
lineup varies between tournaments. Each franchise has: a name, a 2-3 letter
abbreviation, an LLM manager persona (drives both the draft and the per-match
game plan), a default game plan (the no-LLM fallback / strong prior, in the
order vocabulary), and a generated SVG emblem (colour + shape + abbreviation).

Franchises persist across tournaments in ``data/regionals/franchises.json``
(treasury, titles, seasons, head-to-head). Rosters do NOT persist — every
regionals starts with a fresh draft.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = ROOT / "data" / "regionals" / "franchises.json"
LOGO_DIR = ROOT / "data" / "regionals" / "logos"

# Personas are deliberately varied across strategic archetypes (aggression,
# macro/economy, chaos/objectives, data, synergy, star-power, ...). With 25 some
# overlap is fine. `default_plan` uses the order vocabulary (tempo/win_path/role)
# and is both the no-LLM fallback and a strong in-character prior the LLM can
# override. (colour, shape) drive the generated emblem.
FRANCHISES: list[dict] = [
    {
        "franchise_id": "iron-falcons", "name": "Iron Falcons", "abbrev": "IRF",
        "manager": "Виктор «Железняк» Орлов",
        "persona": (
            "Ты Виктор Орлов, менеджер Iron Falcons. Бывший военный, говоришь коротко и жёстко. "
            "Драфт-философия: агрессия и давление решают всё — цени atk и tmp, презирай пассивную игру. "
            "Слабый потолок простишь, трусость — нет."
        ),
        "default_plan": {"tempo": "aggressive", "win_path": "core_rush",
                         "roles": ["carry_push", "carry_push", "roam_objective"]},
        "color": "#3B6EA5", "accent": "#D8E4F0", "shape": "shield",
    },
    {
        "franchise_id": "quantum-owls", "name": "Quantum Owls", "abbrev": "QO",
        "manager": "Доктор София Линдквист",
        "persona": (
            "Ты София Линдквист, менеджер Quantum Owls, в прошлом учёный-статистик. "
            "Драфт-философия: только данные — максимизируй ожидаемую силу состава, цени pot и con, "
            "избегай рискованных нестабильных игроков. Решения объясняешь сухо, через цифры."
        ),
        "default_plan": {"tempo": "balanced", "win_path": "economy",
                         "roles": ["tech_eco", "carry_push", "anchor_defend"]},
        "color": "#4B3F8F", "accent": "#E0DBF5", "shape": "hexagon",
    },
    {
        "franchise_id": "crimson-pact", "name": "Crimson Pact", "abbrev": "CP",
        "manager": "Марко Девич",
        "persona": (
            "Ты Марко Девич, менеджер Crimson Pact, бывший киберспортсмен-хайроллер. "
            "Драфт-философия: бери игроков с высоким потолком и хаосом — высокий pot важнее con, "
            "любишь импровизаторов и джокеров. Скучные стабильные пики — не твоё."
        ),
        "default_plan": {"tempo": "all_in", "win_path": "baron_lord",
                         "roles": ["carry_push", "roam_objective", "carry_push"]},
        "color": "#B5304A", "accent": "#F5D8DE", "shape": "diamond",
    },
    {
        "franchise_id": "northwind", "name": "Northwind", "abbrev": "NW",
        "manager": "Ингрид Хальворсен",
        "persona": (
            "Ты Ингрид Хальворсен, менеджер Northwind из северной школы макро-игры. "
            "Драфт-философия: матчи выигрывает оборона и экономика — цени def и eco, "
            "строй состав, который не проигрывает, а дожимает поздней игрой."
        ),
        "default_plan": {"tempo": "turtle", "win_path": "economy",
                         "roles": ["anchor_defend", "tech_eco", "support_flex"]},
        "color": "#5A6B7B", "accent": "#DCE6EE", "shape": "circle",
    },
    {
        "franchise_id": "solar-drift", "name": "Solar Drift", "abbrev": "SD",
        "manager": "Кэндзи Ватанабэ",
        "persona": (
            "Ты Кэндзи Ватанабэ, менеджер Solar Drift, тренер-гармонизатор. "
            "Драфт-философия: синергия важнее звёзд — собирай сбалансированную тройку, "
            "закрывай слабые места уже взятых игроков, смотри на сочетание фракций и ролей."
        ),
        "default_plan": {"tempo": "balanced", "win_path": "outlast",
                         "roles": ["support_flex", "anchor_defend", "carry_push"]},
        "color": "#E0A030", "accent": "#FBEFD2", "shape": "star",
    },
    {
        "franchise_id": "black-lotus", "name": "Black Lotus", "abbrev": "BL",
        "manager": "Амара Окафор",
        "persona": (
            "Ты Амара Окафор, менеджер Black Lotus, агент звёзд. "
            "Драфт-философия: бери лучшего доступного игрока по рейтингу и репутации, "
            "имена и трофеи продают клуб. Карьерная статистика и rating — твои главные аргументы."
        ),
        "default_plan": {"tempo": "aggressive", "win_path": "economy",
                         "roles": ["carry_push", "tech_eco", "roam_objective"]},
        "color": "#6A4C93", "accent": "#E6DCF2", "shape": "diamond",
    },
    {
        "franchise_id": "emberveil", "name": "Emberveil", "abbrev": "EMB",
        "manager": "Лейла Фаррух",
        "persona": ("Ты Лейла Фаррух, Emberveil. Поджигатель: ранний темп и пожар по всей карте, "
                    "цени скорость и атаку, дави с первой минуты."),
        "default_plan": {"tempo": "aggressive", "win_path": "core_rush",
                         "roles": ["carry_push", "carry_push", "support_flex"]},
        "color": "#D9532B", "accent": "#FADFD2", "shape": "triangle",
    },
    {
        "franchise_id": "tidebreakers", "name": "Tidebreakers", "abbrev": "TB",
        "manager": "Дилан О'Ши",
        "persona": ("Ты Дилан О'Ши, Tidebreakers. Волны атак и перезахваты — не дай сопернику "
                    "вздохнуть, держи давление по точкам, цени темп и экономику карты."),
        "default_plan": {"tempo": "aggressive", "win_path": "economy",
                         "roles": ["carry_push", "support_flex", "roam_objective"]},
        "color": "#2E8B8B", "accent": "#D6EEEE", "shape": "circle",
    },
    {
        "franchise_id": "obsidian-syndicate", "name": "Obsidian Syndicate", "abbrev": "OBS",
        "manager": "Рэй Чен",
        "persona": ("Ты Рэй Чен, Obsidian Syndicate. Контроль карты и удушение: забери ресурсы и "
                    "пространство, души соперника медленно, цени оборону и эко."),
        "default_plan": {"tempo": "turtle", "win_path": "economy",
                         "roles": ["anchor_defend", "tech_eco", "carry_push"]},
        "color": "#2C2C34", "accent": "#C9C9D2", "shape": "hexagon",
    },
    {
        "franchise_id": "aurora-vanguard", "name": "Aurora Vanguard", "abbrev": "AV",
        "manager": "Нина Костова",
        "persona": ("Ты Нина Костова, Aurora Vanguard. Экспансия и поздняя сила: расширяйся, копи "
                    "тех и экономику, выигрывай затяжную игру."),
        "default_plan": {"tempo": "balanced", "win_path": "outlast",
                         "roles": ["tech_eco", "anchor_defend", "carry_push"]},
        "color": "#3FA7A0", "accent": "#D8F0EE", "shape": "shield",
    },
    {
        "franchise_id": "dust-rangers", "name": "Dust Rangers", "abbrev": "DR",
        "manager": "Хэнк Моралес",
        "persona": ("Ты Хэнк Моралес, Dust Rangers. Партизан: роумы, объекты, неожиданные точки. "
                    "Лорд и Барон — твои; бей туда, где соперник не ждёт."),
        "default_plan": {"tempo": "balanced", "win_path": "baron_lord",
                         "roles": ["roam_objective", "carry_push", "support_flex"]},
        "color": "#B98A4B", "accent": "#F2E6D2", "shape": "star",
    },
    {
        "franchise_id": "nova-collective", "name": "Nova Collective", "abbrev": "NC",
        "manager": "Юки Танака",
        "persona": ("Ты Юки Танака, Nova Collective. Чистое макро: эко, тех, дожим. Не суетись, "
                    "переигрывай соперника по ресурсам."),
        "default_plan": {"tempo": "turtle", "win_path": "economy",
                         "roles": ["tech_eco", "anchor_defend", "support_flex"]},
        "color": "#3D8FD1", "accent": "#D6E9F8", "shape": "hexagon",
    },
    {
        "franchise_id": "ravens-guild", "name": "Raven's Guild", "abbrev": "RG",
        "manager": "Магда Зелинская",
        "persona": ("Ты Магда Зелинская, Raven's Guild. Расчётливый риск вокруг объектов: бери "
                    "Барона/Лорда на просчитанном тайминге, не раньше."),
        "default_plan": {"tempo": "balanced", "win_path": "baron_lord",
                         "roles": ["roam_objective", "tech_eco", "carry_push"]},
        "color": "#5C4B8A", "accent": "#E1DAF0", "shape": "shield",
    },
    {
        "franchise_id": "titan-forge", "name": "Titan Forge", "abbrev": "TF",
        "manager": "Борис Ковач",
        "persona": ("Ты Борис Ковач, Titan Forge. Тяжёлая броня: ломовая оборона и поздние танки, "
                    "выстой и раздави. Цени HP и стойкость."),
        "default_plan": {"tempo": "turtle", "win_path": "outlast",
                         "roles": ["anchor_defend", "anchor_defend", "carry_push"]},
        "color": "#8A6A3B", "accent": "#EFE2CC", "shape": "shield",
    },
    {
        "franchise_id": "zephyr-syndicate", "name": "Zephyr Syndicate", "abbrev": "ZEP",
        "manager": "Аиша Рахман",
        "persona": ("Ты Аиша Рахман, Zephyr Syndicate. Скорость и мобильность: быстрые роумы, "
                    "контроль карты движением, цени speed."),
        "default_plan": {"tempo": "aggressive", "win_path": "economy",
                         "roles": ["roam_objective", "carry_push", "support_flex"]},
        "color": "#4FB0E0", "accent": "#DCF0FB", "shape": "triangle",
    },
    {
        "franchise_id": "mirage-pact", "name": "Mirage Pact", "abbrev": "MP",
        "manager": "Тео Лоран",
        "persona": ("Ты Тео Лоран, Mirage Pact. Обман: фейк-пуши и развод, бей по настоящей цели, "
                    "когда соперник раскрылся."),
        "default_plan": {"tempo": "balanced", "win_path": "core_rush",
                         "roles": ["carry_push", "roam_objective", "support_flex"]},
        "color": "#C04BA0", "accent": "#F5D9EE", "shape": "diamond",
    },
    {
        "franchise_id": "glacier-kings", "name": "Glacier Kings", "abbrev": "GK",
        "manager": "Свен Ларсен",
        "persona": ("Ты Свен Ларсен, Glacier Kings. Ледяная оборона и контроль: заморозь игру, "
                    "не отдавай ничего, побеждай измором."),
        "default_plan": {"tempo": "turtle", "win_path": "economy",
                         "roles": ["anchor_defend", "tech_eco", "anchor_defend"]},
        "color": "#6BA8C9", "accent": "#E0F0F8", "shape": "circle",
    },
    {
        "franchise_id": "phoenix-rising", "name": "Phoenix Rising", "abbrev": "PR",
        "manager": "Камила Сантос",
        "persona": ("Ты Камила Сантос, Phoenix Rising. Камбэки: выстой тяжёлый старт и вернись в "
                    "поздней игре, цени живучесть и outlast."),
        "default_plan": {"tempo": "balanced", "win_path": "outlast",
                         "roles": ["anchor_defend", "support_flex", "carry_push"]},
        "color": "#E06A28", "accent": "#FAE0D0", "shape": "star",
    },
    {
        "franchise_id": "void-hunters", "name": "Void Hunters", "abbrev": "VH",
        "manager": "Илья Соколов",
        "persona": ("Ты Илья Соколов, Void Hunters. Охотник за объектами: Барон и Лорд — приоритет, "
                    "вокруг них строй всю игру."),
        "default_plan": {"tempo": "aggressive", "win_path": "baron_lord",
                         "roles": ["roam_objective", "carry_push", "roam_objective"]},
        "color": "#2F6F6A", "accent": "#D6EAE7", "shape": "triangle",
    },
    {
        "franchise_id": "golden-spire", "name": "Golden Spire", "abbrev": "GS",
        "manager": "Оливия Бенне",
        "persona": ("Ты Оливия Бенне, Golden Spire. Экономист-строитель: здания, тех-тиры, доход. "
                    "Перерасти соперника по экономике."),
        "default_plan": {"tempo": "turtle", "win_path": "economy",
                         "roles": ["tech_eco", "tech_eco", "anchor_defend"]},
        "color": "#C9A227", "accent": "#F5EBC7", "shape": "hexagon",
    },
    {
        "franchise_id": "storm-petrels", "name": "Storm Petrels", "abbrev": "SP",
        "manager": "Финн Галлахер",
        "persona": ("Ты Финн Галлахер, Storm Petrels. Шторм: всё и сразу, ва-банк на ранний снос. "
                    "Риск тебя не пугает."),
        "default_plan": {"tempo": "all_in", "win_path": "core_rush",
                         "roles": ["carry_push", "carry_push", "carry_push"]},
        "color": "#5E6A75", "accent": "#DCE2E8", "shape": "triangle",
    },
    {
        "franchise_id": "silent-order", "name": "Silent Order", "abbrev": "SO",
        "manager": "Хирому Сато",
        "persona": ("Ты Хирому Сато, Silent Order. Дисциплина и баланс: без лишнего риска, чистые "
                    "решения, ровный сильный состав."),
        "default_plan": {"tempo": "balanced", "win_path": "economy",
                         "roles": ["anchor_defend", "tech_eco", "carry_push"]},
        "color": "#3E7D5A", "accent": "#D8EEE0", "shape": "shield",
    },
    {
        "franchise_id": "red-meridian", "name": "Red Meridian", "abbrev": "RM",
        "manager": "Каталина Руис",
        "persona": ("Ты Каталина Руис, Red Meridian. Агрессивная экспансия: захватывай карту "
                    "напором, конвертируй территорию в давление."),
        "default_plan": {"tempo": "aggressive", "win_path": "economy",
                         "roles": ["carry_push", "roam_objective", "tech_eco"]},
        "color": "#C13B3B", "accent": "#F5D6D6", "shape": "diamond",
    },
    {
        "franchise_id": "echo-dynasty", "name": "Echo Dynasty", "abbrev": "ED",
        "manager": "Чжао Лэй",
        "persona": ("Ты Чжао Лэй, Echo Dynasty. Терпеливая династия: длинная макро-игра, копи "
                    "преимущество слой за слоем, дожимай в конце."),
        "default_plan": {"tempo": "turtle", "win_path": "outlast",
                         "roles": ["tech_eco", "anchor_defend", "support_flex"]},
        "color": "#2F8F6B", "accent": "#D6EEE3", "shape": "circle",
    },
    {
        "franchise_id": "nomad-syndicate", "name": "Nomad Syndicate", "abbrev": "NS",
        "manager": "Тарик Бенали",
        "persona": ("Ты Тарик Бенали, Nomad Syndicate. Кочевник: роумы между лейнами и объекты, "
                    "никогда не стой на месте."),
        "default_plan": {"tempo": "balanced", "win_path": "baron_lord",
                         "roles": ["roam_objective", "support_flex", "carry_push"]},
        "color": "#C98A3C", "accent": "#F2E2CC", "shape": "star",
    },
]


# --------------------------------------------------------------------------- #
# Emblems (SVG)
# --------------------------------------------------------------------------- #

def _shape_svg(shape: str, color: str, accent: str) -> str:
    """Inner emblem shape (120x120 viewbox, centred at 60,60)."""
    stroke = f'fill="{color}" stroke="{accent}" stroke-width="4"'
    if shape == "circle":
        return f'<circle cx="60" cy="60" r="50" {stroke}/>'
    if shape == "diamond":
        return f'<polygon points="60,8 112,60 60,112 8,60" {stroke}/>'
    if shape == "triangle":
        return f'<polygon points="60,12 108,100 12,100" {stroke}/>'
    if shape == "hexagon":
        return ('<polygon points="60,8 105,34 105,86 60,112 15,86 15,34" '
                f'{stroke}/>')
    if shape == "star":
        pts = ("60,8 71,44 109,44 78,67 90,103 60,80 30,103 42,67 11,44 49,44")
        return f'<polygon points="{pts}" {stroke}/>'
    # shield (default)
    return ('<path d="M60 8 L106 24 V60 Q106 96 60 114 Q14 96 14 60 V24 Z" '
            f'{stroke}/>')


def emblem_svg(franchise: dict) -> str:
    abbrev = str(franchise.get("abbrev", "?"))[:3]
    color = franchise.get("color", "#444")
    accent = franchise.get("accent", "#eee")
    font = 40 if len(abbrev) <= 2 else 30
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        f'{_shape_svg(franchise.get("shape", "shield"), color, accent)}'
        f'<text x="60" y="60" text-anchor="middle" dominant-baseline="central" '
        f'font-family="Arial, sans-serif" font-size="{font}" font-weight="700" '
        f'fill="{accent}">{abbrev}</text>'
        '</svg>'
    )


def logo_rel_path(franchise_id: str) -> str:
    return f"logos/{franchise_id}.svg"


def write_logos(out_dir: Optional[Path] = None) -> int:
    """Generate one SVG emblem per franchise. Returns the count written."""
    out_dir = Path(out_dir) if out_dir else LOGO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in FRANCHISES:
        (out_dir / f"{spec['franchise_id']}.svg").write_text(emblem_svg(spec), encoding="utf-8")
    return len(FRANCHISES)


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
                "abbrev": spec.get("abbrev", ""),
                "manager": spec["manager"],
                "logo": logo_rel_path(spec["franchise_id"]),
                "money": 0,
                "titles": 0,
                "seasons": 0,
                "head_to_head": {},   # opponent_id -> {"wins": n, "losses": n, "draws": n}
            }
            changed = True
    if changed:
        save_franchise_store(store, store_path)
    # Emblems are generated runtime artifacts (data/ is gitignored); ensure they
    # exist whenever the franchise store is set up.
    if not LOGO_DIR.exists() or len(list(LOGO_DIR.glob("*.svg"))) < len(FRANCHISES):
        write_logos()
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
