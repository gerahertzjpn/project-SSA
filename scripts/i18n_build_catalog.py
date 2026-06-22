"""Assemble viewer_pixi/i18n_catalog.js from the workflow translations + supplements."""
import io, json

tr = json.load(io.open("_i18n_translated.json", encoding="utf-8"))
en, ja = tr["en"], tr["ja"]

# Hand-written supplements: paragraphs that fragmented during extraction (inner <strong>),
# so the FULL plain-text key wasn't auto-translated. Keyed by the plain Russian sentence.
SUPP = {
    "Отметь ровно 22 команды. Создание прогоняет SoloQ и расстановку лиг реальными матчами нейронок — это несколько минут, идёт в фоне.": {
        "en": "Pick exactly 22 teams. Creation runs SoloQ and league seeding with real net matches — a few minutes, in the background.",
        "ja": "ちょうど22チームを選択。作成時にSoloQとリーグ配置を本物のネット対戦で実行します — 数分、バックグラウンドで進みます。",
    },
    # world_preview.html demo description (has inner emphasis in some renders)
    "Полная копия World, всё захардкожено: никаких матчей, запросов и тренировок. Сезоны 0·1·2 уже «отыграны», решения приняты за тебя и подсвечены. Все вкладки живые — «🔴 LIVE» (сезон по дням + эфир), «🆚 Очные», «💸 Трансферы», личные страницы игроков. Жми «Сыграть сезон» / «📅 По дням» / «Межсезонье».": {
        "en": "A full copy of World, everything hardcoded: no matches, requests or training. Seasons 0·1·2 are already “played”, decisions are made for you and highlighted. Every tab is live — “🔴 LIVE” (day-by-day season + broadcast), “🆚 Head-to-head”, “💸 Transfers”, player pages. Hit “Play season” / “📅 Day by day” / “Off-season”.",
        "ja": "Worldの完全コピー、すべてハードコード：試合・通信・トレーニングなし。シーズン0・1・2はすでに「プレイ済み」で、判断は代行・ハイライト表示。全タブが稼働 — 「🔴 LIVE」（日ごとのシーズン＋放送）、「🆚 直接対決」、「💸 移籍」、選手ページ。「シーズン開始」/「📅 日ごと」/「オフシーズン」を押そう。",
    },
}

# Enumerable label values used inside "Label: {value}" renders (radar attrs, league
# names) that weren't auto-extracted standalone — so "Стиль/Позиция: {value}" fully localizes.
LABELS = {
    "Атака": ("Attack", "攻撃"),
    "Защита": ("Defense", "防御"),
    "Эконом": ("Economy", "経済"),
    "Темп": ("Tempo", "テンポ"),
    "Потенциал": ("Potential", "ポテンシャル"),
    "Верхняя лига": ("Upper League", "上位リーグ"),
    "Низшая лига": ("Lower League", "下位リーグ"),
    "Низшая": ("Lower", "下位"),
    "Драфт новичков": ("Rookie draft", "ルーキードラフト"),
}
for k, (e, j) in LABELS.items():
    SUPP[k] = {"en": e, "ja": j}

for k, v in SUPP.items():
    en[k] = v["en"]
    ja[k] = v["ja"]


def jdump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=0, separators=(",", ":"))


header = ("/* Auto-generated translation catalog for the viewer UI (RU source -> EN/JA).\n"
          "   Keys are the Russian source string (with {0}/{1} placeholders). Missing key =>\n"
          "   falls back to Russian in i18n.js. Regenerate via the translate-ui-ruenja workflow\n"
          "   then _i18n_build_catalog.py. English-source menu/Match strings live in i18n_extra.js. */\n")

body = "window.__I18N_CATALOG__ = {\n  ru: {},\n  en: %s,\n  ja: %s\n};\n" % (
    jdump(en), jdump(ja))

io.open("viewer_pixi/i18n_catalog.js", "w", encoding="utf-8", newline="\n").write(header + body)
print("wrote i18n_catalog.js  en=%d ja=%d" % (len(en), len(ja)))

# spot-check a few key UI terms
for k in ["🏆 Лиги", "🔴 LIVE", "🆚 Очные", "📊 Рейтинг", "Команда", "Очки", "Создать World",
          "▶ Сыграть сезон", "⟳ Межсезонье"]:
    print("  %-22s EN=%-18s JA=%s" % (k, en.get(k, "—"), ja.get(k, "—")))
