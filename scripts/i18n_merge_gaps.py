"""Merge a gap-translation file ({en,ja}) into the EXISTING viewer_pixi/i18n_catalog.js
(keeping all current translations) + re-assert the hand-written LABELS supplement. Usage:
    python scripts/i18n_merge_gaps.py _i18n_gaps_translated.json
"""
import io, json, re, sys


def grab(js, key):
    m = re.search(r'(?:^|[{,\s])' + key + r'\s*:\s*\{', js)
    i = js.index("{", m.end() - 1)
    d = 0; instr = False; esc = False
    for j in range(i, len(js)):
        c = js[j]
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
            continue
        if c == '"': instr = True
        elif c == "{": d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return json.loads(re.sub(r",(\s*[}\]])", r"\1", js[i:j + 1]))
    return {}


# enumerable label values used inside "Label: {value}" renders (kept in sync with the first build)
LABELS = {
    "Атака": ("Attack", "攻撃"), "Защита": ("Defense", "防御"), "Эконом": ("Economy", "経済"),
    "Темп": ("Tempo", "テンポ"), "Потенциал": ("Potential", "ポテンシャル"),
    "Верхняя лига": ("Upper League", "上位リーグ"), "Низшая лига": ("Lower League", "下位リーグ"),
    "Низшая": ("Lower", "下位"), "Драфт новичков": ("Rookie draft", "ルーキードラフト"),
}

path = sys.argv[1] if len(sys.argv) > 1 else "_i18n_gaps_translated.json"
gaps = json.load(io.open(path, encoding="utf-8"))
g_en, g_ja = gaps["en"], gaps["ja"]

cur = io.open("viewer_pixi/i18n_catalog.js", encoding="utf-8").read()
en = grab(cur, "en")
ja = grab(cur, "ja")
before = len(en)

en.update(g_en)
ja.update(g_ja)
for k, (e, j) in LABELS.items():
    en.setdefault(k, e)
    ja.setdefault(k, j)


def jdump(o):
    return json.dumps(o, ensure_ascii=False, indent=0, separators=(",", ":"))


header = ("/* Auto-generated translation catalog for the viewer UI (RU source -> EN/JA).\n"
          "   Keys are the Russian source string (with {0}/{1} placeholders). Missing key =>\n"
          "   falls back to Russian in i18n.js. Regenerate: scripts/i18n_extract.py -> translate\n"
          "   workflow -> scripts/i18n_merge_gaps.py. English-source menu/Match strings: i18n_extra.js. */\n")
body = "window.__I18N_CATALOG__ = {\n  ru: {},\n  en: %s,\n  ja: %s\n};\n" % (jdump(en), jdump(ja))
io.open("viewer_pixi/i18n_catalog.js", "w", encoding="utf-8", newline="\n").write(header + body)
print("merged %d gap keys -> catalog en %d->%d, ja now %d" % (len(g_en), before, len(en), len(ja)))
