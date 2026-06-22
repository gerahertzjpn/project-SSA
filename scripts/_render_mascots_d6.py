"""Render the 13 designer-6 mascots into shield crests as a PNG contact sheet for visual QA."""
import json
from playwright.sync_api import sync_playwright

data = json.load(open("scripts/_mascots_d6.json", encoding="utf-8"))

# representative crest palettes (dark crest bg, light accent {a}, dark carve {c})
PAL = {
    "falcon":  ("#2e5681", "#1e3752", "#D8E4F0"),
    "eagle":   ("#8d253a", "#5a1825", "#F5D8DE"),
    "owl":     ("#3a3170", "#262048", "#E0DBF5"),
    "raven":   ("#483a6c", "#2e2645", "#E1DAF0"),
    "phoenix": ("#af531f", "#703514", "#FAE0D0"),
    "dragon":  ("#246c6c", "#174646", "#D6EEEE"),
    "wolf":    ("#49535b", "#2f353a", "#DCE2E8"),
    "boar":    ("#906c3a", "#5c4526", "#F2E6D2"),
    "scorpion":("#222229", "#16161a", "#C9C9D2"),
    "bear":    ("#53839d", "#365464", "#E0F0F8"),
    "shark":   ("#246c6c", "#174646", "#D6EEEE"),
    "cobra":   ("#3a3170", "#262048", "#E0DBF5"),
    "stag":    ("#9d6c2f", "#64451e", "#F2E2CC"),
}

def crest(name, body):
    mid, deep, accent = PAL[name]
    inner = body.replace("{a}", accent).replace("{c}", deep)
    gid = "g_" + name
    return f'''
    <div class="cell">
      <svg viewBox="0 0 120 120" width="150" height="150">
        <defs><linearGradient id="{gid}" x1="0" y1="0" x2="0.25" y2="1">
          <stop offset="0" stop-color="{mid}"/><stop offset="1" stop-color="{deep}"/>
        </linearGradient></defs>
        <path d="M60 7 L105 21 V62 Q105 99 60 115 Q15 99 15 62 V21 Z" fill="url(#{gid})" stroke="{accent}" stroke-width="2.5"/>
        {inner}
      </svg>
      <div class="lbl">{name}</div>
    </div>'''

cells = "".join(crest(n, b) for n, b in data.items())
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#0e1116;font-family:Arial,sans-serif}}
.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:14px}}
.cell{{display:flex;flex-direction:column;align-items:center;background:#161a21;border-radius:10px;padding:6px}}
.lbl{{color:#c9d2dd;font-size:13px;font-weight:700;margin-top:2px;text-transform:uppercase;letter-spacing:1px}}
</style></head><body><div class="grid">{cells}</div></body></html>'''

open("scripts/_mascots_d6_preview.html", "w", encoding="utf-8").write(html)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 800, "height": 660}, device_scale_factor=2)
    pg.set_content(html)
    pg.wait_for_timeout(250)
    pg.locator(".grid").screenshot(path="scripts/_mascots_d6_preview.png")
    b.close()
print("wrote scripts/_mascots_d6_preview.png")
