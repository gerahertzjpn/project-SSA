"""Regenerate viewer_pixi/world_preview_franchises.js from the REAL franchises + emblems.
Run after changing the emblem generator: PYTHONPATH=. python scripts/gen_preview_franchises.py
"""
import io, json
from src.regionals.franchises import franchise_specs, emblem_svg

items = [{"franchise_id": f["franchise_id"], "name": f["name"], "abbrev": f.get("abbrev", ""),
          "color": f["color"], "accent": f["accent"], "emblem": emblem_svg(f)}
         for f in franchise_specs()]
header = "/* Auto-generated from src/regionals/franchises.py — REAL franchises + emblems (do not invent). */\n"
body = "window.__PREVIEW_FRANCHISES__ = " + json.dumps(items, ensure_ascii=False) + ";\n"
io.open("viewer_pixi/world_preview_franchises.js", "w", encoding="utf-8", newline="\n").write(header + body)
print("regenerated world_preview_franchises.js with", len(items), "franchises")
