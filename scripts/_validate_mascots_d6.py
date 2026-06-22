"""Validate designer-6 mascot SVG element strings: well-formed XML, bounds, tokens only."""
import re
import sys
import xml.etree.ElementTree as ET

A = "#E8EEF5"
C = "#11161d"

def coords_from_points(s):
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]
    return list(zip(nums[0::2], nums[1::2]))

def check(name, svg):
    wrapped = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">'
               + svg.replace("{a}", A).replace("{c}", C) + "</svg>")
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        return f"XML PARSE ERROR: {e}"
    bad = set(re.findall(r'#[0-9A-Fa-f]{3,6}', svg))
    if bad:
        return f"RAW HEX COLORS: {bad}"
    if "gradient" in svg.lower() or "<text" in svg.lower():
        return "FORBIDDEN gradient/text"
    if "{a}" not in svg:
        return "no {a} fill used"
    minx=miny=1e9; maxx=maxy=-1e9
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag in ('polygon','polyline'):
            for x,y in coords_from_points(el.get('points','')):
                minx=min(minx,x);maxx=max(maxx,x);miny=min(miny,y);maxy=max(maxy,y)
        elif tag=='circle':
            cx=float(el.get('cx',0));cy=float(el.get('cy',0));r=float(el.get('r',0))
            minx=min(minx,cx-r);maxx=max(maxx,cx+r);miny=min(miny,cy-r);maxy=max(maxy,cy+r)
        elif tag=='ellipse':
            cx=float(el.get('cx',0));cy=float(el.get('cy',0));rx=float(el.get('rx',0));ry=float(el.get('ry',0))
            minx=min(minx,cx-rx);maxx=max(maxx,cx+rx);miny=min(miny,cy-ry);maxy=max(maxy,cy+ry)
        elif tag=='path':
            nums=[float(x) for x in re.findall(r"-?\d+\.?\d*", el.get('d',''))]
            xs=nums[0::2]; ys=nums[1::2]
            if xs: minx=min(minx,min(xs));maxx=max(maxx,max(xs))
            if ys: miny=min(miny,min(ys));maxy=max(maxy,max(ys))
    warn=[]
    if maxy>80.5: warn.append(f"y>80 ({maxy:.1f})")
    if miny<21.5: warn.append(f"y<22 ({miny:.1f})")
    if minx<25.5: warn.append(f"x<26 ({minx:.1f})")
    if maxx>94.5: warn.append(f"x>94 ({maxx:.1f})")
    return ("OK" if not warn else "WARN: "+"; ".join(warn)) + f"  bbox=({minx:.0f},{miny:.0f})-({maxx:.0f},{maxy:.0f})"

if __name__ == "__main__":
    import json
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    bad = 0
    for name, svg in data.items():
        res = check(name, svg)
        if not res.startswith("OK"): bad += 1
        print(f"{name:10s} {res}")
    print(f"\n{'ALL OK' if bad==0 else str(bad)+' need attention'} ({len(data)} mascots)")
