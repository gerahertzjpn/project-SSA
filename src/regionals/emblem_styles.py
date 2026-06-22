# -*- coding: utf-8 -*-
"""Team emblem generator v2 — distinctive, esports-style marks.

Two families, dispatched per franchise:
  • BEAST CREST — an angular mascot silhouette (13 unique: falcon/eagle/owl/raven/phoenix/dragon/
    wolf/boar/scorpion/bear/shark/cobra/stag) on a gradient shield + abbreviation banner. Used for
    the animal/predator-themed franchises.
  • APEX MARK — a bold abstract mark on a dark glow ground, in FIVE variants (chevron / blade /
    hexcore / spire / orbit) assigned by theme. Used for the abstract/elemental franchises and as
    the DEFAULT for any future/forked franchise (variant chosen by a stable hash).

Every emblem is a self-contained 120x120 SVG. Internal gradient/clip ids use a per-call counter; the
viewer's uniqSvg() re-suffixes ids per render, so inlining the same emblem many times never collides.
"""
from __future__ import annotations

_EMBLEM_UID = [0]


def _uid() -> int:
    _EMBLEM_UID[0] += 1
    return _EMBLEM_UID[0]


def _shade(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, round(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _svg(inner: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120" '
            'preserveAspectRatio="xMidYMid meet">' + inner + "</svg>")


# ============================================================================ #
#  BEAST CREST — angular mascot silhouettes, centred ~ (60,52). {a}=accent fill,
#  {c}=crest-dark (negative-space eyes/cuts).
# ============================================================================ #
_BEASTS = {
    "falcon": '<g fill="{a}"><path d="M30 34 L52 40 L60 36 L68 40 L90 34 L78 50 L84 58 L70 54 L66 60 L60 78 L54 60 L50 54 L36 58 L42 50 Z"/></g><polygon points="46,44 58,50 58,42" fill="{c}"/><polygon points="74,44 62,50 62,42" fill="{c}"/><polygon points="58,38 60,46 62,38" fill="{c}"/><polygon points="60,58 63,70 57,70" fill="{c}"/>',
    "owl": '<g fill="{a}"><path d="M38 40 L32 22 L51 38 L60 33 L69 38 L88 22 L82 40 L91 57 L75 77 L60 85 L45 77 L29 57 Z"/></g><g fill="{c}"><circle cx="47" cy="56" r="11"/><circle cx="73" cy="56" r="11"/><polygon points="33,45 55,52 60,46 65,52 87,45 87,39 60,43 33,39"/><polygon points="55,62 60,74 65,62 60,66"/></g><g fill="{a}"><circle cx="47" cy="57" r="4.5"/><circle cx="73" cy="57" r="4.5"/></g>',
    "raven": '<g fill="{a}"><path d="M44 32 Q46 26 54 27 Q63 28 65 37 Q75 41 81 55 L90 76 L79 74 L73 62 Q63 65 55 61 Q45 57 43 47 Z"/><polygon points="43,38 24,40 43,43"/></g><circle cx="51" cy="37" r="2.2" fill="{c}"/><g stroke="{a}" stroke-width="2.6" stroke-linecap="round"><line x1="55" y1="60" x2="53" y2="78"/><line x1="63" y1="59" x2="63" y2="78"/></g><path d="M22 82 Q56 77 98 85" stroke="{a}" stroke-width="4" fill="none" stroke-linecap="round"/><path d="M74 80 L88 71" stroke="{a}" stroke-width="3" stroke-linecap="round"/>',
    "phoenix": '<g fill="{a}"><path d="M60 22 L66 30 L60 36 L54 30 Z"/><polygon points="60,34 66,44 54,44"/><path d="M55 42 L65 42 L62 74 L60 80 L58 74 Z"/><path d="M57 46 C41 38 31 27 28 23 C31 33 31 41 37 48 C29 46 27 42 26 40 C28 52 39 62 57 60 Z"/><path d="M63 46 C79 38 89 27 92 23 C89 33 89 41 83 48 C91 46 93 42 94 40 C92 52 81 62 63 60 Z"/><path d="M50 64 L40 78 L52 70 Z"/><path d="M70 64 L80 78 L68 70 Z"/></g><polygon points="56,48 60,56 64,48" fill="{c}"/><circle cx="54" cy="42" r="2" fill="{c}"/><circle cx="66" cy="42" r="2" fill="{c}"/>',
    "dragon": '<g fill="{a}"><path d="M44 32 L34 22 L50 30 Z"/><path d="M76 32 L86 22 L70 30 Z"/><path d="M40 30 L60 26 L80 30 L84 44 L74 49 L80 60 L60 78 L40 60 L46 49 L36 44 Z"/></g><polygon points="46,38 60,45 60,36" fill="{c}"/><polygon points="74,38 60,45 60,36" fill="{c}"/><polygon points="52,57 60,67 68,57 62,60 60,64 58,60" fill="{c}"/>',
    "wolf": '<g fill="{a}"><path d="M32 26 L47 45 L60 39 L73 45 L88 26 L86 50 L78 66 Q60 80 42 66 L34 50 Z"/></g><polygon points="42,31 50,46 36,44" fill="{c}"/><polygon points="78,31 70,46 84,44" fill="{c}"/><polygon points="52,53 60,67 68,53 62,56 60,60 58,56" fill="{c}"/><polygon points="48,47 56,50 48,53" fill="{c}"/><polygon points="72,47 64,50 72,53" fill="{c}"/>',
    "bear": '<g fill="{a}"><circle cx="39" cy="35" r="11"/><circle cx="81" cy="35" r="11"/><path d="M35 42 Q33 40 60 42 Q87 40 85 42 Q88 60 76 70 Q60 80 44 70 Q32 60 35 42 Z"/></g><polygon points="46,47 55,51 46,54" fill="{c}"/><polygon points="74,47 65,51 74,54" fill="{c}"/><path d="M50 60 Q50 72 60 76 Q70 72 70 60 Q60 56 50 60 Z" fill="{c}"/><polygon points="60,58 67,66 53,66" fill="{a}"/>',
    "shark": '<g fill="{a}"><path d="M24 50 Q44 32 64 36 L60 22 L80 40 Q88 47 86 56 L70 56 Q60 60 58 78 Q48 66 50 56 Z"/><path d="M70 56 L96 50 L82 64 Z"/></g><g fill="{c}"><polygon points="42,54 48,63 36,62"/><polygon points="52,55 58,64 46,63"/><polygon points="62,56 68,63 56,64"/></g><polygon points="64,44 72,47 64,50" fill="{c}"/>',
    "cobra": '<g fill="{a}"><path d="M60 26 C40 26 28 42 30 62 C31 72 41 72 46 64 C44 53 50 46 60 46 C70 46 76 53 74 64 C79 72 89 72 90 62 C92 42 80 26 60 26 Z"/><path d="M51 50 Q51 68 60 76 Q69 68 69 50 Z"/></g><path d="M48 35 Q60 30 72 35 Q65 46 60 46 Q55 46 48 35 Z" fill="{c}"/><polygon points="48,53 61,57 52,64" fill="{c}"/><polygon points="72,53 59,57 68,64" fill="{c}"/><polygon points="56,73 59,79 57,80" fill="{a}"/><polygon points="64,73 61,79 63,80" fill="{a}"/>',
    "stag": '<g fill="{a}"><path d="M46 44 L51 66 Q55 80 60 80 Q65 80 69 66 L74 44 Q60 52 46 44 Z"/><g stroke="{a}" stroke-width="4" stroke-linecap="round" fill="none"><path d="M46 44 L33 26"/><path d="M33 26 L23 30"/><path d="M33 26 L30 22"/><path d="M40 34 L27 35"/><path d="M74 44 L87 26"/><path d="M87 26 L97 30"/><path d="M87 26 L90 22"/><path d="M80 34 L93 35"/></g></g><polygon points="53,52 60,56 53,59" fill="{c}"/><polygon points="67,52 60,56 67,59" fill="{c}"/><polygon points="60,64 65,74 55,74" fill="{c}"/>',
    "eagle": '<g fill="{a}"><path d="M58 47 L30 30 L40 44 L19 39 L37 50 L20 55 L41 60 L24 67 L48 64 L58 55 Z"/><path d="M62 47 L90 30 L80 44 L101 39 L83 50 L100 55 L79 60 L96 67 L72 64 L62 55 Z"/><path d="M43 35 Q43 27 55 29 L60 24 L65 29 Q77 27 77 35 L73 50 L60 56 L47 50 Z"/><polygon points="54,52 66,52 63,66 60,73 57,66"/></g><polygon points="46,40 59,47 59,39" fill="{c}"/><polygon points="74,40 61,47 61,39" fill="{c}"/><circle cx="52" cy="44" r="2.8" fill="{c}"/><circle cx="68" cy="44" r="2.8" fill="{c}"/>',
    "boar": '<g fill="{a}"><polygon points="42,34 34,22 52,32"/><polygon points="78,34 86,22 68,32"/><path d="M40 34 L52 28 L60 34 L68 28 L80 34 L82 50 L72 56 L70 64 L60 70 L50 64 L48 56 L38 50 Z"/><polygon points="50,58 70,58 67,70 60,76 53,70"/><path d="M52 68 Q42 70 40 58 Q44 64 51 63 Q54 65 52 68 Z"/><path d="M68 68 Q78 70 80 58 Q76 64 69 63 Q66 65 68 68 Z"/></g><polygon points="42,40 56,46 52,38" fill="{c}"/><polygon points="78,40 64,46 68,38" fill="{c}"/><polygon points="46,47 56,50 47,53" fill="{c}"/><polygon points="74,47 64,50 73,53" fill="{c}"/><ellipse cx="55" cy="63" rx="2.2" ry="3" fill="{c}"/><ellipse cx="65" cy="63" rx="2.2" ry="3" fill="{c}"/>',
    "scorpion": '<g fill="{a}"><path d="M50 48 L34 44 L26 36 L32 32 L40 40 L52 42 Z"/><path d="M32 32 L20 24 L26 22 L36 30 Z"/><path d="M26 36 L18 32 L22 28 L30 34 Z"/><path d="M70 48 L86 44 L94 36 L88 32 L80 40 L68 42 Z"/><path d="M88 32 L100 24 L94 22 L84 30 Z"/><path d="M94 36 L102 32 L98 28 L90 34 Z"/><polygon points="52,44 68,44 66,50 54,50"/><polygon points="54,51 66,51 64,57 56,57"/><polygon points="56,58 64,58 62,63 58,63"/><path d="M58 63 Q50 70 56 74 Q64 78 70 70 Q74 64 70 56 L64 58 Q66 64 63 67 Q60 69 60 64 Z"/><polygon points="68,52 64,62 76,58"/></g><polygon points="55,46 60,48 55,49" fill="{c}"/><polygon points="65,46 60,48 65,49" fill="{c}"/>',
}

# franchise_id -> mascot (the animal/predator-themed clubs; everything else = Apex)
_BEAST_BY_ID = {
    "iron-falcons": "falcon", "storm-petrels": "eagle", "quantum-owls": "owl",
    "ravens-guild": "raven", "phoenix-rising": "phoenix", "emberveil": "dragon",
    "void-hunters": "wolf", "crimson-pact": "boar", "dust-rangers": "scorpion",
    "glacier-kings": "bear", "tidebreakers": "shark", "obsidian-syndicate": "cobra",
    "nomad-syndicate": "stag",
}


def _beast_crest(f: dict) -> str:
    """Just the MASCOT on a TRANSPARENT background — no shield, no banner, no abbreviation (a clean
    logo mark). The body is an accent->lighter gradient so even dark-coloured teams stay visible on
    the app's dark panels; {c} negative-space details stay dark inside the light body. Centered +
    slightly enlarged since there's no shield to leave room for."""
    u = _uid()
    c = f.get("color", "#444444")
    a = f.get("accent", "#eeeeee")
    dark, light = _shade(c, 0.5), _lighten(c, 0.28)
    beast = _BEASTS[_BEAST_BY_ID[f["franchise_id"]]]
    body = beast.replace("{a}", f"url(#tg{u})").replace("{c}", dark)
    return _svg(
        f'<defs><linearGradient id="tg{u}" x1="0" y1="0" x2="0.35" y2="1">'
        f'<stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{light}"/></linearGradient></defs>'
        f'<g transform="translate(60 57) scale(1.12) translate(-60 -50)">{body}</g>')


# ============================================================================ #
#  APEX MARK — bold abstract marks on a dark glow ground (5 variants).
# ============================================================================ #
_INK = "#0a0d15"


def _apex_ground(u: int, c: str, a: str) -> str:
    return (f'<defs><radialGradient id="ar{u}" cx="0.5" cy="0.4" r="0.8">'
            f'<stop offset="0" stop-color="{_shade(c, 0.7)}"/><stop offset="1" stop-color="{_INK}"/>'
            f'</radialGradient><linearGradient id="ab{u}" x1="0" y1="1" x2="0.3" y2="0">'
            f'<stop offset="0" stop-color="{c}"/><stop offset="1" stop-color="{_lighten(a, 0.18)}"/>'
            f'</linearGradient></defs>'
            f'<rect x="8" y="8" width="104" height="104" rx="27" fill="url(#ar{u})" '
            f'stroke="{a}" stroke-opacity="0.22" stroke-width="2"/>')


def _av_chevron(u, c, a, t):
    return (f'<g transform="rotate({t} 60 60)">'
            f'<polygon points="60,18 96,64 80,64 60,38 40,64 24,64" fill="url(#ab{u})"/>'
            f'<polygon points="60,30 78,54 60,54 42,54" fill="{_INK}" opacity="0.22"/>'
            f'<polygon points="60,52 90,92 75,92 60,72 45,92 30,92" fill="{a}"/>'
            f'<polygon points="60,62 74,82 60,82 46,82" fill="{_INK}" opacity="0.20"/></g>')


def _av_blade(u, c, a, t):
    return (f'<g transform="rotate({t} 60 60)">'
            f'<polygon points="60,16 71,52 60,62 49,52" fill="url(#ab{u})"/>'
            f'<polygon points="60,104 71,68 60,58 49,68" fill="url(#ab{u})"/>'
            f'<polygon points="16,60 52,49 62,60 52,71" fill="{a}"/>'
            f'<polygon points="104,60 68,49 58,60 68,71" fill="{a}"/>'
            f'<circle cx="60" cy="60" r="8" fill="{_INK}"/><circle cx="60" cy="60" r="4.2" fill="{a}"/></g>')


def _av_hexcore(u, c, a, t):
    hx = "60,16 98,38 98,82 60,104 22,82 22,38"
    return (f'<g transform="rotate({t} 60 60)">'
            f'<polygon points="{hx}" fill="url(#ab{u})"/>'
            f'<g transform="translate(60 60) scale(0.6) translate(-60 -60)"><polygon points="{hx}" fill="{_INK}"/></g>'
            f'<g transform="translate(60 60) scale(0.3) translate(-60 -60)"><polygon points="{hx}" fill="{a}"/></g>'
            f'<polygon points="{hx}" fill="none" stroke="{a}" stroke-width="2.5" stroke-linejoin="round"/></g>')


def _av_spire(u, c, a, t):
    # a clean faceted obelisk/peak — one tall triangle, a 3D shade, an inner gem, an outline
    return (f'<g transform="rotate({t} 60 60)">'
            f'<polygon points="60,14 86,100 34,100" fill="url(#ab{u})"/>'
            f'<polygon points="60,14 60,100 34,100" fill="{_INK}" opacity="0.16"/>'
            f'<polygon points="60,40 71,64 49,64" fill="{a}"/>'
            f'<polygon points="60,14 86,100 34,100" fill="none" stroke="{a}" '
            f'stroke-width="2.5" stroke-linejoin="round"/></g>')


def _av_orbit(u, c, a, t):
    # a clean full ring + a 4-point nova burst (no broken-looking arc gap)
    return (f'<g transform="rotate({t} 60 60)">'
            f'<circle cx="60" cy="60" r="33" fill="none" stroke="url(#ab{u})" stroke-width="9"/>'
            f'<polygon points="60,36 65,55 84,60 65,65 60,84 55,65 36,60 55,55" fill="{a}"/>'
            f'<circle cx="60" cy="60" r="5" fill="{_INK}"/><circle cx="60" cy="60" r="2.6" fill="{a}"/></g>')


def _av_lotus(u, c, a, t):
    # thematic BLACK LOTUS — a BIGGER symmetric blooming lotus (5 petals) + a water reflection line
    return ('<g transform="translate(60 50) scale(1.2) translate(-60 -50)" '
            'fill="url(#ab%d)" stroke="%s" stroke-width="1.15" stroke-linejoin="round">'
            '<path d="M60 26 C66 42 66 56 60 70 C54 56 54 42 60 26 Z"/>'
            '<path d="M60 70 C49 64 43 53 45 40 C55 46 60 58 60 70 Z"/>'
            '<path d="M60 70 C71 64 77 53 75 40 C65 46 60 58 60 70 Z"/>'
            '<path d="M60 72 C45 71 33 62 31 49 C45 52 56 61 60 72 Z"/>'
            '<path d="M60 72 C75 71 87 62 89 49 C75 52 64 61 60 72 Z"/></g>'
            '<path d="M28 84 Q60 94 92 84" fill="none" stroke="%s" stroke-width="3" '
            'stroke-linecap="round"/>') % (u, a, a)


def _av_tower(u, c, a, t):
    # GOLDEN SPIRE — a tiered tower/spire with a finial + window slits
    return (f'<g transform="rotate({t} 60 60)">'
            f'<g fill="url(#ab{u})" stroke="{a}" stroke-width="1.6" stroke-linejoin="round">'
            f'<polygon points="60,12 67,40 53,40"/>'
            f'<polygon points="55,40 65,40 63,58 57,58"/>'
            f'<polygon points="52,58 68,58 66,78 54,78"/>'
            f'<polygon points="48,78 72,78 70,100 50,100"/></g>'
            f'<circle cx="60" cy="11" r="2.6" fill="{a}"/>'
            f'<g fill="{_INK}"><rect x="58" y="46" width="4" height="8" rx="1"/>'
            f'<rect x="58" y="64" width="4" height="9" rx="1"/>'
            f'<rect x="56.5" y="85" width="7" height="11" rx="1"/></g></g>')


def _av_forge(u, c, a, t):
    # TITAN FORGE — a hammer over an anvil + sparks
    return (f'<g transform="rotate({t} 60 60)">'
            f'<g fill="url(#ab{u})">'
            f'<path d="M34 66 L86 66 Q86 74 78 74 L66 74 L64 84 L56 84 L54 74 L42 74 Q34 74 34 66 Z"/>'
            f'<rect x="49" y="84" width="22" height="7" rx="1"/></g>'
            f'<g transform="rotate(-32 60 42)" fill="{a}">'
            f'<rect x="42" y="32" width="34" height="14" rx="3"/>'
            f'<rect x="57" y="44" width="5" height="24" rx="2"/></g>'
            f'<g fill="{a}"><circle cx="80" cy="48" r="2.6"/><circle cx="86" cy="56" r="1.8"/>'
            f'<circle cx="76" cy="58" r="1.5"/></g></g>')


def _av_aurora(u, c, a, t):
    # AURORA VANGUARD — flowing northern-lights bands + stars
    light = _lighten(a, 0.2)
    return (f'<g transform="rotate({t} 60 60)" fill="none" stroke-linecap="round">'
            f'<path d="M26 52 Q44 34 60 46 T94 42" stroke="url(#ab{u})" stroke-width="9"/>'
            f'<path d="M26 66 Q44 48 60 60 T94 56" stroke="{a}" stroke-width="7" opacity="0.9"/>'
            f'<path d="M30 80 Q46 64 60 74 T90 70" stroke="{light}" stroke-width="5" opacity="0.75"/>'
            f'</g>'
            f'<g fill="{a}"><circle cx="40" cy="34" r="2"/><circle cx="80" cy="32" r="2.4"/>'
            f'<circle cx="64" cy="28" r="1.6"/><circle cx="52" cy="90" r="1.8"/></g>')


def _av_nova(u, c, a, t):
    # NOVA COLLECTIVE — a supernova starburst (4 long axis rays + 4 diagonal rays + glowing core)
    light = _lighten(a, 0.2)
    return (f'<g transform="rotate({t} 60 60)">'
            f'<g fill="url(#ab{u})">'
            f'<polygon points="60,12 66,54 60,62 54,54"/><polygon points="108,60 66,66 58,60 66,54"/>'
            f'<polygon points="60,108 54,66 62,60 66,66"/><polygon points="12,60 54,54 62,60 54,66"/></g>'
            f'<g fill="{a}" transform="rotate(45 60 60)">'
            f'<polygon points="60,28 64,55 60,60 56,55"/><polygon points="92,60 65,64 60,60 65,56"/>'
            f'<polygon points="60,92 56,65 60,60 64,65"/><polygon points="28,60 55,56 60,60 55,64"/></g>'
            f'<circle cx="60" cy="60" r="9" fill="{light}"/>'
            f'<circle cx="60" cy="60" r="5" fill="#ffffff" opacity="0.92"/></g>')


def _av_echo(u, c, a, t):
    # ECHO DYNASTY — concentric ripples fading outward + a source core
    light = _lighten(a, 0.2)
    return (f'<g transform="rotate({t} 60 60)" fill="none">'
            f'<circle cx="60" cy="60" r="13" stroke="url(#ab{u})" stroke-width="7"/>'
            f'<circle cx="60" cy="60" r="26" stroke="{a}" stroke-width="5" opacity="0.7"/>'
            f'<circle cx="60" cy="60" r="39" stroke="{light}" stroke-width="3.5" opacity="0.42"/></g>'
            f'<circle cx="60" cy="60" r="5" fill="{a}"/>')


def _av_meridian(u, c, a, t):
    # RED MERIDIAN — a globe with a grid + a highlighted meridian great-circle + a zenith sun
    return (f'<g transform="rotate({t} 60 60)">'
            f'<circle cx="60" cy="62" r="33" fill="url(#ab{u})"/>'
            f'<g fill="none" stroke="{_INK}" stroke-width="2.5" stroke-opacity="0.42">'
            f'<ellipse cx="60" cy="62" rx="33" ry="11"/><ellipse cx="60" cy="62" rx="20" ry="33"/></g>'
            f'<path d="M60 30 A 9 32 0 0 0 60 94" fill="none" stroke="{a}" stroke-width="4"/>'
            f'<circle cx="60" cy="24" r="4.5" fill="{a}"/></g>')


def _av_seal(u, c, a, t):
    # SILENT ORDER — an arcane order seal: a runic ring + an inscribed sigil
    return (f'<g transform="rotate({t} 60 60)">'
            f'<circle cx="60" cy="60" r="37" fill="none" stroke="url(#ab{u})" stroke-width="4"/>'
            f'<circle cx="60" cy="60" r="30" fill="none" stroke="{a}" stroke-width="1.5" opacity="0.5"/>'
            f'<g stroke="{a}" stroke-width="2.5" stroke-linecap="round">'
            f'<line x1="60" y1="23" x2="60" y2="30"/><line x1="60" y1="90" x2="60" y2="97"/>'
            f'<line x1="23" y1="60" x2="30" y2="60"/><line x1="90" y1="60" x2="97" y2="60"/>'
            f'<line x1="34" y1="34" x2="39" y2="39"/><line x1="86" y1="34" x2="81" y2="39"/>'
            f'<line x1="34" y1="86" x2="39" y2="81"/><line x1="86" y1="86" x2="81" y2="81"/></g>'
            f'<polygon points="60,42 78,74 42,74" fill="none" stroke="{a}" stroke-width="3.5" stroke-linejoin="round"/>'
            f'<line x1="60" y1="42" x2="60" y2="82" stroke="{a}" stroke-width="3"/>'
            f'<circle cx="60" cy="60" r="4.5" fill="{a}"/></g>')


def _av_wind(u, c, a, t):
    # ZEPHYR — flowing wind streams (gusts) curling at the ends
    light = _lighten(a, 0.2)
    return (f'<g transform="rotate({t} 60 60)" fill="none" stroke-linecap="round">'
            f'<path d="M22 42 Q52 32 78 42 Q92 47 86 58 Q81 66 74 60" stroke="url(#ab{u})" stroke-width="7"/>'
            f'<path d="M22 60 Q56 50 84 60" stroke="{a}" stroke-width="6" opacity="0.9"/>'
            f'<path d="M28 78 Q54 70 74 78 Q86 82 80 92 Q76 98 70 92" stroke="{light}" '
            f'stroke-width="5" opacity="0.72"/></g>')


_APEX_VARS = {"chevron": _av_chevron, "blade": _av_blade, "hexcore": _av_hexcore,
              "spire": _av_spire, "orbit": _av_orbit, "lotus": _av_lotus,
              "tower": _av_tower, "forge": _av_forge, "aurora": _av_aurora,
              "nova": _av_nova, "echo": _av_echo, "meridian": _av_meridian,
              "seal": _av_seal, "wind": _av_wind}
_APEX_BY_ID = {
    "northwind": "chevron", "aurora-vanguard": "aurora", "zephyr-syndicate": "wind",
    "solar-drift": "orbit", "nova-collective": "nova", "echo-dynasty": "echo",
    "titan-forge": "forge", "golden-spire": "tower",
    "black-lotus": "lotus", "silent-order": "seal",
    "mirage-pact": "blade", "red-meridian": "meridian",
}
_VARIANT_ORDER = ("chevron", "blade", "hexcore", "spire", "orbit")


def _apex_mark(f: dict) -> str:
    u = _uid()
    c = f.get("color", "#444444")
    a = f.get("accent", "#eeeeee")
    fid = str(f.get("franchise_id", "x"))
    sk = sum(ord(ch) for ch in fid)
    tilt = 0  # upright — the per-franchise tilt made symmetric marks look crooked
    variant = _APEX_BY_ID.get(fid) or _VARIANT_ORDER[sk % len(_VARIANT_ORDER)]
    return _svg(_apex_ground(u, c, a) + _APEX_VARS[variant](u, c, a, tilt))


def emblem_svg(franchise: dict) -> str:
    """Distinctive esports-style emblem: a mascot crest for animal franchises, else a bold Apex mark
    (the default for any unknown/future franchise)."""
    fid = str(franchise.get("franchise_id", "x"))
    if fid in _BEAST_BY_ID:
        return _beast_crest(franchise)
    return _apex_mark(franchise)
