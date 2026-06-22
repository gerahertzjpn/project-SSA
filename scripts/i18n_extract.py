"""Extract user-facing Russian UI phrases from the viewer JS via an esprima AST (reliable: handles
nested templates / comments / regex that the old regex scanner mis-parsed). Output:
_i18n_strings.json = {"phrases":[{"key":...}]}. Keys normalize ${...}->{0} and split on HTML tags so
they match the text nodes world.js renders. Run, then translate the gaps, then i18n_build_catalog.py.
"""
import esprima, io, json, re, collections

CYR = re.compile(r"[А-Яа-яЁё]")
JS_FILES = ["viewer_pixi/world.js", "viewer_pixi/world_preview.js"]

phrases = collections.OrderedDict()


def add(key):
    key = re.sub(r"\s+", " ", key).strip()
    if key and CYR.search(key):
        phrases.setdefault(key, True)


def split_tags(text):
    """`text` has \\x00 markers for interpolations; split on HTML tags; number placeholders PER
    segment (each inter-tag segment becomes its own rendered text node)."""
    for seg in re.split(r"<[^>]*>", text):
        if "\x00" not in seg and not CYR.search(seg):
            continue
        cnt = [0]

        def repl(_m):
            v = "{%d}" % cnt[0]
            cnt[0] += 1
            return v

        seg = re.sub("\x00", repl, seg)
        if CYR.search(seg):
            add(seg)


def cooked_of(q):
    val = getattr(q, "value", None)
    if val is None:
        return ""
    c = getattr(val, "cooked", None)
    if c is None and isinstance(val, dict):
        c = val.get("cooked")
    return c or ""


def walk(node):
    if node is None:
        return
    if isinstance(node, list):
        for x in node:
            walk(x)
        return
    t = getattr(node, "type", None)
    if t is None:
        return
    if t == "Literal":
        v = getattr(node, "value", None)
        if isinstance(v, str) and CYR.search(v):
            if "<" in v:
                split_tags(v.replace("\x00", ""))
            else:
                add(v)
    elif t == "TemplateLiteral":
        quasis = getattr(node, "quasis", []) or []
        exprs = getattr(node, "expressions", []) or []
        parts = []
        for i, q in enumerate(quasis):
            parts.append(cooked_of(q))
            if i < len(exprs):
                parts.append("\x00")
        text = "".join(parts)
        if CYR.search(text):
            split_tags(text)
        for e in exprs:
            walk(e)
        return
    for k, v in getattr(node, "__dict__", {}).items():
        if k == "type":
            continue
        if isinstance(v, list):
            walk(v)
        elif getattr(v, "type", None) is not None:
            walk(v)


def preprocess(code):
    # esprima 4 can't parse ?? / ?. — neutralize for parsing only (string literals are untouched).
    code = code.replace("??", " || ")
    code = re.sub(r"\?\.", ".", code)
    return code


for f in JS_FILES:
    code = preprocess(io.open(f, encoding="utf-8").read())
    tree = esprima.parseScript(code)
    walk(tree)

items = [{"key": k} for k in phrases]
io.open("_i18n_strings.json", "w", encoding="utf-8").write(
    json.dumps({"count": len(items), "phrases": items}, ensure_ascii=False, indent=1))
print("extracted unique phrases:", len(items))
