/* ============================================================================ *
 *  i18n — runtime localization for the World/Match viewer (RU · EN · JA).
 *
 *  Two cooperating mechanisms, both driven by ONE catalog (i18n_catalog.js +
 *  i18n_extra.js), keyed by the SOURCE phrase (Russian for World, English for the
 *  menu/Match), with {0}/{1} placeholders:
 *
 *   1. STATIC SHELL — elements carrying [data-i18n] / [data-i18n-attr]. The source
 *      phrase lives in the attribute (language-independent), so it re-applies
 *      cleanly on every language switch. Text-leaf only (never wipes child nodes).
 *
 *   2. DYNAMIC CONTENT — everything world.js/world_preview.js render is authored in
 *      Russian; a MutationObserver localizes each rendered subtree by walking its
 *      text nodes and translating from the catalog (exact match, then auto-built
 *      {0} patterns, recursively translating captured enumerable values). The
 *      original Russian is remembered per text node so switching language re-
 *      translates from the source. world.js is NEVER edited (it only writes text,
 *      never reads it back), so nothing can break — an unknown phrase simply stays
 *      Russian (graceful fallback).
 *
 *  Language persists to localStorage["worldLang"]; the <head> no-flash script sets
 *  <html lang> early so per-locale CSS applies on first paint.
 * ============================================================================ */
(function () {
  "use strict";

  var LANGS = ["ru", "en", "ja"];
  var DEFAULT = "ru";
  var CYR = /[Ѐ-ӿ]/;
  var SKIP_TAGS = { SCRIPT: 1, STYLE: 1, NOSCRIPT: 1, TEXTAREA: 1, CODE: 1 };
  var SKIP_CLASS = /(?:^|\s)(?:lang-dock|theme-dock|theme-switch)(?:\s|$)/;

  var CATALOG = window.__I18N_CATALOG__ || { ru: {}, en: {}, ja: {} };
  CATALOG.ru = CATALOG.ru || {};
  if (window.__I18N_EXTRA__) {
    LANGS.forEach(function (lg) {
      var src = window.__I18N_EXTRA__[lg];
      if (!src) return;
      CATALOG[lg] = CATALOG[lg] || {};
      for (var k in src) if (Object.prototype.hasOwnProperty.call(src, k)) CATALOG[lg][k] = src[k];
    });
  }

  var lang = DEFAULT;
  try { var saved = localStorage.getItem("worldLang"); if (LANGS.indexOf(saved) >= 0) lang = saved; }
  catch (e) { /* ignore */ }

  function getLang() { return lang; }

  /* -- placeholder fill: {0}/{1} (array or scalar) or {name} (object) ----------- */
  function fill(str, args) {
    if (args == null) return str;
    if (typeof args === "object" && !Array.isArray(args)) {
      return str.replace(/\{(\w+)\}/g, function (m, k) {
        return Object.prototype.hasOwnProperty.call(args, k) ? String(args[k]) : m;
      });
    }
    var arr = Array.isArray(args) ? args : [args];
    return str.replace(/\{(\d+)\}/g, function (m, i) { return arr[+i] != null ? String(arr[+i]) : m; });
  }

  function exact(src, lg) {
    var tbl = CATALOG[lg];
    return tbl && Object.prototype.hasOwnProperty.call(tbl, src) ? tbl[src] : null;
  }

  /* -- patterns: every catalog key containing {N} becomes a regex so interpolated
        phrases ("Сезон {0}") match rendered text ("Сезон 3"). Built once, sorted by
        literal length so the most specific pattern wins. -------------------------- */
  var PATTERNS = null;
  function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function buildPatterns() {
    PATTERNS = [];
    var seen = {};
    LANGS.forEach(function (lg) {
      var tbl = CATALOG[lg]; if (!tbl) return;
      for (var key in tbl) {
        if (!Object.prototype.hasOwnProperty.call(tbl, key) || seen[key]) continue;
        seen[key] = 1;
        if (key.indexOf("{") < 0 || !/\{\d+\}/.test(key)) continue;
        var nums = [], literal = "", re = "^", last = 0, m;
        var rx = /\{(\d+)\}/g;
        while ((m = rx.exec(key))) {
          re += escapeRe(key.slice(last, m.index)) + "([\\s\\S]+?)";
          literal += key.slice(last, m.index);
          nums.push(+m[1]); last = m.index + m[0].length;
        }
        re += escapeRe(key.slice(last)) + "$";
        literal += key.slice(last);
        if (literal.replace(/\s/g, "").length < 3) continue;   // too generic -> skip (avoids false hits)
        PATTERNS.push({ re: new RegExp(re), nums: nums, key: key, lit: literal.length });
      }
    });
    PATTERNS.sort(function (a, b) { return b.lit - a.lit; });
  }

  /** Translate a SOURCE phrase to the active language (exact, then pattern, else source). */
  function translate(src, lg) {
    lg = lg || lang;
    var e = exact(src, lg);
    if (e != null) return e;
    if (PATTERNS === null) buildPatterns();
    for (var i = 0; i < PATTERNS.length; i++) {
      var p = PATTERNS[i], mm = src.match(p.re);
      if (!mm) continue;
      var tmpl = exact(p.key, lg);
      if (tmpl == null) tmpl = p.key;
      var argo = {};
      for (var j = 0; j < p.nums.length; j++) {
        var g = mm[j + 1], gt = exact(g, lg);          // recursively translate enumerable values
        argo[p.nums[j]] = gt != null ? gt : g;
      }
      return fill(tmpl, argo);
    }
    return src;
  }

  /** Programmatic helper (kept for any caller): translate key + fill its own placeholders. */
  function t(key, args) { return fill(translate(key, lang), args); }

  /* -- static [data-i18n] / [data-i18n-attr] (text-leaf only, idempotent) -------- */
  function hasElementChild(el) {
    for (var n = el.firstChild; n; n = n.nextSibling) if (n.nodeType === 1) return true;
    return false;
  }
  function applyStatic(root) {
    var nodes = root.querySelectorAll ? root.querySelectorAll("[data-i18n]") : [];
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (hasElementChild(el)) continue;            // dynamic container filled by world.js -> don't wipe
      var want = translate(el.getAttribute("data-i18n"), lang);
      if (el.textContent !== want) el.textContent = want;
    }
    var an = root.querySelectorAll ? root.querySelectorAll("[data-i18n-attr]") : [];
    for (var j = 0; j < an.length; j++) {
      var spec = an[j].getAttribute("data-i18n-attr").split(";");
      for (var k = 0; k < spec.length; k++) {
        var idx = spec[k].indexOf(":");
        if (idx < 0) continue;
        var attr = spec[k].slice(0, idx).trim(), src = spec[k].slice(idx + 1).trim();
        var v = translate(src, lang);
        if (an[j].getAttribute(attr) !== v) an[j].setAttribute(attr, v);
      }
    }
  }

  /* -- dynamic text-node localization (world.js output) -------------------------- */
  var ORIG = (typeof WeakMap !== "undefined") ? new WeakMap() : null;
  function skipNode(parent) {
    for (var el = parent; el && el.nodeType === 1; el = el.parentNode) {
      if (SKIP_TAGS[el.nodeName]) return true;
      if (el.hasAttribute && el.hasAttribute("data-i18n")) return true;     // owned by applyStatic
      if (el.className && typeof el.className === "string" && SKIP_CLASS.test(el.className)) return true;
    }
    return false;
  }
  function localizeTextNode(node) {
    var parent = node.parentNode;
    if (!parent || skipNode(parent)) return;
    var orig = ORIG ? ORIG.get(node) : node.__i18n;
    if (orig == null) { orig = node.nodeValue; if (ORIG) ORIG.set(node, orig); else node.__i18n = orig; }
    if (!CYR.test(orig)) return;                    // pure data/Latin/numbers -> leave
    var trimmed = orig.replace(/^\s+|\s+$/g, "");
    if (!trimmed) return;
    var lead = orig.slice(0, orig.length - orig.replace(/^\s+/, "").length);
    var trail = orig.slice(orig.replace(/\s+$/, "").length);
    var out = lead + translate(trimmed, lang) + trail;
    if (node.nodeValue !== out) node.nodeValue = out;
  }
  function walkText(root) {
    if (root.nodeType === 3) { localizeTextNode(root); return; }
    if (root.nodeType !== 1 && root.nodeType !== 9 && root.nodeType !== 11) return;
    var tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    var batch = [], n;
    while ((n = tw.nextNode())) batch.push(n);
    for (var i = 0; i < batch.length; i++) localizeTextNode(batch[i]);
  }

  function localizeTree(root) {
    if (!root) return;
    if (root.nodeType === 1 || root.nodeType === 9 || root.nodeType === 11) applyStatic(root);
    walkText(root);
  }

  /* -- observe dynamic renders -------------------------------------------------- */
  var observer = null;
  function startObserver() {
    if (observer || typeof MutationObserver === "undefined" || !document.body) return;
    observer = new MutationObserver(function (muts) {
      observer.disconnect();                         // our own writes won't re-trigger
      for (var i = 0; i < muts.length; i++) {
        var added = muts[i].addedNodes;
        for (var j = 0; j < added.length; j++) {
          var nd = added[j];
          if (nd.nodeType === 1) localizeTree(nd);
          else if (nd.nodeType === 3) localizeTextNode(nd);
        }
      }
      observer.observe(document.body, { childList: true, subtree: true });
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  function setLang(next) {
    if (LANGS.indexOf(next) < 0 || next === lang) return;
    lang = next;
    try { localStorage.setItem("worldLang", next); } catch (e) { /* ignore */ }
    document.documentElement.lang = next;
    if (observer) observer.disconnect();
    localizeTree(document.body);                      // re-applies statics + re-translates from originals
    if (observer) observer.observe(document.body, { childList: true, subtree: true });
    markSwitch();
    try { window.dispatchEvent(new CustomEvent("i18n:changed", { detail: { lang: next } })); }
    catch (e) { /* ignore */ }
  }

  /* -- floating language switcher (self-styled; works on every page) ------------- */
  function injectStyle() {
    if (document.getElementById("i18n-style")) return;
    var css =
      ".lang-dock{position:fixed;z-index:9;right:18px;bottom:64px;display:flex;gap:4px;padding:5px;" +
      "border-radius:var(--r-pill,999px);background:color-mix(in srgb,var(--panel,#141a2a) 82%,transparent);" +
      "border:1px solid var(--border-2,#2b3650);box-shadow:var(--elev-float,0 10px 30px -12px rgba(0,0,0,.55));" +
      "backdrop-filter:blur(10px);}" +
      ".lang-dock button{min-width:34px;height:28px;padding:0 8px;border-radius:var(--r-pill,999px);" +
      "border:1px solid var(--border-2,#2b3650);background:var(--panel-2,#10151f);color:var(--text-2,#a3b3c6);" +
      "font:inherit;font-size:12px;font-weight:800;line-height:1;cursor:pointer;" +
      "transition:background .18s ease,color .18s ease,box-shadow .18s ease,transform .12s ease;}" +
      ".lang-dock button:hover{transform:translateY(-1px);color:var(--text-strong,#eaf1fb);}" +
      ".lang-dock button.is-active{background:linear-gradient(135deg,var(--accent,#6366f1),var(--accent-3,#a855f7));" +
      "border-color:transparent;color:#fff;box-shadow:0 0 12px -3px var(--glow,rgba(99,102,241,.42));}" +
      "body.embed-replay .lang-dock{display:none;}";
    var st = document.createElement("style");
    st.id = "i18n-style"; st.textContent = css;
    document.head.appendChild(st);
  }
  function buildSwitch() {
    if (!document.body || document.querySelector(".lang-dock")) return;
    injectStyle();
    var defs = [["ru", "RU"], ["en", "EN"], ["ja", "日本語"]];
    var dock = document.createElement("div");
    dock.className = "lang-dock";
    dock.setAttribute("role", "group");
    dock.setAttribute("aria-label", "Language / 言語 / Язык");
    dock.innerHTML = defs.map(function (d) {
      return '<button type="button" data-lang-pick="' + d[0] + '" lang="' + d[0] + '" title="' + d[1] + '">' + d[1] + "</button>";
    }).join("");
    dock.addEventListener("click", function (e) {
      var b = e.target.closest("[data-lang-pick]");
      if (b) setLang(b.getAttribute("data-lang-pick"));
    });
    document.body.appendChild(dock);
    markSwitch();
  }
  function markSwitch() {
    var btns = document.querySelectorAll(".lang-dock [data-lang-pick]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].classList.toggle("is-active", btns[i].getAttribute("data-lang-pick") === lang);
    }
  }

  window.t = t;
  window.i18n = { t: t, translate: translate, getLang: getLang, setLang: setLang, localizeTree: localizeTree, langs: LANGS };

  /* Tell the server which language to GENERATE commentator/manager text in. We tag same-origin
     /api/ calls with an X-World-Lang header — so world.js's fetches are never touched. */
  if (window.fetch && !window.__i18nFetchPatched__) {
    window.__i18nFetchPatched__ = true;
    var _origFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      try {
        if (typeof input === "string" && input.indexOf("://") < 0 && input.indexOf("/api/") >= 0) {
          init = init || {};
          var h = new Headers(init.headers || {});
          h.set("X-World-Lang", lang);
          init.headers = h;
        }
      } catch (e) { /* fall through to the original fetch */ }
      return _origFetch(input, init);
    };
  }

  document.documentElement.lang = lang;
  function init() { localizeTree(document.body); buildSwitch(); startObserver(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
