"use strict";

const TEAM_COUNT = 22;
const FACTION = {
  humans: { label: "Humans", color: "#8ec5ff" },
  infected: { label: "Infected", color: "#9ff5b0" },
  robots: { label: "Robots", color: "#ffd58a" },
};
const OP_LABEL = { skill_up: "Скилл", meta_adapt: "Против меты", style_spec: "Стиль",
  obedience_tune: "Послушание", teamplay: "Командная", create_player: "Новичок" };

const state = {
  cards: [],
  selected: new Set(),
  locked: true,
  lockNote: "",
  worldId: null,
  world: null,        // cached overview (season/counts/history/name)
  busy: false,
  tab: "leagues",
  leagueKey: "upper",
  ratingRole: "all",
  ratingFaction: "all",
  players: [],        // cached ranked players for the ratings tab
};

const els = {};
function el(id) {
  // Cache by id, but RE-QUERY if the cached node was detached (e.g. an inline panel like the
  // off-season setup re-renders #osRun / #osReserves via innerHTML — a stale ref would mean the
  // click listener binds to a dead node and the visible button does nothing).
  let e = els[id];
  if (!e || !e.isConnected) e = els[id] = document.getElementById(id);
  return e;
}
function show(id, on) { el(id).classList.toggle("is-hidden", !on); }
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function fetchJson(url, opts) {
  // PREVIEW MODE: a hardcoded data source (world_preview.js) serves every endpoint, so the exact
  // same renderers run with zero server / matches / requests. Real World is untouched.
  if (window.__PREVIEW__) return window.__PREVIEW__(url, opts);
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}
const api = (path) => fetchJson(`/api/world/${state.worldId}${path}`);
const apiPost = (path, body) => fetchJson(`/api/world/${state.worldId}${path}`,
  { method: "POST", headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined });

function factionChip(f) {
  const meta = FACTION[f] || { label: f || "—", color: "#9aa7b4" };
  return `<span class="fac-chip" style="--fac:${meta.color}">${esc(meta.label)}</span>`;
}
function deltaBadge(d) {
  const v = Number(d || 0);
  if (v > 0) return `<span class="delta up">▲ ${v.toFixed(0)}</span>`;
  if (v < 0) return `<span class="delta down">▼ ${Math.abs(v).toFixed(0)}</span>`;
  return `<span class="delta flat">—</span>`;
}
function sparkline(values, w = 88, h = 24, opts = {}) {
  let vals = (values || []).filter((v) => typeof v === "number");
  if (opts.window && vals.length > opts.window) vals = vals.slice(-opts.window);  // last N seasons only
  if (vals.length < 2) return `<span class="spark-empty">—</span>`;
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const pts = vals.map((v, i) =>
    `${(i / (vals.length - 1) * w).toFixed(1)},${(h - (v - min) / span * h).toFixed(1)}`).join(" ");
  // colour: by default the LATEST change; with colorVsFirst, the whole shown window (last vs window-start)
  const rising = opts.colorVsFirst
    ? vals[vals.length - 1] >= vals[0]
    : vals[vals.length - 1] >= vals[vals.length - 2];
  return `<svg class="spark ${rising ? "up" : "down"}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
    `<polyline points="${pts}" fill="none" stroke-width="1.8" /></svg>`;
}
const teamLabelFA = (p) => p.team || (p.role === "streamer" ? "Стример" : "Свободный агент");
let _emblemSeq = 0;
function uniqSvg(svg) {
  // The same emblem SVG is inlined many times across the (always-present) tab panels. Its internal
  // clipPath/gradient ids would then collide — duplicate ids make url(#id) resolve to the FIRST
  // match (often inside a hidden panel), so e.g. the "split" clip fails and the dark half spills.
  // Suffix every id + url(#…) reference with a unique counter per render so each instance is self-contained.
  if (!svg || svg.indexOf("id=") === -1) return svg || "";
  const n = ++_emblemSeq;
  return svg.replace(/id="([^"]+)"/g, (m, id) => `id="${id}-${n}"`)
            .replace(/url\(#([^)]+)\)/g, (m, id) => `url(#${id}-${n})`);
}
function emblem(teamId, cls) {
  return `<span class="tlogo ${cls || ""}">${uniqSvg(state.emblems && state.emblems[teamId])}</span>`;
}
// Primary brand colour for a club (for the broadcast overlay); falls back to the theme accent.
function teamColor(teamId) {
  const c = state.colors && state.colors[teamId];
  return (c && (c.color || c.accent)) || "var(--accent)";
}

// --------------------------------------------------------------------------- //
// New-world team picker
// --------------------------------------------------------------------------- //
function personaShort(text) {
  if (!text) return "";
  const t = text.replace(/^Ты\s+/, "");
  return t.length > 120 ? t.slice(0, 118) + "…" : t;
}
function strategyLabel(plan) {
  if (!plan || !plan.tempo) return "—";
  return `${plan.tempo} · ${plan.win_path}`;
}
function renderCards() {
  const grid = el("teamGrid");
  grid.innerHTML = "";
  const atCap = state.selected.size >= TEAM_COUNT;
  for (const c of state.cards) {
    const picked = state.selected.has(c.franchise_id);
    const card = document.createElement("label");
    card.className = "team-card" + (picked ? " picked" : "");
    card.style.setProperty("--team-color", c.color);
    card.style.setProperty("--team-accent", c.accent);
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "team-check";
    cb.checked = picked;
    cb.disabled = state.locked || (!picked && atCap);
    cb.addEventListener("change", () => toggle(c.franchise_id, cb.checked));
    const emblem = document.createElement("span");
    emblem.className = "team-emblem";
    emblem.innerHTML = uniqSvg(c.emblem);
    const body = document.createElement("div");
    body.className = "team-body";
    body.innerHTML =
      `<div class="team-name">${esc(c.name)} <span class="team-abbrev">${esc(c.abbrev)}</span></div>` +
      `<div class="team-manager">👔 ${esc(c.manager || "—")}</div>` +
      `<div class="team-strategy">Стиль: ${esc(strategyLabel(c.default_plan))}</div>` +
      `<div class="team-persona">${esc(personaShort(c.persona))}</div>`;
    card.append(cb, emblem, body);
    grid.appendChild(card);
  }
}
function toggle(id, on) {
  if (state.locked) return;
  if (on) { if (state.selected.size >= TEAM_COUNT) return; state.selected.add(id); }
  else state.selected.delete(id);
  refreshPicker();
}
function refreshPicker() {
  const n = state.selected.size;
  el("selCount").textContent = `${n} / ${TEAM_COUNT}`;
  el("selCount").classList.toggle("full", n === TEAM_COUNT);
  el("startBtn").disabled = state.locked || state.busy || n !== TEAM_COUNT;
  renderCards();
}

// --------------------------------------------------------------------------- //
// Background-job polling (create / advance / offseason)
// --------------------------------------------------------------------------- //
async function runJob(startUrl, opts, title) {
  state.busy = true;
  show("jobPanel", true);
  el("jobTitle").textContent = title;
  el("jobPhase").textContent = "";
  const t0 = Date.now();
  const tick = setInterval(() => {
    el("jobTimer").textContent = `${Math.round((Date.now() - t0) / 1000)} с`;
  }, 1000);
  try {
    const { job_id } = await fetchJson(startUrl, opts);
    let job;
    do {
      await sleep(2000);
      job = await fetchJson(`/api/world/job/${job_id}`);
      el("jobPhase").textContent = job.phase || "";
    } while (job.status === "running");
    if (job.status === "error") throw new Error(job.error || "job failed");
    return job;
  } finally {
    clearInterval(tick);
    state.busy = false;
    show("jobPanel", false);
  }
}

// --------------------------------------------------------------------------- //
// Tabs
// --------------------------------------------------------------------------- //
function switchTab(name) {
  state.tab = name;
  document.querySelectorAll("#tabbar .tab").forEach((b) =>
    b.classList.toggle("is-active", b.dataset.tab === name));
  ["leagues", "live", "matrix", "ratings", "schedule", "teams", "draft", "transfers", "broadcast", "managers", "history"].forEach((t) =>
    show(`panel-${t}`, t === name));
  if (typeof moveTabInk === "function") requestAnimationFrame(moveTabInk);   // slide the magic line
  loadTab(name);
}
async function loadTab(name) {
  const panel = el(`panel-${name}`);
  if (name === "history") return renderHistory(state.world ? state.world.history : []);
  if (name === "live") return renderLive(state.world ? state.world.live : null);
  if (name === "transfers") return renderOffseasonTx(state.world ? state.world.last_offseason : null);
  if (name === "broadcast") return renderBroadcast(state.world ? state.world.commentary : []);
  if (name === "managers") return renderManagers(state.world ? state.world.manager_comments : []);
  panel.innerHTML = `<div class="muted loading">Загрузка…</div>`;
  try {
    if (name === "leagues") { const d = await api("/leagues"); state._prizes = d.prizes; renderLeagues(d.leagues); }
    else if (name === "matrix") renderMatrix(await api("/matrix"));
    else if (name === "ratings") { state.players = (await api("/players")).players; renderRatings(); }
    else if (name === "schedule") renderSchedule(await api("/schedule"));
    else if (name === "draft") renderDraft(await api("/draft"));
    else if (name === "teams") {
      renderTeams((await api("/teams")).ranking);
      if (state.pendingTeam) { const tid = state.pendingTeam; state.pendingTeam = null; loadTeamPage(tid); }
    }
  } catch (e) {
    panel.innerHTML = `<div class="result-error">Не удалось загрузить: ${esc(e.message)}</div>`;
  }
}

// --------------------------------------------------------------------------- //
// Tab: Лиги
// --------------------------------------------------------------------------- //
const LEAGUE_TABS = [
  { key: "upper", label: "Верхняя лига", sub: "10 команд" },
  { key: "lower_A", label: "Нижняя · группа A", sub: "6 команд" },
  { key: "lower_Z", label: "Нижняя · группа Z", sub: "6 команд" },
  { key: "playoff", label: "🏆 Плей-офф", sub: "обе сетки" },
];
function renderLeagues(leagues) {
  state._leagues = leagues;
  if (!LEAGUE_TABS.some((t) => t.key === state.leagueKey)) state.leagueKey = "upper";
  const seg = `<div class="seg">${LEAGUE_TABS.map((t) =>
    `<button class="seg-btn${t.key === state.leagueKey ? " is-active" : ""}" data-lg="${t.key}">` +
    `${esc(t.label)}<small>${esc(t.sub)}</small></button>`).join("")}</div>`;
  const wireSeg = () => el("panel-leagues").querySelectorAll(".seg-btn").forEach((b) =>
    b.addEventListener("click", () => { state.leagueKey = b.dataset.lg; renderLeagues(state._leagues); }));
  if (state.leagueKey === "playoff") {
    el("panel-leagues").innerHTML = seg + `<div id="lgPlayoff"><div class="muted loading">Загрузка сетки плей-офф…</div></div>`;
    wireSeg();
    api("/schedule").then((d) => { const el2 = document.getElementById("lgPlayoff");
      if (el2) el2.innerHTML = renderBracketGraph(d.bracket || {}); })
      .catch((e) => { const el2 = document.getElementById("lgPlayoff");
        if (el2) el2.innerHTML = `<div class="result-error">${esc(e.message)}</div>`; });
    return;
  }
  const dlt = (d) => `${d > 0 ? "+" : ""}${d}`;
  const rows = (leagues[state.leagueKey] || []).map((t) =>
    `<tr class="ltrow" data-team="${esc(t.team_id)}">` +
      `<td class="lt-place">${t.place}</td>` +
      `<td class="lt-ab">${emblem(t.team_id)}<span>${esc(t.abbrev)}</span></td>` +
      `<td class="lt-nm">${esc(t.name)}</td>` +
      `<td class="lt-wl">${t.series_wins}<span class="muted">–</span>${t.series_losses}</td>` +
      `<td class="lt-dl ${t.map_diff >= 0 ? "up" : "down"}">${dlt(t.map_diff)}</td>` +
      `<td class="lt-pts">${t.points}</td>` +
      `<td class="lt-pw">${t.power}</td>` +
      `<td class="lt-roster">${(t.roster || []).map(esc).join(" · ")}</td>` +
    `</tr>`).join("");
  el("panel-leagues").innerHTML = seg + prizeScheduleBlock(state._prizes, state.leagueKey) +
    `<table class="ltable"><thead><tr>` +
      `<th>#</th><th>Аб.</th><th>Команда</th><th title="Победы–поражения в сериях">В–П</th>` +
      `<th title="Дельта выигранных карт">Δ</th><th title="Очки за победы в сериях">Очки</th>` +
      `<th>Power</th><th>Состав</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
  wireSeg();
  el("panel-leagues").querySelectorAll(".ltrow").forEach((r) =>
    r.addEventListener("click", () => openTeam(r.dataset.team)));
}

// Prize schedule for the current league view (#3): clearly shows what each PLACE pays + the champion
// bonus, so the economy is trackable. Lower = per-group place (A and Z identical).
function prizeScheduleBlock(prizes, lgKey) {
  if (!prizes) return "";
  const lg = lgKey === "upper" ? prizes.upper : prizes.lower;
  if (!lg || lgKey === "playoff") return "";
  const title = lgKey === "upper"
    ? "за итоговое место в лиге" : "за место В ГРУППЕ (A и Z — одинаково)";
  const cells = (lg.places || []).map((amt, i) =>
    `<span class="lg-prz-cell"><span class="lg-prz-pl">${i + 1}</span><span class="money">${amt}</span></span>`).join("");
  return `<details class="lg-prizes" open><summary>💰 Призовые ${esc(title)}</summary>` +
    `<div class="lg-prz-row">${cells}` +
    `<span class="lg-prz-cell champ"><span class="lg-prz-pl">🏆</span><span class="money">+${lg.champion_bonus}</span></span></div>` +
    `<p class="muted lg-prz-note">Клуб получает деньги за итоговое место${lgKey !== "upper" ? " в своей группе" : ""}; ` +
    `🏆 — отдельный бонус победителю гранд-финала. Заработанное командой видно в «Истории» и «Межсезонье · 💰».</p></details>`;
}

// Playoff bracket GRAPH (a real tree with connectors, not a table) — used by the Лиги→Плей-офф
// segment and the Расписание tab. `bracket` = {upper:[{stage,label,matches}], lower:[...]}.
function bgNode(m) {
  const tbd = (ab) => !ab || ab === "TBD";
  const winH = m.played && m.winner && m.winner === m.home, winA = m.played && m.winner && m.winner === m.away;
  const side = (id, ab, sc, win) =>
    `<div class="bg-side${win ? " win" : ""}${tbd(ab) ? " tbd" : ""}">${id && !tbd(ab) ? emblem(id) : `<span class="bg-dot"></span>`}` +
    `<span class="bg-tm">${esc(tbd(ab) ? "TBD" : ab)}</span><span class="bg-sc">${m.played ? sc : "·"}</span></div>`;
  return `<div class="bg-node${m.played ? " played" : ""}" data-key="${esc(m.key || "")}" data-from="${esc((m.from || []).join(","))}" title="Bo${m.best_of || 5}">` +
    side(m.home, m.home_abbrev, m.maps_home, winH) +
    side(m.away, m.away_abbrev, m.maps_away, winA) + `</div>`;
}
// ONE connected double-elim graph per league (liquipedia style): a CSS grid where columns are by
// bracket DEPTH (longest path from a seed), the WINNERS bracket sits on the top row, the LOSERS
// bracket on the bottom row, and the GRAND FINAL spans both rows in the rightmost column. Everything
// lives in a SINGLE .bgraph (one SVG scope), so drawBracketConnectors joins winners→final,
// losers→final AND the loser-drops (ub→lb_final) into one tree.
function bgLeague(cols) {
  const all = (cols || []).flatMap((c) => c.matches || []).filter((m) => m && m.key);
  if (!all.length) return `<div class="muted">Сетка ещё не началась.</div>`;
  const byKey = {};
  all.forEach((m) => (byKey[m.key] = m));
  const depth = {};
  const dof = (k) => {                       // forward depth = 1 + max feeder depth (seeds = 0)
    if (k in depth) return depth[k];
    depth[k] = 0;                            // cycle guard
    const m = byKey[k];
    const fed = ((m && m.from) || []).filter((f) => byKey[f]);
    depth[k] = fed.length ? 1 + Math.max(...fed.map(dof)) : 0;
    return depth[k];
  };
  all.forEach((m) => dof(m.key));
  const maxD = Math.max(0, ...all.map((m) => depth[m.key]));
  const isGf = (m) => m.stage === "grand_final";
  const isWin = (m) => m.stage === "ub";
  const cell = {};                           // "row:col" -> [matches]
  let gf = null;
  all.forEach((m) => {
    if (isGf(m)) { gf = m; return; }
    const key = `${isWin(m) ? 1 : 2}:${depth[m.key] + 1}`;
    (cell[key] = cell[key] || []).push(m);
  });
  let items = "";   // (row labels removed)
  Object.keys(cell).forEach((rc) => {
    const [row, col] = rc.split(":");
    items += `<div class="de2-cell" style="grid-row:${row};grid-column:${col}">${cell[rc].map(bgNode).join("")}</div>`;
  });
  if (gf) items += `<div class="de2-cell de2-gf" style="grid-row:1 / 3;grid-column:${maxD + 1}">${bgNode(gf)}</div>`;
  return `<div class="bgraph de2" style="grid-template-columns:repeat(${maxD + 1}, minmax(176px, 1fr))">` +
    `<svg class="bg-svg" aria-hidden="true"></svg>${items}</div>`;
}
function renderBracketGraph(bracket) {
  const up = bracket.upper || [], lo = bracket.lower || [];
  if (!up.length && !lo.length) return `<div class="muted" style="margin-top:14px">Плей-офф ещё не начался — он откроется, когда сыграется регулярка.</div>`;
  // connectors are drawn after the HTML is in the DOM (see drawBracketConnectors)
  setTimeout(drawBracketConnectors, 0);
  return `<div class="bg-block"><h4 class="bg-h">🏆 Верхняя лига</h4>${bgLeague(up)}</div>` +
    `<div class="bg-block"><h4 class="bg-h">Нижняя лига <span class="muted">(2 повышаются)</span></h4>${bgLeague(lo)}</div>`;
}
// Draw L-shaped connector lines from each feeder match to the one it flows into (read from
// data-key / data-from), measured from the rendered node positions — a real bracket tree.
function drawBracketConnectors() {
  document.querySelectorAll(".bgraph").forEach((g) => {
    const svg = g.querySelector(".bg-svg"); if (!svg) return;
    const gr = g.getBoundingClientRect();
    const ox = g.scrollLeft, oy = g.scrollTop;   // SVG sits in the scrolled content -> add scroll offset
    const nodes = {};
    g.querySelectorAll(".bg-node").forEach((n) => { if (n.dataset.key) nodes[n.dataset.key] = n; });
    let paths = "";
    g.querySelectorAll(".bg-node").forEach((n) => {
      const tr = n.getBoundingClientRect();
      const tx = tr.left - gr.left + ox, ty = tr.top - gr.top + oy + tr.height / 2;
      (n.dataset.from || "").split(",").filter(Boolean).forEach((fk) => {
        const f = nodes[fk]; if (!f) return;
        const fr = f.getBoundingClientRect();
        const fx = fr.right - gr.left + ox, fy = fr.top - gr.top + oy + fr.height / 2;
        // turn the vertical leg DOWN/UP just before the target (in its left gap), so a long
        // winners→final line runs along its own row and never cuts through the losers' nodes.
        const mx = Math.max(fx + 12, tx - 18);
        paths += `<path d="M ${fx} ${fy} H ${mx} V ${ty} H ${tx}" />`;
      });
    });
    svg.setAttribute("width", g.scrollWidth); svg.setAttribute("height", g.scrollHeight);
    svg.innerHTML = paths;
  });
}
window.addEventListener("resize", () => { clearTimeout(window._bgRz); window._bgRz = setTimeout(drawBracketConnectors, 120); });

// --------------------------------------------------------------------------- //
// Tab: Очные — H2H matrix (who-beat-whom + score) + current positions
// --------------------------------------------------------------------------- //
function renderMatrix(data) {
  const groups = (data && data.groups) || [];
  const dlt = (d) => `${d > 0 ? "+" : ""}${d}`;
  const anyPlayed = groups.some((g) => (g.cells || []).some((row) => row.some((c) => c)));
  const block = (g) => {
    const ids = g.teams.map((t) => t.team_id);
    const cols = g.teams.map((t) =>
      `<th class="mx-col" title="${esc(t.name)}">${emblem(t.team_id, "sm")}</th>`).join("");
    const body = g.teams.map((t, i) => {
      const cells = (g.cells[i] || []).map((c, j) => {
        if (i === j) return `<td class="mx-self">—</td>`;
        if (!c) return `<td class="mx-na" title="${esc(g.teams[j].name)}">·</td>`;
        return `<td class="mx-cell ${c.won ? "w" : "l"}" title="${esc(t.abbrev)} ${c.for}:${c.against} ${esc(g.teams[j].abbrev)}">${c.for}:${c.against}</td>`;
      }).join("");
      return `<tr class="mx-tr" data-team="${esc(t.team_id)}">` +
        `<th class="mx-row"><div class="mx-rowin"><span class="mx-pl">${t.place}</span>` +
          `<span class="mx-tm2">${emblem(t.team_id)}<span class="mx-ab">${esc(t.abbrev)}</span></span></div></th>${cells}` +
        `<td class="mx-pts"><b>${t.points}</b></td>` +
        `<td class="mx-wl">${t.series_wins}–${t.series_losses}</td>` +
        `<td class="mx-dl ${t.map_diff >= 0 ? "up" : "down"}">${dlt(t.map_diff)}</td></tr>`;
    }).join("");
    return `<div class="mx-group"><h3>${esc(g.label)}</h3><div class="mx-wrap">` +
      `<table class="mx-grid"><thead><tr><th class="mx-corner">#</th>${cols}` +
      `<th class="mx-h">Очки</th><th class="mx-h">В–П</th><th class="mx-h">Δ</th></tr></thead>` +
      `<tbody>${body}</tbody></table></div></div>`;
  };
  el("panel-matrix").innerHTML =
    `<div class="mx-cap">Сезон ${data.season} · кто кого обыграл и с каким счётом. ` +
      `Позиция в лиге: очки → дельта карт → жребий.</div>` +
    (anyPlayed ? "" : `<div class="muted" style="margin:8px 0">Сезон ещё не сыгран — сыграй сезон, чтобы увидеть очные встречи.</div>`) +
    groups.map(block).join("");
  el("panel-matrix").querySelectorAll(".mx-tr").forEach((r) =>
    r.querySelector(".mx-row").addEventListener("click", () => openTeam(r.dataset.team)));
}

// --------------------------------------------------------------------------- //
// Tab: Рейтинг (all players)
// --------------------------------------------------------------------------- //
function renderRatings() {
  const roles = [["all", "Все"], ["pro", "Про"], ["streamer", "Стримеры"]];
  const facs = [["all", "Все фракции"], ["humans", "Humans"], ["infected", "Infected"], ["robots", "Robots"]];
  let list = state.players.slice();
  if (state.ratingRole !== "all") list = list.filter((p) => p.role === state.ratingRole);
  if (state.ratingFaction !== "all") list = list.filter((p) => p.faction === state.ratingFaction);
  const rows = list.map((p) =>
    `<tr class="rtrow" data-pid="${esc(p.player_id)}">` +
      `<td class="rt-rk">${p.rank}</td>` +
      `<td>${factionChip(p.faction)}</td>` +
      `<td class="rt-nm">${esc(p.nickname)}</td>` +
      `<td class="rt-ar">${esc(p.archetype || "")}</td>` +
      `<td class="rt-tm">${esc(teamLabelFA(p))}</td>` +
      `<td class="rt-rr">${p.rating}</td>` +
      `<td class="rt-dl">${deltaBadge(p.delta)}</td>` +
      `<td class="rt-sp">${sparkline(p.rating_history)}</td>` +
    `</tr>`).join("");
  el("panel-ratings").innerHTML =
    `<div class="filters">` +
      `<div class="seg">${roles.map(([v, l]) =>
        `<button class="seg-btn${state.ratingRole === v ? " is-active" : ""}" data-role="${v}">${l}</button>`).join("")}</div>` +
      `<div class="seg">${facs.map(([v, l]) =>
        `<button class="seg-btn${state.ratingFaction === v ? " is-active" : ""}" data-fac="${v}">${l}</button>`).join("")}</div>` +
      `<span class="muted">${list.length} игроков</span>` +
    `</div>` +
    `<table class="rtable"><thead><tr>` +
      `<th>#</th><th>Фр.</th><th>Игрок</th><th>Архетип</th><th>Команда</th><th>Рейтинг</th><th>Δ сезон</th><th>Динамика</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
  el("panel-ratings").querySelectorAll("[data-role]").forEach((b) =>
    b.addEventListener("click", () => { state.ratingRole = b.dataset.role; renderRatings(); }));
  el("panel-ratings").querySelectorAll("[data-fac]").forEach((b) =>
    b.addEventListener("click", () => { state.ratingFaction = b.dataset.fac; renderRatings(); }));
  el("panel-ratings").querySelectorAll(".rtrow").forEach((r) =>
    r.addEventListener("click", () => openPlayer(r.dataset.pid)));
}
function togglePlayerDetail(row) {
  const next = row.nextElementSibling;
  if (next && next.classList.contains("rt-detail")) { next.remove(); return; }
  const p = state.players.find((x) => x.player_id === row.dataset.pid);
  if (!p) return;
  const pos = Object.entries(p.position_stats || {}).map(([k, v]) =>
    `${k}: ${v.winrate != null ? Math.round(v.winrate * 100) + "%" : "—"} (${v.games}и)`).join(" · ") || "нет данных";
  const tr = document.createElement("tr");
  tr.className = "rt-detail";
  tr.innerHTML = `<td colspan="8"><div class="pdetail">` +
    `<div><b>${esc(p.nickname)}</b> ${factionChip(p.faction)} · ${esc(p.archetype || "")} · ${esc(p.role)}</div>` +
    `<div class="pd-grid">` +
      `<span>Рейтинг: <b>${p.rating}</b> ${deltaBadge(p.delta)}</span>` +
      `<span>Команда: ${esc(teamLabelFA(p))}</span>` +
      `<span>Игр: ${p.games}</span>` +
      `<span>Поколение: ${p.generation || 0}</span>` +
      `<span>Нейронка: <code>${esc(p.net_id || "—")}</code></span>` +
      `<span>Позиции: ${esc(pos)}</span>` +
    `</div>` +
    `<div class="pd-spark">Динамика рейтинга ${sparkline(p.rating_history, 200, 36)}</div>` +
    (p.trophies && p.trophies.length ? `<div>🏆 ${p.trophies.map((t) => `S${t.season} ${esc(t.title)}`).join(", ")}</div>` : "") +
    ((p.soloq_games || []).length ? `<div class="pd-soloq"><b>История SoloQ</b>` +
      p.soloq_games.map((g) =>
        `<div class="sq-game ${g.won ? "w" : "l"}"><span class="sq-res">${g.won ? "W" : "L"}</span>` +
        `<span class="sq-d ${g.delta >= 0 ? "up" : "down"}">${g.delta >= 0 ? "+" : ""}${Math.round(g.delta)}</span>` +
        `<span class="sq-with">с: ${esc((g.teammates || []).join(", "))}</span>` +
        `<span class="sq-vs">против: ${esc((g.opponents || []).join(", "))}</span></div>`).join("") +
    `</div>` : "") +
  `</div></td>`;
  row.after(tr);
}

// --------------------------------------------------------------------------- //
// Player personal page (modal) — full detailed history
// --------------------------------------------------------------------------- //
async function openPlayer(pid) {
  show("playerModal", true);
  el("plBody").innerHTML = `<div class="muted loading">Загрузка…</div>`;
  try { renderPlayerPage((await api(`/player/${pid}`)).player); }
  catch (e) { el("plBody").innerHTML = `<div class="result-error">Ошибка: ${esc(e.message)}</div>`; }
}
// Radar ("spider") chart of a player's 6 derived card attributes (0-100, all derived under the
// SAME absolute-scale gauntlet, so every card is comparable).
const ATTR_ORDER = ["attack", "defense", "economy", "tempo", "consistency", "potential"];
const ATTR_LABELS = { attack: "Атака", defense: "Защита", economy: "Эконом", tempo: "Темп",
  consistency: "Стабильн.", potential: "Потенциал" };
function radarChart(attrs, size) {
  attrs = attrs || {};
  const keys = ATTR_ORDER.filter((k) => k in attrs);
  if (keys.length < 3) return "";
  const S = size || 220, c = S / 2, R = Math.max(8, S / 2 - 34), n = keys.length;
  const ang = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const pt = (i, r) => [c + r * Math.cos(ang(i)), c + r * Math.sin(ang(i))];
  const poly = (r, mapr) => keys.map((k, i) => pt(i, mapr ? r * Math.max(0, Math.min(100, attrs[k])) / 100 : r)
    .map((x) => x.toFixed(1)).join(",")).join(" ");
  let grid = [0.25, 0.5, 0.75, 1].map((f) => `<polygon points="${poly(R * f)}" class="rad-grid"/>`).join("");
  let axes = "", labels = "";
  keys.forEach((k, i) => {
    const [x, y] = pt(i, R);
    axes += `<line x1="${c}" y1="${c}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="rad-axis"/>`;
    const [lx, ly] = pt(i, R + 17);
    const anchor = Math.abs(lx - c) < 2 ? "middle" : lx > c ? "start" : "end";
    labels += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${anchor}" class="rad-lbl">${esc(ATTR_LABELS[k] || k)}</text>` +
      `<text x="${lx.toFixed(1)}" y="${(ly + 11).toFixed(1)}" text-anchor="${anchor}" class="rad-val">${attrs[k]}</text>`;
  });
  return `<svg class="radar" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}" aria-hidden="true">` +
    grid + axes + `<polygon points="${poly(R, true)}" class="rad-fill"/>` + labels + `</svg>`;
}
function renderPlayerPage(p) {
  const pos = Object.entries(p.position_stats || {}).map(([k, v]) =>
    `<span class="pp-pos">${k}: <b>${v.winrate != null ? Math.round(v.winrate * 100) + "%" : "—"}</b> <span class="muted">(${v.games}и)</span></span>`).join("");
  const games = (p.soloq_games || []).map((g) =>
    `<div class="sq-game ${g.won ? "w" : "l"}" data-s="${g.season}" data-r="${g.round}" title="Открыть матч">` +
    `<span class="sq-res">${g.won ? "W" : "L"}</span>` +
    `<span class="sq-d ${g.delta >= 0 ? "up" : "down"}">${g.delta >= 0 ? "+" : ""}${Math.round(g.delta)}</span>` +
    `<span class="sq-with">с: ${esc((g.teammates || []).join(", "))}</span>` +
    `<span class="sq-vs">против: ${esc((g.opponents || []).join(", "))}</span><span class="sq-go">→</span></div>`).join("");
  const teamHist = (p.team_history || []).length
    ? `<div class="pp-timeline">${p.team_history.slice().reverse().map((h) =>
        `<div class="pp-tl"><span class="pp-tl-s">С${h.season}</span>${emblem(h.team_id)}` +
        `<span>${esc(h.team)}</span>${h.result ? `<span class="muted">· ${esc(h.result)}</span>` : ""}</div>`).join("")}</div>`
    : `<div class="muted">Пока без клубной истории.</div>`;
  const troph = (p.trophies || []).length
    ? `🏆 ${p.trophies.map((t) => `S${t.season} ${esc(t.title)}`).join(", ")}` : "—";
  el("plBody").innerHTML =
    `<div class="pp-head">${factionChip(p.faction)}` +
      `<div><div class="pp-nm">${esc(p.nickname)}</div>` +
      `<div class="muted">${esc(p.archetype || "")} · ${p.role === "streamer" ? "стример" : "про"} · ${esc(teamLabelFA(p))}</div></div>` +
      `<div class="pp-rating"><b>${p.rating}</b> ${deltaBadge(p.delta)}<span>рейтинг</span></div></div>` +
    `<div class="pp-stats">` +
      `<div class="pp-st"><b>${p.games}</b><span>игр</span></div>` +
      `<div class="pp-st"><b>${p.official_games || 0}</b><span>офиц.</span></div>` +
      `<div class="pp-st"><b>${p.seasons_played || 0}</b><span>сезонов</span></div>` +
      `<div class="pp-st"><b>${p.generation || 0}</b><span>поколение</span></div>` +
      `<div class="pp-st"><b>${p.salary || 0}</b><span>зарплата</span></div>` +
      `<div class="pp-st"><b>${p.earnings || 0}</b><span>призовые</span></div>` +
    `</div>` +
    (p.attributes && Object.keys(p.attributes).length
      ? `<div class="pp-radar"><h4>Характеристики карточки <span class="muted">(0–100, равные условия)</span></h4>` +
        `<div class="pp-radar-wrap">${radarChart(p.attributes, 240)}</div></div>` : "") +
    `<div class="pp-row">Динамика SoloQ <span class="muted">(5 последних сезонов)</span> ` +
      `${sparkline(p.soloq_history || p.rating_history, 320, 50, { window: 5, colorVsFirst: true })} ` +
      `<span class="muted">всего игр SoloQ: ${p.soloq_total || 0}</span></div>` +
    `<div class="pp-row"><b>Позиции:</b> ${pos || "—"}</div>` +
    `<div class="pp-row"><b>Нейронка:</b> <code>${esc(p.net_id || "—")}</code>` +
      (p.lineage && p.lineage.parent ? ` · форк от ${esc(p.lineage.parent)} (G${p.lineage.generation})` : "") + `</div>` +
    `<div class="pp-row"><b>Трофеи:</b> ${troph}</div>` +
    `<h4>Клубная история</h4>${teamHist}` +
    (games ? `<h4>Последние игры SoloQ <span class="muted">(клик — детали матча)</span></h4><div class="pd-soloq">${games}</div>` : "");
  el("plBody").querySelectorAll(".sq-game[data-s]").forEach((r) =>
    r.addEventListener("click", () => openSoloqMatch(p.player_id, r.dataset.s, r.dataset.r)));
}
// --------------------------------------------------------------------------- //
// Match detail modal — SoloQ 3v3 game OR a league series
// --------------------------------------------------------------------------- //
async function openSoloqMatch(pid, season, round) {
  show("matchModal", true);
  el("mtBody").innerHTML = `<div class="muted loading">Загрузка матча…</div>`;
  try { renderSoloqMatch((await api(`/soloq/${pid}/${season}/${round}`)).match); }
  catch (e) { el("mtBody").innerHTML = `<div class="result-error">Ошибка: ${esc(e.message)}</div>`; }
}
function renderSoloqMatch(m) {
  const plRow = (p) =>
    `<div class="mt-pl" data-pid="${esc(p.player_id)}">${factionChip(p.faction)}` +
    `<span class="mt-pl-nm">${esc(p.nickname)}</span>` +
    `<span class="mt-pl-tm" data-team="${esc(p.team_id || "")}">${esc(p.team_abbrev || (p.role === "streamer" ? "стример" : "—"))}</span>` +
    `<span class="mt-pl-ka" title="сносы вражеских ядер / ассисты за карту">${(p.kills != null || p.assists != null) ? `⚔${p.kills || 0} 🤝${p.assists || 0}` : ""}</span>` +
    `<span class="mt-pl-rt">${p.rating}</span>` +
    `<span class="mt-pl-d ${(p.delta || 0) >= 0 ? "up" : "down"}">${p.delta == null ? "—" : (p.delta >= 0 ? "+" : "") + Math.round(p.delta)}</span></div>`;
  const side = (players, won, label) =>
    `<div class="mt-side ${won ? "win" : "lose"}"><div class="mt-side-h">${label} ${won ? "🏆 победа" : ""}</div>` +
    players.map(plRow).join("") + `</div>`;
  el("mtBody").innerHTML =
    `<div class="mt-head">⚔️ SoloQ матч <span class="muted">· Сезон ${m.season} · раунд ${m.round}</span></div>` +
    `<div class="mt-vs">${side(m.team_a, m.won_a, "Команда A")}<div class="mt-mid">VS</div>${side(m.team_b, !m.won_a, "Команда B")}</div>` +
    `<p class="muted mt-foot">Клик по игроку — его страница; по аббревиатуре клуба — страница команды.</p>`;
  el("mtBody").querySelectorAll(".mt-pl").forEach((r) =>
    r.addEventListener("click", (e) => {
      const tm = e.target.closest(".mt-pl-tm");
      if (tm && tm.dataset.team) { show("matchModal", false); openTeam(tm.dataset.team); return; }
      show("matchModal", false); openPlayer(r.dataset.pid);
    }));
}
function openPlaybook(pb, t) {
  show("matchModal", true);
  const ovs = (pb.overrides || []).length
    ? (pb.overrides || []).map((o, i) =>
        `<div class="pb-ov"><div class="pb-if"><span class="pb-n">${i + 1}</span><b>ЕСЛИ</b> ${esc(o.when)}</div>` +
        `<div class="pb-then"><b>ТО</b> играем «<span class="pb-strat">${esc(o.strategy)}</span>»` +
        (o.note ? ` <span class="muted">— ${esc(o.note)}</span>` : "") + `</div></div>`).join("")
    : `<div class="muted">Условных правил нет — клуб всегда играет базовую стратегию.</div>`;
  el("mtBody").innerHTML =
    `<div class="mt-head">📋 Плейбук · ${esc(t ? t.name : pb.name)} <span class="muted">· тактика на турнир</span></div>` +
    `<div class="pb-base"><b>Базовая стратегия:</b> «<span class="pb-strat">${esc(pb.base_strategy)}</span>»</div>` +
    `<div class="pb-ovs-h muted">Условные правила (проверяются по порядку, срабатывает первое подходящее):</div>` +
    `<div class="pb-ovs">${ovs}</div>` +
    `<p class="muted mt-foot">Менеджер задаёт это перед сезоном; код применяет правило перед каждой картой по живому счёту.</p>`;
}
function openLeagueMatch(m, team) {
  show("matchModal", true);
  const meWon = m.won;
  const seq = (m.maps_seq || []).map((r, i) =>
    `<span class="mt-map ${r === "W" ? "w" : "l"}">Карта ${i + 1}: ${esc(r === "W" ? team.abbrev : m.opponent_abbrev)}</span>`).join("");
  const lteam = (tid, name, score, win) =>
    `<div class="mt-lteam${win ? " win" : ""}" data-team="${esc(tid)}">${emblem(tid, "xl")}` +
    `<div class="mt-lnm">${esc(name)}</div><div class="mt-lsc">${score == null ? "·" : score}</div></div>`;
  el("mtBody").innerHTML =
    `<div class="mt-head">${esc(m.stage_label || "Матч")} <span class="muted">· Сезон ${m.season}${m.day ? " · день " + m.day : ""} · Bo${m.best_of || 3}</span></div>` +
    `<div class="mt-lvs">${lteam(team.team_id, team.name, m.maps_for, meWon)}<div class="mt-mid">:</div>` +
    `${lteam(m.opponent, m.opponent_name || m.opponent_abbrev, m.maps_against, m.played && !meWon)}</div>` +
    (seq ? `<div class="mt-maps">${seq}</div>` : (m.played ? "" : `<p class="muted mt-foot">Матч ещё не сыгран — расписание.</p>`)) +
    `<div class="mt-comm"><button class="ghost mt-commbtn" data-h="${esc(team.team_id)}" data-a="${esc(m.opponent)}" data-s="${m.season}">🎙 Комментарий Вольского</button><div class="mt-commout"></div></div>` +
    `<p class="muted mt-foot">Клик по клубу — его страница.</p>`;
  el("mtBody").querySelectorAll(".mt-lteam").forEach((d) =>
    d.addEventListener("click", () => { show("matchModal", false); openTeam(d.dataset.team); }));
  const cb = el("mtBody").querySelector(".mt-commbtn");
  if (cb) cb.addEventListener("click", () => generateCommentary(cb));
}
async function generateCommentary(btn) {
  const out = btn.parentElement.querySelector(".mt-commout");
  btn.disabled = true; out.innerHTML = `<span class="muted loading">Вольский набирает воздуха…</span>`;
  try {
    const r = await apiPost("/match/commentary", { home: btn.dataset.h, away: btn.dataset.a, season: +btn.dataset.s });
    out.innerHTML = `<div class="mt-quote">🎙 <b>Гарри «Голос» Вольский:</b> ${esc(r.text)}</div>`;
  } catch (e) {
    out.innerHTML = `<div class="result-error">${esc(e.message)}</div>`;
  } finally { btn.disabled = false; }
}

// --------------------------------------------------------------------------- //
// Tab: Расписание + сетка плей-офф
// --------------------------------------------------------------------------- //
function leagueBadge(m) {       // per-match upper/lower mark (leagues are shown together, not split)
  if (m.league === "upper") return `<span class="mc-lg up" title="Высшая лига">Высшая</span>`;
  if (m.league === "lower") {
    const g = m.group ? (String(m.group).endsWith("Z") ? " Z" : " A") : "";
    return `<span class="mc-lg lo" title="Низшая лига${g}">Низшая${g}</span>`;
  }
  return "";
}
function matchCard(m, noDay) {
  const winH = m.played && m.winner === m.home, winA = m.played && m.winner === m.away;
  const side = (id, ab, sc, win) =>
    `<div class="mc-row${win ? " win" : ""}">${emblem(id)}<span class="mc-tm">${esc(ab)}</span>` +
    `<span class="mc-sc">${m.played ? sc : "·"}</span></div>`;
  const tag = (!noDay && m.day) ? `<div class="mc-day">День ${m.day}${m.stage && m.stage !== "league" ? " · " + esc(m.stage_label || m.stage) : ""}</div>` : "";
  return `<div class="mcard${m.played ? " played" : " upcoming"}" title="Bo${m.best_of}">` + leagueBadge(m) + tag +
    side(m.home, m.home_abbrev, m.maps_home, winH) +
    side(m.away, m.away_abbrev, m.maps_away, winA) +
  `</div>`;
}
function renderSchedule(data) {
  const reg = (data.regular || []).slice().sort((a, b) => (a.day || 0) - (b.day || 0));
  const bracket = data.bracket || {};
  const played = reg.filter((m) => m.played).length;
  // The LIVE tab can show any day up to its cursor (active reveal) or the last day (finished, before
  // off-season pops live_replay). Mirror that rule so a calendar day is clickable iff LIVE can open it.
  const live = state.world && state.world.live;
  const maxView = live ? (live.active ? (live.cursor != null ? live.cursor : 0)
                                      : ((live.total_days || 1) - 1)) : -1;
  const note = reg.length
    ? `<div class="sched-note muted">Календарь регулярки — обе лиги вместе, матчи помечены ` +
      `<span class="mc-lg up">Высшая</span>/<span class="mc-lg lo">Низшая</span>. Сыграно: <b>${played}</b> / ${reg.length}.` +
      (maxView >= 0 ? ` Клик по сыгранному дню — открыть его в 🔴 LIVE.` : "") + `</div>` : "";
  // ONE combined calendar grid — every day cell holds that day's 1-2 matches from BOTH leagues.
  const byDay = {};
  reg.forEach((m) => (byDay[m.day || 0] = byDay[m.day || 0] || []).push(m));
  const cells = Object.keys(byDay).map(Number).sort((a, b) => a - b).map((d) => {
    const dms = byDay[d];
    const done = dms.every((m) => m.played), part = !done && dms.some((m) => m.played);
    const clickable = (d - 1) <= maxView;        // d is 1-based; LIVE day param is 0-based
    return `<div class="cal-day${done ? " done" : part ? " part" : ""}${clickable ? " clickable" : ""}"` +
      (clickable ? ` data-liveday="${d - 1}" role="button" tabindex="0" title="Открыть День ${d} в LIVE"` : "") + `>` +
      `<div class="cal-daylbl">День ${d}${clickable ? `<span class="cal-go">▶ LIVE</span>` : ""}</div>` +
      `<div class="cal-matches">${dms.map((m) => matchCard(m, true)).join("")}</div></div>`;
  }).join("");
  const regHtml = reg.length ? `<div class="sched-cal">${cells}</div>`
    : `<div class="muted">Регулярка ещё не сыграна.</div>`;
  el("panel-schedule").innerHTML =
    `<h3>Регулярный сезон</h3>${note}${regHtml}` +
    `<h3>Плей-офф <span class="muted">(сетка)</span></h3>${renderBracketGraph(bracket)}`;
  // clicking an already-played day jumps to the LIVE tab focused on that day
  const openDay = (d) => { switchTab("live"); gotoLiveDay(d); };
  el("panel-schedule").querySelectorAll(".cal-day.clickable[data-liveday]").forEach((c) => {
    c.addEventListener("click", () => openDay(Number(c.dataset.liveday)));
    c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDay(Number(c.dataset.liveday)); } });
  });
}

// --------------------------------------------------------------------------- //
// Tab: Команды + страница команды
// --------------------------------------------------------------------------- //
function renderTeams(ranking) {
  ranking = ranking || [];
  const lgBadge = (t) => t.league === "upper"
    ? `<span class="trk-lg up">Высшая</span>`
    : `<span class="trk-lg lo">Низшая${t.group ? " · " + esc(t.group) : ""}</span>`;
  const rows = ranking.map((t) =>
    `<button class="trk-row" data-team="${esc(t.team_id)}">` +
      `<span class="trk-rank">${t.rank}</span>${emblem(t.team_id, "lg")}` +
      `<div class="trk-body"><div class="trk-nm">${esc(t.name)} <span class="trk-ab">${esc(t.abbrev)}</span></div>` +
      `<div class="trk-sub">${lgBadge(t)} · ${"★".repeat(Math.min(5, t.titles || 0)) || "без титулов"} · ` +
      `<span class="muted">${esc((t.roster || []).join(", "))}</span></div></div>` +
      `<span class="trk-pw">⚡ ${t.power}</span><span class="tc-go">→</span>` +
    `</button>`).join("");
  el("panel-teams").innerHTML =
    `<div id="teamPage" class="team-page is-hidden"></div>` +
    `<div class="trk-head">🌍 Глобальный рейтинг силы <span class="muted">(все клубы вместе, без деления на лиги)</span></div>` +
    `<div class="trk-list">${rows}</div>`;
  el("panel-teams").querySelectorAll(".trk-row").forEach((b) =>
    b.addEventListener("click", () => openTeam(b.dataset.team)));
}
function openTeam(teamId) {
  state.pendingTeam = teamId;
  if (state.tab === "teams") { state.pendingTeam = null; loadTeamPage(teamId); }
  else switchTab("teams");      // loadTab('teams') renders cards then opens the pending team page
}
async function loadTeamPage(teamId) {
  const page = document.getElementById("teamPage");   // fresh (renderTeams just rebuilt it)
  if (!page) return;
  page.innerHTML = `<div class="muted loading">Загрузка команды…</div>`;
  page.classList.remove("is-hidden");
  page.scrollIntoView({ behavior: "smooth", block: "start" });
  try { renderTeamPage((await api(`/team/${teamId}`)).team); }
  catch (e) { page.innerHTML = `<div class="result-error">Ошибка: ${esc(e.message)}</div>`; }
}
function matchLine(m, kind, idx) {
  const stage = m.stage && m.stage !== "league" ? esc(m.stage_label || m.stage) : "Регулярка";
  const venue = m.home ? "дома" : "в гостях";
  const tag = `data-mk="${kind}" data-mi="${idx}"`;
  if (!m.played) {
    return `<div class="tpm upcoming" ${tag}><span class="tpm-day">Д${m.day || "?"}</span>` +
      `<span class="tpm-st">${stage}</span><span class="tpm-vs">${m.home ? "vs" : "@"} ${emblem(m.opponent)}${esc(m.opponent_abbrev)}</span>` +
      `<span class="tpm-res muted">— · ${venue}</span></div>`;
  }
  const won = m.won;
  return `<div class="tpm ${won ? "w" : "l"}" ${tag}><span class="tpm-day">Д${m.day || "?"}</span>` +
    `<span class="tpm-st">${stage}</span><span class="tpm-vs">${m.home ? "vs" : "@"} ${emblem(m.opponent)}${esc(m.opponent_abbrev)}</span>` +
    `<span class="tpm-res"><b>${won ? "W" : "L"}</b> ${m.maps_for}:${m.maps_against}</span></div>`;
}
function renderTeamPage(t) {
  const page = document.getElementById("teamPage");
  if (!page) return;
  const rec = t.record || {};
  const roster = (t.roster || []).map((p) =>
    `<div class="tp-player" data-pid="${esc(p.player_id)}">` +
      `<span class="tpp-fac">${factionChip(p.faction)}</span>` +
      `<span class="tpp-nm">${esc(p.nickname)}</span>` +
      `<span class="tpp-ar">${esc(p.archetype || "")}</span>` +
      `<span class="tpp-rr">${p.rating}</span>` +
      `<span class="tpp-dl">${deltaBadge(p.delta)}</span>` +
      `<span class="tpp-sal money" title="Текущая зарплата за сезон">💸 ${p.salary != null ? p.salary : "—"}</span>` +
      `<span class="tpp-net">${esc(p.net_id || "")}</span>` +
    `</div>`).join("");
  const played = (t.matches_played || []).slice().reverse();
  const upcoming = t.matches_upcoming || [];
  const tf = t.transfers || { in: [], out: [] };
  const txLine = (x) => {
    if (x.dir === "bought") return `🟢 куплен <b>${esc(x.player)}</b> у ${esc(x.from || "?")}${x.price != null ? " за " + x.price : ""}`;
    if (x.dir === "signed") return `🟢 подписан <b>${esc(x.player)}</b> <span class="muted">(${x.kind === "rookie" ? "новичок" : "св. агент"})</span>${x.bonus != null ? ", бонус " + x.bonus : ""}`;
    if (x.dir === "sold") return `🔴 продан <b>${esc(x.player)}</b> в ${esc(x.to || "?")}${x.price != null ? " за " + x.price : ""}`;
    return `🔴 отпущен <b>${esc(x.player)}</b>`;
  };
  const transfers = [...tf.in, ...tf.out];
  const histRows = (t.season_history || []).slice().reverse().map((h) =>
    `<div class="tph-row"><span class="tph-s">Сезон ${h.season}</span>` +
    `<span class="tph-lg ${h.league}">${h.league === "upper" ? "Высшая" : "Низшая"}</span>` +
    `<span class="tph-pl">${h.place} место</span>` +
    `${h.champion ? '<span class="tph-badge ch">🏆 чемпион</span>' : ""}` +
    `${h.promoted ? '<span class="tph-badge up">↑ повышение</span>' : ""}` +
    `${h.relegated ? '<span class="tph-badge down">↓ вылет</span>' : ""}</div>`).join("");
  const sect = (title, body) => `<h4>${title}</h4>${body}`;
  page.innerHTML =
    `<div class="tp-head">` +
      `<button class="ghost tp-close">← к командам</button>` +
      emblem(t.team_id, "xl") +
      `<h3>${esc(t.name)} <span class="muted">${esc(t.abbrev)} · ${t.league === "upper" ? "Высшая лига" : "Низшая лига " + (t.group || "")}</span></h3>` +
      (t.playbook ? `<button class="ghost tp-pb">📋 Плейбук</button>` : "") +
    `</div>` +
    `<div class="tp-stats">` +
      `<div class="tp-stat"><b>#${t.power_rank || "—"}</b><span>в глоб. рейтинге</span></div>` +
      `<div class="tp-stat"><b>${t.power}</b><span>Power</span></div>` +
      `<div class="tp-stat"><b>${rec.series_wins || 0}–${rec.series_losses || 0}</b><span>Серии (сезон)</span></div>` +
      `<div class="tp-stat"><b>${rec.maps_won || 0}–${rec.maps_lost || 0}</b><span>Карты (сезон)</span></div>` +
      `<div class="tp-stat"><b>${t.titles || 0}</b><span>Титулы</span></div>` +
      `<div class="tp-stat"><b>${t.treasury}</b><span>Казна</span></div>` +
      `<div class="tp-stat"><b>${t.certificates || 0}</b><span>Сертификаты</span></div>` +
    `</div>` +
    `<div class="tp-power">Динамика Power (Elo) ${sparkline(t.elo_history, 240, 40)}</div>` +
    (t.trophies && t.trophies.length ? `<div class="tp-trophies">🏆 ${t.trophies.map((x) => `S${x.season} ${esc(x.title)}`).join(", ")}</div>` : "") +
    `<div class="tp-cols">` +
      `<div class="tp-col">` +
        sect("Состав", `<div class="tp-roster">${roster || "<em>пусто</em>"}</div>`) +
        (upcoming.length ? sect(`Предстоящие матчи <span class="muted">(${upcoming.length})</span>`,
          `<div class="tpm-list">${upcoming.map((m, i) => matchLine(m, "u", i)).join("")}</div>`) : "") +
      `</div>` +
      `<div class="tp-col">` +
        sect(`Сыгранные матчи <span class="muted">(${played.length}, клик — детали)</span>`,
          played.length ? `<div class="tpm-list">${played.map((m, i) => matchLine(m, "p", i)).join("")}</div>` : `<div class="muted">Пока не сыграно.</div>`) +
        sect("Трансферы", transfers.length ? `<div class="tp-tx">${transfers.map((x) => `<div class="tp-tx-row">${txLine(x)}</div>`).join("")}</div>` : `<div class="muted">Без движений в последнее межсезонье.</div>`) +
        sect("История по сезонам", histRows ? `<div class="tph">${histRows}</div>` : `<div class="muted">Ещё не сыграно сезонов.</div>`) +
      `</div>` +
    `</div>`;
  page.querySelector(".tp-close").addEventListener("click", () => page.classList.add("is-hidden"));
  const pb = page.querySelector(".tp-pb");
  if (pb) pb.addEventListener("click", () => openPlaybook(t.playbook, t));
  page.querySelectorAll(".tp-player").forEach((r) =>
    r.addEventListener("click", () => openPlayer(r.dataset.pid)));
  const teamCtx = { team_id: t.team_id, name: t.name, abbrev: t.abbrev };
  page.querySelectorAll(".tpm[data-mk]").forEach((r) =>
    r.addEventListener("click", () => openLeagueMatch((r.dataset.mk === "u" ? upcoming : played)[+r.dataset.mi], teamCtx)));
}

// --------------------------------------------------------------------------- //
// Tab: История чемпионов
// --------------------------------------------------------------------------- //
function renderHistory(history) {
  history = (history || []).slice();
  if (!history.length) {
    el("panel-history").innerHTML = `<div class="muted">Ещё не сыграно ни одного сезона.</div>`;
    return;
  }
  const nm = (x) => x && x.name ? x.name : (x || "—");
  const tt = (x) => x && x.team_id
    ? `<span class="hx-tm os-clk" data-os-team="${esc(x.team_id)}">${emblem(x.team_id)}<span>${esc(nm(x))}</span></span>`
    : `<span>${esc(nm(x))}</span>`;
  // <ol> auto-numbers the places — don't print t.place too (that was the duplicate "1. 1.")
  const topList = (rows, withPrize) => `<ol class="hx-top">${(rows || []).map((t) =>
    `<li>${tt(t)}${withPrize && (t.total || t.prize) ? ` <span class="hx-pz money">+${t.total || t.prize}</span>` : ""}</li>`).join("") || "<li>—</li>"}</ol>`;
  // the GRAND-FINAL WINNER, shown clearly (champion + who they beat in the final) — #6.1 / #6.2
  const champLine = (champ, finalists) => {
    if (!champ || !champ.team_id) return "";
    const other = (finalists || []).filter((f) => f && f.team_id !== champ.team_id)[0];
    return `<div class="hx-gf"><span class="hx-gf-cup">🏆</span><b>${tt(champ)}</b>` +
      (other ? `<span class="hx-gf-beat muted">в финале обыграл</span>${tt(other)}` : "") + `</div>`;
  };
  const prizeTable = (pool, lg, title) => {
    const rows = (pool && pool[lg]) || [];
    const head = `<h6>${esc(title)}</h6>`;
    if (!rows.length) return `<div class="hx-prz-col">${head}<div class="muted">—</div></div>`;
    return `<div class="hx-prz-col">${head}<table class="hx-prz"><tbody>${rows.map((r) =>
      `<tr class="os-clk" data-os-team="${esc(r.team_id)}"><td class="hx-pl">${r.place}</td><td>${tt(r)}</td>` +
      `<td class="hx-pz money">${r.prize ? "+" + r.prize : "—"}</td>` +
      `<td>${r.bonus ? `<span class="hx-bonus money">🏆 +${r.bonus}</span>` : ""}</td></tr>`).join("")}</tbody></table></div>`;
  };
  const card = (s) => {
    const promo = (s.promoted || []).map(tt).join(" ") || "—";
    const releg = (s.relegated || []).map(tt).join(" ") || "—";
    const hasBracket = s.bracket && ((s.bracket.upper || []).length || (s.bracket.lower || []).length);
    const promoChips = (s.promoted || []).length ? (s.promoted || []).map(tt).join("") : tt(s.lower_champion);
    return `<details class="hx-card"${s === history[history.length - 1] ? " open" : ""}>` +
      `<summary class="hx-sum"><span class="hx-s">Сезон ${s.season}</span>` +
        `<span class="hx-ch"><span class="hx-ch-ic">🏆</span><span class="hx-ch-lbl">чемпион</span>${tt(s.upper_champion)}</span>` +
        `<span class="hx-ch lo"><span class="hx-ch-ic">⬆</span><span class="hx-ch-lbl">повышены</span>${promoChips}</span></summary>` +
      `<div class="hx-body">` +
        `<div class="hx-col"><h5>🥇 Высшая лига — чемпион</h5>${champLine(s.upper_champion, s.upper_finalists)}` +
          `<div class="hx-toph muted">Топ-3:</div>${topList(s.upper_top, true)}</div>` +
        `<div class="hx-col"><h5>🥈 Низшая лига — чемпион</h5>${champLine(s.lower_champion, s.lower_finalists)}` +
          `<div class="hx-toph muted">Топ-3:</div>${topList(s.lower_top, false)}</div>` +
        `<div class="hx-col"><h5>↕ Движение</h5>` +
          `<div class="hx-pr"><b>↑ Повышены (2):</b> ${promo}</div>` +
          `<div class="hx-pr"><b>↓ Вылетели (2):</b> ${releg}</div>` +
          `<div class="hx-pr muted">${s.n_matches || 85} серий за сезон</div></div>` +
      `</div>` +
      `<details class="hx-sub"><summary>💰 Призовые за сезон</summary><div class="hx-prz-wrap">` +
        prizeTable(s.prize_pool, "upper", "Высшая лига") +
        prizeTable(s.prize_pool, "lower_A", "Низшая · группа A") +
        prizeTable(s.prize_pool, "lower_Z", "Низшая · группа Z") + `</div></details>` +
      (hasBracket ? `<details class="hx-sub hx-brk-d"><summary>🏆 Сетка плей-офф</summary>` +
        `<div class="hx-brk">${renderBracketGraph(s.bracket)}</div></details>` : "") +
      `</details>`;
  };
  el("panel-history").innerHTML =
    `<div class="hx-list">${history.slice().reverse().map(card).join("")}</div>`;
  el("panel-history").querySelectorAll("[data-os-team]").forEach((b) =>
    b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); openTeam(b.dataset.osTeam); }));
  el("panel-history").querySelectorAll(".hx-brk-d").forEach((d) =>   // redraw connectors when expanded
    d.addEventListener("toggle", () => { if (d.open) setTimeout(drawBracketConnectors, 0); }));
}

// --------------------------------------------------------------------------- //
// Tab: ✍️ Драфт
// --------------------------------------------------------------------------- //
function renderDraft(data) {
  const picks = (data && data.picks) || [];
  if (!picks.length) {
    el("panel-draft").innerHTML = `<div class="muted">Драфт ещё не проводился.</div>`;
    return;
  }
  const kind = data.kind === "bootstrap" ? "Стартовый драфт" : "Драфт новичков";
  const rounds = {};
  picks.forEach((p) => { (rounds[p.round] = rounds[p.round] || []).push(p); });
  const html = Object.keys(rounds).sort((a, b) => a - b).map((r) =>
    `<div class="draft-round"><div class="dr-head">Раунд ${r}</div>` +
    `<div class="draft-board">${rounds[r].map((p) =>
      `<div class="dpick">` +
        `<span class="dp-ov">${p.overall}</span>` +
        emblem(p.team_id, "") +
        `<div class="dp-body"><div class="dp-pl">${esc(p.player)} ${factionChip(p.faction)}</div>` +
        `<div class="dp-sub">${esc(p.team)} · ${esc(p.archetype || "")}</div></div>` +
        `<span class="dp-rr">${p.rating ?? ""}</span>` +
      `</div>`).join("")}</div></div>`).join("");
  el("panel-draft").innerHTML =
    `<h3>${kind} <span class="muted">сезон ${data.season} · ${picks.length} пиков</span></h3>${html}`;
}

// --------------------------------------------------------------------------- //
// Tab: 💬 Менеджеры (per-manager comments)
// --------------------------------------------------------------------------- //
function renderManagers(comments) {
  comments = comments || [];
  if (!comments.length) {
    el("panel-managers").innerHTML = `<div class="muted">Комментарии появятся после межсезонья.</div>`;
    return;
  }
  el("panel-managers").innerHTML =
    `<h3>Слово менеджеров <span class="muted">межсезонье</span></h3>` +
    `<div class="mgr-grid">${comments.slice().reverse().map((c) =>
      `<div class="mgr-card">${emblem(c.team_id, "lg")}` +
      `<div class="mgr-body"><div class="mgr-head">${esc(c.team)} <span class="muted">· ${esc(c.manager)}</span></div>` +
      `<div class="mgr-text">«${esc(c.text)}»</div></div></div>`).join("")}</div>`;
}

// --------------------------------------------------------------------------- //
// Tab: 📻 Эфир (commentator)
// --------------------------------------------------------------------------- //
const COMMENT_ICON = { season_open: "🎬", season_close: "🏆", offseason: "🔁", match: "🎙️" };
// --------------------------------------------------------------------------- //
// Tab: 💸 Трансферы — off-season transactions window
// --------------------------------------------------------------------------- //
function osTeamTag(id, name) {
  return `<span class="os-clk" data-os-team="${esc(id)}">${emblem(id)}<span class="tx-tm">${esc(name)}</span></span>`;
}
function osPlayerTag(pid, name) {
  return pid ? `<span class="os-clk tx-who" data-os-pid="${esc(pid)}">${esc(name)}</span>`
             : `<span class="tx-who">${esc(name)}</span>`;
}
function offseasonTxHtml(off, opts) {
  const tx = off.transactions;
  const full = !opts || opts.full !== false;
  const empty = (txt) => `<div class="tx-empty muted">${txt}</div>`;
  const sec = (title, count, body) =>
    `<section class="tx-sec"><h3>${title} <span class="tx-n">${count}</span></h3>${body}</section>`;

  const auctions = (tx.auctions || []).length
    ? `<div class="tx-list">${tx.auctions.map((a) =>
        `<div class="tx-row">${osTeamTag(a.team_id, a.team)}<span class="tx-mid">выиграл слот тренировки</span>` +
        `<span class="tx-price">${a.price}</span></div>`).join("")}</div>`
    : empty("Аукцион тренировок не проводился.");

  const trains = (tx.training_requests || []).length
    ? `<div class="tx-list">${tx.training_requests.map((t) =>
        `<div class="tx-row">${osTeamTag(t.team_id, t.team)}<span class="tx-mid">→ ${esc(t.op_title)}</span>` +
        `${osPlayerTag(t.player_id, t.player)}` +
        `<span class="tx-steps">${t.steps ? (t.steps / 1e6).toFixed(1) + "M ш" : ""}</span></div>`).join("")}</div>`
    : empty("Никто не отправлен на тренировку.");

  const signings = (tx.signings || []).length
    ? `<div class="tx-list">${tx.signings.map((s) =>
        `<div class="tx-row">${osTeamTag(s.team_id, s.team)}<span class="tx-mid">подписал</span>` +
        `${osPlayerTag(s.player_id, s.player)}<span class="tx-kind">${s.kind === "rookie" ? "новичок" : "св. агент"}</span>` +
        `<span class="tx-price">${s.bonus != null ? "бонус " + s.bonus : ""}</span></div>`).join("")}</div>`
    : empty("Подписаний не было.");

  const buys = (tx.buys || []).length
    ? `<div class="tx-list">${tx.buys.map((b) =>
        `<div class="tx-row">${osTeamTag(b.to_team_id, b.to_team)}<span class="tx-mid">купил</span>` +
        `${osPlayerTag(b.player_id, b.player)}<span class="tx-from">у ${esc(b.from_team)}</span>` +
        `<span class="tx-price">${b.price}</span></div>`).join("")}</div>`
    : empty("Сделок между клубами не было.");

  const releases = (tx.releases || []).length
    ? `<div class="tx-list">${tx.releases.map((r) =>
        `<div class="tx-row">${osTeamTag(r.team_id, r.team)}<span class="tx-mid">отпустил</span>` +
        `${osPlayerTag(r.player_id, r.player)}<span class="tx-from">→ свободный агент</span></div>`).join("")}</div>`
    : empty("Никого не отпустили.");

  const requests = (tx.requests || []).length
    ? `<p class="tx-note muted">Аукцион за <b>свободных агентов</b>: на одного игрока могут ставить несколько ` +
      `клубов, выигрывает <b>самая высокая</b> ставка. «✓ выиграл» — ставка победила и игрок подписан ` +
      `(он же появится в «Подписаниях»). Проигравшие ставки ничего не платят. Это НЕ покупка у клуба — ` +
      `тут берут игроков без команды.</p>` +
      `<div class="tx-list">${tx.requests.map((q) =>
        `<div class="tx-row ${q.won ? "tx-won" : ""}">${osTeamTag(q.team_id, q.team)}<span class="tx-mid">ставка на</span>` +
        `${osPlayerTag(q.player_id, q.player)}<span class="tx-price">${q.amount != null ? q.amount : ""}</span>` +
        `<span class="tx-flag">${q.won ? "✓ выиграл" : "проиграл"}</span></div>`).join("")}</div>`
    : empty("Ставок на свободных агентов не было.");

  return (full ? sec("🏷️ Аукцион тренировок", tx.counts.auctions, auctions) +
                 sec("💪 На тренировку", tx.counts.training_requests, trains) : "") +
    sec("✍️ Подписания <small>(кого взяли из свободных агентов)</small>", tx.counts.signings, signings) +
    sec("💱 Покупки между клубами <small>(трансфер за деньги у другого клуба)</small>", tx.counts.buys, buys) +
    sec("✂️ Отпущены (кики)", tx.counts.releases, releases) +
    sec("📨 Ставки на свободных агентов <small>(аукцион: кто на кого ставил)</small>", tx.counts.requests, requests);
}
function renderOffseasonTx(off) {
  const panel = el("panel-transfers");
  if (!off || !off.transactions) {
    panel.innerHTML = `<div class="muted">Межсезонье ещё не проводилось — итоги трансферов появятся здесь.</div>`;
    return;
  }
  panel.innerHTML =
    `<div class="tx-head">Итоги межсезонья <span class="muted">после сезона ${off.season_ended}</span></div>` +
    offseasonTxHtml(off, { full: true });
  // make clubs + players clickable (this standalone tab had no handler wired)
  panel.querySelectorAll("[data-os-team]").forEach((b) =>
    b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); openTeam(b.dataset.osTeam); }));
  panel.querySelectorAll("[data-os-pid]").forEach((b) =>
    b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); openPlayer(b.dataset.osPid); }));
}

// --------------------------------------------------------------------------- //
// Tab: 📻 Эфир
// --------------------------------------------------------------------------- //
function renderBroadcast(lines) {
  lines = lines || [];
  el("panel-broadcast").innerHTML = !lines.length
    ? `<div class="muted">Эфир пуст — сыграй сезон или проведи межсезонье.</div>`
    : `<div class="broadcast"><div class="bc-host">📻 Гарри «Голос» Вольский</div>` +
      lines.slice().reverse().map((c) =>
        `<div class="bc-line"><span class="bc-ic">${COMMENT_ICON[c.kind] || "•"}</span>` +
        `<div class="bc-body"><span class="bc-season">Сезон ${c.season}</span>` +
        `<span class="bc-text">${esc(c.text)}</span></div></div>`).join("") + `</div>`;
}

// --------------------------------------------------------------------------- //
// Training progress (off-season)
// --------------------------------------------------------------------------- //
let trainTimer = null;
async function pollTraining() {
  if (!state.worldId) return;
  let data = {};
  try { data = await fetchJson(`/api/world/${state.worldId}/training`); }
  catch (e) { return; }
  const jobs = data.jobs || [];
  if (data.measured_sps) state._measuredSps = data.measured_sps;   // real machine rate — calibrate the setup ETA
  if (!jobs.length) { show("trainingPanel", false); return; }
  const running = jobs.filter((j) => j.status === "running").length;
  const queued = jobs.filter((j) => j.status === "queued").length;
  if (running + queued) state.trainingDismissed = false;   // a new batch -> show it again
  if (state.trainingDismissed) { show("trainingPanel", false); return; }
  show("trainingPanel", true);
  const allDone = !(running + queued);
  const head = running ? `${running} идёт${queued ? ", " + queued + " в очереди" : ""}`
    : queued ? `${queued} в очереди` : "завершены";
  // accurate ETA: from the running job's OWN observed rate (done/elapsed), measured on this machine
  const etaHead = (running + queued) && data.total_eta_sec
    ? ` · осталось ≈ <b>${fmtEta(data.total_eta_sec)}</b>` : "";
  el("trainingPanel").innerHTML =
    `<h3>🏋️ Тренировки нейронок <span class="muted">${head}${etaHead}</span>` +
    (allDone ? `<button id="trainDismiss" class="ghost tp-x" title="Скрыть">✕</button>` : "") + `</h3>` +
    jobs.map((j) => {
      const pct = Math.min(100, Math.round((j.done / Math.max(1, j.total)) * 100));
      const eta = j.eta_sec != null ? ` · ≈ ${fmtEta(j.eta_sec)}` : "";
      const tag = j.status === "done" ? "✓ карточка пересчитана"
        : j.status === "error" ? "ошибка"
        : j.status === "queued" ? "в очереди" : `${pct}%${eta}`;
      return `<div class="train-row"><span class="tn">${esc(j.player)}</span>` +
        `<span class="tnet">${esc(OP_LABEL[j.op] || j.op || "")}</span>` +
        `<div class="train-bar"><div class="train-fill${j.status === "done" ? " done" : ""}" style="width:${j.status === "done" ? 100 : pct}%"></div></div>` +
        `<span class="tp">${tag}</span></div>`;
    }).join("");
  const dx = document.getElementById("trainDismiss");
  if (dx) dx.addEventListener("click", dismissTraining);
  if (allDone && trainTimer) { clearInterval(trainTimer); trainTimer = null; }
}
async function dismissTraining() {
  state.trainingDismissed = true;
  show("trainingPanel", false);
  try { await apiPost("/training/clear"); } catch (e) { /* best-effort */ }
}
function startTrainingPoll() {
  if (trainTimer) clearInterval(trainTimer);
  pollTraining();
  trainTimer = setInterval(pollTraining, 3000);
}

// --------------------------------------------------------------------------- //
// Open / actions
// --------------------------------------------------------------------------- //
function renderHeader(w) {
  el("viewName").textContent = w.name || w.world_id;
  el("viewSeason").textContent = `Сезон ${w.season}`;
  const c = w.counts || {};
  el("viewCounts").textContent =
    `${c.players} игроков · ${c.rostered} в составах · ${c.streamers} стримеров` +
    (c.reserves ? ` · ${c.reserves} в резерве` : "");
  // Off-season is MANDATORY between seasons: after a played season you must run it before the next.
  const due = w.phase === "offseason_due";
  const live = w.live && w.live.active;        // a day-by-day season is in progress
  show("nextDayBtn", false);                   // the «Следующий день» button lives ONLY on the LIVE tab
  show("liveProgress", !!live);
  show("advanceBtn", !live);
  show("liveStartBtn", !live && !due);
  if (live) {
    const ph = w.live.phase === "playoff" ? "плей-офф" : "регулярка";
    el("liveProgress").innerHTML = `🔴 День ${w.live.day}/${w.live.total_days} · ${ph} <span class="lp-go">→ LIVE</span>`;
    el("liveProgress").onclick = () => switchTab("live");
  }
  el("advanceBtn").disabled = state.busy || due;
  el("liveStartBtn").disabled = state.busy || due;
  // off-season button opens the INLINE view: setup when one is due, otherwise review the last results.
  const canReview = !!(w.last_offseason && w.last_offseason.transactions);
  el("offseasonBtn").disabled = state.busy || (!due && !canReview);
  el("offseasonBtn").textContent = due ? "⟳ Межсезонье" : "⟳ Итоги межсезонья";
  el("offseasonBtn").classList.toggle("required", due && !state.busy);
  el("advanceBtn").title = due ? "Сначала проведите межсезонье" : "";
}
async function openWorld(worldId) {
  try {
    const { world } = await fetchJson(`/api/world/${worldId}`);
    state.worldId = world.world_id;
    state.world = world;
    state.emblems = world.emblems || {};
    state.colors = world.colors || {};
    show("newSection", false);
    show("resultPanel", false);
    show("offseasonView", false);
    show("worldView", true);
    renderHeader(world);
    switchTab(state.tab || "leagues");
    startTrainingPoll();
  } catch (e) {
    show("resultPanel", true);
    el("resultPanel").innerHTML = `<div class="result-error">Не удалось открыть мир: ${esc(e.message)}</div>`;
  }
}

// Season-0 player selection ------------------------------------------------- //
const FAC_SHORT = { humans: "H", infected: "I", robots: "R" };
async function openSelModal() {
  if (state.locked || state.busy || state.selected.size !== TEAM_COUNT) return;
  state.selNamed = new Set();
  el("selList").innerHTML = `<div class="muted">Загрузка…</div>`;
  show("selModal", true);
  try {
    const data = await fetchJson("/api/world/players-pool");
    state.pool = data.pool || [];
    state.minEnter = data.min_enter || 0;     // teams×3 — below this not everyone can draft
    el("selX").min = data.min_enter; el("selX").value = data.total;
    renderSelList();
  } catch (e) { el("selList").innerHTML = `<div class="result-error">${esc(e.message)}</div>`; }
}
function renderSelList() {
  el("selList").innerHTML = state.pool.map((p) =>
    `<label class="sel-row"><input type="checkbox" data-net="${esc(p.net_id)}"${state.selNamed.has(p.net_id) ? " checked" : ""}>` +
    `<span class="sel-ov">${p.overall}</span>` +
    `<span class="sel-fac">${FAC_SHORT[p.faction] || "?"}</span>` +
    `<span class="sel-nm">${esc(p.archetype || p.net_id)}</span>` +
    `<span class="sel-net">${esc(p.net_id)}</span></label>`).join("");
  el("selList").querySelectorAll("input[data-net]").forEach((cb) =>
    cb.addEventListener("change", () => {
      cb.checked ? state.selNamed.add(cb.dataset.net) : state.selNamed.delete(cb.dataset.net);
      updateSelHint();
    }));
  updateSelHint();
}
function updateSelHint() {
  const named = state.selNamed.size;
  const min = state.minEnter || 0;
  const x = Math.max(parseInt(el("selX").value || "0", 10) || 0, named);
  el("selNamed").textContent = `отмечено: ${named}`;
  const tooFew = x < min;
  const hint = el("selHint");
  if (tooFew) {
    hint.textContent = `нужно минимум ${min} (22 клуба × 3 стартера) — сейчас ${x}`;
    hint.classList.add("sel-err");
  } else {
    hint.classList.remove("sel-err");
    hint.textContent = x > named ? `+ ${x - named} случайно из остальных` :
      (x ? "ровно отмеченные" : "");
  }
  const btn = el("selCreate");
  if (btn) btn.disabled = tooFew;     // can't create an under-filled world
}
async function createWithSelection() {
  const min = state.minEnter || 0;
  const x = Math.max(parseInt(el("selX").value || "0", 10) || 0, state.selNamed.size);
  if (x < min) { updateSelHint(); return; }     // guard: never POST an under-filled request
  show("selModal", false);
  refreshPicker();
  try {
    const job = await runJob("/api/world/create", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ franchises: [...state.selected],
        entered_net_ids: [...state.selNamed], n_enter: x }),
    }, "Создаём мир — реальные матчи нейронок");
    await refreshWorldList(job.world_id);
    await openWorld(job.world_id);
  } catch (e) {
    show("resultPanel", true);
    el("resultPanel").innerHTML = `<div class="result-error">Ошибка: ${esc(e.message)}</div>`;
  } finally { refreshPicker(); }
}

function flashResult(html) {
  show("resultPanel", true);
  el("resultPanel").innerHTML = html;
}

async function advance() {
  if (!state.worldId || state.busy) return;
  el("advanceBtn").disabled = true;
  try {
    const job = await runJob(`/api/world/${state.worldId}/advance`, { method: "POST" },
      "Играем сезон — реальные матчи нейронок");
    const s = job.summary || {};
    flashResult(`<div class="season-done">Сезон ${s.season} сыгран — 🏆 <strong>${esc(s.upper_champion)}</strong> ` +
      `(верх), ↑ <strong>${esc(s.lower_champion)}</strong> (низ). ` +
      `Повышены: ${(s.promoted || []).join(", ") || "—"}; вылетели: ${(s.relegated || []).join(", ") || "—"}.</div>`);
    await openWorld(state.worldId);
  } catch (e) {
    flashResult(`<div class="result-error">Ошибка сезона: ${esc(e.message)}</div>`);
  } finally { el("advanceBtn").disabled = state.busy; }
}

// --------------------------------------------------------------------------- //
// Day-by-day LIVE season — play one day at a time + watch a match live (no scrub)
// --------------------------------------------------------------------------- //
const STAGE_LBL = { ub: "Верхняя сетка", lb: "Нижняя сетка", lb_r1: "Нижняя · раунд 1",
  lb_r2: "Нижняя · раунд 2", lb_final: "Финал нижней сетки", grand_final: "Гранд-финал",
  league: "Регулярка" };

async function startLive() {
  if (!state.worldId || state.busy) return;
  state.busy = true; el("liveStartBtn").disabled = true;
  try {
    // Plays the WHOLE season net-real in the background (same matches as «Сыграть сезон»);
    // then we reveal it day by day. Other tabs stay spoiler-free until the reveal finishes.
    await runJob(`/api/world/${state.worldId}/live/start`, { method: "POST" },
      "Играем весь сезон нейросетями — потом раскроем по дням");
    state.lastBroadcast = null;
    await openWorld(state.worldId);
    switchTab("live");
  } catch (e) { flashResult(`<div class="result-error">${esc(e.message)}</div>`); }
  finally { state.busy = false; renderHeader(state.world); }
}

async function gotoLiveDay(day) {        // rewind/advance to an ALREADY-PLAYED day (null = back to the live cursor)
  if (state.busy || !state.world || !state.world.live) return;
  try {
    const { live } = await api(day === null ? "/live" : "/live/" + day);
    if (live) { state.world.live = live; renderLive(live); }
  } catch (e) { flashResult(`<div class="result-error">${esc(e.message)}</div>`); }
}

async function playNextDay() {           // reveal the next already-played day (instant)
  if (state.busy || !state.world || !state.world.live) return;
  state.busy = true; el("nextDayBtn").disabled = true; renderHeader(state.world);
  try {
    const { result, live } = await apiPost("/live/day");
    state.world.live = live;
    if (result.finished) {               // last day revealed -> season settled
      state.world.phase = "offseason_due";
      flashResult(`<div class="season-done">Сезон раскрыт полностью! 🏆 ${esc((live.done || {}).upper_champion || "")} — пора в межсезонье.</div>`);
      await openWorld(state.worldId);
      switchTab("live");
      return;
    }
    renderLive(live);
  } catch (e) { flashResult(`<div class="result-error">${esc(e.message)}</div>`); }
  finally { state.busy = false; renderHeader(state.world); }
}

async function watchMatch(home, away, gid) {  // watch ONE match live (one-time) — COMMITS it
  if (state.busy) return;
  state.busy = true; renderHeader(state.world);
  const bc = document.getElementById(`bcast-${gid}`);
  try {
    const { result, live } = await apiPost("/live/watch", { home, away });
    const maps = bcastMaps(result);
    // The game is committed server-side, but DON'T touch the LIVE tab yet: the modal backdrop fades in
    // (and is semi-transparent), so revealing the card's final score now flashes it BEHIND the эфир for
    // ~0.5s. Defer all reveal to when the эфир modal is CLOSED (after watching) — zero spoiler leak.
    const commitReveal = () => {
      if (live.done && !live.active) {            // that was the season's final game
        state.world.phase = "offseason_due";
        flashResult(`<div class="season-done">Сезон раскрыт полностью! 🏆 ${esc((live.done || {}).upper_champion || "")} — пора в межсезонье.</div>`);
        openWorld(state.worldId).then(() => switchTab("live"));
        return;
      }
      state.world.live = live;                    // the watched game is committed: stats update now
      state.justRevealed = gid;                   // only THIS card's score should pop (not the day's others)
      renderLive(live);                           // card now shows its score + «посмотреть повтор»
    };
    if (maps.some((m) => m.url)) {                // ЭФИР: настоящий матч, без перемотки/спойлеров, со скоростью
      const c = (live.day_matches || []).find((x) => String(x.gid) === String(result.gid));
      openReplayModal(maps, 0, {                  // no score => spoiler-safe broadcast scoreboard
        live: true, bestOf: result.best_of,
        home: { id: result.home, abbrev: result.home_abbrev, name: c ? c.home.name : result.home_abbrev, players: c ? c.home.players : null },
        away: { id: result.away, abbrev: result.away_abbrev, name: c ? c.away.name : result.away_abbrev, players: c ? c.away.players : null },
        onClose: commitReveal,                    // reveal the LIVE card ONLY after the эфир is closed
      });
    } else if (bc) {
      await playBroadcast(bc, result);            // нет записей -> in-card счёт по картам (its own reveal)
      commitReveal();
    } else {
      commitReveal();
    }
  } catch (e) {
    if (bc) bc.innerHTML = `<div class="result-error">${esc(e.message)}</div>`;
  } finally { state.busy = false; renderHeader(state.world); }
}

// Re-watch an ALREADY-revealed match. ПОВТОР: result is known, so full controls (scrubber + map nav).
async function replayMatch(gid, card) {
  if (state.busy) return;
  state.busy = true; renderHeader(state.world);
  try {
    const r = {
      home: card.home.team_id, away: card.away.team_id,
      home_abbrev: card.home.abbrev, away_abbrev: card.away.abbrev,
      best_of: card.best_of, sequence: card.sequence, map_replays: card.map_replays,
      maps_home: card.maps_home, maps_away: card.maps_away, winner: card.winner,
    };
    const maps = bcastMaps(r);
    if (maps.some((m) => m.url)) {
      openReplayModal(maps, 0, {                     // ПОВТОР: счёт серии + табы карт + перемотка
        live: false, bestOf: card.best_of,
        home: { id: card.home.team_id, abbrev: card.home.abbrev, name: card.home.name, players: card.home.players },
        away: { id: card.away.team_id, abbrev: card.away.abbrev, name: card.away.name, players: card.away.players },
        score: { h: card.maps_home, a: card.maps_away },
      });
    } else {
      const bc = document.getElementById(`bcast-${gid}`);
      if (bc) await playBroadcast(bc, r);           // нет записей -> счёт по картам
    }
  } finally { state.busy = false; renderHeader(state.world); }
}

// Per-decisive-map descriptor: which team won + the REAL replay file (null if not recorded).
function bcastMaps(r) {
  const ab = (id) => (id === r.home ? r.home_abbrev : r.away_abbrev);
  const reps = r.map_replays || [];
  return (r.sequence || []).map((w, i) => ({
    n: i + 1, winnerAbbrev: ab(w), homeWon: w === r.home, url: reps[i] || null,
  }));
}

async function playBroadcast(bc, r) {            // reveal a series map-by-map inside `bc`
  const maps = bcastMaps(r);
  const playable = maps.filter((m) => m.url);
  const real = playable.length > 0;              // do we have the actual recorded matches?
  const fast = real ? 1 : 0;                     // when the real match is the show, don't dawdle on the score
  bc.classList.add("playing");
  bc.innerHTML =
    `<div class="bcast">` +
      `<div class="bcast-head"><span class="bcast-live">🔴 LIVE</span> ${real ? "Реальный матч" : "Прямой эфир"}` +
        `<span class="bcast-bo">Bo${r.best_of}${real ? " · клик по карте — настоящий бой" : " · без перемотки"}</span></div>` +
      `<div class="bcast-scoreboard">` +
        `<div class="bcast-team home">${esc(r.home_abbrev)}</div>` +
        `<div class="bcast-score"><span class="bs-h">0</span><span class="bcast-vs">:</span><span class="bs-a">0</span></div>` +
        `<div class="bcast-team away">${esc(r.away_abbrev)}</div>` +
      `</div>` +
      `<div class="bcast-maps"></div>` +
      (real
        ? `<button class="primary bcast-real">▶ Смотреть настоящий матч</button>`
        : `<div class="bcast-norec muted">📼 запись этого матча недоступна — показан счёт по картам</div>`) +
    `</div>`;
  const card = bc.querySelector(".bcast");
  const mapsBox = card.querySelector(".bcast-maps");
  let h = 0, a = 0;
  for (let i = 0; i < maps.length; i++) {
    await sleep(i === 0 ? (fast ? 120 : 600) : (fast ? 160 : 1700));
    const m = maps[i];
    m.homeWon ? h++ : a++;
    card.querySelector(".bs-h").textContent = h;
    card.querySelector(".bs-a").textContent = a;
    mapsBox.insertAdjacentHTML("beforeend", m.url
      ? `<button class="bc-map ${m.homeWon ? "h" : "a"} bc-watch" data-i="${i}" title="Открыть реальный бой этой карты">▶ Карта ${m.n}: ${esc(m.winnerAbbrev)}</button>`
      : `<span class="bc-map ${m.homeWon ? "h" : "a"}">Карта ${m.n}: ${esc(m.winnerAbbrev)}</span>`);
  }
  await sleep(fast ? 80 : 600);
  const win = r.winner === r.home ? r.home_abbrev : r.away_abbrev;
  card.insertAdjacentHTML("beforeend", `<div class="bcast-final">🏆 ${esc(win)} — ${r.maps_home}:${r.maps_away}</div>`);
  bc.classList.remove("playing");
  const realBtn = card.querySelector(".bcast-real");
  if (realBtn) realBtn.addEventListener("click", () => openReplayModal(maps, 0));
  card.querySelectorAll(".bc-watch").forEach((b) =>
    b.addEventListener("click", () => openReplayModal(maps, Number(b.dataset.i))));
  return real;
}

// Esports-broadcast overlay around the real match (embedded pixi player). `maps` = bcastMaps(r);
// opts: { live, home:{id,abbrev,name}, away:{...}, bestOf, score:{h,a}? }.
//   live = ЭФИР: spoiler-free — no scrubber (&live=1), no map tabs, no score; auto-plays + auto-rolls maps.
//   !live = ПОВТОР: full controls (iframe scrubber) + clickable map tabs + series score (result known).
function openReplayModal(maps, startI, opts) {
  opts = opts || {};
  const live = !!opts.live;
  const home = opts.home || {}, away = opts.away || {};
  const bestOf = opts.bestOf || "";
  const score = opts.score || null;                    // {h,a} for повтор; null for эфир (spoiler-safe)
  const playable = maps.map((m, i) => (m.url ? i : -1)).filter((i) => i >= 0);
  if (!playable.length) return;
  let cur = playable.includes(startI) ? startI : playable[0];

  let modal = document.getElementById("replayModal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "replayModal";
    modal.className = "rp-overlay";
    document.body.appendChild(modal);
    modal.addEventListener("click", (e) => { if (e.target === modal && modal._close) modal._close(); });
  }
  if (modal._onKey) { document.removeEventListener("keydown", modal._onKey); modal._onKey = null; }
  if (modal._onMsg) { window.removeEventListener("message", modal._onMsg); modal._onMsg = null; }
  if (modal._onClose) { const cb = modal._onClose; modal._onClose = null; try { cb(); } catch (e) { /* */ } }
  modal._onClose = opts.onClose || null;        // fired on close — e.g. эфир reveals the LIVE card AFTER watching
  modal.classList.toggle("rp-live", live);

  const banner = (t, side) =>
    `<div class="rp-team ${side}" style="--tc:${teamColor(t.id)}">` +
      `<span class="rp-logo">${emblem(t.id)}</span>` +
      `<span class="rp-tn"><b>${esc(t.abbrev || "")}</b><small>${esc(t.name || t.abbrev || "")}</small></span>` +
    `</div>`;
  const center = live
    ? `<div class="rp-center">` +
        `<span class="rp-live-ind"><span class="rp-livedot"></span>LIVE</span>` +
        // эфир score: starts 0:0 and ticks up as each WATCHED map finishes (spoiler-safe — you only
        // see a map count after you've watched it).
        `<div class="rp-score live"><span class="rp-sc h">0</span>` +
          `<span class="rp-scsep">:</span><span class="rp-sc a">0</span></div>` +
        (bestOf ? `<span class="rp-bo">BO${esc(String(bestOf))}</span>` : "") +
      `</div>`
    : `<div class="rp-center">` +
        `<div class="rp-score"><span class="rp-sc h">${score ? esc(String(score.h)) : ""}</span>` +
          `<span class="rp-scsep">:</span><span class="rp-sc a">${score ? esc(String(score.a)) : ""}</span></div>` +
        (bestOf ? `<span class="rp-bo">BO${esc(String(bestOf))}</span>` : "") +
      `</div>`;
  // map tabs live on their OWN row below the scorebar (повтор only) — keeps the bar uncluttered
  const tabrow = live ? "" :
    `<div class="rp-tabrow">${playable.map((idx) =>
      `<button class="rp-maptab" data-i="${idx}">Карта ${esc(String(maps[idx].n))}</button>`).join("")}</div>`;
  // NB: player names are drawn IN the replay, at each player's core (app.js) — not in a top strip.

  modal.innerHTML =
    `<div class="rp-broadcast" role="dialog" aria-modal="true" ` +
      `style="--home-tc:${teamColor(home.id)};--away-tc:${teamColor(away.id)}">` +
      `<div class="rp-scorebar">` +
        banner(home, "home") + center + banner(away, "away") +
        `<button class="rp-close" title="Закрыть (Esc)">✕</button>` +
      `</div>` +
      tabrow +
      `<div class="rp-stage">` +
        `<span class="rp-rail h"></span>` +
        `<iframe class="rp-frame" title="${live ? "Прямой эфир" : "Запись матча"}" allow="fullscreen"></iframe>` +
        `<span class="rp-rail a"></span>` +
      `</div>` +
    `</div>`;

  const frame = modal.querySelector(".rp-frame");
  const tabs = modal.querySelectorAll(".rp-maptab");
  const show = (i) => {
    cur = i;
    let base = `index.html?replay=${encodeURIComponent(maps[i].url)}&chrome=ext&back=about:blank`;
    // chrome=ext => the iframe drops its own title/back; our scorebar is the chrome.
    // LIVE: bake the chosen speed into the URL so the freshly-loaded map applies it itself — no
    // cross-document `load` listener (which would accumulate on re-open and read a stale contentDocument).
    if (live) {
      base += "&live=1";
      if (modal._liveSpeed) base += `&speed=${encodeURIComponent(modal._liveSpeed)}`;
    }
    frame.src = base;
    tabs.forEach((b) => b.classList.toggle("active", Number(b.dataset.i) === i));
  };
  const step = (d) => { const p = playable.indexOf(cur) + d; if (p >= 0 && p < playable.length) show(playable[p]); };
  const onKey = (e) => {
    if (e.key === "Escape") close();
    else if (!live && e.key === "ArrowLeft") step(-1);
    else if (!live && e.key === "ArrowRight") step(1);
  };
  function close() {
    modal.classList.remove("open");
    frame.src = "about:blank";
    if (modal._onKey) { document.removeEventListener("keydown", modal._onKey); modal._onKey = null; }
    if (modal._onMsg) { window.removeEventListener("message", modal._onMsg); modal._onMsg = null; }
    modal._close = null;
    if (modal._onClose) { const cb = modal._onClose; modal._onClose = null; try { cb(); } catch (e) { /* */ } }
  }
  if (live) {                                     // эфир: map ends -> the embedded player pings us, roll the next
    let liveH = 0, liveA = 0;                                   // running series score (revealed map-by-map)
    const onMsg = (e) => {
      if (e.origin !== window.location.origin) {                // only our own embedded player
        console.warn("[world] ignored cross-origin message", e.origin); return;
      }
      if (!e.data || e.data.type !== "worldLiveMapEnded") return;
      try {                                                      // one safe read at map-end: remember the speed
        const sel = frame.contentDocument.getElementById("speedSelect");
        if (sel) modal._liveSpeed = sel.value;
      } catch (er) { /* ignore */ }
      const ended = maps[cur];                                  // the map that just finished -> tick the score
      if (ended) {
        if (ended.homeWon) liveH++; else liveA++;
        const h = modal.querySelector(".rp-score.live .rp-sc.h");
        const a = modal.querySelector(".rp-score.live .rp-sc.a");
        if (h) h.textContent = String(liveH);
        if (a) a.textContent = String(liveA);
      }
      const pos = playable.indexOf(cur);
      if (pos < playable.length - 1) {
        show(playable[pos + 1]);                                // roll the next map
      } else {                                                  // series over
        const ind = modal.querySelector(".rp-live-ind");
        if (ind) { ind.textContent = "ЭФИР ОКОНЧЕН"; ind.classList.add("ended"); }
      }
    };
    window.addEventListener("message", onMsg);
    modal._onMsg = onMsg;
  } else {
    tabs.forEach((b) => b.addEventListener("click", () => show(Number(b.dataset.i))));
  }
  modal.querySelector(".rp-close").addEventListener("click", close);
  modal._close = close;                 // backdrop handler (bound once) calls this
  modal._onKey = onKey;
  document.addEventListener("keydown", onKey);
  modal.classList.add("open");
  show(cur);
}

function liveBracketCols(nodes) {
  if (!nodes || !nodes.length) return "";
  const byStage = {};
  nodes.forEach((n) => (byStage[n.stage] = byStage[n.stage] || []).push(n));
  const order = ["lb_r1", "lb_r2", "ub", "lb", "lb_final", "grand_final"];  // == views.PLAYOFF_STAGE_ORDER
  const slot = (t) => t ? `${emblem(t.team_id)}<span>${esc(t.abbrev)}</span>` : `<span class="br-tbd">—</span>`;
  return `<div class="bracket">${order.filter((s) => byStage[s]).map((s) =>
    `<div class="br-col"><div class="br-stage">${esc(STAGE_LBL[s] || s)}</div><div class="br-matches">` +
    byStage[s].map((n) =>
      `<div class="mcard ${n.played ? "played" : "upcoming"}">` +
      `<div class="mc-row${n.played && n.winner === (n.home && n.home.team_id) ? " win" : ""}">${slot(n.home)}` +
      `<span class="mc-sc">${n.played ? n.maps_home : "·"}</span></div>` +
      `<div class="mc-row${n.played && n.winner === (n.away && n.away.team_id) ? " win" : ""}">${slot(n.away)}` +
      `<span class="mc-sc">${n.played ? n.maps_away : "·"}</span></div></div>`).join("") +
    `</div></div>`).join("")}</div>`;
}

function liveStandingsMini(rows, title) {
  const dlt = (d) => `${d > 0 ? "+" : ""}${d}`;
  return `<div class="lv-tbl"><h5>${esc(title)}</h5><table class="ltable lv-mini"><tbody>${
    rows.map((t) => `<tr><td class="lt-place">${t.place}</td>` +
      `<td class="lt-ab">${emblem(t.team_id)}<span>${esc(t.abbrev)}</span></td>` +
      `<td class="lt-pts">${t.points}</td>` +
      `<td class="lt-dl ${t.map_diff >= 0 ? "up" : "down"}">${dlt(t.map_diff)}</td>` +
      `<td class="lt-wl">${t.series_wins}–${t.series_losses}</td></tr>`).join("")
  }</tbody></table></div>`;
}

function lvFormDots(form) {
  return `<span class="lvm-form">${(form || []).map((r) =>
    `<span class="lvm-dot ${r === "W" ? "w" : "l"}">${r}</span>`).join("") || `<span class="muted">—</span>`}</span>`;
}
function lvTeamSide(t, side) {
  return `<div class="lvm-team ${side}" data-team="${esc(t.team_id)}">` +
    `${emblem(t.team_id, "xl")}` +
    `<div class="lvm-nm">${esc(t.name)}</div><div class="lvm-ab">${esc(t.abbrev)}</div>` +
    `<div class="lvm-stats">` +
      `<div class="lvm-st"><b>⚡ ${t.power}</b><span>Power</span></div>` +
      `<div class="lvm-st"><b>${t.place ? "#" + t.place : "—"}</b><span>место</span></div>` +
      `<div class="lvm-st"><b>${t.points}</b><span>очки</span></div>` +
      `<div class="lvm-st"><b>${t.series_wins}–${t.series_losses}</b><span>В–П</span></div>` +
    `</div>` +
    `<div class="lvm-formrow"><span class="muted">форма</span> ${lvFormDots(t.form)}</div></div>`;
}
// --- Win-probability (UI-only, cosmetic): math from the MOST CURRENT Power (Elo) each day ----- //
function _comb(n, k) { let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); return c; }
function eloExpected(a, b) { return 1 / (1 + Math.pow(10, (b - a) / 400)); }
function seriesWinProb(pMap, bestOf) {          // P(reach ceil(bo/2) map wins first) — race math
  const need = Math.floor((bestOf || 1) / 2) + 1;
  let p = 0;
  for (let j = 0; j < need; j++) p += _comb(need - 1 + j, j) * Math.pow(pMap, need) * Math.pow(1 - pMap, j);
  return p;
}
function lvProbBar(card) {
  // FROZEN multi-factor probability computed server-side BEFORE the match (Power + roster + H2H +
  // form + standing) and stored on the series — it does NOT recompute after the match. Falls back to
  // a live Elo estimate only for legacy worlds that have no stored value.
  let hp = card.win_prob;
  const f = card.win_prob_factors;
  if (hp == null) {
    const pMap = eloExpected(card.home.power, card.away.power);
    let ph = seriesWinProb(pMap, card.best_of); const pa = seriesWinProb(1 - pMap, card.best_of);
    const s = ph + pa; ph = s > 0 ? ph / s : 0.5;
    hp = Math.max(1, Math.min(99, Math.round(ph * 100)));
  }
  const ap = 100 - hp;
  const sgn = (x) => (x >= 0 ? "+" : "") + x;
  const tip = f
    ? `Прогноз ДО матча (зафиксирован, Bo${card.best_of}): Power ${sgn(f.power)}, состав ${sgn(f.roster)}, `
      + `очные ${sgn(f.h2h)}, форма ${sgn(f.form)}, таблица ${sgn(f.standing)}`
    : `Оценка по силе (Bo${card.best_of})`;
  return `<div class="lvm-prob" title="${esc(tip)}">` +
    `<div class="lvp-bar"><div class="lvp-h" style="width:${hp}%"></div><div class="lvp-a" style="width:${ap}%"></div></div>` +
    `<div class="lvp-lbl"><span class="lvp-hn">${hp}%</span>` +
    `<span class="lvp-cap muted">шанс на победу${f ? " · до матча" : ""}</span><span class="lvp-an">${ap}%</span></div></div>`;
}

function lvWatchCard(live, card) {
  const gid = card.gid, rev = card.revealed, isGf = card.stage === "grand_final";
  const justPop = rev && String(gid) === String(state.justRevealed);   // pop only the freshly-watched card
  const mid = rev
    ? `<div class="lvm-score"><span class="${card.winner === card.home.team_id ? "win" : ""}">${card.maps_home}</span>` +
      `<span class="lvm-sc-sep">:</span><span class="${card.winner === card.away.team_id ? "win" : ""}">${card.maps_away}</span></div>` +
      `<button class="ghost lvm-replay" data-gid="${gid}">🔁 Посмотреть повтор</button>`
    : `<div class="lvm-vs-big">VS</div>` +
      `<button class="primary lvm-watch" data-h="${esc(card.home.team_id)}" data-a="${esc(card.away.team_id)}" data-gid="${gid}">📺 Смотреть эфир</button>` +
      `<div class="lvm-hint muted">реальный матч, по картам · одноразово</div>`;
  return `<div class="lvm-card${rev ? " revealed" : ""}${justPop ? " just-pop" : ""}" data-gid="${gid}">` +
    `<div class="lvm-ctx"><span class="lvm-league">${esc(card.league_label || "")}</span>` +
      `<span class="lvm-stage${isGf ? " gf" : ""}">${esc(card.stage_label)}</span>` +
      `<span class="lvm-bo">Bo${card.best_of}</span><span class="lvm-day">День ${card.day} / ${live.total_days}</span></div>` +
    `<div class="lvm-vs">${lvTeamSide(card.home, "home")}` +
      `<div class="lvm-mid">${mid}</div>${lvTeamSide(card.away, "away")}</div>` +
    lvProbBar(card) +
    `<div class="lvm-bcast" id="bcast-${gid}"></div>` +   // empty until «повтор» plays (no duplicate score)
  `</div>`;
}
function renderLive(live) {
  const panel = el("panel-live");
  if (!live || (!live.active && !live.done && !(live.day_matches || []).length)) {
    panel.innerHTML = `<div class="lv-empty"><p class="muted">Сезон по дням ещё не запущен.</p>` +
      `<p>Нажми <b>«📅 По дням»</b> вверху — мы сыграем весь сезон нейросетями, а потом раскроем его ` +
      `день за днём, без подсказок о будущем. Каждый матч можно открыть в <b>прямом эфире</b> (одноразово).</p></div>`;
    return;
  }
  const pct = Math.min(100, Math.round(live.cursor / live.total_days * 100));
  const cards = live.day_matches || [];
  const allWatched = cards.length > 0 && cards.every((c) => c.revealed);
  const isCur = live.is_current_day;            // false => we're rewound to an already-played day
  const body = cards.length
    ? `<div class="lvm-grid">${cards.map((c) => lvWatchCard(live, c)).join("")}</div>`
    : `<div class="lv-done">✅ Сезон раскрыт полностью — открой «⟳ Межсезонье», чтобы продолжить.</div>`;
  const phaseLbl = live.phase === "playoff" ? "плей-офф" : "регулярка";
  // Day-navigation: step across ALREADY-PLAYED days only. A rewound day shows finished results — no
  // live broadcast (it already happened); «Следующий день» (reveal) appears only on the live day.
  const nav =
    `<div class="lv-nav">` +
      `<button id="lvPrev" class="ghost lv-navbtn"${live.can_prev ? "" : " disabled"} title="Предыдущий сыгранный день">◀</button>` +
      `<span class="lv-navlbl ${isCur ? "is-live" : "is-rec"}">${isCur ? "🔴 LIVE" : "📼 запись"} · День <b>${live.day}</b> / ${live.total_days} · ${phaseLbl}</span>` +
      `<button id="lvNextNav" class="ghost lv-navbtn"${live.can_next ? "" : " disabled"} title="Следующий сыгранный день">▶</button>` +
    `</div>`;

  panel.innerHTML =
    `<div class="lv-bar">${nav}` +
    `<div class="lv-prog"><div class="lv-prog-fill" style="width:${pct}%"></div></div>` +
    (!isCur && !live.done ? `<button id="lvToLive" class="ghost" title="Вернуться к текущему дню">↩ К эфиру</button>` : "") +
    (live.active && isCur ? `<button id="lvNext" class="primary${allWatched ? " ready" : ""}" title="Перейти к следующему дню (неотсмотренные матчи дня раскроются сразу, без эфира)">▶ Следующий день</button>` : "") + `</div>` +
    (!isCur && !live.done ? `<div class="lv-recnote muted">📼 День уже сыгран — показаны результаты, прямого эфира больше нет. Можно посмотреть повтор любого матча.</div>` : "") +
    body +
    `<p class="lv-foot muted">Сыгранный матч сразу обновляет таблицы, сетку и расписание — вкладки «Лиги», «Очные», «Расписание». День сменится только по кнопке «Следующий день».</p>`;

  state.justRevealed = null;                  // the one-shot pop has been emitted into the HTML
  const nx = document.getElementById("lvNext");
  if (nx) nx.addEventListener("click", playNextDay);
  const pv = document.getElementById("lvPrev");
  if (pv && live.can_prev) pv.addEventListener("click", () => gotoLiveDay(live.viewing_day - 1));
  const nv = document.getElementById("lvNextNav");
  if (nv && live.can_next) nv.addEventListener("click", () => gotoLiveDay(live.viewing_day + 1));
  const tl = document.getElementById("lvToLive");
  if (tl) tl.addEventListener("click", () => gotoLiveDay(null));
  panel.querySelectorAll(".lvm-watch").forEach((b) =>
    b.addEventListener("click", () => watchMatch(b.dataset.h, b.dataset.a, b.dataset.gid)));
  panel.querySelectorAll(".lvm-replay").forEach((b) =>
    b.addEventListener("click", () => {
      const card = (live.day_matches || []).find((c) => String(c.gid) === String(b.dataset.gid));
      if (card) replayMatch(b.dataset.gid, card);
    }));
  panel.querySelectorAll(".lvm-team").forEach((tdiv) =>
    tdiv.addEventListener("click", (e) => {
      if (e.target.closest(".lvm-watch") || e.target.closest(".lvm-replay")) return;
      openTeam(tdiv.dataset.team);
    }));
}

// =========================================================================== //
// Off-season — INLINE view (REPLACES the main world layout during межсезонье).
// Setup → run (progress bar driven by the chronicle) → per-topic tabs, interactive.
// =========================================================================== //
const STEPS_PER_SEC = 833;   // fallback ≈ 50k/мин; superseded by state._measuredSps (the real machine rate)
const KIND_ICON = { stage: "🗓️", train: "🏋️", draft: "✍️", playbook: "📋", fallback: "⚠️",
  transfer: "🔁", training: "🏋️" };
const STRAT_LABEL = { standard: "Стандарт", core_rush: "Раш ядра", all_in: "Олл-ин",
  turtle: "Черепаха", macro: "Макро-эконом", baron: "Контроль Лорда" };
const OSV_TABS = [
  { key: "overview", label: "📋 Обзор" },
  { key: "soloq", label: "📊 SoloQ" },
  { key: "skills", label: "🕸️ Характеристики" },
  { key: "auction", label: "🏷️ Аукционы" },
  { key: "training", label: "🏋️ Тренировки" },
  { key: "transfers", label: "🔁 Трансферы" },
  { key: "money", label: "💰 Деньги" },
  { key: "draft", label: "✍️ Драфт" },
  { key: "playbooks", label: "📋 Плейбуки" },
];

function fmtMin(min) {
  if (min < 1) return "<1 мин";
  if (min < 90) return `≈ ${Math.round(min)} мин`;
  return `≈ ${(min / 60).toFixed(1)} ч`;
}
function fmtEta(sec) {                       // seconds -> human time for the training ETA
  sec = Math.max(0, Math.round(sec || 0));
  if (sec < 60) return `${sec} с`;
  const m = Math.round(sec / 60);
  if (m < 90) return `${m} мин`;
  const h = Math.floor(m / 60), mm = m % 60;
  return `${h} ч${mm ? " " + mm + " мин" : ""}`;
}
function osSettings() {
  const v = (id) => Math.max(0, parseInt(el(id).value || "0", 10) || 0);
  return {
    soloq_rounds: v("osRounds"),
    auction: { slots: v("osSlots"), steps_per_slot: v("osSlotS") },
    rookies: { n: v("osRookN"), steps: v("osRookS") },
    free_agents: { max: v("osFaN"), steps: v("osFaS") },
  };
}
function updateEta() {
  const s = osSettings();
  const jobs = s.auction.slots;                                 // upper bound — see note below
  const totalSteps = jobs * s.auction.steps_per_slot;           // run serially after the off-season
  const sps = state._measuredSps || STEPS_PER_SEC;              // real measured machine rate when known
  const min = totalSteps / sps / 60;
  el("osEta").innerHTML =
    `⏱ Тренировки нейронок: до <b>${jobs}</b> запусков × ${s.auction.steps_per_slot.toLocaleString("ru")} шагов ` +
    `<b>${fmtMin(min)}</b> <span class="muted">(серийно, в фоне после межсезонья — можно следить в трекере)</span>` +
    (s.rookies.n ? `<br>👶 ${s.rookies.n} новичков (форк) — карточки создаются сразу` : "") +
    `<br><span class="muted">Тренера сами решают: тратить выигранный сертификат сейчас или <b>копить</b> его. ` +
    `Запустятся только заказанные тренировки (может быть и 0, и сильно больше — если копили). ` +
    `Лимитов на число нет, единственный предел — 800k шагов на тренировку.</span>`;
}
// --- view open/close + progress ------------------------------------------------------------- //
function openOffseason() {
  if (!state.worldId) return;
  show("worldView", false);
  show("offseasonView", true);
  const due = state.world && state.world.phase === "offseason_due";
  if (due) renderOsSetup();
  else renderOsResults(state.osTab || "overview");
}
function closeOffseason() {
  show("offseasonView", false);
  show("worldView", true);
  renderHeader(state.world);
  switchTab(state.tab || "leagues");
}
function osProgress(pct, label) {
  show("osvProgress", true);
  el("osvPfill").style.width = `${Math.max(0, Math.min(100, pct))}%`;
  el("osvPlbl").textContent = label || "";
}

// --- setup (replaces the old modal): budgets + reserves + run ------------------------------- //
function osField(id, label, val, min, max, step) {
  const a = [`min="${min}"`];
  if (max != null) a.push(`max="${max}"`);
  if (step) a.push(`step="${step}"`);
  return `<label>${esc(label)}<input id="${id}" type="number" ${a.join(" ")} value="${val}"></label>`;
}
function renderOsSetup() {
  el("osvTitle").textContent = "⟳ Межсезонье — подготовка";
  el("osvSub").textContent = `после сезона ${state.world.season}`;
  show("osvTabs", false);
  show("osvProgress", false);
  const res = (state.world && state.world.reserves) || [];
  const reserves = !res.length ? "" :
    `<div class="osr-head">🎒 Ввести из резерва <span class="muted">(${res.length} — только в межсезонье)</span></div>` +
    `<div class="osr-list">${res.map((p) =>
      `<label class="osr-row"><input type="checkbox" data-pid="${esc(p.player_id)}">` +
      `${factionChip(p.faction)}<span class="osr-nm">${esc(p.nickname)}</span>` +
      `<span class="muted">${esc(p.archetype || "")} · ${p.rating}</span></label>`).join("")}</div>`;
  el("osvBody").innerHTML =
    `<div class="osv-setup">` +
      `<p class="muted">Задай бюджеты и параметры. Менеджеры (LLM) сами решают ставки на аукционе ` +
      `и как тратить выигранные сертификаты.</p>` +
      `<div class="os-fields">` +
        osField("osRounds", "Раундов SoloQ", 10, 1, 30, 1) +
        osField("osSlots", "Слотов аукциона", 6, 0, 20, 1) +
        osField("osSlotS", "Шагов на слот", 1000000, 800000, null, 100000) +
        osField("osRookN", "Новичков (форк)", 0, 0, 12, 1) +
        osField("osRookS", "Шагов на новичка", 1000000, 800000, null, 100000) +
        osField("osFaN", "Свободных агентов", 0, 0, 10, 1) +
        osField("osFaS", "Шагов на агента", 800000, 800000, null, 100000) +
      `</div>` +
      `<div id="osReserves" class="os-reserves">${reserves}</div>` +
      `<div id="osEta" class="os-eta"></div>` +
      `<div class="os-actions"><button id="osRun" class="primary">Запустить межсезонье</button></div>` +
    `</div>`;
  ["osRounds", "osSlots", "osSlotS", "osRookN", "osRookS", "osFaN", "osFaS"].forEach((id) =>
    el(id).addEventListener("input", updateEta));
  el("osRun").addEventListener("click", runOffseasonInline);
  updateEta();
  // pull this machine's measured training rate (persisted from past runs) so the ETA isn't a guess
  if (state.worldId && !state._measuredSps) {
    fetchJson(`/api/world/${state.worldId}/training`)
      .then((d) => { if (d && d.measured_sps) { state._measuredSps = d.measured_sps; updateEta(); } })
      .catch(() => {});
  }
}
async function runOffseasonInline() {
  if (state.busy) return;
  const introduce_ids = [...el("osReserves").querySelectorAll("input:checked")].map((cb) => cb.dataset.pid);
  const settings = { ...osSettings(), introduce_ids };
  el("osRun").disabled = true;
  show("osvTabs", false);
  osProgress(4, "Запускаем межсезонье…");
  try {
    const job = await runJob(`/api/world/${state.worldId}/offseason`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    }, "Межсезонье — SoloQ, аукцион, трансферы");
    const s = job.summary || {};
    const { world } = await fetchJson(`/api/world/${state.worldId}`);   // reload incl. last_offseason
    state.world = world; state.emblems = world.emblems || {}; state.colors = world.colors || {};
    renderHeader(world);
    renderOsResults("overview");                                        // stay in the off-season view
    if ((s.chronicle || []).length) await playOsChronicle(s.chronicle);
    else osProgress(100, "Межсезонье завершено ✓");
  } catch (e) {
    osProgress(0, "");
    el("osvBody").innerHTML = `<div class="result-error">Ошибка межсезонья: ${esc(e.message)}</div>`;
  }
}
function playOsChronicle(chronicle) {
  return new Promise((resolve) => {
    const feed = document.getElementById("osChronFeed");
    if (!feed) { osProgress(100, "Межсезонье завершено ✓"); resolve(); return; }
    feed.innerHTML = "";
    const total = chronicle.length || 1;
    let i = 0;
    const step = () => {
      const live = document.getElementById("osChronFeed");
      if (!live || i >= chronicle.length) { osProgress(100, "Межсезонье завершено ✓"); resolve(); return; }
      const e = chronicle[i++];
      const row = document.createElement("div");
      row.className = "chron-row" + (e.source === "ЛЛМ" ? " llm" : "") + (e.kind === "fallback" ? " warn" : "");
      row.innerHTML = `<span class="cr-ic">${KIND_ICON[e.kind] || "•"}</span>` +
        `<span class="cr-msg">${esc(e.msg)}</span>` +
        (e.source ? `<span class="cr-src">${esc(e.source)}</span>` : "");
      live.appendChild(row);
      live.scrollTop = live.scrollHeight;
      osProgress(Math.round(i / total * 100), `${i}/${total} · ${(e.msg || "").slice(0, 64)}`);
      setTimeout(step, 170);
    };
    step();
  });
}

// --- results: per-topic interactive tabs --------------------------------------------------- //
function renderOsResults(tab) {
  const off = state.world && state.world.last_offseason;
  el("osvTitle").textContent = "⟳ Итоги межсезонья";
  el("osvSub").textContent = off
    ? `сезон ${off.season_ended}${off.next_season ? " → " + off.next_season : ""}` : "";
  if (!off || !off.transactions) {
    show("osvTabs", false); show("osvProgress", false);
    el("osvBody").innerHTML = `<div class="muted">Межсезонье ещё не проводилось — здесь появятся аукционы, ` +
      `трансферы, тренировки, SoloQ и плейбуки.</div>`;
    return;
  }
  state.osTab = OSV_TABS.some((t) => t.key === tab) ? tab : "overview";
  show("osvProgress", false);
  show("osvTabs", true);
  el("osvTabs").innerHTML = OSV_TABS.map((t) =>
    `<button class="osv-tab${t.key === state.osTab ? " is-active" : ""}" data-os="${t.key}">${t.label}</button>`).join("");
  el("osvTabs").querySelectorAll(".osv-tab").forEach((b) =>
    b.addEventListener("click", () => renderOsResults(b.dataset.os)));
  el("osvBody").innerHTML = osTabBody(state.osTab, off);
  if (state.osTab === "auction") wireOsAuction();
  if (!state.osvClickWired) {                 // one delegated handler: click a player/team → its page
    state.osvClickWired = true;
    el("osvBody").addEventListener("click", (e) => {
      const tm = e.target.closest("[data-os-team]");
      if (tm && tm.dataset.osTeam) { closeOffseason(); openTeam(tm.dataset.osTeam); return; }  // team page lives in the main view
      const pl = e.target.closest("[data-os-pid]");
      if (pl && pl.dataset.osPid) openPlayer(pl.dataset.osPid);    // player modal floats over the OS view
    });
  }
}
function osTabBody(tab, off) {
  if (tab === "overview") return osOverviewTab(off);
  if (tab === "soloq") return osSoloqTab(off);
  if (tab === "skills") return osSkillsTab(off);
  if (tab === "auction") return osAuctionTab(off);
  if (tab === "training") return osTrainingTab(off);
  if (tab === "transfers") return offseasonTxHtml(off, { full: false });
  if (tab === "money") return osMoneyTab(off);
  if (tab === "draft") return osDraftTab(off);
  if (tab === "playbooks") return osPlaybooksTab(off);
  return "";
}
function osMoneyTab(off) {
  const m = off.money || {}; const rows = m.rows || [];
  if (!rows.length) return `<div class="muted">Нет данных по деньгам.</div>`;
  // money is yellow (.money); the Δ now spans the WHOLE period and INCLUDES prizes − salaries − spends,
  // so treasury_after = treasury_before + Δ reconciles exactly (the owner can finally track prize money).
  const yel = (n, pre) => n ? `<span class="money">${pre}${n}</span>` : `<span class="muted">—</span>`;
  const body = rows.map((r) => {
    const prize = r.prize || 0, salary = r.salary || 0, net = r.net != null ? r.net : 0;
    const tr = (r.transfer_in ? "+" + r.transfer_in : "") + (r.transfer_out ? " −" + r.transfer_out : "");
    return `<tr class="os-clk" data-os-team="${esc(r.team_id)}"><td class="mn-tm">${osTeamTag(r.team_id, r.team)}</td>` +
      `<td class="mn-tr"><b class="money">${r.treasury_after}</b></td>` +
      `<td>${yel(prize, "+")}</td>` +
      `<td>${salary ? `<span class="money-out">−${salary}</span>` : '<span class="muted">—</span>'}</td>` +
      `<td class="muted">${r.auction_spend ? "−" + r.auction_spend : "—"}</td>` +
      `<td>${tr || "—"}</td><td class="muted">${r.signing_spend ? "−" + r.signing_spend : "—"}</td>` +
      `<td class="mn-net ${net >= 0 ? "up" : "down"}">${net > 0 ? "+" : ""}${net}</td></tr>`;
  }).join("");
  return `<p class="muted">Полный денежный поток за период: <b>казна</b> = казна_до + Δ, где Δ = ` +
    `🏆 призовые − 💸 зарплаты − 🏷️ аукцион ± 🔁 трансферы − ✍️ подписания. Клик по клубу — его страница.</p>` +
    `<table class="ltable mn-tbl"><thead><tr><th>Клуб</th><th>Казна</th>` +
    `<th title="Призовые за сезон (место + бонус чемпиона)">🏆</th>` +
    `<th title="Зарплаты состава за сезон">💸</th><th title="Аукцион тренировок">🏷️</th>` +
    `<th title="Трансферы: + продал / − купил">🔁</th><th title="Подписания (бонусы)">✍️</th>` +
    `<th title="Δ казны за весь период (с призовыми)">Δ</th></tr></thead><tbody>${body}</tbody></table>`;
}
function osSkillsTab(off) {
  const sk = off.skills || [];
  if (!sk.length) return `<div class="muted">В это межсезонье никто не тренировался — характеристики карточек ` +
    `без изменений. После реальных тренировок здесь появятся переоценки (новички получают карточку, ` +
    `опытные обновляют — всё в абсолютно равных условиях).</div>`;
  return `<p class="muted">Карточки тренированных/новых игроков переоценены в равных условиях. Клик — страница игрока.</p>` +
    `<div class="osk-list">${sk.map((s) => {
      const chips = Object.entries(s.changes || {}).map(([k, d]) =>
        `<span class="osk-chg ${d >= 0 ? "up" : "down"}">${esc(ATTR_LABELS[k] || k)} ${d >= 0 ? "+" : ""}${d}</span>`).join("");
      return `<div class="osk-row os-clk" data-os-pid="${esc(s.player_id)}">` +
        `<div class="osk-hd">${factionChip(s.faction)}<span class="osk-nm">${esc(s.player)}</span></div>` +
        `<div class="osk-radar">${radarChart(s.after, 140)}</div>` +
        `<div class="osk-chgs">${chips || '<span class="muted">—</span>'}</div></div>`;
    }).join("")}</div>`;
}
function osStat(n, label) { return `<div class="osv-stat"><b>${n}</b><span>${esc(label)}</span></div>`; }
function osOverviewTab(off) {
  const c = off.transactions.counts || {};
  const sq = off.soloq || {};
  const pv = (off.training && off.training.preview) || {};
  return `<div class="osv-stats">` +
      osStat(c.auctions || 0, "слотов аукциона") +
      osStat(pv.n_requests || 0, "заявок на тренировку") +
      osStat(c.signings || 0, "подписаний") +
      osStat(c.buys || 0, "сделок клубов") +
      osStat(c.releases || 0, "отпущено") +
      osStat(sq.n_moved || 0, "сдвигов SoloQ") +
    `</div>` +
    `<h4 class="osv-h">Хроника межсезонья</h4>` +
    `<div id="osChronFeed" class="osv-chron"></div>`;
}
function osSoloqTab(off) {
  const sq = off.soloq || {}; const empty = `<div class="muted">—</div>`;
  const tbl = (rows) => !rows || !rows.length ? empty :
    `<table class="ltable osv-tbl"><tbody>${rows.map((r) =>
      `<tr class="os-clk" data-os-pid="${esc(r.player_id || "")}"><td>${factionChip(r.faction)}</td><td class="osv-pl">${esc(r.player)}</td>` +
      `<td class="osv-rt">${r.rating}</td>` +
      `<td class="osv-dl ${r.delta >= 0 ? "up" : "down"}">${r.delta >= 0 ? "▲ +" : "▼ "}${Math.round(r.delta)}</td></tr>`).join("")}</tbody></table>`;
  return `<p class="muted">SoloQ обновил рейтинги ${sq.n_moved || 0} игроков. Топ движений:</p>` +
    `<div class="osv-two"><div><h4 class="osv-h">📈 Выросли</h4>${tbl(sq.gainers)}</div>` +
    `<div><h4 class="osv-h">📉 Упали</h4>${tbl(sq.losers)}</div></div>`;
}
function osAuctionTab(off) {
  const aucs = (off.transactions.auctions) || [];
  if (!aucs.length) return `<div class="muted">Аукцион тренировок в этом межсезонье не проводился.</div>`;
  return `<p class="muted">Сертификаты на тренировку разыгрываются по одному (запечатанная макс-ставка). ` +
    `Нажми на слот, чтобы увидеть кто сколько предлагал.</p>` +
    `<div class="osa-list">${aucs.map((a, i) =>
      `<div class="osa-slot" data-i="${i}"><div class="osa-row">` +
        `<span class="osa-n">Слот ${i + 1}</span>${osTeamTag(a.team_id, a.team)}` +
        `<span class="osa-win">выиграл за <b>${a.price}</b>${a.steps ? " · " + (a.steps / 1e6).toFixed(1) + "M ш" : ""}</span>` +
        `<span class="osa-toggle">▼ ставки (${(a.bids || []).length})</span></div>` +
        `<div class="osa-bids is-hidden">${(a.bids || []).length ? a.bids.map((b) =>
          `<div class="osa-bid${b.team_id === a.team_id ? " win" : ""}">${osTeamTag(b.team_id, b.team)}` +
          `<span class="osa-amt">${b.amount}</span></div>`).join("") : `<div class="muted">нет ставок</div>`}</div>` +
      `</div>`).join("")}</div>`;
}
function wireOsAuction() {
  el("osvBody").querySelectorAll(".osa-slot").forEach((slot) => {
    const row = slot.querySelector(".osa-row");
    row.addEventListener("click", () => {
      slot.querySelector(".osa-bids").classList.toggle("is-hidden");
      slot.classList.toggle("open");
    });
  });
}
function osTrainingTab(off) {
  const pv = (off.training && off.training.preview) || {};
  const rows = pv.requests || [];
  const kindLbl = { team: "команда", rookie: "новичок", free_agent: "св. агент" };
  if (!rows.length) return `<div class="muted">Никаких тренировок не заказано (тренеры решили копить сертификаты).</div>`;
  const by = pv.by_kind || {};
  const head = Object.keys(by).map((k) => `${kindLbl[k] || k}: ${by[k].n}`).join(" · ");
  return `<p class="muted">Один общий запуск — ${pv.n_requests || rows.length} заявок (${head}); ` +
    `всего ${((pv.total_steps || 0) / 1e6).toFixed(1)}M шагов ≈ ${pv.total_minutes || "?"} мин (серийно, в фоне).</p>` +
    `<div class="tx-list">${rows.map((r) =>
      `<div class="tx-row"><span class="osv-kind">${esc(kindLbl[r.kind] || r.kind)}</span>` +
      `${r.team_id ? osTeamTag(r.team_id, r.team_id) : `<span class="tx-tm">${esc(r.kind)}</span>`}` +
      `<span class="tx-mid">→ ${esc(r.title)}</span>` +
      `<span class="tx-steps">${((r.steps || 0) / 1e6).toFixed(1)}M ш${r.below_floor ? " ⚠️" : ""}</span></div>`).join("")}</div>`;
}
function osDraftTab(off) {
  const rookies = (off.transactions.signings || []).filter((s) => s.kind === "rookie");
  if (!rookies.length) return `<div class="muted">Новичков в этом межсезонье не подписывали.</div>`;
  return `<p class="muted">Драфт новичков — нижняя лига выбирает проспектов (слабейшие первыми).</p>` +
    `<div class="tx-list">${rookies.map((s) =>
      `<div class="tx-row">${osTeamTag(s.team_id, s.team)}<span class="tx-mid">задрафтил</span>` +
      `${osPlayerTag(s.player_id, s.player)}` +
      `<span class="tx-price">${s.bonus != null ? "бонус " + s.bonus : ""}</span></div>`).join("")}</div>`;
}
function osPlaybooksTab(off) {
  const plans = off.planning || [];
  if (!plans.length) return `<div class="muted">Плейбуки ещё не заданы.</div>`;
  return `<p class="muted">Каждый менеджер задал базовую стратегию на следующий сезон.</p>` +
    `<div class="tx-list">${plans.map((p) =>
      `<div class="tx-row">${osTeamTag(p.team_id, p.team)}<span class="tx-mid">базовый план</span>` +
      `<span class="tx-who">${esc(STRAT_LABEL[p.base] || p.base || "—")}</span></div>`).join("")}</div>`;
}

async function refreshWorldList(selectId) {
  try {
    const info = await fetchJson("/api/world/info");
    const sel = el("worldSelect");
    const worlds = info.worlds || [];
    sel.innerHTML = worlds.length
      ? worlds.map((w) => `<option value="${esc(w)}">${esc(w)}</option>`).join("")
      : `<option value="">— нет миров —</option>`;
    if (selectId) sel.value = selectId;
    el("openBtn").disabled = !worlds.length;
  } catch (e) { /* leave as-is */ }
}

function applyLock(ui) {
  state.locked = !!(ui && ui.locked);
  state.lockNote = (ui && ui.note) || "";
  show("lockBanner", state.locked);
  if (state.locked) {
    el("lockNote").textContent = state.lockNote ||
      "Выбор команд и старт пока заблокированы — включаются одним флагом.";
    document.body.classList.add("world-locked");
  } else document.body.classList.remove("world-locked");
}

function wireEvents() {
  el("startBtn").addEventListener("click", openSelModal);
  el("selCreate").addEventListener("click", createWithSelection);
  el("selCancel").addEventListener("click", () => show("selModal", false));
  el("selX").addEventListener("input", updateSelHint);
  el("advanceBtn").addEventListener("click", advance);
  el("liveStartBtn").addEventListener("click", startLive);
  el("nextDayBtn").addEventListener("click", playNextDay);
  el("offseasonBtn").addEventListener("click", openOffseason);   // inline off-season view (not a modal)
  el("osvBack").addEventListener("click", closeOffseason);
  el("plClose").addEventListener("click", () => show("playerModal", false));
  el("mtClose").addEventListener("click", () => show("matchModal", false));
  el("openBtn").addEventListener("click", () => {
    const wid = el("worldSelect").value;
    if (wid) openWorld(wid);
  });
  el("newBtn").addEventListener("click", () => {
    show("worldView", false);
    show("resultPanel", false);
    show("newSection", true);
  });
  document.querySelectorAll("#tabbar .tab").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
}

// --------------------------------------------------------------------------- //
// Theme switcher — three looks (Neon / Aurora / Vibrant), persisted; pure CSS-var
// swap on <html data-theme>, so functionality is untouched.
// --------------------------------------------------------------------------- //
const THEMES = [
  { id: "neon", label: "Neon" },
  { id: "aurora", label: "Aurora" },
  { id: "vibrant", label: "Vibrant" },
];
function applyTheme(id) {
  if (!THEMES.some((t) => t.id === id)) id = "aurora";
  document.documentElement.dataset.theme = id;
  try { localStorage.setItem("worldTheme", id); } catch (e) { /* ignore */ }
  document.querySelectorAll(".theme-switch__opt").forEach((b) =>
    b.classList.toggle("is-active", b.dataset.set === id));
  const lbl = document.querySelector(".theme-switch__lbl");
  if (lbl) lbl.textContent = (THEMES.find((t) => t.id === id) || {}).label || id;
}
function setupThemeSwitch() {
  let saved = "aurora";
  try { saved = localStorage.getItem("worldTheme") || "aurora"; } catch (e) { /* ignore */ }
  document.documentElement.dataset.theme = THEMES.some((t) => t.id === saved) ? saved : "aurora";
  if (!document.body || document.querySelector(".theme-switch")) return;
  const box = document.createElement("div");
  box.className = "theme-switch";
  box.setAttribute("role", "group");
  box.setAttribute("aria-label", "Тема оформления");
  box.innerHTML = `<span class="theme-switch__lbl"></span>` +
    THEMES.map((t) => `<button class="theme-switch__opt" data-set="${t.id}" title="${t.label}" aria-label="Тема ${t.label}"></button>`).join("");
  document.body.appendChild(box);
  box.querySelectorAll(".theme-switch__opt").forEach((b) =>
    b.addEventListener("click", () => applyTheme(b.dataset.set)));
  applyTheme(document.documentElement.dataset.theme);
}

// --------------------------------------------------------------------------- //
// ✨ Ambient layer + premium micro-interactions (decorative; no functional markup
// touched — injected separately, pointer-events:none where relevant).
// --------------------------------------------------------------------------- //
let _orbRAF = null;
function setupAmbient() {
  if (!document.body || document.querySelector(".world-ambient")) return;
  const amb = document.createElement("div");
  amb.className = "world-ambient";
  amb.setAttribute("aria-hidden", "true");
  amb.innerHTML = ["orb-1", "orb-2", "orb-3", "orb-4", "orb-5"]
    .map((c) => `<span class="orb ${c}"></span>`).join("") + `<span class="world-grid"></span>`;
  document.body.prepend(amb);
  startOrbDrift();
}
// JS-driven drift (requestAnimationFrame) — GUARANTEED to move, immune to mix-blend compositing quirks
// and to prefers-reduced-motion CSS freezes (the owner explicitly wants the live wallpaper).
function startOrbDrift() {
  if (_orbRAF) return;
  const orbs = Array.prototype.slice.call(document.querySelectorAll(".world-ambient .orb"));
  if (!orbs.length) return;
  const P = [                                     // per-orb: amplitude px + angular speed + phase + pulse
    { ax: 180, ay: 140, sx: 0.16, sy: 0.13, ph: 0.0, ss: 0.10 },
    { ax: 150, ay: 165, sx: 0.13, sy: 0.18, ph: 1.7, ss: 0.12 },
    { ax: 200, ay: 120, sx: 0.11, sy: 0.15, ph: 3.0, ss: 0.09 },
    { ax: 165, ay: 155, sx: 0.15, sy: 0.12, ph: 4.5, ss: 0.11 },
    { ax: 145, ay: 150, sx: 0.12, sy: 0.16, ph: 2.2, ss: 0.13 },
  ];
  const t0 = performance.now();
  function frame(now) {
    const t = (now - t0) / 1000;
    for (let i = 0; i < orbs.length; i++) {
      const p = P[i % P.length];
      const x = Math.sin(t * p.sx + p.ph) * p.ax;
      const y = Math.cos(t * p.sy + p.ph) * p.ay;
      const s = 1 + 0.13 * Math.sin(t * p.ss + p.ph);
      orbs[i].style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) scale(${s.toFixed(3)})`;
    }
    _orbRAF = requestAnimationFrame(frame);
  }
  _orbRAF = requestAnimationFrame(frame);
}
// sliding "magic line" under the active tab
function moveTabInk() {
  const bar = document.getElementById("tabbar");
  if (!bar || !bar.offsetWidth) return;
  let ink = bar.querySelector(".tabbar__ink");
  if (!ink) { ink = document.createElement("span"); ink.className = "tabbar__ink"; bar.appendChild(ink); }
  const active = bar.querySelector(".tab.is-active");
  if (!active) { ink.style.opacity = "0"; return; }
  const br = bar.getBoundingClientRect(), ar = active.getBoundingClientRect();
  ink.style.opacity = "1";
  ink.style.width = ar.width + "px";
  // position under the active tab's OWN bottom edge (so a wrapped 2-row tab bar shows ONE line, under
  // the active tab — not a second line at the bottom of the whole bar)
  ink.style.transform = `translate(${ar.left - br.left}px, ${ar.bottom - br.top - 3}px)`;
}
function setupWow() {
  setupAmbient();
  // keep the tab ink glued to the active tab across switches / resizes / show-hide
  const bar = document.getElementById("tabbar");
  if (bar && window.ResizeObserver) { try { new ResizeObserver(() => moveTabInk()).observe(bar); } catch (e) { /* ignore */ } }
  window.addEventListener("resize", () => moveTabInk());
  // tactile ripple on the main controls (skip .tab — its sliding ink is the feedback there)
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button.primary, button.ghost, .seg-btn");
    if (!b || b.disabled) return;
    const rect = b.getBoundingClientRect(), d = Math.max(rect.width, rect.height);
    const r = document.createElement("span");
    r.className = "wm-ripple";
    r.style.width = r.style.height = d + "px";
    r.style.left = (e.clientX - rect.left - d / 2) + "px";
    r.style.top = (e.clientY - rect.top - d / 2) + "px";
    b.appendChild(r);
    setTimeout(() => r.remove(), 620);
  });
}

async function init() {
  setupThemeSwitch();
  setupWow();
  if (window.__PREVIEW__) {                 // hardcoded preview: jump straight into a ready world
    wireEvents();
    try { state.cards = (await fetchJson("/api/world/franchises")).franchises || []; } catch (e) { /* ignore */ }
    show("newSection", false);
    await openWorld("preview");
    return;
  }
  try {
    const info = await fetchJson("/api/world/info");
    applyLock(info.ui);
    const data = await fetchJson("/api/world/franchises");
    state.cards = data.franchises || [];
    await refreshWorldList(null);
  } catch (e) {
    el("teamGrid").innerHTML = `<div class="result-error">Не удалось загрузить: ${esc(e.message)}</div>`;
    return;
  }
  wireEvents();
  refreshPicker();
}

init();
