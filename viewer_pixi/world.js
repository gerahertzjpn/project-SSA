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
  if (!(id in els)) els[id] = document.getElementById(id);
  return els[id];
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
function sparkline(values, w = 88, h = 24) {
  const vals = (values || []).filter((v) => typeof v === "number");
  if (vals.length < 2) return `<span class="spark-empty">—</span>`;
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const pts = vals.map((v, i) =>
    `${(i / (vals.length - 1) * w).toFixed(1)},${(h - (v - min) / span * h).toFixed(1)}`).join(" ");
  const rising = vals[vals.length - 1] >= vals[0];
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
    if (name === "leagues") renderLeagues((await api("/leagues")).leagues);
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
  el("panel-leagues").innerHTML = seg +
    `<table class="ltable"><thead><tr>` +
      `<th>#</th><th>Аб.</th><th>Команда</th><th title="Победы–поражения в сериях">В–П</th>` +
      `<th title="Дельта выигранных карт">Δ</th><th title="Очки за победы в сериях">Очки</th>` +
      `<th>Power</th><th>Состав</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
  wireSeg();
  el("panel-leagues").querySelectorAll(".ltrow").forEach((r) =>
    r.addEventListener("click", () => openTeam(r.dataset.team)));
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
function bgCols(cols) {
  if (!cols || !cols.length) return `<div class="muted">Ещё не начался.</div>`;
  return `<div class="bgraph"><svg class="bg-svg" aria-hidden="true"></svg>${cols.map((col) =>
    `<div class="bg-col"><div class="bg-stage">${esc(col.label || col.stage)}</div>` +
    `<div class="bg-bouts">${(col.matches || []).map(bgNode).join("")}</div></div>`).join("")}</div>`;
}
function renderBracketGraph(bracket) {
  const up = bracket.upper || [], lo = bracket.lower || [];
  if (!up.length && !lo.length) return `<div class="muted" style="margin-top:14px">Плей-офф ещё не начался — он откроется, когда сыграется регулярка.</div>`;
  // connectors are drawn after the HTML is in the DOM (see drawBracketConnectors)
  setTimeout(drawBracketConnectors, 0);
  return `<div class="bg-block"><h4 class="bg-h">🏆 Верхняя лига</h4>${bgCols(up)}</div>` +
    `<div class="bg-block"><h4 class="bg-h">Нижняя лига <span class="muted">(2 повышаются)</span></h4>${bgCols(lo)}</div>`;
}
// Draw L-shaped connector lines from each feeder match to the one it flows into (read from
// data-key / data-from), measured from the rendered node positions — a real bracket tree.
function drawBracketConnectors() {
  document.querySelectorAll(".bgraph").forEach((g) => {
    const svg = g.querySelector(".bg-svg"); if (!svg) return;
    const gr = g.getBoundingClientRect();
    const nodes = {};
    g.querySelectorAll(".bg-node").forEach((n) => { if (n.dataset.key) nodes[n.dataset.key] = n; });
    let paths = "";
    g.querySelectorAll(".bg-node").forEach((n) => {
      const tr = n.getBoundingClientRect();
      const tx = tr.left - gr.left, ty = tr.top - gr.top + tr.height / 2;
      (n.dataset.from || "").split(",").filter(Boolean).forEach((fk) => {
        const f = nodes[fk]; if (!f) return;
        const fr = f.getBoundingClientRect();
        const fx = fr.right - gr.left, fy = fr.top - gr.top + fr.height / 2;
        const mx = fx + Math.max(12, (tx - fx) / 2);
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
        `<th class="mx-row"><span class="mx-pl">${t.place}</span>${emblem(t.team_id)}` +
          `<span class="mx-ab">${esc(t.abbrev)}</span></th>${cells}` +
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
        `<span class="sq-d ${g.delta >= 0 ? "up" : "down"}">${g.delta >= 0 ? "+" : ""}${g.delta}</span>` +
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
function renderPlayerPage(p) {
  const pos = Object.entries(p.position_stats || {}).map(([k, v]) =>
    `<span class="pp-pos">${k}: <b>${v.winrate != null ? Math.round(v.winrate * 100) + "%" : "—"}</b> <span class="muted">(${v.games}и)</span></span>`).join("");
  const games = (p.soloq_games || []).map((g) =>
    `<div class="sq-game ${g.won ? "w" : "l"}"><span class="sq-res">${g.won ? "W" : "L"}</span>` +
    `<span class="sq-d ${g.delta >= 0 ? "up" : "down"}">${g.delta >= 0 ? "+" : ""}${g.delta}</span>` +
    `<span class="sq-with">с: ${esc((g.teammates || []).join(", "))}</span>` +
    `<span class="sq-vs">против: ${esc((g.opponents || []).join(", "))}</span></div>`).join("");
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
    `</div>` +
    `<div class="pp-row">Рейтинг по сезонам ${sparkline(p.rating_history, 260, 44)}</div>` +
    `<div class="pp-row"><b>Позиции:</b> ${pos || "—"}</div>` +
    `<div class="pp-row"><b>Нейронка:</b> <code>${esc(p.net_id || "—")}</code>` +
      (p.lineage && p.lineage.parent ? ` · форк от ${esc(p.lineage.parent)} (G${p.lineage.generation})` : "") + `</div>` +
    `<div class="pp-row"><b>Трофеи:</b> ${troph}</div>` +
    `<h4>Клубная история</h4>${teamHist}` +
    (games ? `<h4>Последние игры SoloQ</h4><div class="pd-soloq">${games}</div>` : "");
}

// --------------------------------------------------------------------------- //
// Tab: Расписание + сетка плей-офф
// --------------------------------------------------------------------------- //
function matchCard(m) {
  const winH = m.played && m.winner === m.home, winA = m.played && m.winner === m.away;
  const side = (id, ab, sc, win) =>
    `<div class="mc-row${win ? " win" : ""}">${emblem(id)}<span class="mc-tm">${esc(ab)}</span>` +
    `<span class="mc-sc">${m.played ? sc : "·"}</span></div>`;
  const tag = m.day ? `<div class="mc-day">День ${m.day}${m.stage && m.stage !== "league" ? " · " + esc(m.stage_label || m.stage) : ""}</div>` : "";
  return `<div class="mcard${m.played ? " played" : " upcoming"}" title="Bo${m.best_of}">` + tag +
    side(m.home, m.home_abbrev, m.maps_home, winH) +
    side(m.away, m.away_abbrev, m.maps_away, winA) +
  `</div>`;
}
function renderSchedule(data) {
  const reg = (data.regular || []).slice().sort((a, b) => (a.day || 0) - (b.day || 0));
  const bracket = data.bracket || {};
  const byLg = {};
  reg.forEach((m) => (byLg[m.league] = byLg[m.league] || []).push(m));
  const played = reg.filter((m) => m.played).length;
  const note = reg.length
    ? `<div class="sched-note muted">Расписание регулярки — кто когда играет (1 день = 1 игра). ` +
      `Сыграно: <b>${played}</b> / ${reg.length}. Остальные — без результата, пока не сыграны.</div>` : "";
  const sect = (title, ms) => !ms || !ms.length ? "" :
    `<h4 class="sched-lg">${title}</h4><div class="sched-grid">${ms.map(matchCard).join("")}</div>`;
  const regHtml = note + (sect("Верхняя лига", byLg.upper) + sect("Нижняя лига", byLg.lower))
    || `<div class="muted">Регулярка ещё не сыграна.</div>`;
  el("panel-schedule").innerHTML =
    `<h3>Регулярный сезон</h3>${regHtml}` +
    `<h3>Плей-офф <span class="muted">(сетка)</span></h3>${renderBracketGraph(bracket)}`;
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
function matchLine(m) {
  const stage = m.stage && m.stage !== "league" ? esc(m.stage_label || m.stage) : "Регулярка";
  const venue = m.home ? "дома" : "в гостях";
  if (!m.played) {
    return `<div class="tpm upcoming"><span class="tpm-day">Д${m.day || "?"}</span>` +
      `<span class="tpm-st">${stage}</span><span class="tpm-vs">${m.home ? "vs" : "@"} ${emblem(m.opponent)}${esc(m.opponent_abbrev)}</span>` +
      `<span class="tpm-res muted">— · ${venue}</span></div>`;
  }
  const won = m.won;
  return `<div class="tpm ${won ? "w" : "l"}"><span class="tpm-day">Д${m.day || "?"}</span>` +
    `<span class="tpm-st">${stage}</span><span class="tpm-vs">${m.home ? "vs" : "@"} ${emblem(m.opponent)}${esc(m.opponent_abbrev)}</span>` +
    `<span class="tpm-res"><b>${won ? "W" : "L"}</b> ${m.maps_for}:${m.maps_against}</span></div>`;
}
function renderTeamPage(t) {
  const page = document.getElementById("teamPage");
  if (!page) return;
  const rec = t.record || {};
  const roster = (t.roster || []).map((p) =>
    `<div class="tp-player" data-pid="${esc(p.player_id)}">${factionChip(p.faction)}` +
      `<span class="tpp-nm">${esc(p.nickname)}</span>` +
      `<span class="tpp-ar">${esc(p.archetype || "")}</span>` +
      `<span class="tpp-rr">${p.rating} ${deltaBadge(p.delta)}</span>` +
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
          `<div class="tpm-list">${upcoming.map(matchLine).join("")}</div>`) : "") +
      `</div>` +
      `<div class="tp-col">` +
        sect(`Сыгранные матчи <span class="muted">(${played.length})</span>`,
          played.length ? `<div class="tpm-list">${played.map(matchLine).join("")}</div>` : `<div class="muted">Пока не сыграно.</div>`) +
        sect("Трансферы", transfers.length ? `<div class="tp-tx">${transfers.map((x) => `<div class="tp-tx-row">${txLine(x)}</div>`).join("")}</div>` : `<div class="muted">Без движений в последнее межсезонье.</div>`) +
        sect("История по сезонам", histRows ? `<div class="tph">${histRows}</div>` : `<div class="muted">Ещё не сыграно сезонов.</div>`) +
      `</div>` +
    `</div>`;
  page.querySelector(".tp-close").addEventListener("click", () => page.classList.add("is-hidden"));
  page.querySelectorAll(".tp-player").forEach((r) =>
    r.addEventListener("click", () => openPlayer(r.dataset.pid)));
}

// --------------------------------------------------------------------------- //
// Tab: История чемпионов
// --------------------------------------------------------------------------- //
function renderHistory(history) {
  history = history || [];
  el("panel-history").innerHTML = !history.length
    ? `<div class="muted">Ещё не сыграно ни одного сезона.</div>`
    : `<div class="history">${history.slice().reverse().map((s) =>
        `<div class="hist-row"><span class="hs">S${s.season}</span>` +
        `<span class="hc">🏆 ${esc(s.upper_champion)}</span>` +
        `<span class="hc lo">↑ ${esc(s.lower_champion)}</span></div>`).join("")}</div>`;
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
function renderOffseasonTx(off) {
  const panel = el("panel-transfers");
  if (!off || !off.transactions) {
    panel.innerHTML = `<div class="muted">Межсезонье ещё не проводилось — итоги трансферов появятся здесь.</div>`;
    return;
  }
  const tx = off.transactions;
  const empty = (txt) => `<div class="tx-empty muted">${txt}</div>`;
  const sec = (title, count, body) =>
    `<section class="tx-sec"><h3>${title} <span class="tx-n">${count}</span></h3>${body}</section>`;
  const teamTag = (id, name) => `${emblem(id)}<span class="tx-tm">${esc(name)}</span>`;

  const auctions = (tx.auctions || []).length
    ? `<div class="tx-list">${tx.auctions.map((a) =>
        `<div class="tx-row">${teamTag(a.team_id, a.team)}<span class="tx-mid">выиграл слот тренировки</span>` +
        `<span class="tx-price">${a.price}</span></div>`).join("")}</div>`
    : empty("Аукцион тренировок не проводился.");

  const trains = (tx.training_requests || []).length
    ? `<div class="tx-list">${tx.training_requests.map((t) =>
        `<div class="tx-row">${teamTag(t.team_id, t.team)}<span class="tx-mid">→ ${esc(t.op_title)}</span>` +
        `<span class="tx-who">${esc(t.player)}</span>` +
        `<span class="tx-steps">${t.steps ? (t.steps / 1e6).toFixed(1) + "M ш" : ""}</span></div>`).join("")}</div>`
    : empty("Никто не отправлен на тренировку.");

  const signings = (tx.signings || []).length
    ? `<div class="tx-list">${tx.signings.map((s) =>
        `<div class="tx-row">${teamTag(s.team_id, s.team)}<span class="tx-mid">подписал</span>` +
        `<span class="tx-who">${esc(s.player)} <span class="tx-kind">${s.kind === "rookie" ? "новичок" : "св. агент"}</span></span>` +
        `<span class="tx-price">${s.bonus != null ? "бонус " + s.bonus : ""}</span></div>`).join("")}</div>`
    : empty("Подписаний не было.");

  const buys = (tx.buys || []).length
    ? `<div class="tx-list">${tx.buys.map((b) =>
        `<div class="tx-row">${teamTag(b.to_team_id, b.to_team)}<span class="tx-mid">купил</span>` +
        `<span class="tx-who">${esc(b.player)}</span><span class="tx-from">у ${esc(b.from_team)}</span>` +
        `<span class="tx-price">${b.price}</span></div>`).join("")}</div>`
    : empty("Сделок между клубами не было.");

  const releases = (tx.releases || []).length
    ? `<div class="tx-list">${tx.releases.map((r) =>
        `<div class="tx-row">${teamTag(r.team_id, r.team)}<span class="tx-mid">отпустил</span>` +
        `<span class="tx-who">${esc(r.player)}</span><span class="tx-from">→ свободный агент</span></div>`).join("")}</div>`
    : empty("Никого не отпустили.");

  const requests = (tx.requests || []).length
    ? `<div class="tx-list">${tx.requests.map((q) =>
        `<div class="tx-row ${q.won ? "tx-won" : ""}">${teamTag(q.team_id, q.team)}<span class="tx-mid">запросил</span>` +
        `<span class="tx-who">${esc(q.player)}</span><span class="tx-price">${q.amount != null ? q.amount : ""}</span>` +
        `<span class="tx-flag">${q.won ? "✓ выиграл" : "—"}</span></div>`).join("")}</div>`
    : empty("Запросов не поступало.");

  panel.innerHTML =
    `<div class="tx-head">Итоги межсезонья <span class="muted">после сезона ${off.season_ended}</span></div>` +
    sec("🏷️ Аукцион тренировок", tx.counts.auctions, auctions) +
    sec("💪 На тренировку", tx.counts.training_requests, trains) +
    sec("✍️ Подписания", tx.counts.signings, signings) +
    sec("💱 Покупки между клубами", tx.counts.buys, buys) +
    sec("✂️ Отпущены (кики)", tx.counts.releases, releases) +
    sec("📨 Запросы (ставки)", tx.counts.requests, requests);
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
  let jobs = [];
  try { jobs = (await fetchJson(`/api/world/${state.worldId}/training`)).jobs || []; }
  catch (e) { return; }
  if (!jobs.length) { show("trainingPanel", false); return; }
  show("trainingPanel", true);
  const running = jobs.filter((j) => j.status === "running").length;
  const queued = jobs.filter((j) => j.status === "queued").length;
  const head = running ? `${running} идёт${queued ? ", " + queued + " в очереди" : ""}`
    : queued ? `${queued} в очереди` : "завершены";
  el("trainingPanel").innerHTML =
    `<h3>🏋️ Тренировки нейронок <span class="muted">${head}</span></h3>` +
    jobs.map((j) => {
      const pct = Math.min(100, Math.round((j.done / Math.max(1, j.total)) * 100));
      const tag = j.status === "done" ? "✓ карточка пересчитана"
        : j.status === "error" ? "ошибка"
        : j.status === "queued" ? "в очереди" : `${pct}%`;
      return `<div class="train-row"><span class="tn">${esc(j.player)}</span>` +
        `<span class="tnet">${esc(OP_LABEL[j.op] || j.op || "")}</span>` +
        `<div class="train-bar"><div class="train-fill${j.status === "done" ? " done" : ""}" style="width:${j.status === "done" ? 100 : pct}%"></div></div>` +
        `<span class="tp">${tag}</span></div>`;
    }).join("");
  if (!(running + queued) && trainTimer) { clearInterval(trainTimer); trainTimer = null; }
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
  el("offseasonBtn").disabled = state.busy || !due;
  el("offseasonBtn").classList.toggle("required", due && !state.busy);
  el("advanceBtn").title = due ? "Сначала проведите межсезонье" : "";
}
async function openWorld(worldId) {
  try {
    const { world } = await fetchJson(`/api/world/${worldId}`);
    state.worldId = world.world_id;
    state.world = world;
    state.emblems = world.emblems || {};
    show("newSection", false);
    show("resultPanel", false);
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
  const x = Math.max(parseInt(el("selX").value || "0", 10) || 0, named);
  el("selNamed").textContent = `отмечено: ${named}`;
  el("selHint").textContent = x > named ? `+ ${x - named} случайно из остальных` :
    (x ? "ровно отмеченные" : "");
}
async function createWithSelection() {
  show("selModal", false);
  refreshPicker();
  try {
    const x = Math.max(parseInt(el("selX").value || "90", 10) || 90, state.selNamed.size);
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

async function watchMatch(home, away) {  // replay one upcoming match map-by-map (no scrub)
  if (state.busy) return;
  state.busy = true; renderHeader(state.world);
  const bc = document.getElementById("liveBroadcast");
  try {
    const { result, live } = await apiPost("/live/watch", { home, away });
    if (bc) await revealMatch(bc, result);
    state.world.live = live;
    renderLive(live);                            // keeps the finished broadcast (state.lastBroadcast)
  } catch (e) {
    if (bc) bc.innerHTML = `<div class="result-error">${esc(e.message)}</div>`;
  } finally { state.busy = false; renderHeader(state.world); }
}

async function revealMatch(bc, r) {
  const seq = r.sequence || [];
  const ab = (id) => id === r.home ? r.home_abbrev : r.away_abbrev;
  bc.innerHTML =
    `<div class="bcast"><div class="bcast-head">📺 Прямой эфир · <b>${esc(r.home_abbrev)}</b> vs <b>${esc(r.away_abbrev)}</b>` +
    ` <span class="muted">Bo${r.best_of} · без перемотки</span></div>` +
    `<div class="bcast-score"><span class="bs-h">0</span><span class="bcast-vs">:</span><span class="bs-a">0</span></div>` +
    `<div class="bcast-maps"></div></div>`;
  const card = bc.querySelector(".bcast");
  let h = 0, a = 0;
  for (let i = 0; i < seq.length; i++) {
    await sleep(i === 0 ? 600 : 1900);           // ~2s between maps in a series (no scrub)
    const homeWon = seq[i] === r.home;
    homeWon ? h++ : a++;
    card.querySelector(".bs-h").textContent = h;
    card.querySelector(".bs-a").textContent = a;
    card.querySelector(".bcast-maps").innerHTML +=
      `<span class="bc-map ${homeWon ? "h" : "a"}">Карта ${i + 1}: ${esc(ab(seq[i]))}</span>`;
  }
  await sleep(700);
  const win = r.winner === r.home ? r.home_abbrev : r.away_abbrev;
  card.innerHTML += `<div class="bcast-final">🏆 ${esc(win)} — ${r.maps_home}:${r.maps_away}</div>`;
  state.lastBroadcast = bc.innerHTML;
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
function renderLive(live) {
  const panel = el("panel-live");
  if (!live || (!live.active && !live.done)) {
    panel.innerHTML = `<div class="lv-empty"><p class="muted">Сезон по дням ещё не запущен.</p>` +
      `<p>Нажми <b>«📅 По дням»</b> вверху — мы сыграем весь сезон нейросетями, а потом раскроем его ` +
      `день за днём, без подсказок о будущем. Каждый матч можно открыть в <b>прямом эфире</b>.</p></div>`;
    return;
  }
  const pct = Math.min(100, Math.round(live.cursor / live.total_days * 100));
  const m = live.match;
  const jp = (live.just_played || [])[0];
  const just = jp ? `<div class="lv-just"><span class="muted">Только что:</span> ` +
    `<b>${esc(jp.home_abbrev)}</b> <span class="lv-just-sc">${jp.maps_home}:${jp.maps_away}</span> <b>${esc(jp.away_abbrev)}</b>` +
    `<span class="lv-just-w">🏆 ${esc(jp.winner === jp.home ? jp.home_abbrev : jp.away_abbrev)}</span></div>` : "";

  const card = m ? (
    `<div class="lvm-card">` +
      `<div class="lvm-ctx"><span class="lvm-stage${m.stage === "grand_final" ? " gf" : ""}">${esc(m.stage_label)}</span>` +
      `<span class="lvm-bo">Bo${m.best_of}</span><span class="lvm-day">День ${m.day} / ${live.total_days}</span></div>` +
      `<div class="lvm-vs">${lvTeamSide(m.home, "home")}` +
        `<div class="lvm-mid"><div class="lvm-vs-big">VS</div>` +
        `<button class="primary lvm-watch" data-h="${esc(m.home.team_id)}" data-a="${esc(m.away.team_id)}">📺 Смотреть эфир</button>` +
        `<div class="lvm-hint muted">реальный матч, по картам</div></div>` +
        `${lvTeamSide(m.away, "away")}</div>` +
    `</div>`) : `<div class="lv-done">✅ Сезон раскрыт полностью — открой «⟳ Межсезонье», чтобы продолжить.</div>`;

  panel.innerHTML =
    `<div class="lv-bar"><div class="lv-day">🔴 День <b>${live.day}</b> / ${live.total_days}</div>` +
    `<div class="lv-prog"><div class="lv-prog-fill" style="width:${pct}%"></div></div>` +
    (live.active ? `<button id="lvNext" class="primary">▶ Следующий день</button>` : "") + `</div>` +
    just +
    `<div id="liveBroadcast" class="lv-broadcast">${state.lastBroadcast || ""}</div>` +
    card +
    `<p class="lv-foot muted">Таблицы, сетка и расписание обновляются по ходу — на вкладках «Лиги», «Очные», «Расписание».</p>`;

  const nx = document.getElementById("lvNext");
  if (nx) nx.addEventListener("click", playNextDay);
  panel.querySelectorAll(".lvm-watch").forEach((b) =>
    b.addEventListener("click", () => watchMatch(b.dataset.h, b.dataset.a)));
  panel.querySelectorAll(".lvm-team").forEach((tdiv) =>
    tdiv.addEventListener("click", (e) => { if (e.target.closest(".lvm-watch")) return; openTeam(tdiv.dataset.team); }));
}

// Off-season settings + ETA -------------------------------------------------- //
const STEPS_PER_SEC = 300;   // ~26 воркеров (оценка); серийно (один тяжёлый запуск за раз)
function fmtMin(min) {
  if (min < 1) return "<1 мин";
  if (min < 90) return `≈ ${Math.round(min)} мин`;
  return `≈ ${(min / 60).toFixed(1)} ч`;
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
  const min = totalSteps / STEPS_PER_SEC / 60;
  el("osEta").innerHTML =
    `⏱ Тренировки нейронок: до <b>${jobs}</b> запусков × ${s.auction.steps_per_slot.toLocaleString("ru")} шагов ` +
    `<b>${fmtMin(min)}</b> <span class="muted">(серийно, в фоне после межсезонья — можно следить в трекере)</span>` +
    (s.rookies.n ? `<br>👶 ${s.rookies.n} новичков (форк) — карточки создаются сразу` : "") +
    `<br><span class="muted">Тренера сами решают: тратить выигранный сертификат сейчас или <b>копить</b> его. ` +
    `Запустятся только заказанные тренировки (может быть и 0, и сильно больше — если копили). ` +
    `Лимитов на число нет, единственный предел — 800k шагов на тренировку.</span>`;
}
function openOsModal() {
  if (!state.worldId || state.busy) return;
  const res = (state.world && state.world.reserves) || [];
  el("osReserves").innerHTML = !res.length ? "" :
    `<div class="osr-head">🎒 Ввести из резерва <span class="muted">(${res.length} — только в межсезонье)</span></div>` +
    `<div class="osr-list">${res.map((p) =>
      `<label class="osr-row"><input type="checkbox" data-pid="${esc(p.player_id)}">` +
      `${factionChip(p.faction)}<span class="osr-nm">${esc(p.nickname)}</span>` +
      `<span class="muted">${esc(p.archetype || "")} · ${p.rating}</span></label>`).join("")}</div>`;
  updateEta();
  show("osModal", true);
}
async function submitOffseason() {
  const introduce_ids = [...el("osReserves").querySelectorAll("input:checked")].map((cb) => cb.dataset.pid);
  show("osModal", false);
  el("offseasonBtn").disabled = true;
  try {
    const job = await runJob(`/api/world/${state.worldId}/offseason`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...osSettings(), introduce_ids }),
    }, "Межсезонье — SoloQ, аукцион, трансферы");
    const s = job.summary || {};
    flashResult(`<div class="season-done">Межсезонье закрыто → сезон <strong>${s.next_season}</strong>. ` +
      `Новичков: ${s.rookies || 0}. Аукцион: ${s.training_auctioned || 0} слотов, запущено тренировок: ${s.training_launched || 0}.</div>`);
    if ((s.chronicle || []).length) replayChronicle(s.chronicle);
    await openWorld(state.worldId);
  } catch (e) {
    flashResult(`<div class="result-error">Ошибка межсезонья: ${esc(e.message)}</div>`);
  } finally { el("offseasonBtn").disabled = state.busy; }
}

// Off-season chronicle replay (simulation of what happened) ------------------- //
const KIND_ICON = { stage: "🗓️", train: "🏋️", draft: "✍️", playbook: "📋", fallback: "⚠️",
  transfer: "🔁", training: "🏋️" };
function replayChronicle(chronicle) {
  const feed = el("chronFeed");
  feed.innerHTML = "";
  show("chronModal", true);
  let i = 0;
  const step = () => {
    if (i >= chronicle.length) return;
    const e = chronicle[i++];
    const row = document.createElement("div");
    row.className = "chron-row" + (e.source === "ЛЛМ" ? " llm" : "") + (e.kind === "fallback" ? " warn" : "");
    row.innerHTML = `<span class="cr-ic">${KIND_ICON[e.kind] || "•"}</span>` +
      `<span class="cr-msg">${esc(e.msg)}</span>` +
      (e.source ? `<span class="cr-src">${esc(e.source)}</span>` : "");
    feed.appendChild(row);
    feed.scrollTop = feed.scrollHeight;
    setTimeout(step, 220);
  };
  step();
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
  el("offseasonBtn").addEventListener("click", openOsModal);
  el("osRun").addEventListener("click", submitOffseason);
  el("osCancel").addEventListener("click", () => show("osModal", false));
  el("chronClose").addEventListener("click", () => show("chronModal", false));
  el("plClose").addEventListener("click", () => show("playerModal", false));
  ["osRounds", "osSlots", "osSlotS", "osRookN", "osRookS", "osFaN", "osFaS"].forEach((id) =>
    el(id).addEventListener("input", updateEta));
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

async function init() {
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
