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
function emblem(teamId, cls) {
  const svg = state.emblems && state.emblems[teamId];
  return `<span class="tlogo ${cls || ""}">${svg || ""}</span>`;
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
    emblem.innerHTML = c.emblem || "";
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
  ["leagues", "ratings", "schedule", "teams", "draft", "broadcast", "managers", "history"].forEach((t) =>
    show(`panel-${t}`, t === name));
  loadTab(name);
}
async function loadTab(name) {
  const panel = el(`panel-${name}`);
  if (name === "history") return renderHistory(state.world ? state.world.history : []);
  if (name === "broadcast") return renderBroadcast(state.world ? state.world.commentary : []);
  if (name === "managers") return renderManagers(state.world ? state.world.manager_comments : []);
  panel.innerHTML = `<div class="muted loading">Загрузка…</div>`;
  try {
    if (name === "leagues") renderLeagues((await api("/leagues")).leagues);
    else if (name === "ratings") { state.players = (await api("/players")).players; renderRatings(); }
    else if (name === "schedule") renderSchedule(await api("/schedule"));
    else if (name === "draft") renderDraft(await api("/draft"));
    else if (name === "teams") {
      renderTeams((await api("/leagues")).leagues);
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
];
function renderLeagues(leagues) {
  state._leagues = leagues;
  if (!LEAGUE_TABS.some((t) => t.key === state.leagueKey)) state.leagueKey = "upper";
  const rows = (leagues[state.leagueKey] || []).map((t) =>
    `<tr class="ltrow" data-team="${esc(t.team_id)}">` +
      `<td class="lt-place">${t.place}</td>` +
      `<td class="lt-ab">${emblem(t.team_id)}<span>${esc(t.abbrev)}</span></td>` +
      `<td class="lt-nm">${esc(t.name)}</td>` +
      `<td class="lt-roster">${(t.roster || []).map(esc).join(" · ")}</td>` +
      `<td class="lt-pw">${t.power}</td>` +
      `<td class="lt-ti">${"★".repeat(Math.min(5, t.titles || 0)) || "—"}</td>` +
    `</tr>`).join("");
  el("panel-leagues").innerHTML =
    `<div class="seg">${LEAGUE_TABS.map((t) =>
      `<button class="seg-btn${t.key === state.leagueKey ? " is-active" : ""}" data-lg="${t.key}">` +
      `${esc(t.label)}<small>${esc(t.sub)}</small></button>`).join("")}</div>` +
    `<table class="ltable"><thead><tr>` +
      `<th>#</th><th>Аб.</th><th>Команда</th><th>Состав</th><th>Power</th><th>Титулы</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
  el("panel-leagues").querySelectorAll(".seg-btn").forEach((b) =>
    b.addEventListener("click", () => { state.leagueKey = b.dataset.lg; renderLeagues(state._leagues); }));
  el("panel-leagues").querySelectorAll(".ltrow").forEach((r) =>
    r.addEventListener("click", () => openTeam(r.dataset.team)));
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
    r.addEventListener("click", () => togglePlayerDetail(r)));
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
// Tab: Расписание + сетка плей-офф
// --------------------------------------------------------------------------- //
function matchCard(m) {
  const winH = m.played && m.winner === m.home, winA = m.played && m.winner === m.away;
  const side = (id, ab, sc, win) =>
    `<div class="mc-row${win ? " win" : ""}">${emblem(id)}<span class="mc-tm">${esc(ab)}</span>` +
    `<span class="mc-sc">${m.played ? sc : "·"}</span></div>`;
  return `<div class="mcard${m.played ? " played" : " upcoming"}" title="Bo${m.best_of}">` +
    side(m.home, m.home_abbrev, m.maps_home, winH) +
    side(m.away, m.away_abbrev, m.maps_away, winA) +
  `</div>`;
}
function renderSchedule(data) {
  const reg = data.regular || [];
  const bracket = data.bracket || {};
  const byLg = {};
  reg.forEach((m) => (byLg[m.league] = byLg[m.league] || []).push(m));
  const sect = (title, ms) => !ms || !ms.length ? "" :
    `<h4 class="sched-lg">${title}</h4><div class="sched-grid">${ms.map(matchCard).join("")}</div>`;
  const regHtml = (sect("Верхняя лига", byLg.upper) + sect("Нижняя лига", byLg.lower))
    || `<div class="muted">Регулярка ещё не сыграна.</div>`;
  const brHtml = Object.keys(bracket).length
    ? Object.entries(bracket).map(([lg, stages]) =>
        `<div class="bracket-block"><h4>${lg === "upper" ? "🏆 Плей-офф · Верхняя" : "Плей-офф · Нижняя"}</h4>` +
        `<div class="bracket">${stages.map((st) =>
          `<div class="br-col"><div class="br-stage">${esc(st.label)}</div>` +
          `<div class="br-matches">${st.matches.map(matchCard).join("")}</div></div>`).join("")}</div></div>`).join("")
    : `<div class="muted">Плей-офф ещё не сыгран.</div>`;
  el("panel-schedule").innerHTML =
    `<h3>Регулярный сезон</h3>${regHtml}<h3>Плей-офф <span class="muted">(сетка)</span></h3>${brHtml}`;
}

// --------------------------------------------------------------------------- //
// Tab: Команды + страница команды
// --------------------------------------------------------------------------- //
function renderTeams(leagues) {
  const group = (title, rows) => !rows.length ? "" :
    `<h3 class="tgroup-h">${title}</h3><div class="tcards">${rows.map((t) =>
      `<button class="tcard" data-team="${esc(t.team_id)}" style="--tc:${state._colors && state._colors[t.team_id] || "#3a4757"}">` +
        emblem(t.team_id, "lg") +
        `<div class="tc-body"><div class="tc-nm">${esc(t.name)} <span class="tc-ab">${esc(t.abbrev)}</span></div>` +
        `<div class="tc-pw">⚡ ${t.power} · ${"★".repeat(Math.min(5, t.titles || 0)) || "без титулов"}</div></div>` +
        `<span class="tc-go">→</span>` +
      `</button>`).join("")}</div>`;
  el("panel-teams").innerHTML =
    `<div id="teamPage" class="team-page is-hidden"></div>` +
    group("Верхняя лига", leagues.upper || []) +
    group("Нижняя · группа A", leagues.lower_A || []) +
    group("Нижняя · группа Z", leagues.lower_Z || []);
  el("panel-teams").querySelectorAll(".tcard").forEach((b) =>
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
function renderTeamPage(t) {
  const page = document.getElementById("teamPage");
  if (!page) return;
  const rec = t.record || {};
  const roster = (t.roster || []).map((p) =>
    `<div class="tp-player">${factionChip(p.faction)}` +
      `<span class="tpp-nm">${esc(p.nickname)}</span>` +
      `<span class="tpp-ar">${esc(p.archetype || "")}</span>` +
      `<span class="tpp-rr">${p.rating} ${deltaBadge(p.delta)}</span>` +
      `<span class="tpp-net">${esc(p.net_id || "")}</span>` +
    `</div>`).join("");
  page.innerHTML =
    `<div class="tp-head">` +
      `<button class="ghost tp-close">← к командам</button>` +
      emblem(t.team_id, "xl") +
      `<h3>${esc(t.name)} <span class="muted">${esc(t.abbrev)} · ${t.league === "upper" ? "Верхняя" : "Нижняя " + (t.group || "")}</span></h3>` +
    `</div>` +
    `<div class="tp-stats">` +
      `<div class="tp-stat"><b>${t.power}</b><span>Power</span></div>` +
      `<div class="tp-stat"><b>${rec.series_wins || 0}–${rec.series_losses || 0}</b><span>Серии</span></div>` +
      `<div class="tp-stat"><b>${rec.maps_won || 0}–${rec.maps_lost || 0}</b><span>Карты</span></div>` +
      `<div class="tp-stat"><b>${t.titles || 0}</b><span>Титулы</span></div>` +
      `<div class="tp-stat"><b>${t.treasury}</b><span>Казна</span></div>` +
      `<div class="tp-stat"><b>${t.certificates || 0}</b><span>Сертификаты</span></div>` +
    `</div>` +
    `<div class="tp-power">Динамика Elo (внутри лиги) ${sparkline(t.elo_history, 240, 40)}</div>` +
    (t.trophies && t.trophies.length ? `<div class="tp-trophies">🏆 ${t.trophies.map((x) => `S${x.season} ${esc(x.title)}`).join(", ")}</div>` : "") +
    `<h4>Состав</h4><div class="tp-roster">${roster || "<em>пусто</em>"}</div>`;
  page.querySelector(".tp-close").addEventListener("click", () => page.classList.add("is-hidden"));
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
  el("advanceBtn").disabled = state.busy || due;
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

// Off-season settings + ETA -------------------------------------------------- //
const STEPS_PER_SEC = 300;   // ~26 воркеров (оценка); серийно (один тяжёлый запуск за раз)
const LAUNCH_CAP = 6;        // server TRAINING_LAUNCH_CAP
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
  const jobs = Math.min(s.auction.slots, LAUNCH_CAP);
  const totalSteps = jobs * s.auction.steps_per_slot;           // launched serially after the off-season
  const min = totalSteps / STEPS_PER_SEC / 60;
  el("osEta").innerHTML =
    `⏱ Тренировки нейронок: до <b>${jobs}</b> запусков × ${s.auction.steps_per_slot.toLocaleString("ru")} шагов ` +
    `<b>${fmtMin(min)}</b> <span class="muted">(серийно, в фоне после межсезонья — можно следить в трекере)</span>` +
    (s.rookies.n ? `<br>👶 ${s.rookies.n} новичков (форк) — карточки создаются сразу` : "") +
    (s.auction.slots > LAUNCH_CAP ? `<br><span class="muted">Аукцион разыграет ${s.auction.slots} слотов, но запустится максимум ${LAUNCH_CAP} тренировок (остальное в очередь/банк).</span>` : "");
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
  el("offseasonBtn").addEventListener("click", openOsModal);
  el("osRun").addEventListener("click", submitOffseason);
  el("osCancel").addEventListener("click", () => show("osModal", false));
  el("chronClose").addEventListener("click", () => show("chronModal", false));
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
