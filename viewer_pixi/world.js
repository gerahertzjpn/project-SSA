"use strict";

const TEAM_COUNT = 22;
const FACTION = {
  humans: { label: "Humans", color: "#8ec5ff" },
  infected: { label: "Infected", color: "#9ff5b0" },
  robots: { label: "Robots", color: "#ffd58a" },
};

const state = {
  cards: [],
  selected: new Set(),
  locked: true,
  lockNote: "",
  worldId: null,
  busy: false,
};

const els = {};
function el(id) {
  if (!(id in els)) els[id] = document.getElementById(id);
  return els[id];
}

async function fetchJson(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

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
      `<div class="team-name">${c.name} <span class="team-abbrev">${c.abbrev}</span></div>` +
      `<div class="team-manager">👔 ${c.manager || "—"}</div>` +
      `<div class="team-strategy">Стиль: ${strategyLabel(c.default_plan)}</div>` +
      `<div class="team-persona">${personaShort(c.persona)}</div>`;

    card.append(cb, emblem, body);
    grid.appendChild(card);
  }
}

function toggle(id, on) {
  if (state.locked) return;
  if (on) {
    if (state.selected.size >= TEAM_COUNT) return;
    state.selected.add(id);
  } else {
    state.selected.delete(id);
  }
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
// Background-job polling (create / advance)
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
// World view
// --------------------------------------------------------------------------- //
function show(id, on) {
  el(id).classList.toggle("is-hidden", !on);
}

function factionChip(f) {
  const meta = FACTION[f] || { label: f || "—", color: "#9aa7b4" };
  return `<span class="fac-chip" style="--fac:${meta.color}">${meta.label}</span>`;
}

function standingsTable(rows, container) {
  const box = el(container);
  box.innerHTML = "";
  rows.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "stand-row";
    row.innerHTML =
      `<span class="rk">${i + 1}</span>` +
      `<span class="ab">${t.abbrev}</span>` +
      `<span class="nm">${t.name}</span>` +
      `<span class="pw">${t.power}</span>` +
      `<span class="ti">${"★".repeat(Math.min(5, t.titles || 0))}</span>`;
    row.addEventListener("click", () => showRoster(t));
    box.appendChild(row);
  });
}

function showRoster(team) {
  const panel = el("rosterPanel");
  const rows = (team.roster || []).map((p) =>
    `<div class="rost-row">${factionChip(p.faction)}` +
    `<span class="rn">${p.nickname}</span>` +
    `<span class="ra">${p.archetype || ""}</span>` +
    `<span class="rr">${p.rating}</span>` +
    `<span class="rnet" title="нейронка">${p.net_id || ""}</span></div>`).join("");
  panel.innerHTML =
    `<div class="rost-head"><strong>${team.name}</strong> · Power ${team.power} ` +
    `<button class="ghost rost-close">✕</button></div>` +
    `<div class="rost-list">${rows || "<em>состав пуст</em>"}</div>`;
  panel.querySelector(".rost-close").addEventListener("click", () => show("rosterPanel", false));
  show("rosterPanel", true);
}

function renderLadder(ladder) {
  el("ladder").innerHTML = ladder.map((p, i) =>
    `<div class="lad-row${p.role === "streamer" ? " streamer" : ""}">` +
    `<span class="rk">${i + 1}</span>${factionChip(p.faction)}` +
    `<span class="ln">${p.nickname}</span>` +
    `<span class="lt">${p.team || (p.role === "streamer" ? "стример" : "FA")}</span>` +
    `<span class="lr">${p.rating}</span></div>`).join("");
}

function renderHistory(history) {
  if (!history.length) {
    el("history").innerHTML = `<div class="muted">Ещё не сыграно ни одного сезона.</div>`;
    return;
  }
  el("history").innerHTML = history.slice().reverse().map((s) =>
    `<div class="hist-row"><span class="hs">S${s.season}</span>` +
    `<span class="hc">🏆 ${s.upper_champion}</span>` +
    `<span class="hc lo">↑ ${s.lower_champion}</span></div>`).join("");
}

function renderView(w) {
  state.worldId = w.world_id;
  show("newSection", false);
  show("resultPanel", false);
  show("worldView", true);
  el("viewName").textContent = w.name || w.world_id;
  el("viewSeason").textContent = `Сезон ${w.season}`;
  const c = w.counts || {};
  el("viewCounts").textContent =
    `${c.players} игроков · ${c.rostered} в составах · ${c.streamers} стримеров`;
  standingsTable(w.upper || [], "upperTable");
  standingsTable(w.lower || [], "lowerTable");
  renderLadder(w.ladder || []);
  renderHistory(w.history || []);
  show("rosterPanel", false);
  el("advanceBtn").disabled = state.busy;
}

async function openWorld(worldId) {
  try {
    const { world } = await fetchJson(`/api/world/${worldId}`);
    renderView(world);
  } catch (e) {
    el("resultPanel").classList.remove("is-hidden");
    el("resultPanel").innerHTML = `<div class="result-error">Не удалось открыть мир: ${e.message}</div>`;
  }
}

// --------------------------------------------------------------------------- //
// Actions
// --------------------------------------------------------------------------- //
async function start() {
  if (state.locked || state.busy || state.selected.size !== TEAM_COUNT) return;
  refreshPicker();
  try {
    const job = await runJob("/api/world/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ franchises: [...state.selected] }),
    }, "Создаём мир — реальные матчи нейронок");
    await refreshWorldList(job.world_id);
    await openWorld(job.world_id);
  } catch (e) {
    el("resultPanel").classList.remove("is-hidden");
    el("resultPanel").innerHTML = `<div class="result-error">Ошибка: ${e.message}</div>`;
  } finally {
    refreshPicker();
  }
}

async function advance() {
  if (!state.worldId || state.busy) return;
  el("advanceBtn").disabled = true;
  try {
    const job = await runJob(`/api/world/${state.worldId}/advance`, { method: "POST" },
      "Играем сезон — реальные матчи нейронок");
    const s = job.summary || {};
    el("resultPanel").classList.remove("is-hidden");
    el("resultPanel").innerHTML =
      `<div class="season-done">Сезон ${s.season} сыгран — 🏆 <strong>${s.upper_champion}</strong> ` +
      `(верх), ↑ <strong>${s.lower_champion}</strong> (низ). ` +
      `Повышены: ${(s.promoted || []).join(", ") || "—"}; вылетели: ${(s.relegated || []).join(", ") || "—"}.</div>`;
    await openWorld(state.worldId);
  } catch (e) {
    el("resultPanel").classList.remove("is-hidden");
    el("resultPanel").innerHTML = `<div class="result-error">Ошибка сезона: ${e.message}</div>`;
  } finally {
    el("advanceBtn").disabled = state.busy;
  }
}

async function refreshWorldList(selectId) {
  try {
    const info = await fetchJson("/api/world/info");
    const sel = el("worldSelect");
    const worlds = info.worlds || [];
    sel.innerHTML = worlds.length
      ? worlds.map((w) => `<option value="${w}">${w}</option>`).join("")
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
  } else {
    document.body.classList.remove("world-locked");
  }
}

async function init() {
  try {
    const info = await fetchJson("/api/world/info");
    applyLock(info.ui);
    const data = await fetchJson("/api/world/franchises");
    state.cards = data.franchises || [];
    await refreshWorldList(null);
  } catch (e) {
    el("teamGrid").innerHTML = `<div class="result-error">Не удалось загрузить: ${e.message}</div>`;
    return;
  }
  el("startBtn").addEventListener("click", start);
  el("advanceBtn").addEventListener("click", advance);
  el("openBtn").addEventListener("click", () => {
    const wid = el("worldSelect").value;
    if (wid) openWorld(wid);
  });
  el("newBtn").addEventListener("click", () => {
    show("worldView", false);
    show("resultPanel", false);
    show("newSection", true);
  });
  refreshPicker();
}

init();
