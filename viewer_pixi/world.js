"use strict";

const TEAM_COUNT = 22;
const state = {
  cards: [],
  selected: new Set(),
  locked: true,
  lockNote: "",
};

const els = {
  grid: document.getElementById("teamGrid"),
  count: document.getElementById("selCount"),
  start: document.getElementById("startBtn"),
  lock: document.getElementById("lockBanner"),
  lockNote: document.getElementById("lockNote"),
  result: document.getElementById("resultPanel"),
};

async function fetchJson(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

function personaShort(text) {
  if (!text) return "";
  const t = text.replace(/^Ты\s+/, "");
  return t.length > 140 ? t.slice(0, 138) + "…" : t;
}

function strategyLabel(plan) {
  if (!plan || !plan.tempo) return "—";
  return `${plan.tempo} · ${plan.win_path}`;
}

function renderCards() {
  els.grid.innerHTML = "";
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
    // block selecting NEW ones once 22 are picked, and lock the whole UI when in dev
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
    els.grid.appendChild(card);
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
  refresh();
}

function refresh() {
  const n = state.selected.size;
  els.count.textContent = `${n} / ${TEAM_COUNT}`;
  els.count.classList.toggle("full", n === TEAM_COUNT);
  els.start.disabled = state.locked || n !== TEAM_COUNT;
  renderCards();
}

async function start() {
  if (state.locked || state.selected.size !== TEAM_COUNT) return;
  els.start.disabled = true;
  els.start.textContent = "Создаём…";
  try {
    const res = await fetchJson("/api/world/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ franchises: [...state.selected] }),
    });
    els.result.classList.remove("is-hidden");
    els.result.innerHTML =
      `<h3>World создан: ${res.world_id} · сезон ${res.season}</h3>` +
      `<div class="result-cols">` +
      `<div><h4>Верхняя лига (${res.upper.length})</h4>${res.upper.map((x) => `<div>${x}</div>`).join("")}</div>` +
      `<div><h4>Нижняя лига (${res.lower.length})</h4>${res.lower.map((x) => `<div>${x}</div>`).join("")}</div>` +
      `</div>`;
  } catch (e) {
    els.result.classList.remove("is-hidden");
    els.result.innerHTML = `<div class="result-error">Ошибка: ${e.message}</div>`;
  } finally {
    els.start.textContent = "Создать World";
    refresh();
  }
}

function applyLock(ui) {
  state.locked = !!(ui && ui.locked);
  state.lockNote = (ui && ui.note) || "";
  els.lock.classList.toggle("is-hidden", !state.locked);
  if (state.locked) {
    els.lockNote.textContent = state.lockNote ||
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
  } catch (e) {
    els.grid.innerHTML = `<div class="result-error">Не удалось загрузить: ${e.message}</div>`;
    return;
  }
  els.start.addEventListener("click", start);
  refresh();
}

init();
