"use strict";
/* Hardcoded World preview — the WHOLE cycle (seasons 0·1·2) for visual debugging.
   No server, no LLM, no real matches: scores are random, everything else is fixed. */

const FACTIONS = { humans: "#8ec5ff", infected: "#9ff5b0", robots: "#ffd58a" };
const FAC = Object.keys(FACTIONS);
const ARCH = ["skirmisher", "berserker", "sentinel", "architect", "warden"];
const TEAM_NAMES = [
  "Iron Falcons", "Golden Spire", "Quantum Owls", "Dust Rangers", "Northwind", "Crimson Apex",
  "Void Wardens", "Solar Kings", "Neon Drakes", "Granite Bears", "Echo Syndicate", "Storm Pact",
  "Ashen Wolves", "Cobalt Guard", "Mirage Legion", "Verdant Host", "Onyx Vanguard", "Pulse Riders",
  "Titan Forge", "Frost Hydra", "Ember Court", "Silent Talons",
];
const NICKS = ["Draar", "Umbux", "Vex", "Kilo", "Nyx", "Praz", "Sol", "Ravi", "Mox", "Juno", "Zane",
  "Quill", "Bram", "Tyso", "Oryn", "Wisp", "Garn", "Lex", "Pyre", "Kova", "Dusk", "Ferro", "Sable",
  "Hex", "Volt", "Cairn", "Rook", "Onyx", "Brisk", "Thane", "Vail", "Lumen", "Crag", "Esk", "Yuki",
  "Drei", "Spire", "Nova", "Kestrel", "Zima", "Bolt", "Arc", "Crux", "Dune", "Flux", "Grim",
  "Halo", "Iris", "Jolt", "Kane", "Loom", "Mire", "Nox", "Orin", "Pike", "Quasar", "Rune", "Slate",
  "Tora", "Ulu", "Vorn", "Wren", "Xan", "Yara", "Zeph", "Ash", "Bex", "Cyl"];

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(20260620);
const ri = (a, b) => a + Math.floor(rng() * (b - a + 1));
const pick = (arr) => arr[Math.floor(rng() * arr.length)];
const clamp = (v) => Math.max(5, Math.min(99, Math.round(v)));

// ---- hardcoded teams + rosters ------------------------------------------- //
let nick = 0;
function makeCard(power) {
  const j = (b) => clamp(b + (rng() - 0.5) * 36);
  return { attack: j(power * 70 + 12), defense: j(power * 55 + 32), economy: j(power * 60 + 18),
           tempo: j(power * 65 + 14), consistency: j(power * 50 + 32), potential: ri(18, 96) };
}
const teams = TEAM_NAMES.map((nm, i) => {
  const power = 0.32 + rng() * 0.55;
  const roster = [0, 1, 2].map(() => ({
    nickname: NICKS[nick++ % NICKS.length], faction: pick(FAC), archetype: pick(ARCH), card: makeCard(power),
  }));
  const abbr = nm.split(" ").map((w) => w[0]).join("").slice(0, 3).toUpperCase();
  return { id: "t" + i, name: nm, abbrev: abbr, power, roster, league: null, w: 0, l: 0, titles: 0,
           elo: 1000 };
});
const byId = Object.fromEntries(teams.map((t) => [t.id, t]));
const teamPower = (t) => t.roster.reduce((s, p) => s + (p.card.attack + p.card.defense + p.card.economy) / 3, 0) / 3;

// ---- hardcoded cycle state ----------------------------------------------- //
const els = {};
const el = (id) => (els[id] || (els[id] = document.getElementById(id)));
const FAC_CHIP = (f) => `<span class="fac-chip" style="--fac:${FACTIONS[f] || "#9aa7b4"}">${f}</span>`;

const state = { stage: 0, season: 0, day: 0, seasonData: null };
let liveTimer = null;

// SEASON SIMULATION (random, weighted by power) ---------------------------- //
function simulateSeason(seasonNum) {
  const upper = teams.filter((t) => t.league === "upper");
  const lower = teams.filter((t) => t.league === "lower");
  teams.forEach((t) => { t.w = 0; t.l = 0; });
  function dayMatches(pool, count) {
    const ms = [];
    const shuffled = [...pool].sort(() => rng() - 0.5);
    for (let i = 0; i + 1 < shuffled.length && ms.length < count; i += 2) {
      const a = shuffled[i], b = shuffled[i + 1];
      const pa = teamPower(a), pb = teamPower(b);
      const need = 2; let ma = 0, mb = 0;
      while (ma < need && mb < need) (rng() < pa / (pa + pb) ? ma++ : mb++);
      const homeWon = ma > mb;
      (homeWon ? a : b).w++; (homeWon ? b : a).l++;
      ms.push({ a, b, ma, mb, winner: homeWon ? a : b });
    }
    return ms;
  }
  const days = [];
  for (let d = 0; d < 5; d++) {
    days.push({ upper: dayMatches(upper, 5), lower: dayMatches(lower, 6) });
  }
  const standUp = [...upper].sort((x, y) => y.w - x.w || y.power - x.power);
  const standLo = [...lower].sort((x, y) => y.w - x.w || y.power - x.power);
  const champUp = standUp[0], champLo = standLo[0];
  champUp.titles++;
  const promoted = standLo.slice(0, 2), relegated = standUp.slice(-2);
  return { seasonNum, days, standUp, standLo, champUp, champLo, promoted, relegated };
}

// ---- stages -------------------------------------------------------------- //
function standings(rows, title) {
  return `<h3>${title}</h3><div class="standings">` + rows.map((t, i) =>
    `<div class="stand-row"><span class="rk">${i + 1}</span><span class="ab">${t.abbrev}</span>` +
    `<span class="nm">${t.name}</span><span class="pw">${t.w}-${t.l}</span>` +
    `<span class="ti">${"★".repeat(Math.min(5, t.titles))}</span></div>`).join("") + `</div>`;
}

function renderIntro() {
  el("previewMain").innerHTML =
    `<p class="world-hint">22 захардкоженных команды, у каждой 3 игрока с фиксированными карточками. ` +
    `Жми «Дальше», чтобы пройти Сезон 0 (SoloQ → драфт → лиги), затем сезоны 1 и 2 с межсезоньем.</p>` +
    `<div class="team-grid">` + teams.map((t) =>
      `<div class="team-card" style="--team-color:#3a6df0">` +
      `<span></span><span class="team-emblem">🛡️</span><div class="team-body">` +
      `<div class="team-name">${t.name} <span class="team-abbrev">${t.abbrev}</span></div>` +
      `<div class="team-strategy">Сила ${(t.power * 100 | 0)}</div>` +
      `<div class="team-persona">${t.roster.map((p) => p.nickname).join(", ")}</div></div></div>`).join("") + `</div>`;
  nav([["Дальше →", next]]);
}

function renderSoloq() {
  const ranked = [...teams].flatMap((t) => t.roster.map((p) => ({ ...p, team: t.abbrev })))
    .sort((a, b) => (b.card.attack + b.card.defense) - (a.card.attack + a.card.defense)).slice(0, 24);
  el("previewMain").innerHTML =
    `<p class="world-hint">SoloQ-лестница: игроки калибруются (захардкожено). Топ по силе:</p>` +
    `<div class="ladder">` + ranked.map((p, i) =>
      `<div class="lad-row"><span class="rk">${i + 1}</span>${FAC_CHIP(p.faction)}` +
      `<span class="ln">${p.nickname}</span><span class="lt">${p.team}</span>` +
      `<span class="lr">${1000 + (24 - i) * 6}</span></div>`).join("") + `</div>`;
  nav([["Драфт →", next]]);
}

function renderDraft() {
  // assign leagues: top 10 power -> upper, next 12 -> lower
  const ranked = [...teams].sort((a, b) => b.power - a.power);
  ranked.forEach((t, i) => { t.league = i < 10 ? "upper" : "lower"; });
  el("previewMain").innerHTML =
    `<p class="world-hint">Драфт (захардкожен, без LLM): каждая команда набрала тройку. Лиги собраны по силе.</p>` +
    `<div class="view-grid"><div class="view-col"><h3>Команды + составы</h3>` +
    teams.map((t) => `<div class="stand-row" style="cursor:default"><span class="ab">${t.abbrev}</span>` +
      `<span class="nm">${t.roster.map((p) => p.nickname).join(" · ")}</span>` +
      `<span class="pw">${t.league === "upper" ? "Верх" : "Низ"}</span></div>`).join("") + `</div></div>`;
  nav([["Лиги →", next]]);
}

function renderLeagues() {
  el("previewMain").innerHTML =
    standings(teams.filter((t) => t.league === "upper"), "Верхняя лига (10)") +
    standings(teams.filter((t) => t.league === "lower"), "Нижняя лига (12)");
  nav([["Сезон 1 →", next]]);
}

function renderSeason() {
  if (!state.seasonData || state.seasonData.seasonNum !== state.season) {
    state.seasonData = simulateSeason(state.season);
    state.day = 0;
  }
  const sd = state.seasonData;
  const day = sd.days[state.day];
  const matchRow = (m, lg) =>
    `<div class="stand-row" style="cursor:default"><span class="ab">${m.a.abbrev}</span>` +
    `<span class="nm" style="text-align:center">${m.ma} : ${m.mb}</span><span class="ab" style="text-align:right">${m.b.abbrev}</span>` +
    `<span class="pw">${m.winner.abbrev}</span>` +
    `<button class="ghost live-btn" data-lg="${lg}" data-i="${day[lg].indexOf(m)}">🔴 Эфир</button></div>`;
  el("previewMain").innerHTML =
    `<div class="preview-day"><strong>День ${state.day + 1} / ${sd.days.length}</strong> — Сезон ${state.season}. ` +
    `Счёт матчей случайный (предпросмотр). «🔴 Эфир» — посмотреть встречу в прямом эфире ×1, без перемотки.</div>` +
    `<div class="view-grid"><div class="view-col"><h3>Верхняя лига — день</h3><div class="standings">` +
    day.upper.map((m) => matchRow(m, "upper")).join("") + `</div></div>` +
    `<div class="view-col"><h3>Нижняя лига — день</h3><div class="standings">` +
    day.lower.map((m) => matchRow(m, "lower")).join("") + `</div></div></div>`;
  el("previewMain").querySelectorAll(".live-btn").forEach((b) =>
    b.addEventListener("click", () => liveBroadcast(day[b.dataset.lg][+b.dataset.i])));
  const last = state.day >= sd.days.length - 1;
  nav([
    [last ? "К плей-офф и итогам →" : "Следующий день →", () => { if (last) next(); else { state.day++; renderSeason(); } }],
  ]);
}

function renderSeasonEnd() {
  const sd = state.seasonData;
  el("previewMain").innerHTML =
    `<div class="season-done">🏆 Сезон ${state.season}: чемпион верхней — <strong>${sd.champUp.name}</strong>, ` +
    `нижней — <strong>${sd.champLo.name}</strong>. ↑ Повышены: ${sd.promoted.map((t) => t.abbrev).join(", ")}; ` +
    `↓ вылетели: ${sd.relegated.map((t) => t.abbrev).join(", ")}.</div>` +
    standings(sd.standUp, "Верхняя лига — итог") + standings(sd.standLo, "Нижняя лига — итог");
  // apply promo/releg for the next season
  sd.promoted.forEach((t) => (t.league = "upper"));
  sd.relegated.forEach((t) => (t.league = "lower"));
  nav([[state.season < 2 ? "Межсезонье →" : "Завершить →", next]]);
}

function renderOffseason() {
  const movers = [...teams].sort(() => rng() - 0.5);
  const transfers = [0, 1, 2].map(() => {
    const p = pick(pick(movers).roster);
    return `${p.nickname} → ${pick(movers).name} (${ri(120, 600)}k¥)`;
  });
  const rookies = [0, 1].map(() => `${pick(NICKS)} (форк, 1.0M шагов, потенциал ${ri(70, 95)})`);
  const training = [0, 1, 2].map(() => {
    const t = pick(teams);
    return `${t.abbrev}: ${pick(["skill_up", "meta_adapt", "teamplay"])} ${ri(1, 2)}.0M шагов → карточка пересчитается`;
  });
  el("previewMain").innerHTML =
    `<div class="preview-banner" style="margin:0 0 14px">Межсезонье (всё захардкожено: LLM-менеджер, ` +
    `аукцион тренировок, трансферы — без запросов).</div>` +
    `<div class="view-grid"><div class="view-col"><h3>🔁 Трансферы</h3><div class="offseason-log">` +
    transfers.map((l) => `<div>· ${l}</div>`).join("") + `</div>` +
    `<h3>🌱 Новички (форки)</h3><div class="offseason-log">` +
    rookies.map((l) => `<div>· ${l}</div>`).join("") + `</div></div>` +
    `<div class="view-col"><h3>🏋️ Тренировки (аукцион)</h3><div class="offseason-log">` +
    training.map((l) => `<div>· ${l}</div>`).join("") + `</div></div></div>`;
  nav([["Сезон 2 →", next]]);
}

function renderEnd() {
  el("previewMain").innerHTML =
    `<div class="season-done">Цикл предпросмотра пройден: Сезон 0 → 1 → межсезонье → 2. ` +
    `Это захардкоженный дебаг-вид; «настоящая» вкладка World делает то же на реальных нейронках, ` +
    `карточках и реплеях.</div>`;
  nav([["↻ С начала", () => location.reload()], ["К вкладке World", () => (location.href = "./world.html")]]);
}

// ---- LIVE broadcast (1x, no scrub) --------------------------------------- //
function liveBroadcast(m) {
  if (liveTimer) clearInterval(liveTimer);
  const total = m.ma + m.mb;
  let shown = 0, a = 0, b = 0;
  const seq = [];
  for (let i = 0; i < m.ma; i++) seq.push("a");
  for (let i = 0; i < m.mb; i++) seq.push("b");
  seq.sort(() => rng() - 0.5);
  const wrap = document.createElement("div");
  wrap.className = "live-overlay";
  wrap.innerHTML =
    `<div class="live-card"><div class="live-head">🔴 ПРЯМОЙ ЭФИР · ×1 · без перемотки` +
    `<button class="ghost live-close">✕</button></div>` +
    `<div class="live-teams"><span>${m.a.name}</span><span id="liveScore" class="live-score">0 : 0</span><span>${m.b.name}</span></div>` +
    `<div class="live-status" id="liveStatus">Матч начинается…</div></div>`;
  document.body.appendChild(wrap);
  wrap.querySelector(".live-close").addEventListener("click", () => { clearInterval(liveTimer); wrap.remove(); });
  liveTimer = setInterval(() => {
    if (shown >= total) {
      clearInterval(liveTimer);
      wrap.querySelector("#liveStatus").textContent = `✓ Матч окончен — победа ${m.winner.name}`;
      return;
    }
    seq[shown] === "a" ? a++ : b++;
    shown++;
    wrap.querySelector("#liveScore").textContent = `${a} : ${b}`;
    wrap.querySelector("#liveStatus").textContent = `Карта ${shown} из ${total} сыграна…`;
  }, 2200);
}

// ---- nav / stepper ------------------------------------------------------- //
const STAGES = [
  { t: "22 команды", r: renderIntro },
  { t: "Сезон 0 · SoloQ", r: renderSoloq },
  { t: "Сезон 0 · Драфт", r: renderDraft },
  { t: "Сезон 0 · Лиги", r: renderLeagues },
  { t: "Сезон 1", r: renderSeason, s: 1 },
  { t: "Сезон 1 · Итоги", r: renderSeasonEnd, s: 1 },
  { t: "Межсезонье 1", r: renderOffseason, s: 1 },
  { t: "Сезон 2", r: renderSeason, s: 2 },
  { t: "Сезон 2 · Итоги", r: renderSeasonEnd, s: 2 },
  { t: "Готово", r: renderEnd },
];

function nav(buttons) {
  el("previewNav").innerHTML = "";
  buttons.forEach(([label, fn], i) => {
    const b = document.createElement("button");
    b.className = i === buttons.length - 1 ? "primary" : "ghost";
    b.textContent = label;
    b.addEventListener("click", fn);
    el("previewNav").appendChild(b);
  });
}

function showStage() {
  const st = STAGES[state.stage];
  if (st.s != null) state.season = st.s;
  el("stageBanner").innerHTML =
    `<strong>${st.t}</strong> <span class="muted">· шаг ${state.stage + 1} / ${STAGES.length}</span>`;
  st.r();
  window.scrollTo(0, 0);
}
function next() { if (state.stage < STAGES.length - 1) { state.stage++; showStage(); } }

showStage();
