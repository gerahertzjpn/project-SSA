const NODE_POSITIONS = {
  A1: [70, 90],
  A2: [70, 210],
  A3: [70, 330],
  CP1: [210, 150],
  CP2: [210, 270],
  CP3: [340, 80],
  CP4: [340, 340],
  CP5: [470, 150],
  CP6: [470, 270],
  Baron: [340, 210],
  B1: [610, 90],
  B2: [610, 210],
  B3: [610, 330],
};

const EDGES = [
  ["A1", "CP1"], ["A1", "CP2"],
  ["A2", "CP1"], ["A2", "CP2"],
  ["A3", "CP1"], ["A3", "CP2"],
  ["CP1", "CP3"], ["CP1", "Baron"],
  ["CP2", "CP4"], ["CP2", "Baron"],
  ["CP3", "CP5"], ["CP4", "CP6"],
  ["CP5", "B1"], ["CP5", "B2"], ["CP5", "B3"],
  ["CP6", "B1"], ["CP6", "B2"], ["CP6", "B3"],
  ["CP5", "Baron"], ["CP6", "Baron"],
];

const CORE_NODES = new Set(["A1", "A2", "A3", "B1", "B2", "B3"]);
const CP_NODES = new Set(["CP1", "CP2", "CP3", "CP4", "CP5", "CP6"]);
const PLAYER_IDS = ["A1", "A2", "A3", "B1", "B2", "B3"];
const CORE_MAX_HP = 50;
const LORD_MAX_HP = 150;
const DEFAULT_UI_RUNS = 20;
const DEFAULT_GENERATION_SEED = 20260602;
const FALLBACK_MODELS = [
  { id: "random", label: "Random bot" },
  { id: "scripted", label: "Scripted bot" },
];
const SAMPLE_REPLAYS = Array.from({ length: DEFAULT_UI_RUNS }, (_, index) => ({
  label: `replay_${index}.json`,
  path: `../replays_viewer_debug_smoke/replay_${index}.json`,
}));

const state = {
  replay: null,
  tick: 0,
  playing: false,
  raf: null,
  stepStartedAt: 0,
  activeTab: "battles",
  samples: [...SAMPLE_REPLAYS],
  models: [...FALLBACK_MODELS],
  maxUiRuns: 100,
  apiAvailable: false,
  generating: false,
};

const els = {
  arena: document.getElementById("arena"),
  edgeLayer: document.getElementById("edgeLayer"),
  nodeLayer: document.getElementById("nodeLayer"),
  routeLayer: document.getElementById("routeLayer"),
  battleLayer: document.getElementById("battleLayer"),
  lordLayer: document.getElementById("lordLayer"),
  replayMeta: document.getElementById("replayMeta"),
  sampleSelect: document.getElementById("sampleSelect"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  fileInput: document.getElementById("fileInput"),
  prevBtn: document.getElementById("prevBtn"),
  playBtn: document.getElementById("playBtn"),
  nextBtn: document.getElementById("nextBtn"),
  tickSlider: document.getElementById("tickSlider"),
  tickNow: document.getElementById("tickNow"),
  tickMax: document.getElementById("tickMax"),
  speedSelect: document.getElementById("speedSelect"),
  diagnosticsGrid: document.getElementById("diagnosticsGrid"),
  playersGrid: document.getElementById("playersGrid"),
  stateList: document.getElementById("stateList"),
  battleList: document.getElementById("battleList"),
  eventList: document.getElementById("eventList"),
  matchCountInput: document.getElementById("matchCountInput"),
  seedInput: document.getElementById("seedInput"),
  playerModelGrid: document.getElementById("playerModelGrid"),
  generateMatchesBtn: document.getElementById("generateMatchesBtn"),
  generationStatus: document.getElementById("generationStatus"),
  generationSpinner: document.getElementById("generationSpinner"),
  generatorSummary: document.getElementById("generatorSummary"),
  panelTabs: Array.from(document.querySelectorAll(".panel-tab")),
  tabPanels: Array.from(document.querySelectorAll("[data-tab-panel]")),
};

function setup() {
  drawEdges();
  populateSamples();
  bindControls();
  initGenerator();
  renderEmpty();
}

function populateSamples(samples = state.samples) {
  state.samples = samples?.length ? samples : [...SAMPLE_REPLAYS];
  els.sampleSelect.innerHTML = "";
  for (const sample of state.samples) {
    const option = document.createElement("option");
    option.value = sample.path;
    option.textContent = sample.label;
    els.sampleSelect.appendChild(option);
  }
}

function bindControls() {
  els.loadSampleBtn.addEventListener("click", loadSampleReplay);
  els.fileInput.addEventListener("change", loadFileReplay);
  els.prevBtn.addEventListener("click", () => seekTo(state.tick - 1));
  els.nextBtn.addEventListener("click", () => seekTo(state.tick + 1));
  els.playBtn.addEventListener("click", togglePlayback);
  els.tickSlider.addEventListener("input", (event) => seekTo(Number(event.target.value)));
  els.matchCountInput.addEventListener("input", validateGeneratorControls);
  els.seedInput.addEventListener("input", validateGeneratorControls);
  els.generateMatchesBtn.addEventListener("click", generateMatches);
  els.panelTabs.forEach((button, index) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
    button.addEventListener("keydown", (event) => handleTabKey(event, index));
  });
  setActiveTab(state.activeTab);
}

function setActiveTab(tab) {
  state.activeTab = tab;
  for (const button of els.panelTabs) {
    const active = button.dataset.tab === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  }

  for (const panel of els.tabPanels) {
    panel.classList.toggle("active", panel.dataset.tabPanel === tab);
  }
}

function handleTabKey(event, index) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    return;
  }
  event.preventDefault();

  let nextIndex = index;
  if (event.key === "ArrowLeft") {
    nextIndex = (index - 1 + els.panelTabs.length) % els.panelTabs.length;
  } else if (event.key === "ArrowRight") {
    nextIndex = (index + 1) % els.panelTabs.length;
  } else if (event.key === "Home") {
    nextIndex = 0;
  } else if (event.key === "End") {
    nextIndex = els.panelTabs.length - 1;
  }

  const next = els.panelTabs[nextIndex];
  next.focus();
  setActiveTab(next.dataset.tab);
}

async function initGenerator() {
  populatePlayerModelSelects(state.models, PLAYER_IDS);
  validateGeneratorControls();
  setGeneratorStatus("Connecting...");

  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    state.apiAvailable = true;
    state.models = payload.models?.length ? payload.models : [...FALLBACK_MODELS];
    state.maxUiRuns = Number(payload.max_ui_runs || 100);
    els.matchCountInput.max = String(state.maxUiRuns);
    populatePlayerModelSelects(state.models, payload.players?.length ? payload.players : PLAYER_IDS);
    if (payload.replays?.length) {
      populateSamples(payload.replays);
    }
    setGeneratorStatus(`Ready | max ${state.maxUiRuns}`);
  } catch (error) {
    state.apiAvailable = false;
    setGeneratorStatus("Generation API offline", "error");
  }

  validateGeneratorControls();
}

function populatePlayerModelSelects(models = FALLBACK_MODELS, players = PLAYER_IDS) {
  els.playerModelGrid.innerHTML = "";
  const choices = models.length ? models : FALLBACK_MODELS;

  for (const playerId of players) {
    const row = document.createElement("label");
    row.className = `model-row ${teamClass(playerId)}`;

    const name = document.createElement("span");
    name.textContent = playerId;

    const select = document.createElement("select");
    select.dataset.player = playerId;
    select.setAttribute("aria-label", `${playerId} model`);
    select.addEventListener("change", validateGeneratorControls);

    for (const model of choices) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      select.appendChild(option);
    }

    row.appendChild(name);
    row.appendChild(select);
    els.playerModelGrid.appendChild(row);
  }
}

function validateGeneratorControls() {
  const runs = Number(els.matchCountInput.value);
  const validRuns = Number.isInteger(runs) && runs >= 1 && runs <= state.maxUiRuns;
  const validSeed = Number.isFinite(Number(els.seedInput.value));
  const canGenerate = state.apiAvailable && !state.generating && validRuns && validSeed;
  els.generateMatchesBtn.disabled = !canGenerate;

  if (state.generating) {
    return false;
  }
  if (!validRuns) {
    setGeneratorStatus(`UI max ${state.maxUiRuns} matches`, "validation");
    return false;
  }
  if (!validSeed) {
    setGeneratorStatus("Seed is invalid", "validation");
    return false;
  }
  if (!state.apiAvailable) {
    setGeneratorStatus("Generation API offline", "error");
    return false;
  }
  if (els.generationStatus.dataset.kind === "validation") {
    setGeneratorStatus(`Ready | max ${state.maxUiRuns}`);
  }
  return true;
}

function playerModelSelection() {
  const players = {};
  for (const select of els.playerModelGrid.querySelectorAll("select[data-player]")) {
    players[select.dataset.player] = select.value;
  }
  return players;
}

async function generateMatches() {
  if (!validateGeneratorControls()) {
    return;
  }

  const runs = Number(els.matchCountInput.value);
  const seed = Number(els.seedInput.value || DEFAULT_GENERATION_SEED);
  const payload = {
    runs,
    seed,
    players: playerModelSelection(),
  };

  state.generating = true;
  setGeneratorLoading(true);
  setGeneratorStatus(`Generating ${runs} matches...`, "loading");
  els.generatorSummary.textContent = "";
  validateGeneratorControls();

  try {
    const response = await fetch("/api/generate-replays", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `${response.status} ${response.statusText}`);
    }

    const replays = data.replays?.length ? data.replays : [];
    populateSamples(replays.length ? replays : SAMPLE_REPLAYS);
    setGeneratorStatus("Generation complete", "ok");
    els.generatorSummary.textContent = generationSummary(data);

    if (replays[0]) {
      els.sampleSelect.value = replays[0].path;
      await loadReplayPath(replays[0].path, replays[0].label);
    }
  } catch (error) {
    setGeneratorStatus(error.message || "Generation failed", "error");
  } finally {
    state.generating = false;
    setGeneratorLoading(false);
    validateGeneratorControls();
  }
}

function generationSummary(data = {}) {
  const winners = countSummary(data.winners);
  const types = countSummary(data.winner_types);
  return [
    `${data.runs || 0} matches`,
    winners ? `winners ${winners}` : "",
    types ? `types ${types}` : "",
  ].filter(Boolean).join(" | ");
}

function countSummary(values = {}) {
  return Object.entries(values)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}:${value}`)
    .join(" ");
}

function setGeneratorLoading(loading) {
  els.generationSpinner.classList.toggle("active", Boolean(loading));
  els.generateMatchesBtn.setAttribute("aria-busy", loading ? "true" : "false");
}

function setGeneratorStatus(text, kind = "") {
  els.generationStatus.textContent = text;
  els.generationStatus.dataset.kind = kind;
}

function drawEdges() {
  els.edgeLayer.innerHTML = "";
  for (const [src, dst] of EDGES) {
    const [x1, y1] = NODE_POSITIONS[src];
    const [x2, y2] = NODE_POSITIONS[dst];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    els.edgeLayer.appendChild(line);
  }
}

async function loadSampleReplay() {
  try {
    const fallback = state.samples[0] || SAMPLE_REPLAYS[0];
    const path = els.sampleSelect.value || fallback.path;
    await loadReplayPath(path);
  } catch (error) {
    els.replayMeta.textContent = "Sample доступен при запуске через локальный сервер";
  }
}

async function loadReplayPath(path, label = "") {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  const replay = await response.json();
  setReplay(replay, label || path.replace("../", ""));
}

function loadFileReplay(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const replay = JSON.parse(String(reader.result));
      setReplay(replay, file.name);
    } catch (error) {
      els.replayMeta.textContent = "JSON не прочитан";
    }
  };
  reader.readAsText(file, "utf-8");
}

function setReplay(replay, label) {
  stopPlayback();
  validateReplay(replay);
  state.replay = replay;
  state.tick = 0;

  const maxTick = replay.ticks.length - 1;
  els.tickSlider.min = "0";
  els.tickSlider.max = String(maxTick);
  els.tickSlider.value = "0";
  els.tickMax.textContent = String(maxTick);

  const meta = replay.meta || {};
  els.replayMeta.textContent = `${label} | schema ${meta.schema_version || "?"} | winner ${meta.winner || "?"}`;
  renderTick();
}

function validateReplay(replay) {
  if (!replay || !Array.isArray(replay.ticks) || replay.ticks.length === 0) {
    throw new Error("Replay has no ticks");
  }
}

function togglePlayback() {
  if (!state.replay) {
    return;
  }
  if (state.playing) {
    stopPlayback();
    return;
  }

  state.playing = true;
  els.playBtn.textContent = "||";
  state.stepStartedAt = performance.now();
  state.raf = window.requestAnimationFrame(playLoop);
}

function playLoop(now) {
  if (!state.replay || !state.playing) {
    return;
  }

  const stepMs = Number(els.speedSelect.value);
  let elapsed = now - state.stepStartedAt;

  while (elapsed >= stepMs) {
    if (state.tick >= state.replay.ticks.length - 1) {
      stopPlayback();
      return;
    }
    state.tick += 1;
    state.stepStartedAt += stepMs;
    elapsed -= stepMs;
  }

  renderTick(Math.max(0, Math.min(1, elapsed / stepMs)));
  state.raf = window.requestAnimationFrame(playLoop);
}

function stopPlayback() {
  if (state.raf) {
    window.cancelAnimationFrame(state.raf);
    state.raf = null;
  }
  state.playing = false;
  els.playBtn.textContent = ">";
}

function seekTo(tick) {
  if (!state.replay) {
    return;
  }
  const maxTick = state.replay.ticks.length - 1;
  state.tick = Math.max(0, Math.min(maxTick, tick));
  state.stepStartedAt = performance.now();
  renderTick(0);
}

function renderEmpty() {
  els.diagnosticsGrid.innerHTML = `<div class="empty">Нет replay</div>`;
  els.playersGrid.innerHTML = `<div class="empty">Нет replay</div>`;
  els.stateList.innerHTML = `<div class="empty">No replay</div>`;
  els.battleList.innerHTML = `<div class="empty">Нет активных боёв</div>`;
  els.eventList.innerHTML = `<div class="empty">Нет событий</div>`;
  els.tickNow.textContent = "0";
  els.tickMax.textContent = "0";
  renderNodes(null);
  renderLordOverlay([]);
}

function renderTick(alpha = 0) {
  const tick = currentTick();
  const nextTick = state.replay?.ticks[Math.min(state.tick + 1, state.replay.ticks.length - 1)] || tick;
  const lordMarkers = alliedLordMarkers(tick, nextTick, alpha);
  const lordNodeIds = new Set(lordMarkers.filter((marker) => marker.nodeId).map((marker) => marker.nodeId));
  els.tickNow.textContent = String(state.tick);
  els.tickSlider.value = String(state.tick);

  renderNodes(tick, lordNodeIds);
  renderRoutes(tick, nextTick, alpha);
  renderBattles(tick);
  renderLordOverlay(lordMarkers);
  renderDiagnostics(tick);
  renderPlayers(tick);
  renderStatePanel(tick, nextTick, alpha);
  renderBattleList(tick, nextTick, alpha);
  renderEvents(tick);
}

function renderDiagnostics(tick) {
  if (!state.replay || !tick) {
    els.diagnosticsGrid.innerHTML = `<div class="empty">Нет replay</div>`;
    return;
  }

  const meta = state.replay.meta || {};
  const counts = eventCounts(state.replay);
  const totalEvents = Object.values(counts).reduce((sum, value) => sum + value, 0);
  const battleEvents = ["BATTLE_STARTED", "BATTLE_TICK", "BATTLE_REINFORCEMENT", "BATTLE_RESOLVED"]
    .reduce((sum, key) => sum + (counts[key] || 0), 0);

  const metrics = [
    ["Schema", meta.schema_version || "?"],
    ["Trace", meta.combat_trace_format || "?"],
    ["Winner", meta.winner || "?"],
    ["Tick", `${state.tick}/${state.replay.ticks.length - 1}`],
    ["Events", totalEvents],
    ["Battle Events", battleEvents],
    ["Moving", tick.moving_groups?.length || 0],
    ["Battles", tick.active_battles?.length || 0],
  ];

  els.diagnosticsGrid.innerHTML = metrics.map(([label, value]) => `
    <div class="metric">
      <strong>${label}</strong>
      <span>${value}</span>
    </div>
  `).join("");
}

function currentTick() {
  return state.replay?.ticks[state.tick] || null;
}

function renderNodes(tick, lordNodeIds = new Set()) {
  els.nodeLayer.innerHTML = "";
  const nodes = tick?.nodes || {};

  for (const [id, [x, y]] of Object.entries(NODE_POSITIONS)) {
    const nodeState = nodes[id] || {};
    const owner = nodeState.owner || "NEUTRAL";
    const node = document.createElement("div");
    node.className = [
      "node",
      CORE_NODES.has(id) ? "core" : "",
      id === "Baron" ? "baron" : "",
      ownerClass(owner),
      nodeActivityClass(id, tick),
      lordNodeIds.has(id) ? "lord-occupied" : "",
      CORE_NODES.has(id) && tick?.players?.[id]?.alive === false ? "dead" : "",
    ].filter(Boolean).join(" ");
    node.style.left = `${scaleX(x)}px`;
    node.style.top = `${scaleY(y)}px`;
    node.dataset.nodeId = id;
    node.title = nodeTitle(id, nodeState);
    node.textContent = lordNodeIds.has(id) ? "" : displayNodeLabel(id);

    const total = nodeDisplayTotal(nodeState, tick, id);
    if (total > 0 && !lordNodeIds.has(id)) {
      const stack = document.createElement("span");
      stack.className = "stack";
      stack.textContent = String(total);
      node.appendChild(stack);
    }

    if (CP_NODES.has(id) && Number(nodeState.neutral_hp || 0) > 0) {
      const neutral = document.createElement("span");
      neutral.className = "neutral-hp";
      neutral.textContent = String(nodeState.neutral_hp);
      node.appendChild(neutral);
    }

    els.nodeLayer.appendChild(node);
  }
}

function renderRoutes(tick, nextTick, alpha) {
  els.routeLayer.innerHTML = "";
  const nextGroups = [...(nextTick?.moving_groups || [])];
  const used = new Set();

  for (const group of tick?.moving_groups || []) {
    if (!isEdgeMovement(group)) {
      continue;
    }
    if (hasLordUnit(group.units)) {
      continue;
    }
    const nextGroup = findNextGroup(group, nextGroups, used);
    const nextProgress = sameMovementSegment(group, nextGroup) ? nextGroup.progress : 1;
    const progress = interpolateValue(
      Number(group.progress || 0),
      Number(nextProgress ?? 0),
      alpha,
    );
    const pos = movingGroupPosition(group, progress);
    if (!pos) {
      continue;
    }
    const marker = document.createElement("div");
    marker.className = `route ${teamClass(group.player)}`;
    marker.style.left = `${pos[0]}px`;
    marker.style.top = `${pos[1]}px`;
    marker.title = `${group.player}: ${routeText(group)} (${phaseLabel(group.phase)})`;
    els.routeLayer.appendChild(marker);
  }
}

function renderLordOverlay(markers = []) {
  els.lordLayer.innerHTML = "";

  for (const markerInfo of markers) {
    if (!markerInfo.pos) {
      continue;
    }

    const marker = document.createElement("div");
    marker.className = [
      "lord-marker",
      teamClass(markerInfo.owner),
      markerInfo.nodeId ? "on-node" : "",
      markerInfo.inBattle ? "in-battle" : "",
    ].filter(Boolean).join(" ");
    marker.style.left = `${markerInfo.pos[0]}px`;
    marker.style.top = `${markerInfo.pos[1]}px`;
    marker.textContent = "L";
    marker.title = `Lord ${markerInfo.owner} | ${markerInfo.action}${markerInfo.target ? ` -> ${displayLocation(markerInfo.target)}` : ""}`;
    els.lordLayer.appendChild(marker);
  }
}

function renderBattles(tick) {
  els.battleLayer.innerHTML = "";
  for (const battle of tick?.active_battles || []) {
    const pos = battlePosition(battle);
    if (!pos) {
      continue;
    }
    const marker = document.createElement("div");
    marker.className = `battle-marker ${battle.location_kind === "edge" ? "edge" : ""}`;
    marker.style.left = `${pos[0]}px`;
    marker.style.top = `${pos[1]}px`;
    marker.textContent = battle.location_kind === "edge" ? "edge" : "node";
    marker.title = `${battle.battle_id} | ${battle.participants.join(", ")}`;
    els.battleLayer.appendChild(marker);
  }
}

function renderPlayers(tick) {
  els.playersGrid.innerHTML = "";
  const players = tick?.players || {};

  for (const playerId of ["A1", "A2", "A3", "B1", "B2", "B3"]) {
    const player = players[playerId] || {};
    const item = document.createElement("div");
    item.className = `player ${teamClass(playerId)} ${player.alive === false ? "dead" : ""}`;
    item.innerHTML = `
      <strong>${playerId}</strong>
      <span>${player.alive === false ? "out" : "alive"}</span>
      <span>${formatNumber(player.resources || 0)} res</span>
    `;
    els.playersGrid.appendChild(item);
  }
}

function renderStatePanel(tick, nextTick, alpha) {
  if (!tick) {
    els.stateList.innerHTML = `<div class="empty">No replay</div>`;
    return;
  }

  const cards = Object.keys(NODE_POSITIONS)
    .map((nodeId) => stateNodeCard(nodeId, tick, nextTick, alpha))
    .filter(Boolean);

  els.stateList.innerHTML = cards.length ? cards.join("") : `<div class="empty">No active node state</div>`;
}

function stateNodeCard(nodeId, tick, nextTick, alpha) {
  const node = tick.nodes?.[nodeId] || {};
  const inside = (tick.moving_groups || []).filter((group) => movementNodeId(group) === nodeId);
  const incoming = (tick.moving_groups || []).filter((group) => isEdgeMovement(group) && group.dst === nodeId);
  const outgoing = (tick.moving_groups || []).filter((group) => isEdgeMovement(group) && group.src === nodeId);
  const battle = (tick.active_battles || []).find((item) => item.location_kind === "node" && item.location_id === nodeId);
  const forceCards = stateForceCards(nodeId, node, tick, inside);
  const coreHp = coreHpMarkup(nodeId, node, tick);
  const orderCards = [
    ...inside.map((group) => movingGroupCard(group, "in node", tick, nextTick, alpha)),
    ...incoming.map((group) => movingGroupCard(group, "in", tick, nextTick, alpha)),
    ...outgoing.map((group) => movingGroupCard(group, "out", tick, nextTick, alpha)),
  ];
  const hasActivity = coreHp || inside.length || incoming.length || outgoing.length || battle || forceCards.length || node.neutral_hp > 0;

  if (!hasActivity) {
    return "";
  }

  const ready = Object.entries(node.unit_ready_tick || {})
    .filter(([, readyTick]) => Number(readyTick) > Number(tick.tick))
    .map(([player, readyTick]) => `<span class="state-tag wait">${player} ready ${readyTick}</span>`)
    .join("");

  return `
    <div class="state-node-card ${battle || inside.length || incoming.length || outgoing.length ? "active" : ""}">
      <div class="state-node-head">
        <strong>${displayNodeLabel(nodeId)}</strong>
        <span>${node.owner || "NEUTRAL"}</span>
      </div>
      <div class="state-tags">
        ${battle ? `<span class="state-tag battle">node battle</span>` : ""}
        ${inside.length ? `<span class="state-tag">inside ${inside.length}</span>` : ""}
        ${incoming.length ? `<span class="state-tag">in ${incoming.length}</span>` : ""}
        ${outgoing.length ? `<span class="state-tag">out ${outgoing.length}</span>` : ""}
        ${node.neutral_hp > 0 ? `<span class="state-tag neutral">neutral ${node.neutral_hp}</span>` : ""}
        ${ready}
      </div>
      ${coreHp}
      <div class="state-section-title">standing</div>
      <div class="state-force-grid">
        ${forceCards.length ? forceCards.join("") : `<div class="state-empty">no stationed force</div>`}
      </div>
      ${orderCards.length ? `
        <div class="state-section-title">orders</div>
        <div class="state-move-grid">${orderCards.join("")}</div>
      ` : ""}
    </div>
  `;
}

function stateForceCards(nodeId, node = {}, tick = {}, insideGroups = []) {
  const cards = [];
  const currentTick = Number(tick.tick || 0);
  const battle = (tick.active_battles || []).find((item) => item.location_kind === "node" && item.location_id === nodeId);
  const alliedLord = alliedLordBattleInfo(battle);

  if (unitTotal(node.units) > 0) {
    const owner = node.owner || "NEUTRAL";
    cards.push(stateForceCard(
      owner,
      node.units,
      forceStatus(node, owner, currentTick, "holding"),
      "node owner",
    ));
  }

  for (const [player, units] of Object.entries(node.visiting_units || {})) {
    if (unitTotal(units) > 0) {
      cards.push(stateForceCard(
        player,
        units,
        forceStatus(node, player, currentTick, "waiting"),
        "visitor",
      ));
    }
  }

  if (Number(node.neutral_hp || 0) > 0) {
    const aggro = (node.neutral_aggro_teams || []).join(", ");
    cards.push(stateForceCard(
      "NEUTRAL",
      { NEUTRAL: node.neutral_hp },
      aggro ? `aggro ${aggro}` : "passive center guard",
      "neutral",
    ));
  }

  for (const group of insideGroups) {
    cards.push(stateForceCard(
      group.player,
      group.units,
      `${phaseLabel(group.phase)} ${Math.round(Number(group.progress || 0) * 100)}%`,
      "moving inside",
      unitCountLabel(group.units),
      hasLordUnit(group.units) ? alliedLordHpMarkup(group.player, tick) : "",
    ));
  }

  if (alliedLord) {
    cards.push(stateForceCard(
      alliedLord.owner,
      { LORD: 1 },
      "node battle",
      "allied lord",
      `${formatNumber(alliedLord.hp)} hp`,
      stateHpBar(alliedLord.hp, alliedLord.maxHp || LORD_MAX_HP, "lord hp", "lord"),
    ));
  }

  if (nodeId === "Baron" && Number(tick.baron?.hp || 0) > 0) {
    cards.push(stateForceCard(
      "LORD",
      { LORD: 1 },
      "guarding center",
      "boss",
      `${formatNumber(tick.baron.hp)} hp`,
      stateHpBar(tick.baron.hp, LORD_MAX_HP, "lord hp", "lord"),
    ));
  }

  return cards;
}

function stateForceCard(player, units, status, role, detail = unitCountLabel(units), extraMarkup = "") {
  return `
    <div class="state-force-card ${formationTeamClass(player)}">
      <div class="state-card-head">
        <strong>${displayForceLabel(player)}</strong>
        <span>${status}</span>
      </div>
      <div class="state-card-meta">
        <span>${role}</span>
        <span>${detail}</span>
      </div>
      ${extraMarkup}
      <div class="state-unit-row">${unitChips(units)}</div>
    </div>
  `;
}

function coreHpMarkup(nodeId, node = {}, tick = {}) {
  if (!CORE_NODES.has(nodeId)) {
    return "";
  }

  const nodeHp = Number(node.core_hp);
  const playerHp = Number(tick.players?.[nodeId]?.core_hp);
  const hp = Number.isFinite(nodeHp) ? nodeHp : (Number.isFinite(playerHp) ? playerHp : 0);
  return stateHpBar(hp, CORE_MAX_HP, "core hp", "core");
}

function alliedLordHpMarkup(player, tick = {}) {
  if (tick.baron?.allied_lord_owner !== player) {
    return "";
  }

  const hp = Number(tick.baron?.allied_lord_hp || 0);
  return hp > 0 ? stateHpBar(hp, LORD_MAX_HP, "lord hp", "lord") : "";
}

function stateHpBar(current, max, label, kind = "") {
  const hp = Math.max(0, Number(current) || 0);
  const maxHp = Math.max(1, Number(max) || 1);
  const pct = Math.max(0, Math.min(100, (hp / maxHp) * 100));
  return `
    <div class="state-hp ${kind}">
      <div class="state-hp-row">
        <span>${label}</span>
        <span>${formatNumber(hp)}/${formatNumber(maxHp)}</span>
      </div>
      <div class="state-hp-track"><span style="width: ${pct}%;"></span></div>
    </div>
  `;
}

function unitCountLabel(units = {}) {
  const total = unitTotal(units);
  return `${formatNumber(total)} unit${total === 1 ? "" : "s"}`;
}

function readyStatus(node, player, currentTick, fallback) {
  const readyTick = Number(node.unit_ready_tick?.[player] || 0);
  return readyTick > currentTick ? `ready t${readyTick}` : fallback;
}

function forceStatus(node, player, currentTick, fallback) {
  const side = node.unit_entry_side?.[player];
  const sideText = side === "attacker" ? "left" : side === "defender" ? "right" : "center";
  return readyStatus(node, player, currentTick, `${fallback} ${sideText}`);
}

function movingGroupCard(group, label, tick, nextTick, alpha) {
  const nextGroups = [...(nextTick?.moving_groups || [])];
  const nextGroup = findNextGroup(group, nextGroups, new Set());
  const nextProgress = sameMovementSegment(group, nextGroup) ? nextGroup.progress : 1;
  const progress = interpolateValue(
    Number(group.progress || 0),
    Number(nextProgress ?? 0),
    alpha,
  );
  const boundedProgress = Math.max(0, Math.min(1, progress));
  const percent = Math.round(boundedProgress * 100);
  const intent = String(group.intent || "MOVE").toLowerCase();
  const eta = Number.isFinite(Number(group.arrives)) ? `arrives t${group.arrives}` : "en route";

  return `
    <div class="state-move-card ${teamClass(group.player)}">
      <div class="state-card-head">
        <strong>${group.player}</strong>
        <span>${label} ${phaseLabel(group.phase)} ${percent}%</span>
      </div>
      <div class="state-route-text">${routeText(group)}</div>
      <div class="state-progress"><span style="width: ${percent}%"></span></div>
      <div class="state-card-meta">
        <span>${intent} · speed ${formatNumber(group.speed || 1)}</span>
        <span>${eta}</span>
      </div>
      ${hasLordUnit(group.units) ? alliedLordHpMarkup(group.player, tick) : ""}
      <div class="state-unit-row">${unitChips(group.units)}</div>
    </div>
  `;
}

function renderBattleList(tick, nextTick, alpha) {
  const battles = tick?.active_battles || [];
  const activeIds = new Set(battles.map((battle) => battle.battle_id));
  const resolved = recentResolvedBattles(state.tick, activeIds);

  if (!battles.length && !resolved.length) {
    els.battleList.innerHTML = battleEventCards(tick?.events || []);
    return;
  }

  els.battleList.innerHTML = "";
  for (const battle of battles) {
    const nextBattle = (nextTick?.active_battles || []).find((item) => item.battle_id === battle.battle_id);
    const battleEvent = battleTickEvent(tick, battle);
    const item = document.createElement("div");
    item.className = "battle-item";
    item.innerHTML = battleCardMarkup(battle, nextBattle, alpha, battleEvent);
    els.battleList.appendChild(item);
  }

  for (const resolvedBattle of resolved) {
    const item = document.createElement("div");
    item.className = "battle-item resolved";
    item.innerHTML = battleCardMarkup(
      resolvedBattle.battle,
      null,
      0,
      resolvedBattle.event,
      `resolved ${resolvedBattle.ago}t ago`,
    );
    els.battleList.appendChild(item);
  }
}

function renderBattleListLegacy(tick, nextTick, alpha) {
  const battles = tick?.active_battles || [];
  if (!battles.length) {
    els.battleList.innerHTML = `<div class="empty">Нет активных боёв</div>`;
    return;
  }

  els.battleList.innerHTML = "";
  for (const battle of battles) {
    const nextBattle = (nextTick?.active_battles || []).find((item) => item.battle_id === battle.battle_id);
    const item = document.createElement("div");
    item.className = "battle-item";
    item.innerHTML = `
      <strong>${battle.battle_type} · ${battle.location_id}</strong>
      <div class="muted">age ${battle.age_ticks} · ${battle.participants.join(", ") || "no players"}</div>
      <div>${sideSummary(battle.sides)}</div>
      ${formationMarkup(battle.formation || [], nextBattle?.formation || [], alpha)}
    `;
    els.battleList.appendChild(item);
  }
}

function renderEvents(tick) {
  const events = tick?.events || [];
  if (!events.length) {
    els.eventList.innerHTML = `<div class="empty">Нет событий</div>`;
    return;
  }

  els.eventList.innerHTML = "";
  for (const event of events.slice(-12).reverse()) {
    const item = document.createElement("div");
    item.className = "event-item";
    item.innerHTML = eventMarkup(event);
    els.eventList.appendChild(item);
  }
}

function battleTickEvent(tick, battle) {
  return (tick?.events || []).find((event) => (
    event.type === "BATTLE_TICK" && event.battle_id === battle.battle_id
  ));
}

function battleCardMarkup(battle, nextBattle, alpha, battleEvent = null, status = null) {
  const attacks = battleEvent?.summary?.attacks || [];
  return `
    <div class="battle-header">
      <strong>${battleLabel(battle)}</strong>
      <span>${status || `${battle.age_ticks}t`}</span>
    </div>
    <div class="battle-participants">
      ${battle.participants?.join(", ") || "no players"}
      ${battleEvent?.winner ? ` · winner ${battleEvent.winner}` : ""}
    </div>
    ${battleSideCards(battle.sides)}
    ${deathSummaryMarkup(battleEvent?.summary?.deaths || {})}
    ${formationMarkup(battle.formation || [], nextBattle?.formation || [], alpha, attacks)}
  `;
}

function recentResolvedBattles(tickIndex, activeIds = new Set()) {
  if (!state.replay) {
    return [];
  }

  const result = [];
  const seen = new Set();
  const from = Math.max(0, tickIndex - 4);

  for (let index = tickIndex; index >= from; index -= 1) {
    const tick = state.replay.ticks[index];
    for (const event of [...(tick?.events || [])].reverse()) {
      if (event.type !== "BATTLE_RESOLVED" || !event.battle_id || activeIds.has(event.battle_id) || seen.has(event.battle_id)) {
        continue;
      }

      const battle = previousBattleSnapshot(index, event.battle_id);
      if (!battle) {
        continue;
      }

      const tickEvent = (tick?.events || []).find((item) => item.type === "BATTLE_TICK" && item.battle_id === event.battle_id);
      seen.add(event.battle_id);
      result.push({
        battle,
        event: tickEvent ? { ...event, summary: tickEvent.summary } : event,
        ago: tickIndex - index,
      });
    }
  }

  return result;
}

function previousBattleSnapshot(tickIndex, battleId) {
  const from = Math.max(0, tickIndex - 8);
  for (let index = tickIndex; index >= from; index -= 1) {
    const battle = (state.replay?.ticks[index]?.active_battles || []).find((item) => item.battle_id === battleId);
    if (battle) {
      return battle;
    }
  }
  return null;
}

function battleEventCards(events = []) {
  const battleEvents = events.filter((event) => (
    event.type === "BATTLE_TICK" ||
    event.type === "BATTLE_RESOLVED" ||
    event.type === "BATTLE_STARTED"
  ));

  if (!battleEvents.length) {
    return `<div class="empty">No active battles</div>`;
  }

  return `
    <div class="empty">No active battles</div>
    ${battleEvents.slice(-5).reverse().map((event) => `
      <div class="battle-item compact">
        <div class="battle-header">
          <strong>${event.type} - ${displayLocation(event.location_id || "?")}</strong>
          <span>${event.winner ? `win ${event.winner}` : event.battle_type || ""}</span>
        </div>
        <div class="battle-participants">${event.participants?.join(", ") || sideSummary(event.summary?.alive || {})}</div>
        ${deathSummaryMarkup(event.summary?.deaths || {})}
        ${attackSummaryMarkup(event.summary?.attacks || [])}
      </div>
    `).join("")}
  `;
}

function battleLabel(battle) {
  return `${String(battle.battle_type || "battle").replace("_", " ")} - ${displayLocation(battle.location_id)}`;
}

function battleSideCards(sides = {}) {
  const entries = Object.entries(sides || {});
  if (!entries.length) {
    return `<div class="battle-sides"><span class="muted">empty</span></div>`;
  }

  return `
    <div class="battle-sides">
      ${entries.map(([side, info]) => `
        <div class="battle-side ${formationTeamClass(side)}">
          <strong>${side}</strong>
          <div class="battle-units">${unitChips(info.units)}</div>
          <span class="battle-hp">${formatNumber(info.hp || 0)}hp</span>
        </div>
      `).join("")}
    </div>
  `;
}

function unitChips(units = {}) {
  const entries = orderedUnitEntries(units);
  if (!entries.length) {
    return `<span class="unit-chip empty-chip">empty</span>`;
  }

  return entries
    .map(([type, count]) => `<span class="unit-chip">${unitCode(type)}${count}</span>`)
    .join("");
}

function eventMarkup(event) {
  if (event.type === "BATTLE_TICK") {
    return `
      <strong>${event.battle_type} · ${displayLocation(event.location_id)}</strong>
      <div class="muted">${event.type}</div>
      <div>${sideSummary(event.summary?.alive || {})}</div>
      <div class="muted">dmg ${kv(event.summary?.damage_by_team || {})}</div>
    `;
  }
  if (event.type === "BATTLE_STARTED" || event.type === "BATTLE_RESOLVED") {
    return `
      <strong>${event.type} · ${displayLocation(event.location_id)}</strong>
      <div class="muted">${event.battle_type || ""}</div>
      <div>${event.winner ? `winner ${event.winner}` : event.participants?.join(", ") || ""}</div>
    `;
  }
  if (event.type === "MOVE") {
    return `
      <strong>MOVE · ${event.player}</strong>
      <div>${displayLocation(event.src)} -> ${displayLocation(event.dst)}</div>
      <div class="muted">${unitText(event.units)} · ${event.intent}</div>
    `;
  }
  if (event.type === "REINFORCE" || event.type === "CAPTURE") {
    return `
      <strong>${event.type} · ${event.player}</strong>
      <div>${event.node}</div>
    `;
  }
  return `
    <strong>${event.type}</strong>
    <div class="muted">${displayLocation(event.location_id || event.node || event.player || "")}</div>
  `;
}

function eventCounts(replay) {
  const counts = {};
  for (const tick of replay.ticks || []) {
    for (const event of tick.events || []) {
      counts[event.type] = (counts[event.type] || 0) + 1;
    }
  }
  return counts;
}

function battlePosition(battle) {
  if (battle.location_kind === "node") {
    return scalePoint(NODE_POSITIONS[battle.location_id]);
  }

  const [src, dst] = String(battle.location_id).split("|");
  return interpolate(src, dst, Number(battle.edge_position ?? 0.5));
}

function alliedLordMarkers(tick, nextTick, alpha = 0) {
  if (!tick) {
    return [];
  }

  const markers = [];
  const nextGroups = [...(nextTick?.moving_groups || [])];
  const used = new Set();

  for (const group of tick.moving_groups || []) {
    if (!hasLordUnit(group.units) || !isPlayerId(group.player)) {
      continue;
    }

    const nextGroup = findNextGroup(group, nextGroups, used);
    const nextProgress = sameMovementSegment(group, nextGroup) ? nextGroup.progress : 1;
    const progress = interpolateValue(
      Number(group.progress || 0),
      Number(nextProgress ?? 0),
      alpha,
    );
    const nodeId = isEdgeMovement(group) ? null : movementNodeId(group);
    const pos = nodeId ? scalePoint(NODE_POSITIONS[nodeId]) : movingGroupPosition(group, progress);

    markers.push({
      owner: group.player,
      pos,
      nodeId,
      target: group.final_dst || group.dst,
      action: phaseLabel(group.phase),
      inBattle: false,
      priority: nodeId ? 2 : 1,
    });
  }

  for (const battle of tick.active_battles || []) {
    const owner = alliedLordBattleOwner(battle);
    if (!owner) {
      continue;
    }

    const nodeId = battle.location_kind === "node" ? battle.location_id : null;
    markers.push({
      owner,
      pos: battlePosition(battle),
      nodeId,
      target: nodeId,
      action: battle.location_kind === "node" ? "node battle" : "edge battle",
      inBattle: true,
      priority: 3,
    });
  }

  for (const [nodeId, node] of Object.entries(tick.nodes || {})) {
    const owner = nodeAlliedLordOwner(node);
    if (!owner) {
      continue;
    }

    markers.push({
      owner,
      pos: scalePoint(NODE_POSITIONS[nodeId]),
      nodeId,
      target: nodeId,
      action: "holding",
      inBattle: false,
      priority: 2,
    });
  }

  return dedupeLordMarkers(markers);
}

function alliedLordBattleInfo(battle = {}) {
  for (const unit of normalizeFormation(battle.formation || [])) {
    if (unit.type === "L" && isPlayerId(unit.team)) {
      return {
        owner: unit.team,
        hp: unit.hp,
        maxHp: unit.maxHp || LORD_MAX_HP,
      };
    }
  }

  for (const [player, info] of Object.entries(battle.sides || {})) {
    if (isPlayerId(player) && hasLordUnit(info.units)) {
      return {
        owner: player,
        hp: Number(info.hp || 0),
        maxHp: LORD_MAX_HP,
      };
    }
  }

  return null;
}

function alliedLordBattleOwner(battle = {}) {
  return alliedLordBattleInfo(battle)?.owner || null;
}

function nodeAlliedLordOwner(node = {}) {
  if (hasLordUnit(node.units) && isPlayerId(node.owner)) {
    return node.owner;
  }

  for (const [player, units] of Object.entries(node.visiting_units || {})) {
    if (isPlayerId(player) && hasLordUnit(units)) {
      return player;
    }
  }

  return null;
}

function dedupeLordMarkers(markers = []) {
  const bySlot = new Map();
  for (const marker of markers) {
    if (!marker.pos) {
      continue;
    }
    const key = marker.nodeId || `${Math.round(marker.pos[0])}:${Math.round(marker.pos[1])}`;
    const existing = bySlot.get(key);
    if (!existing || marker.priority > existing.priority) {
      bySlot.set(key, marker);
    }
  }
  return Array.from(bySlot.values());
}

function interpolate(src, dst, progress) {
  const a = NODE_POSITIONS[src];
  const b = NODE_POSITIONS[dst];
  if (!a || !b) {
    return null;
  }
  const t = Math.max(0, Math.min(1, progress));
  return [
    scaleX(a[0] + (b[0] - a[0]) * t),
    scaleY(a[1] + (b[1] - a[1]) * t),
  ];
}

function movingGroupPosition(group, progress) {
  if (!isEdgeMovement(group)) {
    return null;
  }
  return interpolate(group.src, group.dst, progress);
}

function nodePhasePoint(nodeId, src, dst, phase, progress) {
  const node = NODE_POSITIONS[nodeId];
  if (!node) {
    return null;
  }

  const offset = 24;
  const t = Math.max(0, Math.min(1, Number(progress) || 0));
  const fromVec = directionFromNode(nodeId, src);
  const toVec = directionFromNode(nodeId, dst);

  if (phase === "node_exit") {
    return [
      node[0] + toVec[0] * offset * t,
      node[1] + toVec[1] * offset * t,
    ];
  }

  if (phase === "node_enter") {
    return [
      node[0] + fromVec[0] * offset * (1 - t),
      node[1] + fromVec[1] * offset * (1 - t),
    ];
  }

  const start = [node[0] + fromVec[0] * offset, node[1] + fromVec[1] * offset];
  const end = [node[0] + toVec[0] * offset, node[1] + toVec[1] * offset];
  return [
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
  ];
}

function directionFromNode(nodeId, otherId) {
  const node = NODE_POSITIONS[nodeId];
  const other = NODE_POSITIONS[otherId];
  if (!node || !other || nodeId === otherId) {
    return [0, 0];
  }

  const dx = other[0] - node[0];
  const dy = other[1] - node[1];
  const length = Math.max(1, Math.hypot(dx, dy));
  return [dx / length, dy / length];
}

function scalePoint(point) {
  if (!point) {
    return null;
  }
  return [scaleX(point[0]), scaleY(point[1])];
}

function scaleX(x) {
  return (x / 680) * els.arena.clientWidth;
}

function scaleY(y) {
  return (y / 410) * els.arena.clientHeight;
}

function shortNodeLabel(id) {
  return id.startsWith("CP") ? id.slice(2) : id;
}

function displayNodeLabel(id) {
  return id === "Baron" ? "Lord" : shortNodeLabel(id);
}

function displayLocation(id) {
  return String(id || "").replace("Baron", "Lord");
}

function routeText(group) {
  const path = group.path?.length ? group.path : [group.src, group.dst];
  return path.map(displayLocation).join(" -> ");
}

function phaseLabel(phase) {
  return {
    node_exit: "exit",
    edge: "edge",
    node_transit: "inside",
    node_enter: "enter",
  }[phase] || "move";
}

function displayForceLabel(id) {
  if (id === "NEUTRAL") {
    return "Neutral";
  }
  if (id === "LORD" || id === "BARON") {
    return "Lord";
  }
  return id;
}

function ownerClass(owner) {
  if (String(owner).startsWith("A")) {
    return "team-a";
  }
  if (String(owner).startsWith("B")) {
    return "team-b";
  }
  return "";
}

function nodeActivityClass(nodeId, tick) {
  if (!tick) {
    return "";
  }

  const hasBattle = (tick.active_battles || []).some((battle) => (
    battle.location_kind === "node" && battle.location_id === nodeId
  ));
  const hasMovement = (tick.moving_groups || []).some((group) => (
    group.src === nodeId || group.dst === nodeId || movementNodeId(group) === nodeId
  ));
  const hasNodeEvent = (tick.events || []).some((event) => (
    event.node === nodeId || event.location_id === nodeId || event.src === nodeId || event.dst === nodeId
  ));

  return hasBattle || hasMovement || hasNodeEvent ? "active" : "";
}

function isEdgeMovement(group) {
  return !group.phase || group.phase === "edge";
}

function movementNodeId(group) {
  if (isEdgeMovement(group)) {
    return null;
  }
  if (group.segment_node) {
    return group.segment_node;
  }
  if (group.phase === "node_exit") {
    return group.src;
  }
  if (group.phase === "node_enter") {
    return group.dst;
  }
  return null;
}

function sameMovementSegment(group, candidate) {
  if (!candidate) {
    return false;
  }
  return (
    group.phase === candidate.phase &&
    Number(group.segment_index ?? 0) === Number(candidate.segment_index ?? 0) &&
    group.src === candidate.src &&
    group.dst === candidate.dst
  );
}

function teamClass(player) {
  return String(player).startsWith("A") ? "team-a" : "team-b";
}

function isPlayerId(id) {
  return /^[AB]\d+$/.test(String(id || ""));
}

function hasLordUnit(units = {}) {
  return Number(units.LORD || units.BARON || units.L || units.B || 0) > 0;
}

function interpolateValue(from, to, alpha) {
  const start = Number.isFinite(Number(from)) ? Number(from) : 0;
  const end = Number.isFinite(Number(to)) ? Number(to) : start;
  const t = Math.max(0, Math.min(1, Number(alpha) || 0));
  return start + (end - start) * t;
}

function findNextGroup(group, nextGroups, used) {
  for (let index = 0; index < nextGroups.length; index += 1) {
    if (used.has(index)) {
      continue;
    }

    const candidate = nextGroups[index];
    if (
      candidate.player === group.player &&
      candidate.src === group.src &&
      candidate.dst === group.dst &&
      candidate.intent === group.intent &&
      unitSignature(candidate.units) === unitSignature(group.units)
    ) {
      used.add(index);
      return candidate;
    }
  }
  return null;
}

function unitSignature(units = {}) {
  return Object.entries(units || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([type, count]) => `${type}:${count}`)
    .join("|");
}

function formationMarkup(formation = [], nextFormation = [], alpha = 0, attacks = []) {
  const units = interpolateFormation(formation, nextFormation, alpha);
  const bounds = formationBounds(units);
  const placed = [];
  const stacks = formationStacks(units);
  const stackByUid = stackUidIndex(stacks);
  const positioned = stacks.map((stack) => {
    const hpRatio = stack.maxHp > 0 ? stack.hp / stack.maxHp : 1;
    const x = formationX(stack.pos, bounds);
    let y = formationLane(stack.team, stack.type);
    const collisions = placed.filter((item) => Math.abs(item.x - x) < 10 && Math.abs(item.y - y) < 9).length;
    const offsets = [0, -8, 8, -16, 16];
    y = Math.max(12, Math.min(88, y + offsets[collisions % offsets.length]));
    placed.push({ x, y });
    return { ...stack, x, y, hpRatio };
  });

  const links = attackLinks(attacks, positioned, stackByUid);
  const markers = positioned.map((stack) => {
    const hpText = `${formatNumber(stack.hp)}/${formatNumber(stack.maxHp)}`;
    const hpPct = Math.max(0, Math.min(100, stack.hpRatio * 100));
    return `
      <span
        class="formation-stack ${formationTeamClass(stack.team)} ${stack.hpRatio < 0.35 ? "low" : ""}"
        style="left:${stack.x}%;top:${stack.y}%;"
        title="${stack.team} ${stack.type} x${stack.count} ${hpText} range ${formatNumber(stack.range)}"
      >
        <span class="stack-kind">${stack.type}</span>
        <span class="stack-count">${stack.count}</span>
        <span class="stack-health"><span style="width:${hpPct}%;"></span></span>
      </span>
    `;
  }).join("");

  return `
    <div class="formation">
      ${attackLayerMarkup(links)}
      ${markers}
    </div>
    ${attackSummaryMarkup(attacks, stackByUid)}
  `;
}

function formationStacks(units = []) {
  const stacks = new Map();

  for (const unit of units) {
    const { uid, team, type, hp, maxHp, pos, range } = unit;
    const key = `${team}|${type}`;
    if (!stacks.has(key)) {
      stacks.set(key, {
        id: key,
        team,
        type,
        count: 0,
        hp: 0,
        maxHp: 0,
        posTotal: 0,
        range: 0,
        uids: [],
      });
    }

    const stack = stacks.get(key);
    stack.count += 1;
    stack.hp += Number(hp) || 0;
    stack.maxHp += Number(maxHp) || 0;
    stack.posTotal += Number(pos) || 0;
    stack.range = Math.max(stack.range, Number(range) || 0);
    if (uid !== null && uid !== undefined) {
      stack.uids.push(uid);
    }
  }

  return Array.from(stacks.values())
    .map((stack) => ({
      ...stack,
      pos: stack.count > 0 ? stack.posTotal / stack.count : 0,
    }))
    .sort((left, right) => {
      const laneDelta = formationLane(left.team, left.type) - formationLane(right.team, right.type);
      return laneDelta || left.pos - right.pos || left.team.localeCompare(right.team);
    });
}

function formationBounds(units = []) {
  const positions = units
    .map((unit) => Number(unit.pos))
    .filter((pos) => Number.isFinite(pos));

  if (!positions.length) {
    return { min: 0, max: 20 };
  }

  const radius = Math.max(10, ...positions.map((pos) => Math.abs(pos - 10)));
  return { min: 10 - radius, max: 10 + radius };
}

function formationX(pos, bounds) {
  const value = Number(pos);
  if (!Number.isFinite(value)) {
    return 50;
  }

  const span = Math.max(1, bounds.max - bounds.min);
  return 4 + Math.max(0, Math.min(1, (value - bounds.min) / span)) * 92;
}

function stackUidIndex(stacks = []) {
  const index = new Map();
  for (const stack of stacks) {
    for (const uid of stack.uids) {
      index.set(uid, stack);
    }
  }
  return index;
}

function attackLinks(attacks = [], stacks = [], stackByUid = new Map()) {
  const positionedById = new Map(stacks.map((stack) => [stack.id, stack]));
  const links = new Map();

  for (const attack of attacks) {
    const attacker = stackByUid.get(attack[0]);
    const target = stackByUid.get(attack[1]);
    if (!attacker || !target) {
      continue;
    }

    const key = `${attacker.id}->${target.id}`;
    const existing = links.get(key) || {
      from: positionedById.get(attacker.id),
      to: positionedById.get(target.id),
      damage: 0,
      ranged: attacker.type === "R",
    };
    existing.damage += Number(attack[2]) || 0;
    links.set(key, existing);
  }

  return Array.from(links.values()).filter((link) => link.from && link.to);
}

function attackLayerMarkup(links = []) {
  if (!links.length) {
    return "";
  }

  return `
    <svg class="attack-layer" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      ${links.map((link) => `
        <line
          class="attack-link ${formationTeamClass(link.from.team)} ${link.ranged ? "ranged" : ""}"
          x1="${link.from.x}" y1="${link.from.y}"
          x2="${link.to.x}" y2="${link.to.y}"
        />
      `).join("")}
    </svg>
  `;
}

function attackSummaryMarkup(attacks = [], stackByUid = new Map()) {
  if (!attacks.length) {
    return `<div class="attack-summary muted">closing / no units in range</div>`;
  }

  const rows = attacks.slice(0, 4).map((attack) => {
    const attacker = stackByUid.get(attack[0]);
    const target = stackByUid.get(attack[1]);
    const attackerText = attacker ? `${attacker.team} ${attacker.type}` : `${attack[3]} ${displayUnitCode(attack[4])}`;
    const targetText = target ? `${target.team} ${target.type}` : `${attack[5]} ${displayUnitCode(attack[6])}`;
    return `<span>${attackerText} -> ${targetText} ${formatNumber(attack[2])}</span>`;
  }).join("");

  const more = attacks.length > 4 ? `<span>+${attacks.length - 4} more</span>` : "";
  return `<div class="attack-summary">${rows}${more}</div>`;
}

function deathSummaryMarkup(deaths = {}) {
  const rows = Object.entries(deaths || {}).flatMap(([team, units]) => (
    orderedUnitEntries(units).map(([type, count]) => `${team} -${unitCode(type)}${count}`)
  ));

  if (!rows.length) {
    return "";
  }

  return `<div class="death-summary">${rows.map((row) => `<span>${row}</span>`).join("")}</div>`;
}

function interpolateFormation(formation = [], nextFormation = [], alpha = 0) {
  const current = normalizeFormation(formation);
  const next = normalizeFormation(nextFormation);
  const nextByUid = new Map(next.filter((unit) => unit.uid !== null).map((unit) => [unit.uid, unit]));
  const nextBuckets = formationBuckets(next);

  return current.map((unit) => {
    const bucket = nextBuckets.get(`${unit.team}|${unit.type}`) || [];
    const nextUnit = unit.uid !== null ? nextByUid.get(unit.uid) : bucket.shift();

    return {
      ...unit,
      hp: interpolateValue(unit.hp, nextUnit?.hp ?? unit.hp, alpha),
      pos: interpolateValue(unit.pos, nextUnit?.pos ?? unit.pos, alpha),
    };
  });
}

function formationBuckets(formation = []) {
  const buckets = new Map();
  const ordered = [...formation].sort((left, right) => Number(left.pos) - Number(right.pos));

  for (const unit of ordered) {
    const key = `${unit.team}|${unit.type}`;
    if (!buckets.has(key)) {
      buckets.set(key, []);
    }
    buckets.get(key).push(unit);
  }
  return buckets;
}

function normalizeFormation(formation = []) {
  return formation.map((row, index) => {
    if (Array.isArray(row) && row.length >= 7) {
      return {
        uid: row[0],
        team: row[1],
        type: displayUnitCode(row[2]),
        hp: Number(row[3]) || 0,
        maxHp: Number(row[4]) || 0,
        pos: Number(row[5]) || 0,
        range: Number(row[6]) || 0,
      };
    }

    return {
      uid: null,
      team: row[0],
      type: displayUnitCode(row[1]),
      hp: Number(row[2]) || 0,
      maxHp: Number(row[3]) || 0,
      pos: Number(row[4]) || 0,
      range: defaultUnitRange(row[1]),
      fallbackIndex: index,
    };
  });
}

function formationLane(team, type = "") {
  if (team === "NEUTRAL" || team === "BARON" || team === "LORD") {
    return 50;
  }
  const lanes = {
    V: 38,
    S: 25,
    R: 14,
    A: 14,
    C: 18,
    N: 50,
    L: 50,
    B: 50,
  };
  const lane = lanes[type] ?? 30;
  return String(team).startsWith("A") ? lane : 100 - lane;
}

function formationTeamClass(team) {
  if (team === "NEUTRAL" || team === "BARON" || team === "LORD") {
    return "neutral";
  }
  return teamClass(team);
}

function unitTotal(units = {}) {
  return Object.values(units || {}).reduce((sum, count) => sum + Number(count || 0), 0);
}

function nodeDisplayTotal(nodeState = {}, tick = {}, nodeId = "") {
  const movingInside = (tick.moving_groups || [])
    .filter((group) => movementNodeId(group) === nodeId)
    .reduce((sum, group) => sum + unitTotal(group.units), 0);
  return unitTotal(nodeState.units) + visitingTotal(nodeState.visiting_units) + movingInside;
}

function visitingTotal(visitors = {}) {
  return Object.values(visitors || {}).reduce((sum, units) => sum + unitTotal(units), 0);
}

function nodeTitle(id, nodeState) {
  return `${id} | owner ${nodeState.owner || "?"} | units ${unitText(nodeState.units)} | neutral ${nodeState.neutral_hp || 0}`;
}

function unitText(units = {}) {
  const parts = orderedUnitEntries(units).map(([type, count]) => `${unitCode(type)}${count}`);
  return parts.length ? parts.join(" ") : "empty";
}

function orderedUnitEntries(units = {}) {
  const order = {
    VANGUARD: 0,
    SKIRMISHER: 1,
    RANGER: 2,
    ARCHER: 2,
    CORE: 3,
    NEUTRAL: 4,
    BARON: 5,
    LORD: 5,
  };

  return Object.entries(units || {})
    .filter(([, count]) => Number(count) > 0)
    .sort(([left], [right]) => (order[left] ?? 99) - (order[right] ?? 99) || left.localeCompare(right));
}

function unitCode(type) {
  return {
    VANGUARD: "V",
    SKIRMISHER: "S",
    RANGER: "R",
    ARCHER: "R",
    CORE: "C",
    NEUTRAL: "N",
    BARON: "L",
    LORD: "L",
  }[type] || String(type).slice(0, 1);
}

function displayUnitCode(type) {
  return {
    A: "R",
    B: "L",
    V: "V",
    S: "S",
    R: "R",
    C: "C",
    N: "N",
    L: "L",
    VANGUARD: "V",
    SKIRMISHER: "S",
    RANGER: "R",
    ARCHER: "R",
    CORE: "C",
    NEUTRAL: "N",
    BARON: "L",
    LORD: "L",
  }[type] || String(type).slice(0, 1);
}

function defaultUnitRange(type) {
  return {
    V: 1,
    S: 1.5,
    R: 3,
    A: 3,
    C: 0.5,
    N: 4,
    B: 2,
    L: 2,
  }[type] || 1;
}

function sideSummary(sides = {}) {
  const parts = Object.entries(sides).map(([side, info]) => {
    const hp = formatNumber(info.hp || 0);
    return `${side}: ${unitText(info.units)} (${hp}hp)`;
  });
  return parts.length ? parts.join("<br>") : `<span class="muted">empty</span>`;
}

function kv(values = {}) {
  const parts = Object.entries(values).map(([key, value]) => `${key}:${formatNumber(value)}`);
  return parts.length ? parts.join(" ") : "none";
}

function formatNumber(value) {
  return Number(value).toFixed(Number(value) % 1 === 0 ? 0 : 1);
}

window.addEventListener("resize", () => {
  if (state.replay) {
    renderTick();
  } else {
    renderNodes(null);
  }
});

setup();
