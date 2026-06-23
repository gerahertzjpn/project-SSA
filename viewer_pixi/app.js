import { Application, Container, Graphics, Text } from "pixi.js";

const WORLD_W = 680;
const WORLD_H = 410;
const BASE_TICK_MS = 700;
const HANDOFF_DISTANCE = 90;
const MAX_EXPANDED_UNITS = 260;
const HANDOFF_BACKFILL_DISTANCE = 135;
const CORE_HP = 80;
const ATTACK_PREVIEW_ALPHA = 0.5;
// Mirror the engine's melee/ranged split (combat2d `_uses_melee_engagement`,
// combat `_uses_melee_engagement`: range <= 1.25 fights in contact). Units above
// this threshold shoot from a distance, so their attack lines render as ranged.
const MELEE_RANGE_THRESHOLD = 1.25;
const SPRITE_EASE = {
  moving: 7.0,
  battle: 8.5,
  fighting: 8.5,
  stationed: 10.5,
  visiting: 10.5,
  neutral: 10.5,
  unit: 9.0,
};
const FADE_IN_PER_SECOND = 5.5;
const FADE_OUT_PER_SECOND = 3.6;
const DEATH_HOLD_SECONDS = 0.24;
const DEATH_FADE_OUT_PER_SECOND = 1.45;

const FALLBACK_REPLAYS = Array.from({ length: 20 }, (_, index) => ({
  label: `replay_${index + 1}.json`,
  path: `../replays/replay_${index + 1}.json`,
}));

const COLORS = {
  bg: 0x0b0f14,
  bgDeep: 0x080b10,
  lane: 0x4a5763,
  laneGlow: 0x2a3742,
  tunnel: 0x2b343f,
  room: 0x222932,
  roomLine: 0x758292,
  teamA: 0xff5b6b,
  teamABright: 0xff8a96,
  teamABg: 0x3c1c22,
  teamAHalf: 0x1a1013,
  teamB: 0x4fa8ff,
  teamBBright: 0x8cc8ff,
  teamBBg: 0x16314f,
  teamBHalf: 0x0d1620,
  neutral: 0xe2b056,
  neutralBright: 0xffd27a,
  neutralBg: 0x33280f,
  deadCore: 0xffffff,
  deadCoreBg: 0xf2f5f8,
  deadCoreText: 0x151b22,
  deadUnit: 0x6c747e,
  outline: 0x0a0d12,
  hpTrack: 0x05080c,
  text: 0xeef4fa,
  muted: 0x9facba,
  danger: 0xff8f8f,
  attack: 0xf0d18a,
  buildingBg: 0x0c1016,
  buildingProgress: 0xf0d18a,
  spark: 0xfff2c4,
  baron: 0xc77dff,
  baronGlow: 0x7b3fb0,
  baronBright: 0xddb0ff,
  baronBg: 0x241634,
};

// Faction identity: every unit's BODY is its team colour (red A / blue B) so the
// two sides read instantly; the FACTION is carried by silhouette + an accent
// colour used for trim/lens/spikes/glow, distinct from both team colours.
const FACTION_ACCENT = {
  humans: 0xeaf2fb,   // clean steel-white
  infected: 0x8fe34a, // toxic lime
  robots: 0xffb547,   // amber
  neutral: 0xd8c08a,
  core: 0xeef4fa,
  lord: 0xffd27a,
};

// Per-unit-type visual spec. faction drives accent+silhouette family; klass
// drives base size + tier markers; ranged adds a muzzle cue + drives projectiles.
const UNIT_SPEC = {
  // Humans — angular / military
  V:  { fac: "humans", klass: "std", ranged: false, dir: false },
  R:  { fac: "humans", klass: "std", ranged: true, dir: true },
  S:  { fac: "humans", klass: "light", ranged: false, dir: true },
  RD: { fac: "humans", klass: "tech1", ranged: false, dir: false },
  MP: { fac: "humans", klass: "tech2", ranged: true, dir: true },
  // Infected — organic / spiky
  SW: { fac: "infected", klass: "swarm", ranged: false, dir: false },
  SP: { fac: "infected", klass: "std", ranged: true, dir: true },
  BR: { fac: "infected", klass: "heavy", ranged: false, dir: false },
  RV: { fac: "infected", klass: "tech1", ranged: false, dir: false },
  BH: { fac: "infected", klass: "tech2", ranged: false, dir: false },
  // Robots — mechanical / hard-edged
  WK: { fac: "robots", klass: "std", ranged: false, dir: false },
  LS: { fac: "robots", klass: "std", ranged: true, dir: true },
  DR: { fac: "robots", klass: "swarm", ranged: false, dir: true },
  GM: { fac: "robots", klass: "tech1", ranged: false, dir: false },
  SG: { fac: "robots", klass: "tech2", ranged: true, dir: true },
  // Objectives / neutral
  N:  { fac: "neutral", klass: "neutral", ranged: false, dir: false },
  C:  { fac: "core", klass: "core", ranged: false, dir: false },
  L:  { fac: "lord", klass: "lord", ranged: false, dir: false },
};

const KLASS_SIZE = {
  swarm: 3.7,
  light: 4.5,
  std: 5.3,
  tech1: 5.9,
  heavy: 6.7,
  tech2: 7.3,
  neutral: 5.0,
  core: 9.0,
  lord: 12.0,
};

// Live effect pools (projectiles + particles), advanced every frame in real time.
const PROJECTILE_LIFE = 0.16;     // seconds in flight, fast tracers
const PARTICLE_GRAVITY = 38;      // px/s^2 for debris fall
const MAX_PROJECTILES = 220;
const MAX_PARTICLES = 520;
const MAX_SPAWN_PER_TICK = 70;

const BUILDING_LABELS = {
  TURRET: "T",
  BOOSTER: "BO",
  POWER: "PW",
  LIFE: "LF",
  ECONOMY_COLLECTOR: "$",
  TECH_1: "T1",
  TECH_2: "T2",
};

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

const TUNNELS = new Set([
  "A1|CP1", "A1|CP2", "A2|CP1", "A2|CP2", "A3|CP1", "A3|CP2",
  "B1|CP5", "B1|CP6", "B2|CP5", "B2|CP6", "B3|CP5", "B3|CP6",
]);

const PLAYER_IDS = ["A1", "A2", "A3", "B1", "B2", "B3"];
const TEAM_PLAYERS = {
  TEAM_A: ["A1", "A2", "A3"],
  TEAM_B: ["B1", "B2", "B3"],
};
const TEAM_LABELS = {
  TEAM_A: "Team A",
  TEAM_B: "Team B",
};
const DEFAULT_FACTIONS = ["humans", "infected", "robots"];
const FACTION_LABELS = {
  humans: "Humans",
  infected: "Infected",
  robots: "Robots",
};

// Coherent strategy presets — mirrors the server; used offline. One preset per
// player (no incoherent tempo×win_path combos).
const FALLBACK_STRATEGIES = {
  none: { value: "none", label: "— нет приказа —" },
  presets: [
    { value: "standard", label: "Стандарт" },
    { value: "core_rush", label: "Раш ядра" },
    { value: "all_in", label: "Олл-ин" },
    { value: "turtle", label: "Черепаха" },
    { value: "macro", label: "Макро-эконом" },
    { value: "baron", label: "Контроль Лорда" },
  ],
};
const FALLBACK_MODELS = [
  { id: "random", label: "Random Bot", kind: "baseline", show_by_default: true },
  { id: "scripted", label: "Scripted Bot", kind: "baseline", show_by_default: true },
  { id: "scripted_core", label: "Core Bot", kind: "baseline", show_by_default: true },
  { id: "scripted_core_raider", label: "Raider Core", kind: "baseline", show_by_default: true },
  { id: "scripted_core_bmp", label: "BMP Core", kind: "baseline", show_by_default: true },
];

const appState = {
  app: null,
  replay: null,
  samples: [...FALLBACK_REPLAYS],
  models: [...FALLBACK_MODELS],
  players: [...PLAYER_IDS],
  teams: { ...TEAM_PLAYERS },
  factions: [...DEFAULT_FACTIONS],
  strategies: null,
  apiAvailable: false,
  apiStale: false,
  serverInfo: null,
  replayLabel: "",
  tickFloat: 0,
  playing: false,
  playStartedAt: 0,
  playStartTick: 0,
  lastNow: performance.now(),
  camera: { x: 0, y: 0, scale: 1 },
  desired: new Map(),
  sprites: new Map(),
  selected: null,
  nodeMeta: {},
  lastMapSignature: "",
  generating: false,
  maxUiRuns: 100,
  configExpanded: readStoredFlag("pixiConfigExpanded", false),
  showAllModels: readStoredFlag("pixiShowAllModels", false),
};

const layers = {
  root: new Container(),
  bg: new Container(),
  map: new Container(),
  labels: new Container(),
  units: new Container(),
  effects: new Container(),
  particles: new Container(),
  screen: new Container(),
};

// Live, frame-advanced visual effects (independent of the per-tick replay data).
const fx = {
  projectiles: [],
  particles: [],
  shake: 0,
  shakeX: 0,
  shakeY: 0,
  coreFlash: 0,
  coreFlashTeam: null,
  lastEffectTick: -1,
  particleG: new Graphics(),
  projectileG: new Graphics(),
  screenG: new Graphics(),
};

const EMBED = readEmbedParams();

const els = {
  topHud: document.getElementById("topHud"),
  stage: document.getElementById("pixiStage"),
  backBtn: document.getElementById("backBtn"),
  embedTitle: document.getElementById("embedTitle"),
  metaText: document.getElementById("metaText"),
  replaySelect: document.getElementById("replaySelect"),
  loadBtn: document.getElementById("loadBtn"),
  configToggleBtn: document.getElementById("configToggleBtn"),
  modelsToggleBtn: document.getElementById("modelsToggleBtn"),
  playerConfigGrid: document.getElementById("playerConfigGrid"),
  generateCountInput: document.getElementById("generateCountInput"),
  generateSeedInput: document.getElementById("generateSeedInput"),
  generateBtn: document.getElementById("generateBtn"),
  generateStatus: document.getElementById("generateStatus"),
  prevBtn: document.getElementById("prevBtn"),
  playBtn: document.getElementById("playBtn"),
  nextBtn: document.getElementById("nextBtn"),
  tickSlider: document.getElementById("tickSlider"),
  tickNow: document.getElementById("tickNow"),
  tickMax: document.getElementById("tickMax"),
  speedSelect: document.getElementById("speedSelect"),
  inspectPanel: document.getElementById("inspectPanel"),
  inspectCloseBtn: document.getElementById("inspectCloseBtn"),
  inspectBody: document.getElementById("inspectBody"),
};

setup();

async function setup() {
  const app = new Application();
  await app.init({
    resizeTo: els.stage,
    background: COLORS.bg,
    antialias: true,
    autoDensity: true,
    resolution: Math.min(window.devicePixelRatio || 1, 2),
    preference: "webgl",
  });
  appState.app = app;
  els.stage.appendChild(app.canvas);

  layers.root.addChild(layers.bg, layers.map, layers.labels, layers.units, layers.effects, layers.particles);
  layers.effects.addChild(fx.projectileG);
  layers.particles.addChild(fx.particleG);
  app.stage.addChild(layers.root);
  // Screen-space overlay (vignette / core-hit flash), not affected by camera.
  layers.screen.addChild(fx.screenG);
  app.stage.addChild(layers.screen);
  drawBackground();

  applyConfigMode();
  applyModelScope();
  applyEmbedMode();
  // applyEmbedMode() adds .embed-replay, which shrinks #pixiStage (reserves the transport-bar band so
  // it no longer overlaps the match). resizeTo only re-measures on window 'resize', so force a one-time
  // re-measure now that the stage box changed — otherwise the canvas keeps full height and the bar
  // covers the bottom of the map.
  if (EMBED.active && typeof app.resize === "function") app.resize();
  bindControls();
  await loadReplayList();
  if (EMBED.active) {
    appState.samples = [{ label: EMBED.label, path: EMBED.replay }];
    populateReplaySelect();
    els.replaySelect.value = "0";
  }
  drawStaticMap();
  fitCamera();
  await loadSelectedReplay();
  app.ticker.add((ticker) => frame(ticker.deltaMS));
}

function readEmbedParams() {
  const params = new URLSearchParams(window.location.search);
  const replay = params.get("replay");
  if (!replay) {
    return { active: false };
  }
  return {
    active: true,
    replay,
    live: params.get("live") === "1",   // LIVE broadcast: no scrubbing, no winner spoiler, auto-play
    chrome: params.get("chrome"),       // "ext" => host page supplies the chrome; hide our title/back
    speed: params.get("speed"),         // optional playback-speed carry-over across LIVE map reloads
    back: params.get("back") || "./home.html",
    label: params.get("label") || "Match recording",
  };
}

function applyEmbedMode() {
  if (!EMBED.active) {
    return;
  }
  document.body.classList.add("embed-replay");
  if (EMBED.live) {
    // LIVE broadcast: hide only the SEEKING controls (scrubber + step + tick readout) — no перемотка —
    // but KEEP play/pause and the speed selector so the viewer can still control playback speed.
    document.body.classList.add("live-broadcast");
    [els.prevBtn, els.nextBtn, els.tickSlider].forEach((elm) => { if (elm) elm.style.display = "none"; });
    const readout = document.querySelector(".tick-readout");
    if (readout) readout.style.display = "none";
  }
  // chrome=ext: the host page (World broadcast overlay) provides the title/scoreboard/close — keep the
  // iframe's own floating title + back button hidden so the match canvas stays clean underneath.
  if (EMBED.chrome === "ext") {
    return;
  }
  if (els.backBtn) {
    els.backBtn.href = EMBED.back;
    els.backBtn.classList.remove("is-hidden");
  }
  if (els.embedTitle) {
    els.embedTitle.textContent = EMBED.label;
    els.embedTitle.classList.remove("is-hidden");
  }
}

function bindControls() {
  window.addEventListener("resize", () => {
    fitCamera(false);
    drawStaticMap();
  });
  els.loadBtn.addEventListener("click", loadSelectedReplay);
  els.configToggleBtn.addEventListener("click", toggleConfigMode);
  els.modelsToggleBtn.addEventListener("click", toggleModelScope);
  els.generateBtn.addEventListener("click", generateReplays);
  els.generateCountInput.addEventListener("input", validateGeneratorControls);
  els.generateSeedInput.addEventListener("input", validateGeneratorControls);
  els.prevBtn.addEventListener("click", () => seekTo(Math.floor(appState.tickFloat) - 1));
  els.nextBtn.addEventListener("click", () => seekTo(Math.floor(appState.tickFloat) + 1));
  els.playBtn.addEventListener("click", togglePlay);
  els.tickSlider.addEventListener("input", (event) => seekTo(Number(event.target.value)));
  els.speedSelect.addEventListener("change", () => resetPlaybackClock(performance.now()));
  els.stage.addEventListener("click", (event) => selectAt(event));
  els.inspectCloseBtn.addEventListener("click", closeInspector);
}

function toggleConfigMode() {
  appState.configExpanded = !appState.configExpanded;
  storeFlag("pixiConfigExpanded", appState.configExpanded);
  applyConfigMode();
}

function applyConfigMode() {
  els.topHud.classList.toggle("compact", !appState.configExpanded);
  els.configToggleBtn.textContent = appState.configExpanded ? "Compact" : "Config";
  els.configToggleBtn.setAttribute("aria-expanded", appState.configExpanded ? "true" : "false");
}

function toggleModelScope() {
  appState.showAllModels = !appState.showAllModels;
  storeFlag("pixiShowAllModels", appState.showAllModels);
  applyModelScope();
  populatePlayerConfigGrid();
  validateGeneratorControls();
}

function applyModelScope() {
  els.topHud.classList.toggle("show-all-models", appState.showAllModels);
  els.modelsToggleBtn.textContent = appState.showAllModels ? "Latest only" : "All models";
  els.modelsToggleBtn.setAttribute("aria-pressed", appState.showAllModels ? "true" : "false");
  els.modelsToggleBtn.title = appState.showAllModels
    ? "Show only current latest models"
    : "Show every compatible checkpoint";
}

function closeInspector() {
  appState.selected = null;
  updateInspector();
}

function readStoredFlag(key, fallback) {
  try {
    const value = window.localStorage.getItem(key);
    return value === null ? fallback : value === "1";
  } catch {
    return fallback;
  }
}

function storeFlag(key, value) {
  try {
    window.localStorage.setItem(key, value ? "1" : "0");
  } catch {
    // Ignore storage failures; the current session still updates.
  }
}

async function loadReplayList() {
  try {
    const payload = await fetchJson("/api/models");
    appState.apiAvailable = true;
    appState.serverInfo = payload.server || null;
    appState.apiStale = Boolean(appState.serverInfo?.stale);
    appState.samples = payload.replays?.length ? payload.replays : [...FALLBACK_REPLAYS];
    appState.models = payload.models?.length ? payload.models : [...FALLBACK_MODELS];
    appState.strategies = payload.strategies || appState.strategies;
    appState.players = payload.players?.length ? payload.players : [...PLAYER_IDS];
    appState.teams = payload.teams || { ...TEAM_PLAYERS };
    appState.factions = payload.factions?.length ? payload.factions : [...DEFAULT_FACTIONS];
    appState.maxUiRuns = Number(payload.max_ui_runs || 100);
    els.generateCountInput.max = String(appState.maxUiRuns);
    setGenerateStatus(
      appState.apiStale ? "API stale" : "Ready",
      appState.apiStale ? "error" : "",
      serverStatusLabel(appState.serverInfo),
    );
  } catch (error) {
    appState.apiAvailable = false;
    appState.apiStale = false;
    appState.serverInfo = null;
    appState.samples = [...FALLBACK_REPLAYS];
    appState.models = [...FALLBACK_MODELS];
    appState.players = [...PLAYER_IDS];
    appState.teams = { ...TEAM_PLAYERS };
    appState.factions = [...DEFAULT_FACTIONS];
    setGenerateStatus("API offline", "error", error.message || "Generation API offline");
  }
  populateReplaySelect();
  populatePlayerConfigGrid();
  validateGeneratorControls();
}

function populateReplaySelect() {
  els.replaySelect.innerHTML = appState.samples
    .map((item, index) => `<option value="${index}">${escapeHtml(item.label || `replay_${index + 1}.json`)}</option>`)
    .join("");
}

function populatePlayerConfigGrid() {
  if (!els.playerConfigGrid) {
    return;
  }
  const selectedModels = playerModelSelection({ allowMissing: true });
  const selectedFactions = playerFactionSelection({ allowMissing: true });
  const defaultModel = preferredModelId();

  els.playerConfigGrid.innerHTML = "";
  for (const [teamId, teamPlayers] of Object.entries(normalizedTeams())) {
    const group = document.createElement("section");
    group.className = `team-config ${teamClass(teamId)}`;

    const header = document.createElement("div");
    header.className = "team-config-head";

    const title = document.createElement("strong");
    title.textContent = TEAM_LABELS[teamId] || teamId;

    const teamModel = document.createElement("select");
    teamModel.className = "team-model-select";
    teamModel.dataset.teamModel = teamId;
    teamModel.setAttribute("aria-label", `${TEAM_LABELS[teamId] || teamId} model preset`);
    populateModelOptions(teamModel, defaultModel);

    const teamFaction = document.createElement("select");
    teamFaction.className = "team-faction-select";
    teamFaction.dataset.teamFaction = teamId;
    teamFaction.setAttribute("aria-label", `${TEAM_LABELS[teamId] || teamId} faction preset`);
    populateFactionOptions(teamFaction, "humans");

    const apply = document.createElement("button");
    apply.className = "mini-command";
    apply.type = "button";
    apply.textContent = "Apply";
    apply.addEventListener("click", () => applyTeamConfig(teamId));

    header.append(title, teamModel, teamFaction, apply);
    group.appendChild(header);

    const rows = document.createElement("div");
    rows.className = "player-config-rows";
    for (const playerId of teamPlayers) {
      const row = document.createElement("div");
      row.className = "player-config-row";

      const slot = document.createElement("span");
      slot.className = "player-slot";
      slot.textContent = playerId;

      const model = document.createElement("select");
      model.dataset.playerModel = playerId;
      model.setAttribute("aria-label", `${playerId} model`);
      populateModelOptions(model, selectedModels[playerId] || defaultModel);

      const faction = document.createElement("select");
      faction.dataset.playerFaction = playerId;
      faction.setAttribute("aria-label", `${playerId} faction`);
      populateFactionOptions(faction, selectedFactions[playerId] || "humans");
      faction.addEventListener("change", validateGeneratorControls);

      model.addEventListener("change", () => {
        syncFactionLock(model, faction);
        validateGeneratorControls();
      });
      syncFactionLock(model, faction);

      const strategy = buildStrategyControls(playerId);
      row.append(slot, model, faction, strategy);
      rows.appendChild(row);
    }
    group.appendChild(rows);
    els.playerConfigGrid.appendChild(group);
  }
}

function addOption(select, value, label) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  select.appendChild(opt);
}

// Per-player strategy (coach order): ONE coherent preset, or "no order" (-> the net
// plays freely). Roles/tempo/win_path are derived server-side from the preset, so
// there are no incoherent combos (e.g. turtle+core_rush). Sent as `orders`.
function buildStrategyControls(playerId) {
  const voc = appState.strategies || FALLBACK_STRATEGIES;
  const wrap = document.createElement("span");
  wrap.className = "player-strategy";

  const strat = document.createElement("select");
  strat.dataset.playerStrategy = playerId;
  strat.title = "Стратегия (приказ тренера) — «нет приказа» = играет свободно";
  strat.setAttribute("aria-label", `${playerId} стратегия`);
  addOption(strat, voc.none.value, voc.none.label);
  for (const o of (voc.presets || [])) addOption(strat, o.value, o.label);
  strat.value = "none";

  wrap.append(strat);
  return wrap;
}

// Read the per-player strategy selectors into the orders payload.
function playerOrderSelection() {
  const grid = els.playerConfigGrid;
  const orders = {};
  for (const playerId of appState.players.length ? appState.players : PLAYER_IDS) {
    const strat = grid?.querySelector(`select[data-player-strategy="${playerId}"]`);
    if (!strat || strat.value === "none") {
      orders[playerId] = { mode: "none" };
    } else {
      orders[playerId] = { strategy: strat.value };
    }
  }
  return orders;
}

function populateModelOptions(select, selectedId) {
  const models = visibleModels();
  select.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = modelLabel(model);
    option.title = modelTitle(model);
    select.appendChild(option);
  }
  select.value = models.some((model) => model.id === selectedId) ? selectedId : models[0]?.id || "random";
}

function visibleModels() {
  const models = appState.models.length ? appState.models : FALLBACK_MODELS;
  if (appState.showAllModels) {
    return [...models];
  }
  const latest = models.filter((model) => (
    model.kind === "baseline"
    || model.show_by_default === true
    || (model.show_by_default !== false && model.is_latest)
  ));
  return latest.length ? latest : [...models];
}

function factionForModel(modelId) {
  return appState.models.find((m) => m.id === modelId)?.faction || null;
}

function syncFactionLock(modelSelect, factionSelect) {
  const faction = factionForModel(modelSelect.value);
  if (faction) {
    factionSelect.value = faction;
    factionSelect.disabled = true;
    factionSelect.title = `Locked to ${FACTION_LABELS[faction] || faction} by selected model`;
  } else {
    factionSelect.disabled = false;
    factionSelect.title = "";
  }
}

function populateFactionOptions(select, selectedFaction) {
  const factions = appState.factions.length ? appState.factions : DEFAULT_FACTIONS;
  select.innerHTML = "";
  for (const faction of factions) {
    const option = document.createElement("option");
    option.value = faction;
    option.textContent = FACTION_LABELS[faction] || faction;
    select.appendChild(option);
  }
  select.value = factions.includes(selectedFaction) ? selectedFaction : factions[0] || "humans";
}

function modelLabel(model = {}) {
  return model.label || model.id || "?";
}

function modelTitle(model = {}) {
  const checkpoint = model.checkpoint || model.label || model.id;
  if (model.policy_state) {
    return `${checkpoint} | policy state ${model.policy_state} | assignable to any player`;
  }
  return checkpoint;
}

function preferredModelId() {
  const models = visibleModels();
  const policies = models
    .filter((model) => model.kind === "policy" || String(model.id || "").startsWith("policy:"))
    .sort((left, right) => Number(right.updated_at || 0) - Number(left.updated_at || 0));
  return policies[0]?.id || models.find((model) => model.id === "scripted")?.id || models[0]?.id || "random";
}

function normalizedTeams() {
  return {
    TEAM_A: appState.teams?.TEAM_A?.length ? appState.teams.TEAM_A : TEAM_PLAYERS.TEAM_A,
    TEAM_B: appState.teams?.TEAM_B?.length ? appState.teams.TEAM_B : TEAM_PLAYERS.TEAM_B,
  };
}

function teamClass(value) {
  const text = String(value || "");
  return text === "TEAM_A" || text.startsWith("A") ? "team-a" : "team-b";
}

function applyTeamConfig(teamId) {
  const model = els.playerConfigGrid.querySelector(`select[data-team-model="${teamId}"]`)?.value;
  const faction = els.playerConfigGrid.querySelector(`select[data-team-faction="${teamId}"]`)?.value;
  for (const playerId of normalizedTeams()[teamId] || []) {
    const modelSelect = els.playerConfigGrid.querySelector(`select[data-player-model="${playerId}"]`);
    const factionSelect = els.playerConfigGrid.querySelector(`select[data-player-faction="${playerId}"]`);
    if (modelSelect && model) {
      modelSelect.value = model;
    }
    if (factionSelect && faction) {
      factionSelect.value = faction;
    }
    if (modelSelect && factionSelect) {
      syncFactionLock(modelSelect, factionSelect);
    }
  }
  validateGeneratorControls();
}

function playerModelSelection(options = {}) {
  const players = {};
  for (const playerId of appState.players.length ? appState.players : PLAYER_IDS) {
    const select = els.playerConfigGrid?.querySelector(`select[data-player-model="${playerId}"]`);
    if (select?.value) {
      players[playerId] = select.value;
    } else if (!options.allowMissing) {
      players[playerId] = preferredModelId();
    }
  }
  return players;
}

function playerFactionSelection(options = {}) {
  const factions = {};
  for (const playerId of appState.players.length ? appState.players : PLAYER_IDS) {
    const select = els.playerConfigGrid?.querySelector(`select[data-player-faction="${playerId}"]`);
    if (select?.value) {
      factions[playerId] = select.value;
    } else if (!options.allowMissing) {
      factions[playerId] = "humans";
    }
  }
  return factions;
}

async function generateReplays() {
  if (!validateGeneratorControls()) {
    return;
  }

  const runs = Number(els.generateCountInput.value || 5);
  const seedRaw = els.generateSeedInput.value.trim();
  const seed = seedRaw === "" ? null : Number(seedRaw);  // blank = fresh random (server picks + records)
  const players = playerModelSelection();
  const factions = playerFactionSelection();
  const orders = playerOrderSelection();
  appState.generating = true;
  validateGeneratorControls();
  setGenerateStatus("Generating", "loading", `${runs} matches | ${matchConfigSummary(players, factions)}`);

  try {
    const payload = await fetchJson("/api/generate-replays", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runs, seed, players, factions, orders }),
    });
    appState.serverInfo = payload.server || appState.serverInfo;
    appState.apiStale = Boolean(appState.serverInfo?.stale);
    appState.samples = payload.replays?.length ? payload.replays : [...FALLBACK_REPLAYS];
    populateReplaySelect();
    els.replaySelect.value = "0";
    setGenerateStatus(`${payload.runs || 0} generated`, "ok", generationSummary(payload));
    await loadSelectedReplay();
  } catch (error) {
    setGenerateStatus(statusLabelForError(error), "error", error.message || "Generation failed");
  } finally {
    appState.generating = false;
    validateGeneratorControls();
  }
}

function validateGeneratorControls() {
  const runs = Number(els.generateCountInput.value || 0);
  const seedRaw = els.generateSeedInput.value.trim();
  const validRuns = Number.isInteger(runs) && runs >= 1 && runs <= appState.maxUiRuns;
  const validSeed = seedRaw === "" || Number.isFinite(Number(seedRaw));  // blank = random
  const validPlayers = hasCompletePlayerConfig();
  const disabled = appState.generating || !appState.apiAvailable || appState.apiStale || !validRuns || !validSeed || !validPlayers;
  els.generateBtn.disabled = disabled;
  els.generateBtn.setAttribute("aria-busy", appState.generating ? "true" : "false");
  if (!validRuns) {
    els.generateBtn.title = `Matches must be 1..${appState.maxUiRuns}`;
  } else if (!validSeed) {
    els.generateBtn.title = "Seed must be a number (leave blank for random)";
  } else if (!validPlayers) {
    els.generateBtn.title = "Every player needs a model and faction";
  } else if (appState.apiStale) {
    els.generateBtn.title = "Restart the viewer API";
  } else if (!appState.apiAvailable) {
    els.generateBtn.title = "Generation API offline";
  } else {
    els.generateBtn.title = "Generate replay files";
  }
  return !disabled;
}

function hasCompletePlayerConfig() {
  const models = playerModelSelection();
  const factions = playerFactionSelection();
  const allowedModels = new Set(visibleModels().map((model) => model.id));
  const allowedFactions = new Set(appState.factions.length ? appState.factions : DEFAULT_FACTIONS);
  return (appState.players.length ? appState.players : PLAYER_IDS).every((playerId) => (
    allowedModels.has(models[playerId]) && allowedFactions.has(factions[playerId])
  ));
}

function setGenerateStatus(message, tone = "", title = "") {
  els.generateStatus.textContent = message || "";
  els.generateStatus.title = title || message || "";
  els.generateStatus.dataset.tone = tone;
}

function statusLabelForError(error) {
  const message = String(error?.message || "");
  if (/old code|stale|restart viewer/i.test(message)) {
    return "API stale";
  }
  if (/offline|failed to fetch|networkerror|load failed/i.test(message)) {
    return "API offline";
  }
  return "Error";
}

function generationSummary(payload = {}) {
  const winners = countSummary(payload.winners);
  const types = countSummary(payload.winner_types);
  return [
    `${payload.runs || 0} matches generated`,
    winners ? `winners ${winners}` : "",
    types ? `types ${types}` : "",
    payload.players && payload.factions ? matchConfigSummary(payload.players, payload.factions) : "",
  ].filter(Boolean).join(" | ");
}

function countSummary(values = {}) {
  return Object.entries(values)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}:${value}`)
    .join(" ");
}

function matchConfigSummary(players = {}, factions = {}) {
  return Object.entries(normalizedTeams())
    .map(([teamId, teamPlayers]) => {
      const playerText = teamPlayers
        .map((playerId) => `${playerId}:${shortModelId(players[playerId])}/${shortFaction(factions[playerId])}`)
        .join(" ");
      return `${TEAM_LABELS[teamId] || teamId} ${playerText}`;
    })
    .join(" | ");
}

function shortModelId(modelId) {
  const model = (appState.models.length ? appState.models : FALLBACK_MODELS).find((item) => item.id === modelId);
  if (model?.checkpoint) {
    return model.label || model.checkpoint.split("/").slice(-3).join("/");
  }
  return String(model?.label || modelId || "?").replace(/\s+bot$/i, "");
}

function shortFaction(faction) {
  return String(faction || "humans").slice(0, 3);
}

function serverStatusLabel(info) {
  if (!info) {
    return "";
  }
  const commit = info.commit || "?";
  const schema = info.schema_version || "?";
  return `build ${commit} | schema ${schema}${info.stale ? " | stale" : ""}`;
}

async function loadSelectedReplay() {
  const sample = appState.samples[Number(els.replaySelect.value || 0)] || appState.samples[0];
  if (!sample) {
    return;
  }
  const replay = await fetchJson(sample.path);
  appState.replay = replay;
  appState.replayLabel = sample.label || sample.path;
  appState.tickFloat = 0;
  appState.playing = false;
  appState.sprites.clear();
  layers.units.removeChildren();
  resetFx();
  appState.nodeMeta = replay.meta?.map2d?.nodes || {};
  appState.lastMapSignature = "";
  els.tickSlider.max = String(Math.max(0, (replay.ticks || []).length - 1));
  els.tickSlider.value = "0";
  els.tickNow.textContent = "0";
  els.tickMax.textContent = String(Math.max(0, (replay.ticks || []).length - 1));
  els.playBtn.textContent = ">";
  if (EMBED.active) {
    // ANY embedded replay (эфир AND повтор) auto-plays — it was opened to be watched, not to sit at frame 0.
    // LIVE additionally hides the winner (replayMetaText prints meta.winner); повтор shows it (result known).
    if (EMBED.live) {
      els.metaText.textContent = "🔴 Прямой эфир";
      els.metaText.title = "";
    } else {
      els.metaText.textContent = replayMetaText(replay);
      els.metaText.title = replayMetaTitle(replay);
    }
    // carry the viewer-chosen speed across LIVE map reloads (passed in the embed URL)
    if (EMBED.speed && els.speedSelect && [...els.speedSelect.options].some((o) => o.value === EMBED.speed)) {
      els.speedSelect.value = EMBED.speed;
    }
    appState.liveEnded = false;
    appState.playing = true;
    els.playBtn.textContent = "||";
    resetPlaybackClock(performance.now());
  } else {
    els.metaText.textContent = replayMetaText(replay);
    els.metaText.title = replayMetaTitle(replay);
  }
  drawStaticMap();
  updateInspector();
}

function replayMetaText(replay) {
  const meta = replay?.meta || {};
  const factions = meta.player_factions || {};
  const teamA = TEAM_PLAYERS.TEAM_A.map((playerId) => `${playerId}:${shortFaction(factions[playerId])}`).join(" ");
  const teamB = TEAM_PLAYERS.TEAM_B.map((playerId) => `${playerId}:${shortFaction(factions[playerId])}`).join(" ");
  return `schema ${meta.schema_version || "?"} | ${meta.winner || "?"} | ${meta.total_ticks || 0}t | ${teamA} vs ${teamB}`;
}

function replayMetaTitle(replay) {
  const meta = replay?.meta || {};
  if (!meta.player_models && !meta.player_factions) {
    return "";
  }
  return matchConfigSummary(meta.player_models || {}, meta.player_factions || {});
}

function frame(deltaMS) {
  const dt = Math.max(0.001, Math.min(0.05, deltaMS / 1000));
  if (!appState.replay) {
    advanceParticles(dt);
    advanceProjectiles(dt);
    renderEffectsLayer();
    updateScreenFx(dt);
    return;
  }
  if (appState.playing) {
    const speed = Number(els.speedSelect.value || 1);
    const elapsed = performance.now() - appState.playStartedAt;
    appState.tickFloat = appState.playStartTick + (elapsed / BASE_TICK_MS) * speed;
    if (appState.tickFloat >= maxTick()) {
      appState.tickFloat = maxTick();
      appState.playing = false;
      els.playBtn.textContent = ">";
      // LIVE broadcast: tell the World page the map finished so it can roll the next one
      if (EMBED.active && EMBED.live && !appState.liveEnded) {
        appState.liveEnded = true;
        try { window.parent.postMessage({ type: "worldLiveMapEnded" }, window.location.origin); } catch (e) { /* cross-origin: ignore */ }
      }
    }
  }
  renderReplay(deltaMS);
  advanceProjectiles(dt);
  advanceParticles(dt);
  renderEffectsLayer();
  updateScreenFx(dt);
  applyCameraShake(dt);
}

function renderReplay(deltaMS) {
  const replay = appState.replay;
  const ticks = replay?.ticks || [];
  if (!ticks.length) {
    return;
  }
  const tickIndex = Math.max(0, Math.min(maxTick(), Math.floor(appState.tickFloat)));
  const alpha = Math.max(0, Math.min(1, appState.tickFloat - tickIndex));
  const tick = ticks[tickIndex];
  const nextTick = ticks[Math.min(ticks.length - 1, tickIndex + 1)];
  els.tickSlider.value = String(Math.round(appState.tickFloat));
  els.tickNow.textContent = String(Math.round(appState.tickFloat));

  const signature = mapStateSignature(tick);
  if (signature !== appState.lastMapSignature) {
    appState.lastMapSignature = signature;
    drawStaticMap();
  }

  const hasVisualTrack = Array.isArray(tick.visual_units);
  const desired = desiredUnits(ticks, tickIndex, tick, nextTick, alpha);
  syncSprites(desired, Math.max(0.001, deltaMS / 1000), { replayInterpolated: hasVisualTrack });

  // Discrete combat effects (projectiles, hit flashes, death bursts, core
  // shake) fire once when the integer tick advances. Fire the effects for the
  // tick the HP bars are interpolating TOWARD (nextTick) at the START of the
  // interval, so the strike is shown first and the HP then slides down over the
  // interval — instead of HP dropping before the hit visibly lands.
  if (tickIndex !== fx.lastEffectTick) {
    spawnTickEffects(nextTick, tick);
    fx.lastEffectTick = tickIndex;
  }
}

function desiredUnits(ticks, tickIndex, tick, nextTick, alpha) {
  if (Array.isArray(tick.visual_units)) {
    return visualTrackDesired(ticks, tickIndex, tick, nextTick, alpha);
  }
  const desired = [];
  desired.push(...stationedDesired(tick));
  desired.push(...movingDesired(tick, nextTick, alpha));
  desired.push(...battleDesired(tick, nextTick, alpha));
  return desired;
}

function visualTrackDesired(ticks, tickIndex, tick, nextTick, alpha) {
  const current = normalizeVisualUnits(tick.visual_units || []);
  const previousById = new Map(normalizeVisualUnits(ticks?.[tickIndex - 1]?.visual_units || []).map((unit) => [unit.id, unit]));
  const nextById = new Map(normalizeVisualUnits(nextTick?.visual_units || []).map((unit) => [unit.id, unit]));
  const deathIds = deathUnitIds(tick);
  const mapped = current.map((unit) => {
    if (deathIds.has(unit.id)) {
      return {
        ...unit,
        hpRatio: 0,
        dead: true,
        layer: "fallen",
        state: "fallen",
      };
    }
    const previous = previousById.get(unit.id);
    const next = nextById.get(unit.id);
    if (shouldBridgeTransientStationed(unit, previous, next)) {
      const eased = smoothstep(alpha);
      return {
        ...unit,
        x: lerp(previous.x, next.x, eased),
        y: lerp(previous.y, next.y, eased),
        hpRatio: lerp(previous.hpRatio, next.hpRatio, eased),
        layer: "battle",
        state: "fighting",
        battleId: previous.battleId || next.battleId,
        locationId: unit.nodeId,
        locationKind: "node",
      };
    }
    const layer = unit.layer || unit.state || "unit";
    const interpolated = {
      ...unit,
      x: lerp(unit.x, next?.x ?? unit.x, alpha),
      y: lerp(unit.y, next?.y ?? unit.y, alpha),
      hpRatio: lerp(unit.hpRatio, next?.hpRatio ?? unit.hpRatio, alpha),
      layer,
      route: unit.route || (unit.path?.length ? unit.path.join(" -> ") : ""),
      battleId: unit.battleId,
    };
    return interpolated;
  });
  return arrangeCoreBattlePlates(mapped);
}

function deathUnitIds(tick) {
  const ids = new Set();
  for (const event of tick?.events || []) {
    for (const unit of event?.death_units || []) {
      if (unit?.id) {
        ids.add(String(unit.id));
      }
    }
  }
  return ids;
}

function shouldBridgeTransientStationed(unit, previous, next) {
  const nodeId = unit.nodeId;
  if (!nodeId || !previous || !next) {
    return false;
  }
  const layer = unit.layer || unit.state || "unit";
  if (!["stationed", "visiting"].includes(layer)) {
    return false;
  }
  return isBattleAtNode(previous, nodeId) && isBattleAtNode(next, nodeId);
}

function isBattleAtNode(unit, nodeId) {
  const layer = unit?.layer || unit?.state || "unit";
  if (!isBattleLayer(layer)) {
    return false;
  }
  return unit.locationId === nodeId || unit.battleId === `node:${nodeId}`;
}

function applyVisualHandoffBackfill(item, previous, alpha) {
  if (!shouldBackfillHandoff(previous, item)) {
    return item;
  }
  const eased = smoothstep(alpha);
  return {
    ...item,
    x: lerp(previous.x, item.x, eased),
    y: lerp(previous.y, item.y, eased),
    hpRatio: lerp(previous.hpRatio, item.hpRatio, eased),
  };
}

function shouldBackfillHandoff(previous, item) {
  if (!previous || previous.type !== item.type || displayTeamBucket(previous.team) !== displayTeamBucket(item.team)) {
    return false;
  }
  const previousLayer = previous.layer || previous.state || "unit";
  const nextLayer = item.layer || item.state || "unit";
  if (previousLayer === nextLayer) {
    return false;
  }
  const distance = Math.hypot(Number(item.x) - Number(previous.x), Number(item.y) - Number(previous.y));
  if (distance > HANDOFF_BACKFILL_DISTANCE) {
    return false;
  }
  return isBattleLayer(previousLayer) || isBattleLayer(nextLayer) || previousLayer === "moving" || nextLayer === "moving";
}

function normalizeVisualUnits(units = []) {
  return units
    .filter((unit) => unit && unit.id && Number.isFinite(Number(unit.x)) && Number.isFinite(Number(unit.y)))
    .map((unit) => ({
      id: String(unit.id),
      x: Number(unit.x),
      y: Number(unit.y),
      team: unit.owner || unit.team || "NEUTRAL",
      type: displayUnitCode(unit.unit_type || unit.type),
      hpRatio: Number.isFinite(Number(unit.hp_ratio))
        ? Number(unit.hp_ratio)
        : Number.isFinite(Number(unit.hpRatio))
          ? Number(unit.hpRatio)
          : 1,
      dead: Boolean(unit.dead || unit.state === "fallen" || unit.layer === "fallen"),
      layer: unit.layer || unit.state || "unit",
      state: unit.state || unit.layer || "unit",
      nodeId: unit.node,
      route: unit.path?.length ? unit.path.join(" -> ") : "",
      path: Array.isArray(unit.path) ? unit.path : [],
      intent: unit.intent,
      battleId: unit.battle_id || unit.battleId,
      locationId: unit.location_id || unit.locationId,
      locationKind: unit.location_kind || unit.locationKind,
      uid: unit.uid,
      phase: unit.phase,
      progress: unit.progress,
    }));
}

function stationedDesired(tick) {
  const result = [];
  const battleNodes = new Set((tick.active_battles || [])
    .filter((battle) => battle.location_kind === "node" || battle.location_kind === "approach")
    .map((battle) => battle.location_id));

  for (const [nodeId, node] of Object.entries(tick.nodes || {})) {
    if (battleNodes.has(nodeId)) {
      continue;
    }
    const base = nodePoint(nodeId);
    if (!base) {
      continue;
    }
    if (node.units && Object.keys(node.units).length) {
      result.push(...expandDesired(node.units, node.owner, `node:${nodeId}:${node.owner}`, base, { layer: "stationed", nodeId }));
    }
    for (const [visitor, units] of Object.entries(node.visiting_units || {})) {
      result.push(...expandDesired(units, visitor, `visit:${nodeId}:${visitor}`, base, { layer: "stationed", nodeId, visitor: true }));
    }
    if (Number(node.neutral_hp || 0) > 0) {
      result.push(...expandDesired({ NEUTRAL: node.neutral_hp }, "NEUTRAL", `neutral:${nodeId}`, base, { layer: "neutral", nodeId }));
    }
    if (nodeId === "Baron" && Number(tick.baron?.hp || 0) > 0 && node.owner === "NEUTRAL") {
      result.push(...expandDesired({ BARON: 1 }, "BARON", "baron:neutral", base, { layer: "neutral", nodeId, lord: true }));
    }
  }
  return result;
}

function movingDesired(tick, nextTick, alpha) {
  const current = tick.moving_groups || [];
  const next = nextTick?.moving_groups || [];
  const used = new Set();
  const result = [];
  for (const group of current) {
    const nextGroup = findNextGroup(group, next, used);
    const currentPos = groupVisualPoint(group);
    const nextPos = nextGroup ? groupVisualPoint(nextGroup) : currentPos;
    if (!currentPos) {
      continue;
    }
    const point = nextPos
      ? [lerp(currentPos[0], nextPos[0], alpha), lerp(currentPos[1], nextPos[1], alpha)]
      : currentPos;
    const axis = group.visual?.heading || edgeAxis(group.segment_src || group.src, group.segment_dst || group.dst);
    result.push(...expandDesired(group.units, group.player, `move:${movementSignature(group)}`, point, {
      layer: "moving",
      axis,
      route: (group.path || []).join(" -> "),
      intent: group.intent,
    }));
  }
  return result;
}

function battleDesired(tick, nextTick, alpha) {
  const result = [];
  for (const battle of tick.active_battles || []) {
    const nextBattle = (nextTick?.active_battles || []).find((item) => item.battle_id === battle.battle_id);
    const units = interpolateFormation(battle.formation || [], nextBattle?.formation || [], alpha);
    const fallback = battlePosition(battle);
    const axis = battleAxis(battle);
    for (const unit of units) {
      const point = Number.isFinite(unit.x) && Number.isFinite(unit.y)
        ? [unit.x, unit.y]
        : fallbackFormationPoint(fallback, axis, unit);
      if (!point) {
        continue;
      }
      result.push({
        id: `battle:${battle.battle_id}:${unit.uid}`,
        x: point[0],
        y: point[1],
        team: unit.team,
        type: unit.type,
        hpRatio: unit.maxHp > 0 ? unit.hp / unit.maxHp : 1,
        layer: "battle",
        battleId: battle.battle_id,
        uid: unit.uid,
        range: unit.range,
        rangeMap: unit.rangeMap,
      });
    }
  }
  return result;
}

function expandDesired(units, owner, baseId, anchor, options = {}) {
  const entries = expandUnitEntries(units);
  const axis = options.axis || [1, 0];
  const perp = [-axis[1], axis[0]];
  const result = [];
  for (let index = 0; index < entries.length; index += 1) {
    const unit = entries[index];
    const offset = clusterOffset(index, entries.length, options.lord ? 26 : 18);
    const typeShift = typeLane(unit.type);
    result.push({
      id: `${baseId}:${unit.type}:${unit.index}`,
      x: anchor[0] + axis[0] * offset[0] + perp[0] * (offset[1] + typeShift),
      y: anchor[1] + axis[1] * offset[0] + perp[1] * (offset[1] + typeShift),
      team: owner,
      type: unit.type,
      hpRatio: 1,
      layer: options.layer || "unit",
      nodeId: options.nodeId,
      route: options.route,
      intent: options.intent,
      axis: options.axis,
    });
  }
  return result;
}

function syncSprites(desiredList, dt, options = {}) {
  const desired = new Map(desiredList.map((item) => [item.id, item]));
  const used = new Set();
  const nextSprites = new Map();

  for (const item of desired.values()) {
    let sprite = appState.sprites.get(item.id);
    if (sprite) {
      used.add(item.id);
    } else {
      const match = findHandoffSprite(item, used);
      if (match) {
        sprite = match.sprite;
        used.add(match.id);
        appState.sprites.delete(match.id);
      } else {
        sprite = createUnitSprite(item);
        layers.units.addChild(sprite.container);
        sprite.x = item.x;
        sprite.y = item.y;
        sprite.alpha = 0;
      }
    }

    sprite.dying = false;
    sprite.deathAge = 0;
    sprite.target = item;
    if (sprite.hitFlash > 0) {
      sprite.hitFlash = Math.max(0, sprite.hitFlash - dt * 7);
    }
    const ease = spriteEaseFor(item, sprite, dt, options);
    sprite.x = lerp(sprite.x, item.x, ease);
    sprite.y = lerp(sprite.y, item.y, ease);
    sprite.alpha = Math.min(1, sprite.alpha + dt * FADE_IN_PER_SECOND);
    updateUnitSprite(sprite, item);
    nextSprites.set(item.id, sprite);
  }

  for (const [id, sprite] of appState.sprites.entries()) {
    if (used.has(id) || nextSprites.has(id)) {
      continue;
    }
    if (!sprite.dying) {
      const prev = sprite.target || {};
      const wasCombat = isBattleLayer(prev.layer || prev.state) || Number(prev.hpRatio ?? 1) < 0.6;
      if (wasCombat && !BUILDING_TYPE_CODES.has(prev.type) && prev.type !== "C" && prev.type !== "N") {
        spawnDeathBurst(sprite.x, sprite.y, teamColor(prev.team));
      }
      sprite.dying = true;
      sprite.deathAge = 0;
      sprite.target = {
        ...prev,
        hpRatio: 0,
        dead: true,
        state: "fallen",
        layer: "fallen",
      };
    }
    sprite.deathAge += dt;
    updateUnitSprite(sprite, sprite.target || {});
    if (sprite.deathAge > DEATH_HOLD_SECONDS) {
      sprite.alpha -= dt * DEATH_FADE_OUT_PER_SECOND;
    } else {
      sprite.alpha = Math.max(sprite.alpha, 0.92);
    }
    sprite.container.alpha = Math.max(0, sprite.alpha);
    if (sprite.alpha > 0.04) {
      nextSprites.set(id, sprite);
    } else {
      layers.units.removeChild(sprite.container);
      sprite.container.destroy({ children: true });
    }
  }

  appState.sprites = nextSprites;
  layers.units.children.sort((left, right) => Number(left.y || 0) - Number(right.y || 0));
}

function spriteEaseFor(item, sprite, dt, options = {}) {
  if (options.replayInterpolated) {
    return 1;
  }
  const rate = SPRITE_EASE[item.layer] || SPRITE_EASE[item.state] || SPRITE_EASE.unit;
  const distance = Math.hypot(Number(item.x) - sprite.x, Number(item.y) - sprite.y);
  const layerChanged = sprite.target && sprite.target.layer && sprite.target.layer !== item.layer;
  const handoffBoost = layerChanged || distance > 120 ? 1.35 : 1.0;
  return Math.max(0.02, Math.min(0.92, 1 - Math.exp(-dt * rate * handoffBoost)));
}

function findHandoffSprite(item, used) {
  let best = null;
  let bestDistance = HANDOFF_DISTANCE;
  for (const [id, sprite] of appState.sprites.entries()) {
    if (used.has(id)) {
      continue;
    }
    const old = sprite.target || {};
    if (displayTeamBucket(old.team) !== displayTeamBucket(item.team) || old.type !== item.type) {
      continue;
    }
    const distance = Math.hypot(sprite.x - item.x, sprite.y - item.y);
    if (distance < bestDistance) {
      best = { id, sprite };
      bestDistance = distance;
    }
  }
  return best;
}

function createUnitSprite(item) {
  const container = new Container();
  const glow = new Graphics();
  const body = new Graphics();
  const flash = new Graphics();
  const hp = new Graphics();
  glow.blendMode = "add";
  flash.blendMode = "add";
  container.addChild(glow, body, flash, hp);
  return {
    container, glow, body, flash, hp,
    target: item, x: item.x, y: item.y, alpha: 0,
    hitFlash: 0, bodyKey: "", hpKey: "", glowKey: "",
  };
}

function updateUnitSprite(sprite, item) {
  const dead = Boolean(item.dead || item.layer === "fallen" || item.state === "fallen" || item.hpRatio <= 0);
  const spec = UNIT_SPEC[item.type] || null;
  const teamBucket = displayTeamBucket(item.team);
  const color = dead ? COLORS.deadUnit : teamColor(item.team);
  const accent = dead ? COLORS.deadUnit : factionAccent(item.type);
  const size = unitSize(item);
  sprite.container.x = sprite.x;
  sprite.container.y = sprite.y;
  sprite.container.alpha = sprite.alpha;
  sprite.container.eventMode = "none";

  // Body silhouette: redraw only when identity/size/state changes (facing is a
  // cheap transform applied below, not a redraw).
  const sizeBucket = Math.round(size * 4);
  const bodyKey = `${item.type}|${dead ? "d" : "a"}|${teamBucket}|${sizeBucket}`;
  if (bodyKey !== sprite.bodyKey) {
    sprite.bodyKey = bodyKey;
    sprite.body.clear();
    drawUnitShape(sprite.body, item.type, color, accent, size, dead);
  }
  sprite.body.rotation = spec && spec.dir && !dead ? bodyFacing(item) : 0;

  // Soft glow halo for heavy / tech-2 / objective units (fake bloom, no filters).
  if (bodyKey !== sprite.glowKey) {
    sprite.glowKey = bodyKey;
    sprite.glow.clear();
    const gr = dead ? 0 : glowRadius(item.type, size);
    if (gr > 0) {
      const gc = glowColor(item.type, item.team);
      sprite.glow.circle(0, 0, gr).fill({ color: gc, alpha: 0.15 });
      sprite.glow.circle(0, 0, gr * 0.62).fill({ color: gc, alpha: 0.13 });
    }
  }

  // Hit flash overlay (set on damage in spawnTickEffects, decays per frame).
  sprite.flash.clear();
  if (sprite.hitFlash > 0.02 && !dead) {
    sprite.flash.circle(0, 0, size * 1.4).fill({ color: 0xffffff, alpha: Math.min(0.8, sprite.hitFlash) });
  }

  // HP bar: redraw only when ratio/size bucket changes; colour shifts as it drains.
  const hpRatio = Math.max(0, Math.min(1, Number(item.hpRatio ?? 1)));
  const showHp = !hideSpriteHpBar(item) && (dead || hpRatio < 0.999);
  const hpKey = showHp ? `${Math.round(hpRatio * 24)}|${sizeBucket}|${teamBucket}|${dead ? "d" : "a"}` : "none";
  if (hpKey !== sprite.hpKey) {
    sprite.hpKey = hpKey;
    sprite.hp.clear();
    if (showHp) {
      const barW = size * 2;
      sprite.hp.roundRect(-size, -size - 8, barW, 3, 1).fill(COLORS.hpTrack);
      const hpWidth = barW * hpRatio;
      if (hpWidth > 0.5) {
        const hpColor = dead ? COLORS.deadUnit
          : hpRatio > 0.5 ? color
            : hpRatio > 0.22 ? COLORS.neutralBright : COLORS.danger;
        sprite.hp.roundRect(-size, -size - 8, hpWidth, 3, 1).fill(hpColor);
      }
    }
  }
}

function hideSpriteHpBar(item) {
  return item?.type === "L" && item?.team === "BARON";
}

const OUTLINE = COLORS.outline;

// Geometry primitives → flat [x0,y0,x1,y1,...] point arrays for g.poly().
function regularPoly(sides, radius, rot = -Math.PI / 2) {
  const pts = [];
  for (let i = 0; i < sides; i += 1) {
    const a = rot + i * (Math.PI * 2 / sides);
    pts.push(Math.cos(a) * radius, Math.sin(a) * radius);
  }
  return pts;
}

function starPoly(points, rOuter, rInner, rot = -Math.PI / 2) {
  const pts = [];
  for (let i = 0; i < points * 2; i += 1) {
    const r = i % 2 === 0 ? rOuter : rInner;
    const a = rot + i * (Math.PI / points);
    pts.push(Math.cos(a) * r, Math.sin(a) * r);
  }
  return pts;
}

// Each unit type gets a recognizable silhouette so you can read WHO is fighting
// at a glance: faction = shape family (humans angular, infected spiky, robots
// hard-edged), klass = size, ranged units carry a muzzle/lens. Body fill is the
// TEAM colour (red A / blue B); accent is the faction trim colour.
const UNIT_DRAWERS = {
  // ── Humans: angular / military ──────────────────────────────────────────
  V: (g, c, a, s) => { // Vanguard — shield (melee frontline)
    g.poly([-s * 0.95, -s * 0.8, s * 0.95, -s * 0.8, s * 0.95, s * 0.2, 0, s * 1.25, -s * 0.95, s * 0.2])
      .fill(c).stroke({ color: OUTLINE, width: 1.3 });
    g.rect(-s * 0.16, -s * 0.5, s * 0.32, s * 1.15).fill(a);
  },
  R: (g, c, a, s) => { // Ranger — forward arrowhead (ranged)
    g.poly([s * 1.35, 0, -s * 0.25, s * 0.92, -s * 0.85, 0, -s * 0.25, -s * 0.92])
      .fill(c).stroke({ color: OUTLINE, width: 1.2 });
    g.circle(s * 0.45, 0, s * 0.24).fill(a);
  },
  S: (g, c, a, s) => { // Skirmisher — light dart
    g.poly([s * 1.2, 0, -s * 0.55, s * 0.82, -s * 0.05, 0, -s * 0.55, -s * 0.82])
      .fill(c).stroke({ color: OUTLINE, width: 1.05 });
  },
  RD: (g, c, a, s) => { // Raider — swept hexagon w/ slash (tech-1 striker)
    g.poly(regularPoly(6, s * 1.12, 0)).fill(c).stroke({ color: OUTLINE, width: 1.25 });
    g.moveTo(-s * 0.55, s * 0.4).lineTo(s * 0.55, -s * 0.4).stroke({ color: a, width: 1.8 });
  },
  MP: (g, c, a, s) => { // BMP — tracked vehicle w/ barrel (tech-2)
    g.roundRect(-s * 1.2, -s * 0.78, s * 2.4, s * 1.56, 2).fill(c).stroke({ color: OUTLINE, width: 1.3 });
    g.rect(s * 0.3, -s * 0.13, s * 1.45, s * 0.26).fill(a);
    g.rect(-s * 0.6, -s * 0.66, s * 1.2, s * 0.26).fill(a);
    g.circle(-s * 0.5, s * 0.86, s * 0.24).fill(OUTLINE);
    g.circle(s * 0.5, s * 0.86, s * 0.24).fill(OUTLINE);
  },
  // ── Infected: organic / spiky ───────────────────────────────────────────
  SW: (g, c, a, s) => { // Swarmling — tiny trefoil
    g.poly(starPoly(3, s * 1.35, s * 0.55)).fill(c).stroke({ color: OUTLINE, width: 1.0 });
    g.circle(0, 0, s * 0.3).fill(a);
  },
  SP: (g, c, a, s) => { // Spitter — blob w/ nozzle (ranged)
    g.circle(-s * 0.1, 0, s * 0.95).fill(c).stroke({ color: OUTLINE, width: 1.2 });
    g.poly([-s * 1.25, -s * 0.5, -s * 0.5, 0, -s * 1.25, s * 0.5]).fill(c).stroke({ color: OUTLINE, width: 0.8 });
    g.poly([s * 1.35, 0, s * 0.2, s * 0.5, s * 0.2, -s * 0.5]).fill(a);
  },
  BR: (g, c, a, s) => { // Brute — heavy spiked disc
    g.poly(starPoly(7, s * 1.2, s * 0.86)).fill(c).stroke({ color: OUTLINE, width: 1.4 });
    g.circle(0, 0, s * 0.5).fill(a);
    g.circle(0, 0, s * 0.22).fill(OUTLINE);
  },
  RV: (g, c, a, s) => { // Ravager — sharp 6-point star (tech-1)
    g.poly(starPoly(6, s * 1.35, s * 0.5)).fill(c).stroke({ color: OUTLINE, width: 1.25 });
    g.circle(0, 0, s * 0.34).fill(a);
  },
  BH: (g, c, a, s) => { // Behemoth — huge jagged mass (tech-2)
    g.poly(starPoly(9, s * 1.4, s * 0.82)).fill(c).stroke({ color: OUTLINE, width: 1.6 });
    g.circle(0, 0, s * 0.62).fill(OUTLINE);
    g.circle(0, 0, s * 0.42).fill(a);
  },
  // ── Robots: mechanical / hard-edged ─────────────────────────────────────
  WK: (g, c, a, s) => { // Walker — legged hexagon
    g.rect(-s * 0.72, s * 0.35, s * 0.3, s * 0.85).fill(OUTLINE);
    g.rect(s * 0.42, s * 0.35, s * 0.3, s * 0.85).fill(OUTLINE);
    g.poly(regularPoly(6, s * 1.04, -Math.PI / 2)).fill(c).stroke({ color: OUTLINE, width: 1.3 });
    g.circle(0, -s * 0.08, s * 0.3).fill(a);
  },
  LS: (g, c, a, s) => { // Laser — chassis w/ barrel + lens (ranged)
    g.roundRect(-s * 0.9, -s * 0.72, s * 1.65, s * 1.44, 2).fill(c).stroke({ color: OUTLINE, width: 1.25 });
    g.rect(s * 0.2, -s * 0.1, s * 1.3, s * 0.2).fill(OUTLINE);
    g.circle(s * 1.5, 0, s * 0.26).fill(a);
  },
  DR: (g, c, a, s) => { // Drone — rotor diamond
    g.poly(regularPoly(4, s * 1.18, 0)).fill(c).stroke({ color: OUTLINE, width: 1.0 });
    g.circle(0, 0, s * 0.34).fill(a);
  },
  GM: (g, c, a, s) => { // Guardian Mech — octagon w/ bolt (tech-1)
    g.poly(regularPoly(8, s * 1.08, -Math.PI / 8)).fill(c).stroke({ color: OUTLINE, width: 1.3 });
    g.circle(0, 0, s * 0.42).fill(a);
    g.circle(0, 0, s * 0.18).fill(OUTLINE);
  },
  SG: (g, c, a, s) => { // Siege Platform — artillery w/ big cannon (tech-2)
    g.poly([-s * 1.0, s * 0.85, s * 0.7, s * 0.85, s * 1.0, -s * 0.3, -s * 0.7, -s * 0.85])
      .fill(c).stroke({ color: OUTLINE, width: 1.4 });
    g.rect(0, -s * 0.2, s * 1.7, s * 0.4).fill(OUTLINE);
    g.circle(s * 1.7, 0, s * 0.3).fill(a);
  },
  // ── Objectives ──────────────────────────────────────────────────────────
  N: (g) => { // Neutral creep — hollow ring
    g.circle(0, 0, 5).fill({ color: COLORS.neutralBg, alpha: 0.55 }).stroke({ color: COLORS.neutral, width: 1.6 });
    g.circle(0, 0, 1.7).fill(COLORS.neutralBright);
  },
  C: (g, c, a, s) => { // Core — fortified octagon
    g.poly(regularPoly(8, s * 1.18, -Math.PI / 8)).fill(COLORS.bgDeep).stroke({ color: c, width: 2.6 });
    g.poly(regularPoly(4, s * 0.62, 0)).fill(c);
    g.circle(0, 0, s * 0.26).fill(0xffffff);
  },
  L: (g, c, a, s) => { // Lord / Baron — crowned star
    g.poly(starPoly(5, s * 1.75, s * 0.9)).fill(COLORS.neutral).stroke({ color: c, width: 2.4, join: "round" });
    g.poly(regularPoly(4, s * 0.82, 0)).fill(c).stroke({ color: OUTLINE, width: 1.2 });
    g.circle(0, 0, s * 0.32).fill(0xffffff);
  },
};

function drawUnitShape(g, type, color, accent, size, dead = false) {
  const drawer = UNIT_DRAWERS[type];
  if (drawer) {
    drawer(g, color, accent, size, dead);
    return;
  }
  if (BUILDING_TYPE_CODES.has(type)) {
    drawPlateShape(g, type, color, accent, size);
    return;
  }
  g.circle(0, 0, size).fill(color).stroke({ color: OUTLINE, width: 1.2 });
}

// Buildings fight as defensive "plates" shielding the core (engine gates a core
// assault through them layer by layer). Draw them as a team-edged plate so the
// in-battle sprite reads as a structure, not a generic blob.
function drawPlateShape(g, type, color, accent, size) {
  const w = size * 1.7;
  const h = size * 1.05;
  g.roundRect(-w, -h, w * 2, h * 2, 3).fill(COLORS.buildingBg).stroke({ color, width: 1.9 });
  if (type === "T") { // turret — barrel
    g.moveTo(0, 0).lineTo(w * 1.15, 0).stroke({ color: accent, width: 2.4 });
    g.circle(0, 0, size * 0.5).fill(color);
    return;
  }
  if (type === "T1" || type === "T2") { // tech — diamond core
    g.poly(regularPoly(4, size * 0.7, 0)).fill(accent);
    if (type === "T2") g.poly(regularPoly(4, size * 0.34, 0)).fill(OUTLINE);
    return;
  }
  g.circle(0, 0, size * 0.42).fill(color);
  g.circle(0, 0, size * 0.2).fill(accent);
}

function factionAccent(type) {
  const spec = UNIT_SPEC[type];
  return FACTION_ACCENT[spec ? spec.fac : "humans"] || FACTION_ACCENT.humans;
}

// Heavy / tech-2 / objective units get a soft additive halo (fake bloom).
function glowRadius(type, size) {
  const spec = UNIT_SPEC[type];
  if (!spec) return 0;
  if (spec.klass === "lord") return size * 2.4;
  if (spec.klass === "core") return size * 1.95;
  if (spec.klass === "tech2" || spec.klass === "heavy") return size * 1.85;
  return 0;
}

function glowColor(type, team) {
  const spec = UNIT_SPEC[type];
  if (spec && spec.klass === "lord") return COLORS.neutralBright;
  return teamColor(team);
}

// Directional units face their travel axis if known, else the enemy side
// (Team A points right, Team B points left) so armies read as facing off.
function bodyFacing(item) {
  if (item.axis && (Number(item.axis[0]) || Number(item.axis[1]))) {
    return Math.atan2(Number(item.axis[1]) || 0, Number(item.axis[0]) || 0);
  }
  return displayTeamBucket(item.team) === "TEAM_B" ? Math.PI : 0;
}

// ── Live combat FX: projectiles, particles, hit flashes, core shake ──────────

function resetFx() {
  fx.projectiles = [];
  fx.particles = [];
  fx.shake = 0;
  fx.shakeX = 0;
  fx.shakeY = 0;
  fx.coreFlash = 0;
  fx.coreFlashTeam = null;
  fx.lastEffectTick = -1;
  fx.particleG.clear();
  fx.projectileG.clear();
  fx.screenG.clear();
}

function spawnProjectile(x0, y0, x1, y1, color, accent, targetId) {
  if (fx.projectiles.length >= MAX_PROJECTILES) {
    return;
  }
  fx.projectiles.push({ x0, y0, x1, y1, t: 0, dur: PROJECTILE_LIFE, color, accent, targetId, hit: false });
}

function spawnParticle(p) {
  if (fx.particles.length >= MAX_PARTICLES) {
    return;
  }
  fx.particles.push(p);
}

function spawnBurst(x, y, color, count, opts = {}) {
  const speed = opts.speed ?? 70;
  const life = opts.life ?? 0.4;
  const size = opts.size ?? 1.7;
  const gravity = opts.gravity ?? 0;
  for (let i = 0; i < count; i += 1) {
    const a = Math.random() * Math.PI * 2;
    const sp = speed * (0.45 + Math.random() * 0.85);
    spawnParticle({
      x, y,
      vx: Math.cos(a) * sp,
      vy: Math.sin(a) * sp,
      life,
      maxLife: life,
      size: size * (0.7 + Math.random() * 0.7),
      color,
      gravity,
    });
  }
}

function spawnDeathBurst(x, y, color) {
  spawnBurst(x, y, color, 7, { speed: 78, life: 0.5, size: 2.0, gravity: PARTICLE_GRAVITY });
  spawnBurst(x, y, 0xffffff, 3, { speed: 50, life: 0.26, size: 1.6 });
}

function advanceProjectiles(dt) {
  const live = [];
  for (const p of fx.projectiles) {
    p.t += dt;
    if (p.t >= p.dur) {
      if (!p.hit) {
        p.hit = true;
        spawnBurst(p.x1, p.y1, p.accent, 4, { speed: 58, life: 0.26, size: 1.5 });
        const sprite = p.targetId != null ? appState.sprites.get(p.targetId) : null;
        if (sprite) {
          sprite.hitFlash = Math.max(sprite.hitFlash, 0.7);
        }
      }
      continue;
    }
    live.push(p);
  }
  fx.projectiles = live;
}

function advanceParticles(dt) {
  const live = [];
  for (const p of fx.particles) {
    p.life -= dt;
    if (p.life <= 0) {
      continue;
    }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.gravity) {
      p.vy += p.gravity * dt;
    }
    p.vx *= 1 - dt * 1.3;
    p.vy *= 1 - dt * 1.3;
    live.push(p);
  }
  fx.particles = live;
}

function renderEffectsLayer() {
  const pg = fx.projectileG;
  pg.clear();
  for (const p of fx.projectiles) {
    const f = Math.min(1, p.t / p.dur);
    const x = lerp(p.x0, p.x1, f);
    const y = lerp(p.y0, p.y1, f);
    const tailF = Math.max(0, f - 0.3);
    pg.moveTo(lerp(p.x0, p.x1, tailF), lerp(p.y0, p.y1, tailF)).lineTo(x, y)
      .stroke({ color: p.color, width: 1.7, alpha: 0.5 });
    pg.circle(x, y, 3.4).fill({ color: p.accent, alpha: 0.28 });
    pg.circle(x, y, 2.0).fill({ color: COLORS.spark, alpha: 0.95 });
  }
  const g = fx.particleG;
  g.clear();
  for (const p of fx.particles) {
    const a = Math.max(0, p.life / p.maxLife);
    g.circle(p.x, p.y, p.size * (0.5 + a * 0.5)).fill({ color: p.color, alpha: a });
  }
}

function applyCameraShake(dt) {
  if (fx.shake > 0.05) {
    fx.shake = Math.max(0, fx.shake - dt * 26);
    fx.shakeX = (Math.random() - 0.5) * fx.shake;
    fx.shakeY = (Math.random() - 0.5) * fx.shake;
  } else {
    fx.shake = 0;
    fx.shakeX = 0;
    fx.shakeY = 0;
  }
  if (appState.app) {
    layers.root.x = appState.camera.x + fx.shakeX;
    layers.root.y = appState.camera.y + fx.shakeY;
  }
}

function updateScreenFx(dt) {
  const g = fx.screenG;
  if (fx.coreFlash <= 0.002) {
    g.clear();
    return;
  }
  fx.coreFlash = Math.max(0, fx.coreFlash - dt * 2.2);
  const w = appState.app?.screen.width || 0;
  const h = appState.app?.screen.height || 0;
  const col = fx.coreFlashTeam != null ? teamColor(fx.coreFlashTeam) : COLORS.danger;
  const k = fx.coreFlash;
  const band = Math.max(6, Math.min(w, h) * 0.07);
  g.clear();
  g.rect(0, 0, w, h).fill({ color: col, alpha: 0.04 * k });
  g.rect(0, 0, w, band).fill({ color: col, alpha: 0.26 * k });
  g.rect(0, h - band, w, band).fill({ color: col, alpha: 0.26 * k });
  g.rect(0, 0, band, h).fill({ color: col, alpha: 0.26 * k });
  g.rect(w - band, 0, band, h).fill({ color: col, alpha: 0.26 * k });
}

// Fired once per integer tick advance: turn the tick's combat data into live FX.
function spawnTickEffects(tick, prevTick) {
  let totalCoreDmg = 0;
  for (const nodeId of PLAYER_IDS) {
    const cur = coreHpValue(nodeId, tick.nodes?.[nodeId] || {}, tick);
    const prev = coreHpValue(nodeId, prevTick.nodes?.[nodeId] || {}, prevTick);
    const dmg = prev - cur;
    if (dmg > 0.5) {
      totalCoreDmg += dmg;
      const p = nodePoint(nodeId);
      if (p) {
        spawnBurst(p[0], p[1], teamColor(nodeId), Math.min(11, 3 + Math.round(dmg)), { speed: 80, life: 0.55, size: 2.1, gravity: PARTICLE_GRAVITY });
      }
      fx.coreFlash = Math.min(1, fx.coreFlash + Math.min(0.85, dmg / 9));
      fx.coreFlashTeam = nodeId;
    }
  }
  if (totalCoreDmg > 0) {
    fx.shake = Math.min(18, fx.shake + 5 + totalCoreDmg * 0.9);
  }

  let spawned = 0;
  for (const battle of tick.active_battles || []) {
    if (spawned >= MAX_SPAWN_PER_TICK) {
      break;
    }
    const event = (tick.events || []).find((e) => e.type === "BATTLE_TICK" && e.battle_id === battle.battle_id);
    const attacks = event?.summary?.attacks || [];
    if (!attacks.length) {
      continue;
    }
    const byUid = new Map();
    for (const [id, sprite] of appState.sprites.entries()) {
      const t = sprite.target || {};
      if (t.battleId === battle.battle_id && t.uid != null) {
        byUid.set(Number(t.uid), { id, sprite });
      }
    }
    for (const attack of attacks) {
      if (spawned >= MAX_SPAWN_PER_TICK) {
        break;
      }
      const from = byUid.get(Number(attack[0]));
      const to = byUid.get(Number(attack[1]));
      // Attack-time combat positions recorded by the engine [7..10]: the units were
      // in range HERE. Draw the effect at these so it never stretches across the map
      // to a stale/absent sprite (the attacker may have died, the target may have
      // marched off as the battle resolved). Fall back to live sprites for old replays.
      const hasPos = attack.length >= 11 && attack[7] != null && attack[9] != null;
      const fx0 = hasPos ? Number(attack[7]) : from?.sprite.x;
      const fy0 = hasPos ? Number(attack[8]) : from?.sprite.y;
      const tx0 = hasPos ? Number(attack[9]) : to?.sprite.x;
      const ty0 = hasPos ? Number(attack[10]) : to?.sprite.y;
      if (fx0 == null || fy0 == null || tx0 == null || ty0 == null) {
        continue;
      }
      spawned += 1;
      const col = teamColor(attack[3]);       // attacker team from the record
      const acc = factionAccent(attack[4]);   // attacker type code (sprite may be gone)
      if (isRangedAttacker(from?.sprite, attack[4])) {
        spawnProjectile(fx0, fy0, tx0, ty0, col, acc, to?.id);
        spawnBurst(fx0, fy0, acc, 2, { speed: 30, life: 0.15, size: 1.3 });
      } else {
        const mx = (fx0 + tx0) / 2;
        const my = (fy0 + ty0) / 2;
        spawnBurst(mx, my, COLORS.spark, 3, { speed: 52, life: 0.22, size: 1.6 });
        if (to) {
          to.sprite.hitFlash = Math.max(to.sprite.hitFlash, 0.7);
        }
      }
    }
  }
}

function drawBackground() {
  layers.bg.removeChildren();
  const g = new Graphics();
  g.rect(0, 0, WORLD_W, WORLD_H).fill(COLORS.bgDeep);
  // Team-tinted territory halves so each side of the field reads at a glance.
  g.rect(0, 0, WORLD_W * 0.4, WORLD_H).fill({ color: COLORS.teamAHalf, alpha: 0.65 });
  g.rect(WORLD_W * 0.6, 0, WORLD_W * 0.4, WORLD_H).fill({ color: COLORS.teamBHalf, alpha: 0.65 });
  // Faint tech-grid dots.
  const step = 34;
  for (let x = step / 2; x < WORLD_W; x += step) {
    for (let y = step / 2; y < WORLD_H; y += step) {
      g.circle(x, y, 1).fill({ color: 0x24323f, alpha: 0.5 });
    }
  }
  // Soft inset vignette frame.
  g.rect(0.5, 0.5, WORLD_W - 1, WORLD_H - 1).stroke({ color: 0x1a242e, width: 3, alpha: 0.7 });
  g.rect(4, 4, WORLD_W - 8, WORLD_H - 8).stroke({ color: 0x131b23, width: 6, alpha: 0.45 });
  layers.bg.addChild(g);
}

// Fallback when a sprite has no formation range yet (e.g. mid-handoff): codes the
// engine resolves as melee (combat range <= 1.25). Everything else shoots.
const MELEE_TYPE_CODES = new Set(["V", "S", "SW", "BR", "WK", "DR"]);

// Building unit-type codes (see _unit_type_code in engine.py). In an active
// battle these are emitted as combatants stacked on the node centre; the viewer
// fans them out as shield plates in front of the core so the layering reads.
const BUILDING_TYPE_CODES = new Set(["T", "BO", "PW", "LF", "EC", "T1", "T2"]);
const PLATE_FRONT_GAP = 11;      // core -> first plate, world units
const PLATE_LAYER_SPACING = 9;   // between successive plate layers
const PLATE_CORE_BACKSET = 5;    // nudge the core away from the attackers

function buildingSlotIndexFromId(id) {
  // visual_id format: building:<node>:<slot>:<TYPE>
  const parts = String(id || "").split(":");
  if (parts[0] === "building" && parts.length >= 4) {
    const slot = Number(parts[2]);
    return Number.isFinite(slot) ? slot : 0;
  }
  return 0;
}

// Direction the assault comes from: cores sit on the map edges, attackers push
// in from the middle, so plates face the map centre. Stable per node (no jitter).
function plateThreatDir(anchor) {
  const dx = WORLD_W / 2 - anchor[0];
  const dy = WORLD_H / 2 - anchor[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-3) {
    return [1, 0];
  }
  return [dx / len, dy / len];
}

// Pure presentation: re-place building/core battle sprites (all stacked on the
// node centre by the engine) as a layered shield — outermost slot nearest the
// attackers (it dies first), core behind. Combat math is untouched; attack lines
// follow these sprites, so hits visibly land on the front plate.
function arrangeCoreBattlePlates(items) {
  const groups = new Map();
  for (const item of items) {
    if (!item.battleId || !isBattleLayer(item.layer)) {
      continue;
    }
    const isCore = item.type === "C";
    const isBuilding = BUILDING_TYPE_CODES.has(item.type);
    if (!isCore && !isBuilding) {
      continue;
    }
    const key = `${item.battleId}|${item.team}`;
    let group = groups.get(key);
    if (!group) {
      group = { core: null, plates: [] };
      groups.set(key, group);
    }
    if (isCore) {
      group.core = item;
    } else {
      group.plates.push(item);
    }
  }

  for (const group of groups.values()) {
    if (!group.plates.length) {
      continue;
    }
    const anchor = group.core
      ? [group.core.x, group.core.y]
      : (nodePoint(group.plates[0].locationId) || [group.plates[0].x, group.plates[0].y]);
    const dir = plateThreatDir(anchor);
    const perp = [-dir[1], dir[0]];
    group.plates.sort((a, b) => buildingSlotIndexFromId(b.id) - buildingSlotIndexFromId(a.id));
    const bySlot = new Map();
    for (const plate of group.plates) {
      const slot = buildingSlotIndexFromId(plate.id);
      bySlot.set(slot, (bySlot.get(slot) || 0) + 1);
    }
    const placedPerSlot = new Map();
    group.plates.forEach((plate, layer) => {
      const slot = buildingSlotIndexFromId(plate.id);
      const lateralIndex = placedPerSlot.get(slot) || 0;
      placedPerSlot.set(slot, lateralIndex + 1);
      const lateral = (lateralIndex - (bySlot.get(slot) - 1) / 2) * PLATE_LAYER_SPACING * 1.4;
      const along = PLATE_FRONT_GAP + layer * PLATE_LAYER_SPACING;
      plate.x = anchor[0] + dir[0] * along + perp[0] * lateral;
      plate.y = anchor[1] + dir[1] * along + perp[1] * lateral;
    });
    if (group.core) {
      group.core.x = anchor[0] - dir[0] * PLATE_CORE_BACKSET;
      group.core.y = anchor[1] - dir[1] * PLATE_CORE_BACKSET;
    }
  }
  return items;
}

function isRangedAttacker(sprite, typeCode) {
  const range = Number(sprite?.target?.range);
  if (Number.isFinite(range) && range > 0) {
    return range > MELEE_RANGE_THRESHOLD;
  }
  return !MELEE_TYPE_CODES.has(displayUnitCode(typeCode));
}

function battleTickEventForVisual(tick, nextTick, battle, alpha) {
  const currentEvent = (tick.events || []).find((item) => item.type === "BATTLE_TICK" && item.battle_id === battle.battle_id);
  if (currentEvent) {
    return { event: currentEvent, alpha: 1 };
  }
  if (alpha < ATTACK_PREVIEW_ALPHA) {
    return { event: null, alpha: 0 };
  }
  const nextEvent = (nextTick?.events || []).find((item) => item.type === "BATTLE_TICK" && item.battle_id === battle.battle_id);
  if (!nextEvent) {
    return { event: null, alpha: 0 };
  }
  return {
    event: nextEvent,
    alpha: (alpha - ATTACK_PREVIEW_ALPHA) / (1 - ATTACK_PREVIEW_ALPHA),
  };
}

function drawStaticMap() {
  layers.map.removeChildren();
  layers.labels.removeChildren();
  // Background (territory tint + grid + frame) lives in the persistent bg layer.

  for (const [src, dst] of EDGES) {
    const key = edgeKey(src, dst);
    const [sx, sy] = nodePoint(src);
    const [dx, dy] = nodePoint(dst);
    const tunnel = TUNNELS.has(key);
    const glow = new Graphics();
    glow.moveTo(sx, sy).lineTo(dx, dy).stroke({ color: COLORS.laneGlow, width: tunnel ? 16 : 10, alpha: 0.4, cap: "round" });
    layers.map.addChild(glow);
    const line = new Graphics();
    line.moveTo(sx, sy).lineTo(dx, dy).stroke({
      color: tunnel ? COLORS.tunnel : COLORS.lane,
      width: tunnel ? 9 : 4.5,
      alpha: tunnel ? 0.95 : 0.9,
      cap: "round",
    });
    layers.map.addChild(line);
  }

  for (const nodeId of Object.keys(NODE_POSITIONS)) {
    const [x, y] = nodePoint(nodeId);
    const meta = appState.nodeMeta[nodeId] || {};
    const radius = Number(meta.radius || (nodeId === "Baron" ? 42 : nodeId.startsWith("CP") ? 30 : 34));
    const node = currentTick()?.nodes?.[nodeId] || {};
    const tick = currentTick() || {};
    const owner = node.owner || (nodeId.startsWith("B") ? "B1" : nodeId.startsWith("A") ? "A1" : "NEUTRAL");
    const deadCore = isDeadCoreNode(nodeId, node, tick);
    const isBaron = nodeId === "Baron";
    const ringColor = deadCore ? COLORS.deadCore : isBaron ? COLORS.baron : teamColor(owner);
    const fillColor = deadCore ? COLORS.deadCoreBg : isBaron ? COLORS.baronBg : nodeFill(owner);

    const room = new Graphics();
    if (!deadCore) {
      const halo = isBaron ? COLORS.baronGlow : owner === "NEUTRAL" ? COLORS.neutral : teamColor(owner);
      room.circle(x, y, radius + 6).fill({ color: halo, alpha: 0.1 });
    }
    room.circle(x, y, radius).fill(fillColor).stroke({ color: ringColor, width: isBaron ? 4 : 2.6, alpha: 0.95 });
    room.circle(x, y, Math.max(2, radius - 4)).stroke({ color: ringColor, width: 1, alpha: 0.35 });
    layers.map.addChild(room);

    if (isCoreNode(nodeId)) {
      const fort = new Graphics();
      fort.poly(regularPoly(8, radius - 8, -Math.PI / 8).map((v, i) => v + (i % 2 === 0 ? x : y)))
        .stroke({ color: ringColor, width: 1.4, alpha: 0.45 });
      layers.map.addChild(fort);
      drawCoreHpBar(nodeId, node, tick, x, y, radius, deadCore, ringColor);
    }
    if (nodeId.startsWith("CP") && Number(node.neutral_hp || 0) > 0) {
      drawNeutralCountLabel(nodeId, node, x, y, radius);
    }
    if (isBaron && node.owner === "NEUTRAL" && Number(tick.baron?.hp || 0) > 0) {
      drawNeutralLordHpBar(tick, x, y, radius);
    }
    drawNodeBuildings(nodeId, node, tick, x, y, radius);

    const label = new Text({
      // VISUAL-ONLY rename CP -> P (owner refers to control points as P1..P6 now).
      // Internal node ids stay "CP1".."CP6" everywhere in code/replays — this only
      // relabels the on-canvas text, so nothing downstream breaks.
      text: isBaron ? "LORD" : (nodeId.startsWith("CP") ? "P" + nodeId.slice(2) : nodeId),
      style: { fill: deadCore ? "#151b22" : isBaron ? "#ecd0ff" : "#dfe8f2", fontSize: isBaron ? 12 : 12, fontWeight: "700" },
    });
    label.anchor.set(0.5);
    label.x = x;
    label.y = isBaron ? y - radius - 9 : y;
    label.alpha = isBaron ? 0.95 : 0.7;
    layers.labels.addChild(label);

    // Player NAME beside the core they play (World broadcast): to the OUTER SIDE of the core, off the
    // HP bar — home (A*) cores sit on the left so their names hang left; away (B*) hang right.
    // Reads meta.player_names (slot -> nickname); absent on legacy replays -> simply not drawn.
    const pname = appState.replay?.meta?.player_names?.[nodeId];
    if (isCoreNode(nodeId) && pname) {
      const homeSide = nodeId.charAt(0) === "A";
      const nameLbl = new Text({
        text: pname,
        style: { fill: deadCore ? "#5a6470" : teamColor(owner), fontSize: 13, fontWeight: "800",
                 stroke: { color: "#070b12", width: 3.5 } },
      });
      nameLbl.anchor.set(homeSide ? 1 : 0, 0.5);     // text hangs to the OUTER side, beside the core
      nameLbl.x = homeSide ? (x - radius - 8) : (x + radius + 8);
      nameLbl.y = y;                                  // vertically centred on the core, clear of the HP bar
      nameLbl.alpha = deadCore ? 0.5 : 0.97;
      layers.labels.addChild(nameLbl);
    }
  }

  fitCamera(false);
}

function drawNodeBuildings(nodeId, node, tick, x, y, radius) {
  const buildings = (node?.buildings || [])
    .filter((building) => Number(building?.max_hp ?? building?.hp ?? 0) > 0)
    // A complete building that is fighting renders as a battle plate at the core
    // (see arrangeCoreBattlePlates); skip its corner marker so it is not duplicated.
    .filter((building) => !buildingFightingInBattle(tick, nodeId, building))
    .slice(0, 3);
  if (!buildings.length) {
    return;
  }
  for (let index = 0; index < buildings.length; index += 1) {
    const building = buildings[index];
    const [ox, oy] = buildingSlotOffset(index, buildings.length, radius, nodeId);
    drawBuildingMarker(building, x + ox, y + oy, index);
  }
}

function buildingFightingInBattle(tick, nodeId, building) {
  if (!building?.complete) {
    return false;
  }
  const owner = building.owner;
  const typeCode = displayUnitCode(building.building_type);
  return (tick?.visual_units || []).some((unit) => {
    if (!isBattleLayer(unit.layer || unit.state || "")) {
      return false;
    }
    if ((unit.location_id || unit.locationId) !== nodeId) {
      return false;
    }
    if ((unit.owner || unit.team) !== owner) {
      return false;
    }
    return displayUnitCode(unit.unit_type || unit.type) === typeCode;
  });
}

function buildingSlotOffset(index, count, radius, nodeId) {
  if (count === 1) {
    return [radius * 0.62, -radius * 0.62];
  }
  if (count === 2) {
    return [
      (index === 0 ? -1 : 1) * radius * 0.58,
      -radius * 0.58,
    ];
  }
  const angle = -Math.PI / 2 + index * (Math.PI * 2 / Math.max(3, count));
  const scale = nodeId === "Baron" ? 0.78 : 0.72;
  return [Math.cos(angle) * radius * scale, Math.sin(angle) * radius * scale];
}

function drawBuildingMarker(building, x, y) {
  const type = String(building.building_type || "");
  const label = BUILDING_LABELS[type] || type.slice(0, 2) || "?";
  const complete = Boolean(building.complete);
  const ownerColor = teamColor(building.owner);
  const size = type === "TURRET" ? 10 : 9;
  const hpRatio = Math.max(0, Math.min(1, Number(building.hp || 0) / Math.max(1, Number(building.max_hp || 1))));
  const alpha = !complete ? 0.62 : (hpRatio <= 0 ? 0.38 : 1);
  const progress = Math.max(0, Math.min(1, Number(building.progress ?? (complete ? 1 : 0))));
  const marker = new Graphics();
  marker.alpha = alpha;
  drawBuildingShape(marker, type, x, y, size, ownerColor);
  layers.map.addChild(marker);

  const progressWidth = size * 2.2;
  const barY = y + size + 3;
  const bars = new Graphics();
  bars.roundRect(x - progressWidth / 2, barY, progressWidth, 2.5, 1).fill(COLORS.hpTrack);
  if (complete) {
    bars.roundRect(x - progressWidth / 2, barY, progressWidth * hpRatio, 2.5, 1).fill(ownerColor);
  } else {
    bars.roundRect(x - progressWidth / 2, barY, progressWidth * progress, 2.5, 1).fill(COLORS.buildingProgress);
  }
  layers.map.addChild(bars);

  const text = new Text({
    text: label,
    style: { fill: "#eef4fa", fontSize: label.length > 1 ? 6 : 8, fontWeight: "800" },
  });
  text.anchor.set(0.5);
  text.x = x;
  text.y = y;
  layers.labels.addChild(text);
}

function drawBuildingShape(g, type, x, y, size, color) {
  if (type === "TURRET") {
    const turretSize = size;
    g.roundRect(x - turretSize, y - turretSize, turretSize * 2, turretSize * 2, 3)
      .fill(COLORS.buildingBg)
      .stroke({ color, width: 2.0 });
    g.moveTo(x, y - turretSize * 0.15).lineTo(x + turretSize * 1.45, y - turretSize * 0.15)
      .stroke({ color, width: 2.4 });
    return;
  }
  if (type === "BOOSTER") {
    g.roundRect(x - size, y - size * 0.75, size * 2, size * 1.5, 5)
      .fill(COLORS.buildingBg)
      .stroke({ color, width: 1.8 });
    g.moveTo(x - size * 0.55, y + size * 0.45)
      .lineTo(x, y - size * 0.55)
      .lineTo(x + size * 0.55, y + size * 0.45)
      .stroke({ color, width: 2.0 });
    return;
  }
  if (type === "POWER") {
    g.circle(x, y, size * 1.05).fill(COLORS.buildingBg).stroke({ color, width: 1.8 });
    g.moveTo(x, y - size * 0.8).lineTo(x - size * 0.25, y).lineTo(x + size * 0.25, y).lineTo(x, y + size * 0.8)
      .stroke({ color, width: 2.0 });
    return;
  }
  if (type === "LIFE") {
    g.circle(x, y, size * 1.05).fill(COLORS.buildingBg).stroke({ color, width: 1.8 });
    g.circle(x - size * 0.28, y - size * 0.12, size * 0.28).fill(color);
    g.circle(x + size * 0.22, y + size * 0.18, size * 0.36).fill(color);
    return;
  }
  if (type === "ECONOMY_COLLECTOR") {
    g.circle(x, y, size * 1.12).fill(COLORS.buildingBg).stroke({ color, width: 1.8 });
    g.circle(x, y, size * 0.42).fill(color);
    return;
  }
  if (type === "TECH_1" || type === "TECH_2") {
    g.moveTo(x, y - size * 1.18)
      .lineTo(x + size * 1.18, y)
      .lineTo(x, y + size * 1.18)
      .lineTo(x - size * 1.18, y)
      .closePath()
      .fill(COLORS.buildingBg)
      .stroke({ color, width: type === "TECH_2" ? 2.3 : 1.8 });
    return;
  }
  g.circle(x, y, size).fill(COLORS.buildingBg).stroke({ color, width: 1.6 });
}

function drawCoreHpBar(nodeId, node, tick, x, y, radius, deadCore, color) {
  const hp = coreHpValue(nodeId, node, tick);
  const ratio = Math.max(0, Math.min(1, hp / CORE_HP));
  const width = Math.max(30, Math.min(48, radius * 1.25));
  const height = 4;
  const barY = y + radius + 5;
  const bar = new Graphics();
  bar.roundRect(x - width / 2, barY, width, height, 2).fill(COLORS.hpTrack);
  if (ratio > 0.001) {
    bar.roundRect(x - width / 2, barY, width * ratio, height, 2).fill(deadCore ? COLORS.deadUnit : color);
  }
  layers.map.addChild(bar);

  const hpText = new Text({
    text: `${Math.max(0, Math.ceil(hp))}/${CORE_HP}`,
    style: { fill: deadCore ? "#151b22" : "#eef4fa", fontSize: 8, fontWeight: "700" },
  });
  hpText.anchor.set(0.5, 0);
  hpText.x = x;
  hpText.y = barY + 5;
  layers.labels.addChild(hpText);
}

function drawNeutralCountLabel(nodeId, node, x, y, radius) {
  const current = Math.max(0, Math.ceil(Number(node.neutral_hp || 0)));
  const maxHp = Math.max(current, Math.ceil(neutralMaxValue(nodeId, node)));
  if (maxHp <= 0) {
    return;
  }
  const text = new Text({
    text: `${current}/${maxHp}`,
    style: {
      fill: "#f0d18a",
      fontSize: 8,
      fontWeight: "800",
      stroke: { color: "#101419", width: 3 },
    },
  });
  text.anchor.set(0.5, 0);
  text.x = x;
  text.y = y + radius + 4;
  layers.labels.addChild(text);
}

function drawNeutralLordHpBar(tick, x, y, radius) {
  const hp = neutralLordHpValue(tick);
  const maxHp = neutralLordMaxHp(tick);
  const ratio = Math.max(0, Math.min(1, hp / Math.max(1, maxHp)));
  const width = Math.max(42, Math.min(60, radius * 1.35));
  const height = 4;
  const barY = y + radius + 6;
  const bar = new Graphics();
  bar.roundRect(x - width / 2, barY, width, height, 2).fill(COLORS.hpTrack);
  if (ratio > 0.001) {
    bar.roundRect(x - width / 2, barY, width * ratio, height, 2).fill(COLORS.baron);
  }
  layers.map.addChild(bar);

  const hpText = new Text({
    text: `${Math.max(0, Math.ceil(hp))}/${Math.ceil(maxHp)}`,
    style: { fill: "#e2b0ff", fontSize: 8, fontWeight: "800", stroke: { color: "#101419", width: 3 } },
  });
  hpText.anchor.set(0.5, 0);
  hpText.x = x;
  hpText.y = barY + 5;
  layers.labels.addChild(hpText);
}

function selectAt(event) {
  const rect = els.stage.getBoundingClientRect();
  const world = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
  let best = null;
  let bestDistance = 18 / appState.camera.scale;
  for (const [id, sprite] of appState.sprites.entries()) {
    const distance = Math.hypot(sprite.x - world[0], sprite.y - world[1]);
    if (distance < bestDistance) {
      best = { type: "unit", id, data: sprite.target, distance };
      bestDistance = distance;
    }
  }
  for (const [nodeId, point] of Object.entries(NODE_POSITIONS)) {
    const distance = Math.hypot(point[0] - world[0], point[1] - world[1]);
    if (distance < bestDistance + 10) {
      best = { type: "node", id: nodeId, data: currentTick()?.nodes?.[nodeId], distance };
      bestDistance = distance;
    }
  }
  appState.selected = best;
  updateInspector();
}

function updateInspector() {
  const selected = appState.selected;
  if (!selected) {
    els.inspectPanel.classList.add("is-hidden");
    els.inspectBody.innerHTML = "";
    return;
  }
  els.inspectPanel.classList.remove("is-hidden");
  if (selected.type === "unit") {
    const data = selected.data || {};
    els.inspectBody.innerHTML = `
      <div class="row"><span>Unit</span><strong>${escapeHtml(data.type || "?")}</strong></div>
      <div class="row"><span>Owner</span><strong>${escapeHtml(data.team || "?")}</strong></div>
      <div class="row"><span>Layer</span><strong>${escapeHtml(data.layer || "?")}</strong></div>
      ${data.route ? `<div class="row"><span>Route</span><strong>${escapeHtml(data.route)}</strong></div>` : ""}
      ${data.intent ? `<div class="row"><span>Intent</span><strong>${escapeHtml(data.intent)}</strong></div>` : ""}
    `;
    return;
  }
  const node = selected.data || {};
  const buildings = buildingSummary(node.buildings || {});
  const player = currentTick()?.players?.[selected.id];
  els.inspectBody.innerHTML = `
    <div class="row"><span>Node</span><strong>${escapeHtml(selected.id)}</strong></div>
    <div class="row"><span>Owner</span><strong>${escapeHtml(node.owner || "?")}</strong></div>
    ${player?.faction ? `<div class="row"><span>Faction</span><strong>${escapeHtml(player.faction)}</strong></div>` : ""}
    <div class="row"><span>Units</span><strong>${unitSummary(node.units || {}) || "-"}</strong></div>
    <div class="row"><span>Visitors</span><strong>${Object.keys(node.visiting_units || {}).length}</strong></div>
    <div class="row"><span>Neutral HP</span><strong>${Number(node.neutral_hp || 0)}</strong></div>
    <div class="row"><span>Buildings</span><strong>${buildings || "-"}</strong></div>
  `;
}

function fitCamera(center = true) {
  if (!appState.app) {
    return;
  }
  const width = Math.max(1, appState.app.screen.width);
  const height = Math.max(1, appState.app.screen.height);
  // In the World broadcast the canvas is much wider than the map, so the map is height-constrained and
  // leaves top/bottom slack — fill more there. The standalone Match tab keeps 0.92 (room for its HUDs).
  const fill = (typeof EMBED !== "undefined" && EMBED && EMBED.active) ? 0.98 : 0.92;
  const scale = Math.min(width / WORLD_W, height / WORLD_H) * fill;
  appState.camera.scale = scale;
  if (center) {
    appState.camera.x = (width - WORLD_W * scale) / 2;
    appState.camera.y = (height - WORLD_H * scale) / 2;
  }
  layers.root.x = appState.camera.x;
  layers.root.y = appState.camera.y;
  layers.root.scale.set(scale);
}

function screenToWorld(x, y) {
  return [
    (x - appState.camera.x) / appState.camera.scale,
    (y - appState.camera.y) / appState.camera.scale,
  ];
}

function seekTo(value) {
  appState.tickFloat = Math.max(0, Math.min(maxTick(), Number(value || 0)));
  resetPlaybackClock(performance.now());
  updateInspector();
}

function togglePlay() {
  appState.playing = !appState.playing;
  els.playBtn.textContent = appState.playing ? "||" : ">";
  resetPlaybackClock(performance.now());
}

function resetPlaybackClock(now) {
  appState.playStartedAt = now;
  appState.playStartTick = appState.tickFloat;
}

function maxTick() {
  return Math.max(0, (appState.replay?.ticks || []).length - 1);
}

function currentTick() {
  return appState.replay?.ticks?.[Math.max(0, Math.min(maxTick(), Math.floor(appState.tickFloat)))] || null;
}

function mapStateSignature(tick) {
  return Object.entries(tick?.nodes || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([nodeId, node]) => {
      const buildings = (node.buildings || [])
        .map((building) => [
          building.owner || "",
          building.building_type || "",
          Math.round(Number(building.hp || 0) * 10) / 10,
          Number(building.progress || 0).toFixed(2),
          building.complete ? 1 : 0,
        ].join(","))
        .join(";");
      return `${nodeId}:${node.owner || ""}:${node.neutral_hp || 0}:${node.neutral_max_hp || 0}:${node.core_hp ?? ""}:${tick?.players?.[nodeId]?.alive ?? ""}:${buildings}`;
    })
    .join("|") + `|baron:${neutralLordHpValue(tick)}:${neutralLordMaxHp(tick)}`;
}

function battlePosition(battle) {
  if (battle.location_kind === "node") {
    return nodePoint(battle.location_id);
  }
  if (battle.location_kind === "approach" && battle.approach_id) {
    const [src, dst] = String(battle.approach_id).split("->");
    return edgePoint(src, dst, Number(battle.edge_position ?? 0.75));
  }
  const [src, dst] = String(battle.location_id || "").split("|");
  return edgePoint(src, dst, Number(battle.edge_position ?? 0.5));
}

function battleAxis(battle) {
  if (battle.location_kind === "approach" && battle.approach_id) {
    const [src, dst] = String(battle.approach_id).split("->");
    return edgeAxis(src, dst);
  }
  if (battle.location_kind === "edge") {
    const [src, dst] = String(battle.location_id || "").split("|");
    return edgeAxis(src, dst);
  }
  return [1, 0];
}

function fallbackFormationPoint(base, axis, unit) {
  if (!base) {
    return null;
  }
  const pos = Math.max(-1, Math.min(1, (Number(unit.pos || 10) - 10) / 10));
  const perp = [-axis[1], axis[0]];
  const lane = typeLane(unit.type) * 1.6;
  return [base[0] + axis[0] * pos * 82 + perp[0] * lane, base[1] + axis[1] * pos * 82 + perp[1] * lane];
}

function interpolateFormation(currentRows, nextRows, alpha) {
  const current = normalizeFormation(currentRows);
  const next = normalizeFormation(nextRows);
  const nextByUid = new Map(next.map((unit) => [unit.uid, unit]));
  return current.map((unit) => {
    const nextUnit = nextByUid.get(unit.uid);
    return {
      ...unit,
      hp: lerp(unit.hp, nextUnit?.hp ?? unit.hp, alpha),
      pos: lerp(unit.pos, nextUnit?.pos ?? unit.pos, alpha),
      x: Number.isFinite(unit.x) ? lerp(unit.x, nextUnit?.x ?? unit.x, alpha) : unit.x,
      y: Number.isFinite(unit.y) ? lerp(unit.y, nextUnit?.y ?? unit.y, alpha) : unit.y,
    };
  });
}

function normalizeFormation(rows) {
  return (rows || []).map((row) => ({
    uid: Number(row[0]),
    team: row[1],
    type: displayUnitCode(row[2]),
    hp: Number(row[3]) || 0,
    maxHp: Number(row[4]) || 0,
    pos: Number(row[7] ?? row[5] ?? 10),
    range: Number(row[6]) || 0,
    x: row.length >= 10 ? Number(row[8]) : NaN,
    y: row.length >= 10 ? Number(row[9]) : NaN,
    rangeMap: row.length >= 11 ? Number(row[10]) : null,
  }));
}

function findNextGroup(group, groups, used) {
  const signature = movementSignature(group);
  let best = null;
  let bestScore = Number.POSITIVE_INFINITY;
  for (let index = 0; index < groups.length; index += 1) {
    if (used.has(index)) {
      continue;
    }
    const candidate = groups[index];
    if (movementSignature(candidate) !== signature) {
      continue;
    }
    const score = Math.abs(Number(candidate.progress || 0) - Number(group.progress || 0));
    if (score < bestScore) {
      best = { index, candidate };
      bestScore = score;
    }
  }
  if (!best) {
    return null;
  }
  used.add(best.index);
  return best.candidate;
}

function movementSignature(group) {
  return [
    group.player,
    group.phase,
    group.segment_src || group.src,
    group.segment_dst || group.dst,
    group.segment_node || "",
    (group.path || []).join(">"),
    JSON.stringify(group.units || {}),
  ].join("|");
}

function groupVisualPoint(group) {
  if (Number.isFinite(group?.visual?.x) && Number.isFinite(group?.visual?.y)) {
    return [Number(group.visual.x), Number(group.visual.y)];
  }
  if (group.phase === "edge") {
    return edgePoint(group.src, group.dst, Number(group.progress || 0));
  }
  return nodePoint(group.segment_node || group.dst || group.src);
}

function expandUnitEntries(units) {
  const entries = [];
  for (const [rawType, rawCount] of Object.entries(units || {})) {
    const type = displayUnitCode(rawType);
    const count = Math.max(0, Math.min(MAX_EXPANDED_UNITS - entries.length, Number(rawCount || 0)));
    for (let index = 0; index < count; index += 1) {
      entries.push({ type, index });
    }
    if (entries.length >= MAX_EXPANDED_UNITS) {
      break;
    }
  }
  return entries;
}

function clusterOffset(index, count, spread) {
  if (count <= 1) {
    return [0, 0];
  }
  const columns = Math.ceil(Math.sqrt(count));
  const row = Math.floor(index / columns);
  const col = index % columns;
  const rows = Math.ceil(count / columns);
  return [
    (col - (columns - 1) / 2) * spread * 0.58,
    (row - (rows - 1) / 2) * spread * 0.58,
  ];
}

function nodePoint(nodeId) {
  const meta = appState.nodeMeta?.[nodeId];
  if (meta?.center) {
    return [Number(meta.center[0]), Number(meta.center[1])];
  }
  return NODE_POSITIONS[nodeId] || null;
}

function edgePoint(src, dst, progress) {
  const start = nodePoint(src);
  const end = nodePoint(dst);
  if (!start || !end) {
    return null;
  }
  const t = Math.max(0, Math.min(1, Number(progress || 0)));
  return [lerp(start[0], end[0], t), lerp(start[1], end[1], t)];
}

function edgeAxis(src, dst) {
  const start = nodePoint(src);
  const end = nodePoint(dst);
  if (!start || !end) {
    return [1, 0];
  }
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const length = Math.max(1, Math.hypot(dx, dy));
  return [dx / length, dy / length];
}

function edgeKey(src, dst) {
  return [src, dst].sort().join("|");
}

function displayUnitCode(type) {
  const value = String(type || "").toUpperCase();
  if (value === "VANGUARD" || value === "V") return "V";
  if (value === "RANGER" || value === "ARCHER" || value === "A" || value === "R") return "R";
  if (value === "SKIRMISHER" || value === "S") return "S";
  if (value === "RAIDER" || value === "SPECIALIST_1" || value === "RD" || value === "D") return "RD";
  if (value === "BMP" || value === "SPECIALIST_2" || value === "MP") return "MP";
  if (value === "SWARMLING" || value === "SW") return "SW";
  if (value === "SPITTER" || value === "SP") return "SP";
  if (value === "BRUTE" || value === "BR") return "BR";
  if (value === "RAVAGER" || value === "RV") return "RV";
  if (value === "BEHEMOTH" || value === "BH") return "BH";
  if (value === "WALKER" || value === "WK") return "WK";
  if (value === "LASER" || value === "LS") return "LS";
  if (value === "DRONE" || value === "DR") return "DR";
  if (value === "GUARDIAN_MECH" || value === "GM") return "GM";
  if (value === "SIEGE_PLATFORM" || value === "SG") return "SG";
  if (value === "NEUTRAL" || value === "N") return "N";
  if (value === "CORE" || value === "C") return "C";
  if (value === "BARON" || value === "LORD" || value === "B" || value === "L") return "L";
  if (value === "TURRET" || value === "T") return "T";
  if (value === "BOOSTER" || value === "BO") return "BO";
  if (value === "POWER" || value === "PW") return "PW";
  if (value === "LIFE" || value === "LF") return "LF";
  if (value === "ECONOMY_COLLECTOR" || value === "EC") return "EC";
  if (value === "TECH_1" || value === "T1") return "T1";
  if (value === "TECH_2" || value === "T2") return "T2";
  return value || "?";
}

function teamColor(team) {
  if (team === "BARON") {
    return COLORS.baron;  // Lord renders purple, matching its node ring/halo
  }
  if (team === "NEUTRAL") {
    return COLORS.neutral;
  }
  return String(team || "").startsWith("B") ? COLORS.teamB : COLORS.teamA;
}

function nodeFill(owner) {
  if (owner === "NEUTRAL" || owner === "BARON") {
    return COLORS.neutralBg;
  }
  return String(owner || "").startsWith("B") ? COLORS.teamBBg : COLORS.teamABg;
}

function isCoreNode(nodeId) {
  return /^[AB][123]$/.test(String(nodeId || ""));
}

function coreHpValue(nodeId, node = {}, tick = {}) {
  const value = Number(node.core_hp ?? tick.players?.[nodeId]?.core_hp ?? CORE_HP);
  if (!Number.isFinite(value)) {
    return CORE_HP;
  }
  return Math.max(0, Math.min(CORE_HP, value));
}

function neutralMaxValue(nodeId, node = {}) {
  const direct = Number(node.neutral_max_hp || 0);
  if (Number.isFinite(direct) && direct > 0) {
    return direct;
  }
  let maxHp = Number(node.neutral_hp || 0);
  for (const tick of appState.replay?.ticks || []) {
    const tickNode = tick?.nodes?.[nodeId];
    maxHp = Math.max(maxHp, Number(tickNode?.neutral_max_hp || 0), Number(tickNode?.neutral_hp || 0));
  }
  return Number.isFinite(maxHp) ? maxHp : 0;
}

function neutralLordHpValue(tick = {}) {
  for (const battle of tick.active_battles || []) {
    if (battle.location_id !== "Baron" && battle.battle_id !== "node:Baron") {
      continue;
    }
    for (const row of battle.formation || []) {
      if (String(row?.[1] || "") === "BARON" && displayUnitCode(row?.[2]) === "L") {
        const hp = Number(row?.[3]);
        if (Number.isFinite(hp)) {
          return hp;
        }
      }
    }
  }
  const hp = Number(tick.baron?.hp || 0);
  return Number.isFinite(hp) ? hp : 0;
}

function neutralLordMaxHp(tick = {}) {
  const direct = Number(tick.baron?.max_hp || appState.replay?.meta?.rules?.baron_hp || 300);
  return Number.isFinite(direct) && direct > 0 ? direct : 300;
}

function isDeadCoreNode(nodeId, node = {}, tick = {}) {
  if (!isCoreNode(nodeId)) {
    return false;
  }
  const coreHp = coreHpValue(nodeId, node, tick);
  const alive = tick.players?.[nodeId]?.alive;
  return alive === false || coreHp <= 0;
}

function displayTeamBucket(team) {
  if (team === "NEUTRAL" || team === "BARON") {
    return "NEUTRAL";
  }
  return String(team || "").startsWith("B") ? "TEAM_B" : "TEAM_A";
}

function isBattleLayer(layer) {
  return ["battle", "fighting", "fallen"].includes(String(layer || ""));
}

function unitSize(item) {
  const spec = UNIT_SPEC[item.type];
  if (spec) {
    const base = KLASS_SIZE[spec.klass] || 5.3;
    // Objectives are absolute; mobile units bump slightly in a battle cluster.
    if (spec.klass === "lord" || spec.klass === "core" || spec.klass === "neutral") {
      return base;
    }
    return item.layer === "battle" ? base * 1.08 : base;
  }
  if (BUILDING_TYPE_CODES.has(item.type)) {
    return item.layer === "battle" ? 5.6 : 4.8;
  }
  return item.layer === "battle" ? 5.4 : 4.6;
}

function typeLane(type) {
  return {
    R: -7,
    MP: -10,
    SP: -6,
    LS: -9,
    SG: -11,
    S: 0,
    RD: 2,
    SW: 1,
    RV: 2,
    DR: 2,
    GM: 1,
    V: 7,
    BR: 8,
    BH: 9,
    WK: 8,
    C: 2,
    L: 0,
    N: 0,
  }[displayUnitCode(type)] || 0;
}

function unitSummary(units) {
  return Object.entries(units || {})
    .filter(([, count]) => Number(count || 0) > 0)
    .map(([type, count]) => `${displayUnitCode(type)}${count}`)
    .join(" ");
}

function buildingSummary(buildings = []) {
  return (buildings || [])
    .filter((building) => Number(building?.max_hp ?? building?.hp ?? 0) > 0)
    .map((building) => {
      const label = BUILDING_LABELS[building.building_type] || String(building.building_type || "?");
      const hp = `${Math.ceil(Number(building.hp || 0))}/${Math.ceil(Number(building.max_hp || 0))}`;
      const status = building.complete ? hp : `${Math.round(Number(building.progress || 0) * 100)}%`;
      return `${label}:${building.owner || "?"} ${status}`;
    })
    .join(" | ");
}

function lerp(left, right, alpha) {
  return Number(left || 0) + (Number(right || 0) - Number(left || 0)) * Math.max(0, Math.min(1, alpha));
}

function smoothstep(value) {
  const t = Math.max(0, Math.min(1, Number(value || 0)));
  return t * t * (3 - 2 * t);
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}: ${path}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      // Keep the HTTP status fallback.
    }
    throw new Error(message);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
