import { Application, Container, Graphics, Text } from "pixi.js";

const WORLD_W = 680;
const WORLD_H = 410;
const BASE_TICK_MS = 700;
const HANDOFF_DISTANCE = 90;
const MAX_EXPANDED_UNITS = 260;
const HANDOFF_BACKFILL_DISTANCE = 135;
const CORE_HP = 80;
const ATTACK_PREVIEW_ALPHA = 0.5;
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
  label: `replay_${index}.json`,
  path: `../replays_viewer_debug_smoke/replay_${index}.json`,
}));

const COLORS = {
  bg: 0x0f1419,
  lane: 0x44505d,
  tunnel: 0x242c35,
  room: 0x222932,
  roomLine: 0x758292,
  teamA: 0xf0646e,
  teamABg: 0x55282d,
  teamB: 0x5da0ff,
  teamBBg: 0x203a60,
  neutral: 0xd8a84b,
  neutralBg: 0x332818,
  deadCore: 0xffffff,
  deadCoreBg: 0xf2f5f8,
  deadCoreText: 0x151b22,
  deadUnit: 0x858c95,
  hpTrack: 0x101419,
  text: 0xeef4fa,
  muted: 0x9facba,
  danger: 0xff8f8f,
  attack: 0xf0d18a,
  buildingBg: 0x101419,
  buildingProgress: 0xf0d18a,
};

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

const appState = {
  app: null,
  replay: null,
  samples: [...FALLBACK_REPLAYS],
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
};

const layers = {
  root: new Container(),
  map: new Container(),
  labels: new Container(),
  units: new Container(),
  effects: new Container(),
};

const els = {
  stage: document.getElementById("pixiStage"),
  metaText: document.getElementById("metaText"),
  replaySelect: document.getElementById("replaySelect"),
  loadBtn: document.getElementById("loadBtn"),
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

  layers.root.addChild(layers.map, layers.labels, layers.effects, layers.units);
  app.stage.addChild(layers.root);

  bindControls();
  await loadReplayList();
  drawStaticMap();
  fitCamera();
  await loadSelectedReplay();
  app.ticker.add((ticker) => frame(ticker.deltaMS));
}

function bindControls() {
  window.addEventListener("resize", () => {
    fitCamera(false);
    drawStaticMap();
  });
  els.loadBtn.addEventListener("click", loadSelectedReplay);
  els.generateBtn.addEventListener("click", generateReplays);
  els.generateCountInput.addEventListener("input", validateGeneratorControls);
  els.generateSeedInput.addEventListener("input", validateGeneratorControls);
  els.prevBtn.addEventListener("click", () => seekTo(Math.floor(appState.tickFloat) - 1));
  els.nextBtn.addEventListener("click", () => seekTo(Math.floor(appState.tickFloat) + 1));
  els.playBtn.addEventListener("click", togglePlay);
  els.tickSlider.addEventListener("input", (event) => seekTo(Number(event.target.value)));
  els.speedSelect.addEventListener("change", () => resetPlaybackClock(performance.now()));
  els.stage.addEventListener("click", (event) => selectAt(event));
}

async function loadReplayList() {
  try {
    const payload = await fetchJson("/api/models");
    appState.samples = payload.replays?.length ? payload.replays : [...FALLBACK_REPLAYS];
    appState.maxUiRuns = Number(payload.max_ui_runs || 100);
    els.generateCountInput.max = String(appState.maxUiRuns);
    setGenerateStatus("");
  } catch {
    appState.samples = [...FALLBACK_REPLAYS];
    setGenerateStatus("Generation API offline", "error");
  }
  populateReplaySelect();
  validateGeneratorControls();
}

function populateReplaySelect() {
  els.replaySelect.innerHTML = appState.samples
    .map((item, index) => `<option value="${index}">${escapeHtml(item.label || `replay_${index}.json`)}</option>`)
    .join("");
}

async function generateReplays() {
  if (!validateGeneratorControls()) {
    return;
  }

  const runs = Number(els.generateCountInput.value || 20);
  const seed = Number(els.generateSeedInput.value || 20260602);
  appState.generating = true;
  validateGeneratorControls();
  setGenerateStatus("Generating", "loading", `Generating ${runs} matches`);

  try {
    const payload = await fetchJson("/api/generate-replays", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runs, seed }),
    });
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
  const seed = Number(els.generateSeedInput.value || 0);
  const validRuns = Number.isInteger(runs) && runs >= 1 && runs <= appState.maxUiRuns;
  const validSeed = Number.isFinite(seed);
  const disabled = appState.generating || !validRuns || !validSeed;
  els.generateBtn.disabled = disabled;
  els.generateBtn.setAttribute("aria-busy", appState.generating ? "true" : "false");
  if (!validRuns) {
    els.generateBtn.title = `Matches must be 1..${appState.maxUiRuns}`;
  } else if (!validSeed) {
    els.generateBtn.title = "Seed must be a number";
  } else {
    els.generateBtn.title = "Generate replay files";
  }
  return !disabled;
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
  ].filter(Boolean).join(" | ");
}

function countSummary(values = {}) {
  return Object.entries(values)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}:${value}`)
    .join(" ");
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
  appState.nodeMeta = replay.meta?.map2d?.nodes || {};
  appState.lastMapSignature = "";
  els.tickSlider.max = String(Math.max(0, (replay.ticks || []).length - 1));
  els.tickSlider.value = "0";
  els.tickNow.textContent = "0";
  els.tickMax.textContent = String(Math.max(0, (replay.ticks || []).length - 1));
  els.playBtn.textContent = ">";
  els.metaText.textContent = `schema ${replay.meta?.schema_version || "?"} | ${replay.meta?.winner || "?"} | ${replay.meta?.total_ticks || 0}t`;
  drawStaticMap();
  updateInspector();
}

function frame(deltaMS) {
  if (!appState.replay) {
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
    }
  }
  renderReplay(deltaMS);
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
  drawEffects(tick, nextTick, alpha);
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
  return current.map((unit) => {
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
      sprite.dying = true;
      sprite.deathAge = 0;
      sprite.target = {
        ...(sprite.target || {}),
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
  const body = new Graphics();
  const hp = new Graphics();
  container.addChild(body, hp);
  return { container, body, hp, target: item, x: item.x, y: item.y, alpha: 0 };
}

function updateUnitSprite(sprite, item) {
  const dead = Boolean(item.dead || item.layer === "fallen" || item.state === "fallen" || item.hpRatio <= 0);
  const color = dead ? COLORS.deadUnit : teamColor(item.team);
  const size = unitSize(item);
  sprite.container.x = sprite.x;
  sprite.container.y = sprite.y;
  sprite.container.alpha = sprite.alpha;
  sprite.container.eventMode = "none";
  sprite.body.clear();
  drawUnitShape(sprite.body, item.type, color, size);
  sprite.hp.clear();
  const hpRatio = Math.max(0, Math.min(1, Number(item.hpRatio ?? 1)));
  if (!hideSpriteHpBar(item) && (dead || hpRatio < 0.999)) {
    sprite.hp.roundRect(-size, -size - 8, size * 2, 3, 1).fill(COLORS.hpTrack);
    const hpWidth = size * 2 * hpRatio;
    if (hpWidth > 0.5) {
      sprite.hp.roundRect(-size, -size - 8, hpWidth, 3, 1).fill(color);
    }
  }
}

function hideSpriteHpBar(item) {
  return item?.type === "L" && item?.team === "BARON";
}

function drawUnitShape(g, type, color, size) {
  if (type === "RD") {
    const points = 6;
    for (let index = 0; index < points; index += 1) {
      const angle = -Math.PI / 2 + index * (Math.PI * 2 / points);
      const x = Math.cos(angle) * size * 1.08;
      const y = Math.sin(angle) * size * 1.08;
      if (index === 0) {
        g.moveTo(x, y);
      } else {
        g.lineTo(x, y);
      }
    }
    g.closePath().fill(color).stroke({ color: 0x101419, width: 1.2 });
    return;
  }
  if (type === "MP") {
    g.roundRect(-size * 1.25, -size * 0.82, size * 2.5, size * 1.64, 2)
      .fill(color)
      .stroke({ color: 0x101419, width: 1.3 });
    g.moveTo(size * 0.55, 0).lineTo(size * 1.7, 0).stroke({ color: 0x101419, width: 2 });
    g.circle(-size * 0.55, size * 0.9, size * 0.23).fill(0x101419);
    g.circle(size * 0.55, size * 0.9, size * 0.23).fill(0x101419);
    return;
  }
  if (type === "R") {
    g.moveTo(0, -size).lineTo(size, 0).lineTo(0, size).lineTo(-size, 0).closePath().fill(color).stroke({ color: 0x101419, width: 1.2 });
    return;
  }
  if (type === "S") {
    g.moveTo(0, -size * 1.15).lineTo(size, size * 0.9).lineTo(-size, size * 0.9).closePath().fill(color).stroke({ color: 0x101419, width: 1.2 });
    return;
  }
  if (type === "C") {
    g.roundRect(-size * 1.15, -size * 1.15, size * 2.3, size * 2.3, 3).fill(color).stroke({ color: 0x101419, width: 1.4 });
    return;
  }
  if (type === "L" || type === "B") {
    drawLordShape(g, color, size);
    return;
  }
  g.circle(0, 0, size).fill(color).stroke({ color: 0x101419, width: 1.2 });
}

function drawLordShape(g, color, size) {
  const outer = size * 1.75;
  const inner = size * 0.92;
  for (let index = 0; index < 10; index += 1) {
    const radius = index % 2 === 0 ? outer : inner;
    const angle = -Math.PI / 2 + index * Math.PI / 5;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    if (index === 0) {
      g.moveTo(x, y);
    } else {
      g.lineTo(x, y);
    }
  }
  g.closePath().fill(COLORS.neutral).stroke({ color, width: 2.5, join: "round" });
  g.moveTo(0, -size * 1.02)
    .lineTo(size * 0.84, 0)
    .lineTo(0, size * 1.02)
    .lineTo(-size * 0.84, 0)
    .closePath()
    .fill(color)
    .stroke({ color: 0x101419, width: 1.4, join: "round" });
  g.circle(0, 0, size * 0.34).fill(0x101419);
}

function drawEffects(tick, nextTick, alpha) {
  layers.effects.removeChildren();
  for (const battle of tick.active_battles || []) {
    drawAttackLines(tick, battle, nextTick, alpha);
  }
}

function drawAttackLines(tick, battle, nextTick, alpha) {
  const visualEvent = battleTickEventForVisual(tick, nextTick, battle, alpha);
  const attacks = visualEvent.event?.summary?.attacks || [];
  if (!attacks.length) {
    return;
  }
  const byUid = new Map();
  for (const sprite of appState.sprites.values()) {
    const target = sprite.target || {};
    if (target.battleId === battle.battle_id && target.uid != null) {
      byUid.set(Number(target.uid), sprite);
    }
  }
  const lines = new Graphics();
  for (const attack of attacks.slice(0, 64)) {
    const from = byUid.get(Number(attack[0]));
    const to = byUid.get(Number(attack[1]));
    if (!from || !to) {
      continue;
    }
    lines.moveTo(from.x, from.y).lineTo(to.x, to.y).stroke({
      color: teamColor(from.target.team),
      width: displayUnitCode(attack[4]) === "R" ? 1.1 : 1.6,
      alpha: (displayUnitCode(attack[4]) === "R" ? 0.52 : 0.72) * visualEvent.alpha,
    });
  }
  layers.effects.addChild(lines);
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
  const bg = new Graphics();
  bg.rect(0, 0, WORLD_W, WORLD_H).fill(0x101419);
  layers.map.addChild(bg);

  for (const [src, dst] of EDGES) {
    const key = edgeKey(src, dst);
    const [sx, sy] = nodePoint(src);
    const [dx, dy] = nodePoint(dst);
    const line = new Graphics();
    line.moveTo(sx, sy).lineTo(dx, dy).stroke({
      color: TUNNELS.has(key) ? COLORS.tunnel : COLORS.lane,
      width: TUNNELS.has(key) ? 9 : 4,
      alpha: TUNNELS.has(key) ? 0.95 : 0.86,
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
    const ringColor = deadCore ? COLORS.deadCore : teamColor(owner);
    const fillColor = deadCore ? COLORS.deadCoreBg : nodeFill(owner);
    const room = new Graphics();
    room.circle(x, y, radius).fill(fillColor).stroke({ color: ringColor, width: nodeId === "Baron" ? 4 : 2.4, alpha: 0.95 });
    layers.map.addChild(room);
    if (isCoreNode(nodeId)) {
      drawCoreHpBar(nodeId, node, tick, x, y, radius, deadCore, ringColor);
    }
    if (nodeId.startsWith("CP") && Number(node.neutral_hp || 0) > 0) {
      drawNeutralCountLabel(nodeId, node, x, y, radius);
    }
    if (nodeId === "Baron" && node.owner === "NEUTRAL" && Number(tick.baron?.hp || 0) > 0) {
      drawNeutralLordHpBar(tick, x, y, radius);
    }
    drawNodeBuildings(nodeId, node, tick, x, y, radius);

    const label = new Text({
      text: nodeId === "Baron" ? "L" : nodeId,
      style: { fill: deadCore ? "#151b22" : "#eef4fa", fontSize: nodeId === "Baron" ? 18 : 13, fontWeight: "700" },
    });
    label.anchor.set(0.5);
    label.x = x;
    label.y = y;
    layers.labels.addChild(label);
  }

  fitCamera(false);
}

function drawNodeBuildings(nodeId, node, tick, x, y, radius) {
  const buildings = (node?.buildings || [])
    .filter((building) => Number(building?.max_hp ?? building?.hp ?? 0) > 0)
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
    bar.roundRect(x - width / 2, barY, width * ratio, height, 2).fill(COLORS.neutral);
  }
  layers.map.addChild(bar);

  const hpText = new Text({
    text: `${Math.max(0, Math.ceil(hp))}/${Math.ceil(maxHp)}`,
    style: { fill: "#f0d18a", fontSize: 8, fontWeight: "800", stroke: { color: "#101419", width: 3 } },
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
    els.inspectBody.innerHTML = `<span class="muted">Click a unit, room, or fight.</span>`;
    return;
  }
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
  els.inspectBody.innerHTML = `
    <div class="row"><span>Node</span><strong>${escapeHtml(selected.id)}</strong></div>
    <div class="row"><span>Owner</span><strong>${escapeHtml(node.owner || "?")}</strong></div>
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
  const scale = Math.min(width / WORLD_W, height / WORLD_H) * 0.92;
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
  return value.slice(0, 1) || "?";
}

function teamColor(team) {
  if (team === "NEUTRAL" || team === "BARON") {
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
  if (item.type === "L") {
    return 12;
  }
  if (item.type === "MP") {
    return item.layer === "battle" ? 6.6 : 6.0;
  }
  if (["BH", "GM", "SG"].includes(item.type)) {
    return item.layer === "battle" ? 7.0 : 6.2;
  }
  if (["SW", "DR"].includes(item.type)) {
    return item.layer === "battle" ? 5.0 : 4.2;
  }
  if (item.type === "C") {
    return 9;
  }
  if (item.type === "N") {
    return 5;
  }
  return item.layer === "battle" ? 5.5 : 4.6;
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
