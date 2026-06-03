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

const TUNNEL_EDGES = new Set([
  "A1|CP1", "A1|CP2", "A2|CP1", "A2|CP2", "A3|CP1", "A3|CP2",
  "B1|CP5", "B1|CP6", "B2|CP5", "B2|CP6", "B3|CP5", "B3|CP6",
]);
const NODE_IDS = Object.keys(NODE_POSITIONS);
const CORE_NODES = new Set(["A1", "A2", "A3", "B1", "B2", "B3"]);
const CP_NODES = new Set(["CP1", "CP2", "CP3", "CP4", "CP5", "CP6"]);
const CORE_HP = 50;
const LORD_HP = 120;
const WORLD_W = 680;
const WORLD_H = 410;
const BASE_TICK_MS = 700;
const HANDOFF_DISTANCE = 95;
const MAX_EXPANDED_UNITS = 220;
const SAMPLE_REPLAYS = Array.from({ length: 20 }, (_, index) => ({
  label: `replay_${index}.json`,
  path: `../replays_viewer_debug_smoke/replay_${index}.json`,
}));

const COLORS = {
  bg: "#101419",
  road: "#34404d",
  roadDark: "#06080b",
  room: "#2a3038",
  roomLine: "#737d8a",
  teamA: "#f0646e",
  teamABg: "#55282d",
  teamB: "#5da0ff",
  teamBBg: "#203a60",
  neutral: "#d8a84b",
  neutralBg: "#332818",
  text: "#eef4fa",
  muted: "#9facba",
  green: "#75c47b",
  danger: "#ff8f8f",
};

const UNIT_SHAPE = {
  V: "circle",
  S: "triangle",
  R: "diamond",
  C: "core",
  N: "circle",
  L: "lord",
  B: "lord",
};

const state = {
  replay: null,
  replayLabel: "",
  samples: [...SAMPLE_REPLAYS],
  tickFloat: 0,
  playing: false,
  lastFrame: 0,
  snapVisuals: true,
  visuals: new Map(),
  selected: null,
  dragging: false,
  dragStart: null,
  cameraStart: null,
  camera: { x: 0, y: 0, scale: 1 },
};

const els = {
  canvas: document.getElementById("gameCanvas"),
  metaText: document.getElementById("metaText"),
  replaySelect: document.getElementById("replaySelect"),
  loadBtn: document.getElementById("loadBtn"),
  fileInput: document.getElementById("fileInput"),
  prevBtn: document.getElementById("prevBtn"),
  playBtn: document.getElementById("playBtn"),
  nextBtn: document.getElementById("nextBtn"),
  tickSlider: document.getElementById("tickSlider"),
  tickNow: document.getElementById("tickNow"),
  tickMax: document.getElementById("tickMax"),
  speedSelect: document.getElementById("speedSelect"),
  inspectBody: document.getElementById("inspectBody"),
};

function setup() {
  fitCamera();
  bindControls();
  loadReplayList();
  requestAnimationFrame(frameLoop);
}

function bindControls() {
  window.addEventListener("resize", () => {
    fitCamera(false);
    state.snapVisuals = true;
  });
  els.loadBtn.addEventListener("click", loadSelectedReplay);
  els.fileInput.addEventListener("change", loadFileReplay);
  els.playBtn.addEventListener("click", togglePlay);
  els.prevBtn.addEventListener("click", () => seekTo(Math.floor(state.tickFloat) - 1));
  els.nextBtn.addEventListener("click", () => seekTo(Math.floor(state.tickFloat) + 1));
  els.tickSlider.addEventListener("input", (event) => seekTo(Number(event.target.value)));
  els.canvas.addEventListener("mousedown", startDrag);
  window.addEventListener("mousemove", dragMove);
  window.addEventListener("mouseup", endDrag);
  els.canvas.addEventListener("wheel", zoomAtPointer, { passive: false });
  els.canvas.addEventListener("click", selectAtPointer);
}

async function loadReplayList() {
  try {
    const response = await fetch("/api/replays", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    state.samples = payload.replays?.length ? payload.replays : [...SAMPLE_REPLAYS];
  } catch (error) {
    state.samples = [...SAMPLE_REPLAYS];
  }
  populateReplaySelect();
  if (state.samples[0]) {
    await loadReplayPath(state.samples[0].path, state.samples[0].label);
  }
}

function populateReplaySelect() {
  els.replaySelect.innerHTML = "";
  for (const sample of state.samples) {
    const option = document.createElement("option");
    option.value = sample.path;
    option.textContent = sample.label;
    els.replaySelect.appendChild(option);
  }
}

async function loadSelectedReplay() {
  const selected = state.samples.find((sample) => sample.path === els.replaySelect.value);
  await loadReplayPath(els.replaySelect.value, selected?.label || els.replaySelect.value);
}

async function loadReplayPath(path, label = "") {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  setReplay(await response.json(), label || path);
}

function loadFileReplay(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = () => setReplay(JSON.parse(String(reader.result)), file.name);
  reader.readAsText(file, "utf-8");
}

function setReplay(replay, label) {
  if (!Array.isArray(replay?.ticks) || !replay.ticks.length) {
    throw new Error("Replay has no ticks");
  }
  state.replay = replay;
  state.replayLabel = label;
  state.tickFloat = 0;
  state.playing = false;
  state.visuals.clear();
  state.selected = null;
  state.snapVisuals = true;
  fitCamera();
  updatePlaybackUi();
  updateMeta();
  updateInspector();
}

function updateMeta() {
  const meta = state.replay?.meta || {};
  els.metaText.textContent = state.replay
    ? `${state.replayLabel} | schema ${meta.schema_version || "?"} | winner ${meta.winner || "?"}`
    : "load replay";
}

function togglePlay() {
  if (!state.replay) {
    return;
  }
  state.playing = !state.playing;
  els.playBtn.textContent = state.playing ? "||" : ">";
}

function seekTo(tick) {
  if (!state.replay) {
    return;
  }
  const max = state.replay.ticks.length - 1;
  state.tickFloat = Math.max(0, Math.min(max, tick));
  state.snapVisuals = true;
  updatePlaybackUi();
}

function frameLoop(now) {
  const dt = state.lastFrame ? Math.min(0.08, (now - state.lastFrame) / 1000) : 0;
  state.lastFrame = now;

  if (state.replay && state.playing) {
    const speed = Number(els.speedSelect.value || 1);
    const maxTick = state.replay.ticks.length - 1;
    state.tickFloat = Math.min(maxTick, state.tickFloat + (dt * 1000 / BASE_TICK_MS) * speed);
    if (state.tickFloat >= maxTick) {
      state.playing = false;
      els.playBtn.textContent = ">";
    }
  }

  render(dt);
  requestAnimationFrame(frameLoop);
}

function render(dt) {
  const { ctx, width, height } = prepareCanvas();
  ctx.clearRect(0, 0, width, height);
  drawBackdrop(ctx, width, height);

  ctx.save();
  applyCamera(ctx);
  drawWorld(ctx);

  const tick = currentTick();
  const nextTick = nextReplayTick();
  const alpha = tickAlpha();
  if (tick) {
    const desired = desiredUnits(tick, nextTick, alpha);
    syncVisuals(desired, dt);
    drawBattleEffects(ctx, tick);
    drawVisualUnits(ctx);
    drawSelection(ctx, tick);
  }
  ctx.restore();

  updatePlaybackUi();
}

function prepareCanvas() {
  const canvas = els.canvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const pixelW = Math.round(width * dpr);
  const pixelH = Math.round(height * dpr);
  if (canvas.width !== pixelW || canvas.height !== pixelH) {
    canvas.width = pixelW;
    canvas.height = pixelH;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawBackdrop(ctx, width, height) {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#15191f");
  gradient.addColorStop(0.52, "#101419");
  gradient.addColorStop(1, "#111821");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
}

function applyCamera(ctx) {
  ctx.translate(state.camera.x, state.camera.y);
  ctx.scale(state.camera.scale, state.camera.scale);
}

function drawWorld(ctx) {
  drawEdges(ctx);
  drawRooms(ctx);
}

function drawEdges(ctx) {
  ctx.save();
  ctx.lineCap = "round";
  for (const [src, dst] of EDGES) {
    const a = NODE_POSITIONS[src];
    const b = NODE_POSITIONS[dst];
    const tunnel = isTunnelEdge(src, dst);

    ctx.strokeStyle = COLORS.roadDark;
    ctx.lineWidth = tunnel ? 22 : 15;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();

    ctx.strokeStyle = tunnel ? "#26313d" : COLORS.road;
    ctx.lineWidth = tunnel ? 13 : 8;
    ctx.setLineDash(tunnel ? [12, 7] : []);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawRooms(ctx) {
  const tick = currentTick();
  for (const nodeId of NODE_IDS) {
    const node = tick?.nodes?.[nodeId] || {};
    const [x, y] = NODE_POSITIONS[nodeId];
    const owner = node.owner || "NEUTRAL";
    const room = roomSize(nodeId);

    ctx.save();
    ctx.fillStyle = ownerFill(owner, nodeId);
    ctx.strokeStyle = ownerStroke(owner, nodeId);
    ctx.lineWidth = selectedMatches("node", nodeId) ? 3.5 : 2;
    roundedRectPath(ctx, x - room.w / 2, y - room.h / 2, room.w, room.h, room.r);
    ctx.fill();
    ctx.stroke();

    if (CORE_NODES.has(nodeId)) {
      drawHpBar(ctx, x - room.w / 2 + 8, y + room.h / 2 - 12, room.w - 16, nodeCoreHp(nodeId, node, tick), CORE_HP);
    }

    if (nodeId === "Baron" && Number(tick?.baron?.hp || 0) > 0) {
      drawHpBar(ctx, x - room.w / 2 + 8, y + room.h / 2 - 12, room.w - 16, tick.baron.hp, LORD_HP, COLORS.neutral);
    }

    ctx.fillStyle = COLORS.text;
    ctx.font = `${nodeId === "Baron" ? 15 : 13}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(displayNode(nodeId), x, y - 1);
    ctx.fillStyle = COLORS.muted;
    ctx.font = "8px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(owner === "NEUTRAL" ? "N" : owner, x, y + room.h * 0.28);
    ctx.restore();
  }
}

function desiredUnits(tick, nextTick, alpha) {
  const desired = [];
  const battleNodeIds = new Set((tick.active_battles || [])
    .filter((battle) => battle.location_kind === "node")
    .map((battle) => battle.location_id));

  for (const nodeId of NODE_IDS) {
    if (!battleNodeIds.has(nodeId)) {
      desired.push(...stationedDesired(nodeId, tick));
    }
  }
  desired.push(...movingDesired(tick, nextTick, alpha));
  desired.push(...battleDesired(tick, nextTick, alpha));
  return desired;
}

function stationedDesired(nodeId, tick) {
  const node = tick.nodes?.[nodeId] || {};
  const forces = [];
  if (unitTotal(node.units) > 0) {
    forces.push({ owner: node.owner || "NEUTRAL", units: node.units, role: "hold" });
  }
  for (const [owner, units] of Object.entries(node.visiting_units || {})) {
    if (unitTotal(units) > 0) {
      forces.push({ owner, units, role: "visit" });
    }
  }
  if (Number(node.neutral_hp || 0) > 0) {
    forces.push({ owner: "NEUTRAL", units: { NEUTRAL: node.neutral_hp }, role: "neutral" });
  }
  if (nodeId === "Baron" && Number(tick.baron?.hp || 0) > 0) {
    forces.push({ owner: "LORD", units: { LORD: 1 }, role: "lord" });
  }

  const result = [];
  forces.forEach((force, forceIndex) => {
    const anchor = nodeAnchor(nodeId, forceIndex, forces.length);
    result.push(...expandDesiredUnits(
      force.units,
      force.owner,
      `node:${nodeId}:${force.owner}:${force.role}`,
      anchor,
      {
        spread: CORE_NODES.has(nodeId) ? 20 : 23,
        layer: "stationed",
        nodeId,
      },
    ));
  });
  return result;
}

function movingDesired(tick, nextTick, alpha) {
  const result = [];
  const nextGroups = [...(nextTick?.moving_groups || [])];
  const used = new Set();
  const edgeGroups = (tick.moving_groups || []).filter((group) => isEdgeMovement(group));
  const edgeTotals = edgeGroupTotals(edgeGroups);
  const edgeIndexes = new Map();

  for (const group of tick.moving_groups || []) {
    const nextGroup = findNextGroup(group, nextGroups, used);
    const nextProgress = sameMovementSegment(group, nextGroup) ? nextGroup.progress : 1;
    const progress = lerp(Number(group.progress || 0), Number(nextProgress ?? 0), alpha);

    let anchor = null;
    let axis = [1, 0];
    if (isEdgeMovement(group)) {
      const key = edgeKey(group.src, group.dst);
      const laneIndex = edgeIndexes.get(key) || 0;
      edgeIndexes.set(key, laneIndex + 1);
      const laneTotal = edgeTotals.get(key) || 1;
      anchor = edgePoint(group.src, group.dst, progress, laneIndex, laneTotal);
      axis = edgeAxis(group.src, group.dst);
    } else {
      const nodeId = movementNodeId(group);
      anchor = nodePhasePoint(nodeId, group.src, group.dst, group.phase, progress);
      axis = nodeId && group.dst ? edgeAxis(nodeId, group.dst) : [1, 0];
    }

    if (!anchor) {
      continue;
    }
    result.push(...expandDesiredUnits(
      group.units,
      group.player,
      `move:${movementSignature(group)}`,
      anchor,
      {
        axis,
        spread: hasLordUnit(group.units) ? 24 : 18,
        layer: "moving",
        route: routeText(group),
        intent: group.intent,
      },
    ));
  }
  return result;
}

function battleDesired(tick, nextTick, alpha) {
  const result = [];
  for (const battle of tick.active_battles || []) {
    const nextBattle = (nextTick?.active_battles || []).find((item) => item.battle_id === battle.battle_id);
    const units = interpolateFormation(battle.formation || [], nextBattle?.formation || [], alpha);
    const base = battlePosition(battle);
    const axis = battleAxis(battle);
    const perp = [-axis[1], axis[0]];
    const span = battle.location_kind === "edge" ? 72 : 82;
    const laneCounts = new Map();

    units.forEach((unit, index) => {
      const key = `${unit.team}:${unit.type}:${Math.round(unit.pos)}`;
      const count = laneCounts.get(key) || 0;
      laneCounts.set(key, count + 1);
      const pos = Math.max(-1, Math.min(1, (Number(unit.pos || 10) - 10) / 10));
      const lane = battleLane(unit, count);
      const jitter = deterministicJitter(`${battle.battle_id}:${unit.uid ?? index}`, 2.2);
      result.push({
        id: `battle:${battle.battle_id}:${unit.uid ?? `${unit.team}:${unit.type}:${index}`}`,
        x: base[0] + axis[0] * pos * span + perp[0] * lane + jitter[0],
        y: base[1] + axis[1] * pos * span + perp[1] * lane + jitter[1],
        team: unit.team,
        type: unit.type,
        hpRatio: unit.maxHp > 0 ? unit.hp / unit.maxHp : 1,
        layer: "battle",
        battleId: battle.battle_id,
        uid: unit.uid,
      });
    });
  }
  return result;
}

function expandDesiredUnits(units, owner, baseId, anchor, options = {}) {
  const expanded = expandUnitEntries(units, MAX_EXPANDED_UNITS);
  const result = [];
  const axis = options.axis || [1, 0];
  const perp = [-axis[1], axis[0]];
  const spread = Number(options.spread || 20);
  const count = expanded.length;

  expanded.forEach((unit, index) => {
    const offset = clusterOffset(index, count, spread);
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
  });
  return result;
}

function syncVisuals(desired, dt) {
  const previous = state.visuals;
  const usedPrevious = new Set();
  const nextVisuals = new Map();
  const ease = state.snapVisuals ? 1 : 1 - Math.exp(-Math.max(0.001, dt) * 18);

  for (const target of desired) {
    let visual = previous.get(target.id);
    if (visual) {
      usedPrevious.add(target.id);
    } else {
      const match = findHandoffVisual(previous, usedPrevious, target);
      if (match) {
        visual = { ...match.visual, id: target.id };
        usedPrevious.add(match.id);
      } else {
        visual = { ...target, alpha: 0, x: target.x, y: target.y };
      }
    }

    visual.targetX = target.x;
    visual.targetY = target.y;
    visual.team = target.team;
    visual.type = target.type;
    visual.hpRatio = target.hpRatio;
    visual.layer = target.layer;
    visual.nodeId = target.nodeId;
    visual.battleId = target.battleId;
    visual.route = target.route;
    visual.intent = target.intent;
    visual.x = lerp(visual.x, target.x, ease);
    visual.y = lerp(visual.y, target.y, ease);
    visual.alpha = Math.min(1, visual.alpha + dt * 8);
    visual.missing = false;
    nextVisuals.set(target.id, visual);
  }

  for (const [id, visual] of previous.entries()) {
    if (usedPrevious.has(id) || nextVisuals.has(id)) {
      continue;
    }
    const faded = { ...visual, alpha: visual.alpha - dt * 5, missing: true };
    if (faded.alpha > 0.04) {
      nextVisuals.set(id, faded);
    }
  }

  state.visuals = nextVisuals;
  state.snapVisuals = false;
}

function findHandoffVisual(previous, usedPrevious, target) {
  let best = null;
  let bestDist = HANDOFF_DISTANCE;
  for (const [id, visual] of previous.entries()) {
    if (usedPrevious.has(id) || visual.type !== target.type || teamBucket(visual.team) !== teamBucket(target.team)) {
      continue;
    }
    const dist = Math.hypot(visual.x - target.x, visual.y - target.y);
    if (dist < bestDist) {
      best = { id, visual };
      bestDist = dist;
    }
  }
  return best;
}

function drawBattleEffects(ctx, tick) {
  for (const battle of tick.active_battles || []) {
    const pos = battlePosition(battle);
    if (!pos) {
      continue;
    }
    ctx.save();
    ctx.strokeStyle = battle.location_id === "Baron" ? COLORS.neutral : "#f0d18a";
    ctx.globalAlpha = selectedMatches("battle", battle.battle_id) ? 0.8 : 0.42;
    ctx.lineWidth = selectedMatches("battle", battle.battle_id) ? 3 : 1.6;
    ctx.setLineDash([7, 6]);
    ctx.beginPath();
    ctx.ellipse(pos[0], pos[1], battle.location_kind === "edge" ? 48 : 58, 26, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    drawAttackLines(ctx, battle, tick);
  }
}

function drawAttackLines(ctx, battle, tick) {
  const event = (tick.events || []).find((item) => item.type === "BATTLE_TICK" && item.battle_id === battle.battle_id);
  const attacks = event?.summary?.attacks || [];
  if (!attacks.length) {
    return;
  }

  const byUid = new Map();
  for (const visual of state.visuals.values()) {
    if (visual.battleId === battle.battle_id && visual.id.startsWith("battle:")) {
      const uid = Number(visual.id.split(":").at(-1));
      if (Number.isFinite(uid)) {
        byUid.set(uid, visual);
      }
    }
  }

  ctx.save();
  ctx.lineWidth = 1.2;
  ctx.globalAlpha = 0.72;
  for (const attack of attacks.slice(0, 48)) {
    const from = byUid.get(attack[0]);
    const to = byUid.get(attack[1]);
    if (!from || !to) {
      continue;
    }
    ctx.strokeStyle = teamColor(from.team);
    ctx.setLineDash(displayUnitCode(attack[4]) === "R" ? [4, 4] : []);
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawVisualUnits(ctx) {
  const visuals = Array.from(state.visuals.values())
    .sort((left, right) => layerOrder(left.layer) - layerOrder(right.layer) || left.y - right.y);
  for (const visual of visuals) {
    drawUnit(ctx, visual);
  }
}

function drawUnit(ctx, unit) {
  const selected = selectedMatches("unit", unit.id);
  const size = unitSize(unit);
  const color = teamColor(unit.team);

  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(1, unit.alpha));
  ctx.translate(unit.x, unit.y);
  ctx.lineWidth = selected ? 2.4 : 1.2;
  ctx.strokeStyle = selected ? COLORS.text : color;
  ctx.fillStyle = unitFill(unit.team);

  if (selected) {
    ctx.globalAlpha *= 0.95;
    ctx.beginPath();
    ctx.arc(0, 0, size + 7, 0, Math.PI * 2);
    ctx.stroke();
  }

  drawUnitShape(ctx, unit.type, size);

  if (unit.hpRatio < 0.98) {
    ctx.fillStyle = unit.hpRatio < 0.35 ? COLORS.danger : COLORS.green;
    ctx.fillRect(-size, size + 3, Math.max(1, size * 2 * unit.hpRatio), 2);
  }
  ctx.restore();
}

function drawUnitShape(ctx, type, size) {
  const shape = UNIT_SHAPE[type] || "circle";
  if (shape === "triangle") {
    ctx.beginPath();
    ctx.moveTo(0, -size * 1.25);
    ctx.lineTo(size * 1.1, size * 0.9);
    ctx.lineTo(-size * 1.1, size * 0.9);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    return;
  }
  if (shape === "diamond") {
    ctx.beginPath();
    ctx.moveTo(0, -size * 1.25);
    ctx.lineTo(size * 1.25, 0);
    ctx.lineTo(0, size * 1.25);
    ctx.lineTo(-size * 1.25, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    return;
  }
  if (shape === "core") {
    roundedRectPath(ctx, -size * 1.5, -size * 1.3, size * 3, size * 2.6, size * 0.35);
    ctx.fill();
    ctx.stroke();
    return;
  }
  if (shape === "lord") {
    drawStar(ctx, 0, 0, size * 0.75, size * 1.55, 6);
    ctx.fill();
    ctx.stroke();
    return;
  }

  ctx.beginPath();
  ctx.arc(0, 0, size, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawSelection(ctx, tick) {
  const selected = state.selected;
  if (!selected || selected.type !== "node") {
    return;
  }
  const point = NODE_POSITIONS[selected.id];
  if (!point) {
    return;
  }
  const room = roomSize(selected.id);
  ctx.save();
  ctx.strokeStyle = COLORS.text;
  ctx.globalAlpha = 0.8;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  roundedRectPath(ctx, point[0] - room.w / 2 - 5, point[1] - room.h / 2 - 5, room.w + 10, room.h + 10, room.r + 4);
  ctx.stroke();
  ctx.restore();
}

function updatePlaybackUi() {
  if (!state.replay) {
    els.tickNow.textContent = "0";
    els.tickMax.textContent = "0";
    els.tickSlider.max = "0";
    els.tickSlider.value = "0";
    return;
  }
  const tick = Math.floor(state.tickFloat);
  const max = state.replay.ticks.length - 1;
  els.tickNow.textContent = String(tick);
  els.tickMax.textContent = String(max);
  els.tickSlider.max = String(max);
  if (document.activeElement !== els.tickSlider) {
    els.tickSlider.value = String(tick);
  }
}

function updateInspector() {
  const selected = state.selected;
  const tick = currentTick();
  if (!selected || !tick) {
    els.inspectBody.innerHTML = "Click a room, squad, or fight.";
    return;
  }

  if (selected.type === "node") {
    const node = tick.nodes?.[selected.id] || {};
    els.inspectBody.innerHTML = `
      <div class="inspect-title"><strong>${displayNode(selected.id)}</strong><span>${node.owner || "NEUTRAL"}</span></div>
      ${inspectRow("units", unitText(node.units))}
      ${inspectRow("visitors", visitorsText(node.visiting_units))}
      ${CORE_NODES.has(selected.id) ? inspectRow("core", `${formatNumber(nodeCoreHp(selected.id, node, tick))}/${CORE_HP}`) : ""}
      ${Number(node.neutral_hp || 0) > 0 ? inspectRow("neutral", formatNumber(node.neutral_hp)) : ""}
      ${selected.id === "Baron" ? inspectRow("lord", `${formatNumber(tick.baron?.hp || 0)}/${LORD_HP}`) : ""}
    `;
    return;
  }

  if (selected.type === "battle") {
    const battle = (tick.active_battles || []).find((item) => item.battle_id === selected.id);
    if (!battle) {
      els.inspectBody.innerHTML = "Fight resolved.";
      return;
    }
    els.inspectBody.innerHTML = `
      <div class="inspect-title"><strong>${displayLocation(battle.location_id)}</strong><span>${battle.age_ticks}t</span></div>
      ${inspectRow("kind", battle.battle_type)}
      ${inspectRow("sides", Object.keys(battle.sides || {}).join(", ") || "none")}
      <div class="chips">${Object.entries(battle.sides || {}).map(([side, info]) => `<span class="chip">${side}: ${unitText(info.units)} ${formatNumber(info.hp || 0)}hp</span>`).join("")}</div>
    `;
    return;
  }

  if (selected.type === "unit") {
    const unit = state.visuals.get(selected.id);
    if (!unit) {
      els.inspectBody.innerHTML = "Unit no longer visible.";
      return;
    }
    els.inspectBody.innerHTML = `
      <div class="inspect-title"><strong>${displayForceLabel(unit.team)}</strong><span>${displayUnitCode(unit.type)}</span></div>
      ${inspectRow("layer", unit.layer)}
      ${unit.nodeId ? inspectRow("room", displayNode(unit.nodeId)) : ""}
      ${unit.route ? inspectRow("route", unit.route) : ""}
      ${unit.intent ? inspectRow("intent", unit.intent) : ""}
    `;
  }
}

function inspectRow(label, value) {
  return `<div class="inspect-row"><span>${label}</span><span>${value || "none"}</span></div>`;
}

function startDrag(event) {
  state.dragging = true;
  state.dragStart = [event.clientX, event.clientY];
  state.cameraStart = { ...state.camera };
  els.canvas.classList.add("dragging");
}

function dragMove(event) {
  if (!state.dragging) {
    return;
  }
  const dx = event.clientX - state.dragStart[0];
  const dy = event.clientY - state.dragStart[1];
  state.camera.x = state.cameraStart.x + dx;
  state.camera.y = state.cameraStart.y + dy;
}

function endDrag() {
  state.dragging = false;
  els.canvas.classList.remove("dragging");
}

function zoomAtPointer(event) {
  event.preventDefault();
  const rect = els.canvas.getBoundingClientRect();
  const sx = event.clientX - rect.left;
  const sy = event.clientY - rect.top;
  const before = screenToWorld(sx, sy);
  const factor = event.deltaY > 0 ? 0.9 : 1.1;
  state.camera.scale = Math.max(0.55, Math.min(4.2, state.camera.scale * factor));
  state.camera.x = sx - before[0] * state.camera.scale;
  state.camera.y = sy - before[1] * state.camera.scale;
}

function selectAtPointer(event) {
  if (state.dragStart && Math.hypot(event.clientX - state.dragStart[0], event.clientY - state.dragStart[1]) > 6) {
    return;
  }
  const rect = els.canvas.getBoundingClientRect();
  const world = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
  state.selected = pickAt(world[0], world[1]);
  updateInspector();
}

function pickAt(x, y) {
  let bestUnit = null;
  let bestDist = 14 / state.camera.scale;
  for (const [id, visual] of state.visuals.entries()) {
    const dist = Math.hypot(visual.x - x, visual.y - y);
    if (dist < bestDist) {
      bestDist = dist;
      bestUnit = { type: "unit", id };
    }
  }
  if (bestUnit) {
    return bestUnit;
  }

  const tick = currentTick();
  for (const battle of tick?.active_battles || []) {
    const pos = battlePosition(battle);
    if (pos && Math.hypot(pos[0] - x, pos[1] - y) < 44) {
      return { type: "battle", id: battle.battle_id };
    }
  }

  for (const nodeId of NODE_IDS) {
    const [nx, ny] = NODE_POSITIONS[nodeId];
    const room = roomSize(nodeId);
    if (Math.abs(x - nx) <= room.w / 2 && Math.abs(y - ny) <= room.h / 2) {
      return { type: "node", id: nodeId };
    }
  }
  return null;
}

function fitCamera(resetPan = true) {
  const rect = els.canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width || window.innerWidth);
  const height = Math.max(1, rect.height || window.innerHeight);
  const scale = Math.min((width - 140) / WORLD_W, (height - 150) / WORLD_H);
  if (resetPan || !state.camera.scale) {
    state.camera.scale = Math.max(0.55, scale);
    state.camera.x = (width - WORLD_W * state.camera.scale) / 2;
    state.camera.y = (height - WORLD_H * state.camera.scale) / 2 + 12;
  }
}

function screenToWorld(x, y) {
  return [
    (x - state.camera.x) / state.camera.scale,
    (y - state.camera.y) / state.camera.scale,
  ];
}

function currentTick() {
  return state.replay?.ticks[Math.floor(state.tickFloat)] || null;
}

function nextReplayTick() {
  if (!state.replay) {
    return null;
  }
  return state.replay.ticks[Math.min(Math.floor(state.tickFloat) + 1, state.replay.ticks.length - 1)];
}

function tickAlpha() {
  return Math.max(0, Math.min(1, state.tickFloat - Math.floor(state.tickFloat)));
}

function roomSize(nodeId) {
  if (nodeId === "Baron") {
    return { w: 68, h: 54, r: 9 };
  }
  if (CORE_NODES.has(nodeId)) {
    return { w: 62, h: 50, r: 7 };
  }
  return { w: 56, h: 46, r: 8 };
}

function ownerFill(owner, nodeId) {
  if (String(owner).startsWith("A")) {
    return COLORS.teamABg;
  }
  if (String(owner).startsWith("B")) {
    return COLORS.teamBBg;
  }
  return nodeId === "Baron" ? COLORS.neutralBg : COLORS.room;
}

function ownerStroke(owner, nodeId) {
  if (String(owner).startsWith("A")) {
    return COLORS.teamA;
  }
  if (String(owner).startsWith("B")) {
    return COLORS.teamB;
  }
  return nodeId === "Baron" ? COLORS.neutral : COLORS.roomLine;
}

function drawHpBar(ctx, x, y, width, hp, maxHp, color = COLORS.green) {
  const pct = Math.max(0, Math.min(1, Number(hp || 0) / Math.max(1, Number(maxHp || 1))));
  ctx.fillStyle = "#07090c";
  roundedRectPath(ctx, x, y, width, 5, 3);
  ctx.fill();
  ctx.fillStyle = pct < 0.35 ? COLORS.danger : color;
  roundedRectPath(ctx, x, y, Math.max(2, width * pct), 5, 3);
  ctx.fill();
}

function nodeAnchor(nodeId, index, total) {
  const [x, y] = NODE_POSITIONS[nodeId];
  if (total <= 1) {
    return [x, y + 22];
  }
  const angle = -Math.PI / 2 + (Math.PI * 2 * index) / total;
  const radius = CORE_NODES.has(nodeId) ? 34 : 38;
  return [x + Math.cos(angle) * radius, y + Math.sin(angle) * radius];
}

function clusterOffset(index, count, spread) {
  const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
  const rows = Math.ceil(count / columns);
  const col = index % columns;
  const row = Math.floor(index / columns);
  return [
    (col - (columns - 1) / 2) * Math.min(8, spread / Math.max(1, columns - 1 || 1)),
    (row - (rows - 1) / 2) * Math.min(8, spread / Math.max(1, rows - 1 || 1)),
  ];
}

function expandUnitEntries(units = {}, limit = MAX_EXPANDED_UNITS) {
  const result = [];
  for (const [type, count] of orderedUnitEntries(units)) {
    const code = unitCode(type);
    const total = Math.max(0, Math.round(Number(count) || 0));
    for (let index = 0; index < total && result.length < limit; index += 1) {
      result.push({ type: code, index });
    }
  }
  return result;
}

function edgePoint(src, dst, progress, laneIndex = 0, laneTotal = 1) {
  const a = NODE_POSITIONS[src];
  const b = NODE_POSITIONS[dst];
  if (!a || !b) {
    return null;
  }
  const t = Math.max(0, Math.min(1, Number(progress) || 0));
  const axis = edgeAxis(src, dst);
  const perp = [-axis[1], axis[0]];
  const offset = (laneIndex - (laneTotal - 1) / 2) * 10;
  return [
    lerp(a[0], b[0], t) + perp[0] * offset,
    lerp(a[1], b[1], t) + perp[1] * offset,
  ];
}

function edgeAxis(src, dst) {
  const a = NODE_POSITIONS[src];
  const b = NODE_POSITIONS[dst];
  if (!a || !b) {
    return [1, 0];
  }
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.max(1, Math.hypot(dx, dy));
  return [dx / len, dy / len];
}

function nodePhasePoint(nodeId, src, dst, phase, progress) {
  const node = NODE_POSITIONS[nodeId];
  if (!node) {
    return null;
  }
  const t = Math.max(0, Math.min(1, Number(progress) || 0));
  const offset = 30;
  const from = directionFromNode(nodeId, src);
  const to = directionFromNode(nodeId, dst);

  if (phase === "node_exit") {
    return [node[0] + to[0] * offset * t, node[1] + to[1] * offset * t];
  }
  if (phase === "node_enter") {
    return [node[0] + from[0] * offset * (1 - t), node[1] + from[1] * offset * (1 - t)];
  }
  return [
    node[0] + lerp(from[0] * offset, to[0] * offset, t),
    node[1] + lerp(from[1] * offset, to[1] * offset, t),
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
  const len = Math.max(1, Math.hypot(dx, dy));
  return [dx / len, dy / len];
}

function battlePosition(battle) {
  if (battle.location_kind === "node") {
    return NODE_POSITIONS[battle.location_id] || null;
  }
  const [src, dst] = String(battle.location_id).split("|");
  return edgePoint(src, dst, Number(battle.edge_position ?? 0.5));
}

function battleAxis(battle) {
  if (battle.location_kind === "edge") {
    const [src, dst] = String(battle.location_id).split("|");
    return edgeAxis(src, dst);
  }
  return [1, 0];
}

function battleLane(unit, laneIndex = 0) {
  if (unit.team === "NEUTRAL" || unit.team === "BARON" || unit.team === "LORD") {
    return ((laneIndex % 5) - 2) * 4;
  }
  const base = { R: -15, S: 0, V: 14, C: 20, L: 0 }[unit.type] ?? 0;
  return (String(unit.team).startsWith("A") ? -base : base) + ((laneIndex % 3) - 1) * 3;
}

function interpolateFormation(formation = [], nextFormation = [], alpha = 0) {
  const current = normalizeFormation(formation);
  const next = normalizeFormation(nextFormation);
  const byUid = new Map(next.filter((unit) => unit.uid !== null).map((unit) => [unit.uid, unit]));
  const buckets = formationBuckets(next);
  return current.map((unit) => {
    const bucket = buckets.get(`${unit.team}|${unit.type}`) || [];
    const nextUnit = unit.uid !== null ? byUid.get(unit.uid) : bucket.shift();
    return {
      ...unit,
      hp: lerp(unit.hp, nextUnit?.hp ?? unit.hp, alpha),
      pos: lerp(unit.pos, nextUnit?.pos ?? unit.pos, alpha),
    };
  });
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
      fallbackIndex: index,
    };
  });
}

function formationBuckets(formation = []) {
  const buckets = new Map();
  for (const unit of [...formation].sort((left, right) => left.pos - right.pos)) {
    const key = `${unit.team}|${unit.type}`;
    if (!buckets.has(key)) {
      buckets.set(key, []);
    }
    buckets.get(key).push(unit);
  }
  return buckets;
}

function findNextGroup(group, nextGroups, used) {
  const signature = movementSignature(group);
  const progress = Number(group.progress || 0);
  let bestIndex = -1;
  let bestScore = Number.POSITIVE_INFINITY;
  for (let index = 0; index < nextGroups.length; index += 1) {
    if (used.has(index)) {
      continue;
    }
    const candidate = nextGroups[index];
    if (movementSignature(candidate) !== signature) {
      continue;
    }
    const candidateProgress = Number(candidate.progress || 0);
    if (candidateProgress + 0.001 < progress) {
      continue;
    }
    const score = candidateProgress - progress;
    if (score < bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }
  if (bestIndex >= 0) {
    used.add(bestIndex);
    return nextGroups[bestIndex];
  }
  return null;
}

function sameMovementSegment(group, candidate) {
  return Boolean(candidate) &&
    group.phase === candidate.phase &&
    Number(group.segment_index ?? 0) === Number(candidate.segment_index ?? 0) &&
    group.src === candidate.src &&
    group.dst === candidate.dst;
}

function movementSignature(group = {}) {
  return [
    group.player || "",
    group.intent || "",
    group.phase || "edge",
    Number(group.segment_index ?? 0),
    group.src || "",
    group.dst || "",
    group.final_dst || "",
    group.started ?? "",
    (group.path || []).join(">"),
    unitSignature(group.units),
  ].join("|");
}

function edgeGroupTotals(groups = []) {
  const totals = new Map();
  for (const group of groups) {
    const key = edgeKey(group.src, group.dst);
    totals.set(key, (totals.get(key) || 0) + 1);
  }
  return totals;
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

function nodeCoreHp(nodeId, node = {}, tick = {}) {
  const nodeHp = Number(node.core_hp);
  const playerHp = Number(tick.players?.[nodeId]?.core_hp);
  return Number.isFinite(nodeHp) ? nodeHp : (Number.isFinite(playerHp) ? playerHp : 0);
}

function hasLordUnit(units = {}) {
  return Number(units.LORD || units.BARON || units.L || units.B || 0) > 0;
}

function unitTotal(units = {}) {
  return Object.values(units || {}).reduce((sum, value) => sum + Number(value || 0), 0);
}

function orderedUnitEntries(units = {}) {
  const order = { VANGUARD: 0, SKIRMISHER: 1, RANGER: 2, ARCHER: 2, CORE: 3, NEUTRAL: 4, BARON: 5, LORD: 5 };
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
  }[type] || displayUnitCode(type);
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
  }[type] || String(type || "?").slice(0, 1);
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

function displayNode(id) {
  return id === "Baron" ? "Lord" : (id.startsWith("CP") ? id.slice(2) : id);
}

function displayLocation(id) {
  return String(id || "").replace("Baron", "Lord");
}

function unitText(units = {}) {
  const parts = orderedUnitEntries(units).map(([type, count]) => `${unitCode(type)}${count}`);
  return parts.length ? parts.join(" ") : "empty";
}

function visitorsText(visitors = {}) {
  const parts = Object.entries(visitors || {})
    .filter(([, units]) => unitTotal(units) > 0)
    .map(([owner, units]) => `${owner}: ${unitText(units)}`);
  return parts.length ? parts.join(" | ") : "none";
}

function routeText(group) {
  return (group.path?.length ? group.path : [group.src, group.dst]).map(displayLocation).join(" -> ");
}

function unitSignature(units = {}) {
  return Object.entries(units || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([type, count]) => `${type}:${count}`)
    .join("|");
}

function edgeKey(src, dst) {
  return [src, dst].sort().join("|");
}

function isTunnelEdge(src, dst) {
  return TUNNEL_EDGES.has(edgeKey(src, dst));
}

function layerOrder(layer) {
  return { stationed: 0, moving: 1, battle: 2 }[layer] ?? 1;
}

function teamBucket(team) {
  if (String(team).startsWith("A")) {
    return "A";
  }
  if (String(team).startsWith("B")) {
    return "B";
  }
  return "N";
}

function teamColor(team) {
  if (String(team).startsWith("A")) {
    return COLORS.teamA;
  }
  if (String(team).startsWith("B")) {
    return COLORS.teamB;
  }
  return COLORS.neutral;
}

function unitFill(team) {
  if (String(team).startsWith("A")) {
    return "#1a0d10";
  }
  if (String(team).startsWith("B")) {
    return "#0b1421";
  }
  return "#17130b";
}

function unitSize(unit) {
  if (unit.type === "L" || unit.type === "B") {
    return 8.5;
  }
  if (unit.type === "C") {
    return 7.5;
  }
  if (unit.layer === "battle") {
    return 4.8;
  }
  return 4.4;
}

function typeLane(type) {
  return { R: -7, S: 0, V: 6, C: 0, L: 0, N: 0 }[type] ?? 0;
}

function selectedMatches(type, id) {
  return state.selected?.type === type && state.selected?.id === id;
}

function lerp(a, b, t) {
  return Number(a || 0) + (Number(b || 0) - Number(a || 0)) * Math.max(0, Math.min(1, Number(t) || 0));
}

function deterministicJitter(key, amount = 1) {
  let hash = 2166136261;
  const text = String(key);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return [
    ((hash & 0xff) / 255 - 0.5) * amount * 2,
    (((hash >> 8) & 0xff) / 255 - 0.5) * amount * 2,
  ];
}

function roundedRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
}

function drawStar(ctx, x, y, inner, outer, points) {
  ctx.beginPath();
  for (let index = 0; index < points * 2; index += 1) {
    const radius = index % 2 === 0 ? outer : inner;
    const angle = -Math.PI / 2 + (Math.PI * index) / points;
    const px = x + Math.cos(angle) * radius;
    const py = y + Math.sin(angle) * radius;
    if (index === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.closePath();
}

function formatNumber(value) {
  return Number(value || 0).toFixed(Number(value || 0) % 1 === 0 ? 0 : 1);
}

setup();
