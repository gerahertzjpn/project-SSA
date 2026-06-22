"use strict";
/* World PREVIEW — a fully hardcoded, deterministic clone of the real World.
 * It defines window.__PREVIEW__(url, opts): world.js then runs UNCHANGED, hitting this fake API
 * instead of the server. Zero matches / requests / training — every page + stage is 1:1, with all
 * events randomly generated "as if a season was really simulated". Seasons 0/1/2; owner decisions
 * are made + highlighted. */
(function () {
  // ---- deterministic PRNG ----------------------------------------------- //
  function mulberry32(a) {
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  const R = mulberry32(20260621);
  const ri = (a, b) => a + Math.floor(R() * (b - a + 1));
  const pk = (arr) => arr[Math.floor(R() * arr.length)];
  const shuf = (arr) => { arr = arr.slice(); for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(R() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const FACTIONS = ["humans", "infected", "robots"];
  const ARCH = ["skirmisher", "architect", "sentinel", "warden", "berserker"];
  // synthetic card attributes (0-100) for the demo radar — biased by archetype, potential ~ headroom
  const ARCH_BIAS = { skirmisher: { attack: 16, tempo: 12, defense: -8 }, berserker: { attack: 20, tempo: 8, defense: -12, consistency: -6 },
    sentinel: { defense: 18, consistency: 10, attack: -10 }, architect: { economy: 18, consistency: 8, attack: -8 }, warden: { defense: 12, economy: 8 } };
  function sqAttrs(base, arch) {
    const lvl = Math.max(12, Math.min(90, Math.round((base - 900) / 2.4) + 32));
    const bias = ARCH_BIAS[arch] || {}, o = {};
    ["attack", "defense", "economy", "tempo", "consistency"].forEach((k) =>
      (o[k] = Math.max(6, Math.min(99, lvl + ri(-15, 15) + (bias[k] || 0)))));
    o.potential = Math.max(8, Math.min(99, 96 - Math.round((base - 900) / 3.2) + ri(-8, 14)));  // weaker → more upside
    return o;
  }
  const STRAT = ["standard", "core_rush", "all_in", "turtle", "macro", "baron"];
  const OPS = ["skill_up", "meta_adapt", "style_spec", "obedience_tune", "teamplay"];
  // Prize money — 1:1 with src/world/economy.py. Upper = one pool by league rank (10). Lower = PER
  // GROUP pool (6) — groups A and Z are SEPARATE but EQUAL (1st in A earns what 1st in Z earns).
  const PRIZE_U = [1000, 760, 580, 450, 360, 290, 235, 190, 150, 120];
  const PRIZE_L_GROUP = [300, 230, 175, 135, 105, 80];
  const BONUS_U = 500, BONUS_L = 200;
  // readable playbook conditions (mirror views._readable_when) for the team Плейбук modal
  const STRAT_RU = { standard: "Стандарт", core_rush: "Раш ядра", all_in: "Олл-ин",
    turtle: "Черепаха", macro: "Макро-эконом", baron: "Контроль Лорда" };
  const MGR_T = [
    "Сезон {s} — наш. Состав готов, верю в {p}. {rival} сильны, но мы дадим бой.",
    "Делаем ставку на агрессию. {p} в форме, остальные подтянутся.",
    "Не лучшее межсезонье по деньгам, но цель прежняя — титул. Будем грызть.",
    "Скаутили {rival} — у них дыры. {p} их разберёт.",
    "Молодёжь голодная, ветераны злые. Иначе зачем всё это.",
    "Рынок дорогой. Сохранили костяк, {p} — наш лидер. Идём за результатом.",
    "Уважаю {rival}, но боюсь только себя. Сезон {s} покажет, кто чего стоит.",
    "Перестроились. Риск есть, но без риска нет титулов.",
  ];
  // Curated realistic esports-style handles (mirrors src/world/world.py _NICK_POOL); shuffled, drawn
  // unique; once spent, append a number to a base handle so big rosters keep authentic gamer-tags.
  const NICK_POOL = [
    "Reaper", "Venom", "Havoc", "Rogue", "Riot", "Fury", "Vandal", "Savage", "Carnage", "Menace",
    "Wrath", "Ruin", "Slayer", "Ravage", "Outlaw", "Renegade", "Maverick", "Marauder", "Berserk", "Rampage",
    "Specter", "Phantom", "Wraith", "Shadow", "Ghost", "Cipher", "Enigma", "Mirage", "Eclipse", "Oblivion",
    "Nocturne", "Vesper", "Raven", "Onyx", "Obsidian", "Cinder", "Ember", "Smoke", "Dusk", "Nightfall",
    "Twilight", "Umbra", "Void", "Abyss", "Hollow", "Grim", "Mortis", "Shade", "Revenant", "Spectre",
    "Pixel", "Vector", "Glitch", "Static", "Surge", "Pulse", "Volt", "Bolt", "Spark", "Flux",
    "Circuit", "Proxy", "Vertex", "Apex", "Zenith", "Crux", "Nexus", "Matrix", "Cortex", "Synth",
    "Quantum", "Photon", "Neon", "Laser", "Razor", "Edge", "Blade", "Saber", "Talon", "Spike",
    "Hydra", "Kraken", "Drake", "Wyvern", "Cobra", "Viper", "Mamba", "Falcon", "Raptor", "Lynx",
    "Panther", "Jaguar", "Fenrir", "Cerberus", "Titan", "Atlas", "Orion", "Nova", "Comet", "Meteor",
    "Solar", "Lunar", "Frost", "Blaze", "Inferno", "Tempest", "Cyclone", "Quake", "Glacier", "Tundra",
    "Vex", "Nyx", "Sol", "Kade", "Zane", "Rune", "Jinx", "Koda", "Riven", "Ozz",
    "Ryu", "Kai", "Zed", "Neo", "Axl", "Dex", "Rix", "Fyn", "Lux", "Vox",
    "Nox", "Syx", "Zyl", "Kex", "Vyn", "Dro", "Qix", "Hux", "Jet", "Ace",
    "Zer0", "Sn1per", "Gh0st", "Fr0st", "Pyr0", "N0va", "V1per", "Kha0s", "Sl1ck", "Hav0c",
    "T0xic", "P1xel", "Bl1tz", "Sk0pe", "Wr3ck", "Cl0ud", "Fl4re", "Cr1t", "Vyr4l", "Pr0xy",
    "Specs", "Drift", "Pace", "Clutch", "Frag", "Recoil", "Scope", "Flick", "Whiff", "Peek",
    "Strafe", "Bunny", "Dash", "Juke", "Rush", "Camp", "Bait", "Smurf", "Tilt", "Carry",
    "Anchor", "Pivot", "Relay", "Vanguard", "Sentry", "Echo", "Pyre", "Rift",
  ];
  const nickPool = shuf(NICK_POOL);
  let nickI = 0;
  const usedNick = new Set();
  function nick() {
    while (nickI < nickPool.length) { const n = nickPool[nickI++]; if (!usedNick.has(n)) { usedNick.add(n); return n; } }
    let k = 0, n;                                  // pool spent -> base + number, e.g. "Vortex7"
    do { n = nickPool[k % nickPool.length] + (Math.floor(k / nickPool.length) + 2); k++; } while (usedNick.has(n));
    usedNick.add(n); return n;
  }

  const WID = "world-preview";

  // ---- teams + leagues (REAL franchises + emblems from src/regionals/franchises.py) ----- //
  const PF = (window.__PREVIEW_FRANCHISES__ || []).slice(0, 22);
  let teams = PF.map((f) => ({
    team_id: `${WID}-t-${f.franchise_id}`, name: f.name, abbrev: f.abbrev,
    franchise_id: f.franchise_id, color: f.color, accent: f.accent, base: ri(965, 1175),
  }));
  teams.sort((a, b) => b.base - a.base);
  teams.forEach((t, i) => {
    t.league = i < 10 ? "upper" : "lower";
    t.group = i < 10 ? null : ((i - 10) % 2 === 0 ? "A" : "Z");
    t.power = i < 10 ? t.base + 120 : t.base;     // upper seeded above lower (earned promotion)
    t.titles = 0; t.trophies = []; t.eloHist = [Math.round(t.power)]; t.certs = 0;
    t.treasury = t.league === "upper" ? 300 : 200; t.roster = [];
  });
  const byId = Object.fromEntries(teams.map((t) => [t.team_id, t]));
  const tname = (id) => (byId[id] ? byId[id].name : id);
  const tabbr = (id) => (byId[id] ? byId[id].abbrev : id);
  // REAL emblems (emblem_svg output), keyed by the preview team_id — no invented emblems.
  const EMBLEMS = Object.fromEntries(PF.map((f) => [`${WID}-t-${f.franchise_id}`, f.emblem]));
  const COLORS = Object.fromEntries(PF.map((f) => [`${WID}-t-${f.franchise_id}`, { color: f.color, accent: f.accent || f.color }]));

  // ---- players (90: 78 debut, 12 reserve) + Season-0 draft --------------- //
  let players = [];
  for (let i = 0; i < 90; i++) {
    const fac = pk(FACTIONS), ar = pk(ARCH);
    const base = ri(905, 1130);
    const p = { player_id: `${WID}-p${String(i).padStart(3, "0")}`, nickname: nick(), faction: fac,
      archetype: ar, net_id: `${fac}_${ri(1, 6)}__${ar}`, role: "pro", base,
      teamId: null, reserve: false, games: ri(24, 90), gen: 1, posStats: {},
      attributes: sqAttrs(base, ar), ratingHist: [], soloq: [], trophies: [] };
    ["A1", "A2", "A3"].forEach((pos) => { const g = ri(3, 22), w = ri(0, g); p.posStats[pos] = { games: g, wins: w }; });
    players.push(p);
  }
  const pById = Object.fromEntries(players.map((p) => [p.player_id, p]));
  // selection: 78 debut, 12 reserve
  const shuffledIds = shuf(players.map((p) => p.player_id));
  shuffledIds.slice(78).forEach((id) => { pById[id].reserve = true; });
  const enteredIds = shuffledIds.slice(0, 78);
  // bootstrap draft: strongest first, snake order, 3 per club (66 picks)
  const pool = enteredIds.map((id) => pById[id]).sort((a, b) => b.base - a.base);
  let draftPicks = [], di = 0;
  for (let rnd = 1; rnd <= 3; rnd++) {
    const order = rnd % 2 ? teams : teams.slice().reverse();
    for (const t of order) {
      const pl = pool[di++]; if (!pl) break;
      pl.teamId = t.team_id; t.roster.push(pl.player_id);
      draftPicks.push({ round: rnd, overall: draftPicks.length + 1, team_id: t.team_id, player_id: pl.player_id });
    }
  }
  const freeAgents = enteredIds.map((id) => pById[id]).filter((p) => !p.teamId);  // entered, undrafted
  // 18 STREAMERS: ancestor-bound, SoloQ-only free agents, never drafted (role "streamer").
  for (let i = 0; i < 18; i++) {
    const fac = pk(FACTIONS), ar = pk(ARCH);
    const p = { player_id: `${WID}-s${String(i).padStart(2, "0")}`, nickname: nick(), faction: fac,
      archetype: ar, net_id: `${fac}_anc_${i}`, role: "streamer", base: ri(945, 1090), teamId: null,
      reserve: false, games: ri(30, 130), gen: 0, posStats: {}, ratingHist: [], soloq: [], trophies: [] };
    ["A1", "A2", "A3"].forEach((pos) => { const g = ri(3, 20), w = ri(0, g); p.posStats[pos] = { games: g, wins: w }; });
    players.push(p); pById[p.player_id] = p;
  }

  // ---- season simulation helpers ---------------------------------------- //
  const STAGE_LABEL = { ub: "Верхняя сетка", lb: "Нижняя сетка", lb_r1: "Нижняя сетка — 1",
    lb_r2: "Нижняя сетка — 2", lb_final: "Финал нижней сетки", grand_final: "Гранд-финал" };
  function mrow(league, stage, label, home, away, mh, ma, day) {
    return { day, league, stage, stage_label: label, home, away,
      home_abbrev: tabbr(home), away_abbrev: tabbr(away), home_name: tname(home), away_name: tname(away),
      best_of: stage === "league" ? 3 : 5, played: true, maps_home: mh, maps_away: ma,
      winner: mh > ma ? home : away, home_strategy: pk(STRAT), away_strategy: pk(STRAT) };
  }
  function series(home, away, needed, label, stage, league, day, rec) {
    const ph = byId[home].power + ri(-70, 70), pa = byId[away].power + ri(-70, 70);
    const hw = ph >= pa;
    const mh = hw ? needed : ri(0, needed - 1), ma = hw ? ri(0, needed - 1) : needed;
    if (rec) {
      rec[home].sw += hw ? 1 : 0; rec[home].sl += hw ? 0 : 1; rec[home].mw += mh; rec[home].ml += ma;
      rec[away].sw += hw ? 0 : 1; rec[away].sl += hw ? 1 : 0; rec[away].mw += ma; rec[away].ml += mh;
    }
    return { row: mrow(league, stage, label, home, away, mh, ma, day), winner: hw ? home : away };
  }
  const _col = (stage, matches) => ({ stage, label: STAGE_LABEL[stage], matches });
  // Upper playoff (mirrors brackets.upper_playoff): UB 1v2, LB 3v4, LB-final, grand final.
  function upperPlayoff(seeds, dayRef) {
    const ub = series(seeds[0], seeds[1], 3, STAGE_LABEL.ub, "ub", "upper", dayRef.d++);
    const lb = series(seeds[2], seeds[3], 3, STAGE_LABEL.lb, "lb", "upper", dayRef.d++);
    const ubLoser = ub.winner === seeds[0] ? seeds[1] : seeds[0];
    const lbf = series(lb.winner, ubLoser, 3, STAGE_LABEL.lb_final, "lb_final", "upper", dayRef.d++);
    const gf = series(ub.winner, lbf.winner, 3, STAGE_LABEL.grand_final, "grand_final", "upper", dayRef.d++);
    Object.assign(ub.row, { key: "ub", from: [] });            // winner-advance feeders for the lines
    Object.assign(lb.row, { key: "lb", from: [] });
    Object.assign(lbf.row, { key: "lbf", from: ["lb"] });      // ub loser drops here — no line
    Object.assign(gf.row, { key: "gf", from: ["ub", "lbf"] });
    return { cols: [_col("ub", [ub.row]), _col("lb", [lb.row]), _col("lb_final", [lbf.row]),
      _col("grand_final", [gf.row])], champ: gf.winner, finalists: [ub.winner, lbf.winner] };
  }
  // Lower playoff (mirrors brackets.lower_playoff): UB A1vZ1; LB-r1 A2vZ3 & Z2vA3; LB-r2; LB-final;
  // grand final. Both grand finalists promote. Cols ordered as views.PLAYOFF_STAGE_ORDER.
  function lowerPlayoff(a, z, dayRef) {
    const ub = series(a[0], z[0], 3, STAGE_LABEL.ub, "ub", "lower", dayRef.d++);
    const lb1a = series(a[1], z[2], 3, STAGE_LABEL.lb_r1, "lb_r1", "lower", dayRef.d++);
    const lb1b = series(z[1], a[2], 3, STAGE_LABEL.lb_r1, "lb_r1", "lower", dayRef.d++);
    const lb2 = series(lb1a.winner, lb1b.winner, 3, STAGE_LABEL.lb_r2, "lb_r2", "lower", dayRef.d++);
    const ubLoser = ub.winner === a[0] ? z[0] : a[0];
    const lbf = series(lb2.winner, ubLoser, 3, STAGE_LABEL.lb_final, "lb_final", "lower", dayRef.d++);
    const gf = series(ub.winner, lbf.winner, 3, STAGE_LABEL.grand_final, "grand_final", "lower", dayRef.d++);
    Object.assign(ub.row, { key: "ub", from: [] });            // winner-advance feeders for the lines
    Object.assign(lb1a.row, { key: "lb1a", from: [] });
    Object.assign(lb1b.row, { key: "lb1b", from: [] });
    Object.assign(lb2.row, { key: "lb2", from: ["lb1a", "lb1b"] });
    Object.assign(lbf.row, { key: "lbf", from: ["lb2"] });     // ub loser drops here — no line
    Object.assign(gf.row, { key: "gf", from: ["ub", "lbf"] });
    return { cols: [_col("lb_r1", [lb1a.row, lb1b.row]), _col("lb_r2", [lb2.row]), _col("ub", [ub.row]),
      _col("lb_final", [lbf.row]), _col("grand_final", [gf.row])],
      champ: gf.winner, finalists: [ub.winner, lbf.winner] };
  }

  // --- band-based SoloQ (mirrors src/world/soloq.py): STRICT 100-step bands, balanced 3v3, Elo ----
  const SQ_K = 32, SQ_SPREAD = 0.35, SQ_CAP = 100;
  const sqExpected = (a, b) => 1 / (1 + Math.pow(10, (b - a) / 400));
  function sqBalanced(group, R0) {                 // 6 → two 3s with closest average (greedy, desc)
    const ordered = group.slice().sort((x, y) => R0[y] - R0[x]);
    const a = [], b = []; let sa = 0, sb = 0;
    ordered.forEach((pid) => {
      if (a.length < 3 && (sa <= sb || b.length >= 3)) { a.push(pid); sa += R0[pid]; }
      else { b.push(pid); sb += R0[pid]; } });
    return [a, b];
  }
  const sqDelta = (rating, oppAvg, won, impact) =>
    SQ_K * ((won ? 1 : 0) - sqExpected(rating, oppAvg)) * (1 + SQ_SPREAD * (2 * impact - 1));
  function sqRunRounds(sqR, pids, nRounds, roundStart) {
    let rnd = roundStart;
    for (let k = 0; k < nRounds; k++, rnd++) {
      const bands = {};                            // a player can ONLY meet others in their 100-band
      pids.forEach((pid) => { const bd = Math.floor(sqR[pid] / 100) * 100; (bands[bd] = bands[bd] || []).push(pid); });
      Object.keys(bands).forEach((bd) => {
        const members = shuf(bands[bd].slice());   // random lobby composition; tiny bands replay each round
        const full = Math.floor(members.length / 6) * 6;
        for (let i = 0; i < full; i += 6) {
          const chunk = members.slice(i, i + 6);
          const [ta, tb] = sqBalanced(chunk, sqR);
          const avgA = ta.reduce((s, p) => s + sqR[p], 0) / 3, avgB = tb.reduce((s, p) => s + sqR[p], 0) / 3;
          const aWon = R() < sqExpected(avgA, avgB);
          const atTime = {}; chunk.forEach((p) => (atTime[p] = Math.round(sqR[p])));
          const mband = Math.floor(sqR[chunk[0]] / 100) * 100;   // matchmaking band, shared by all 6
          const imp = {}; chunk.forEach((p) => { let im = 0.25 + R() * 0.5;
            if ((aWon && ta.indexOf(p) >= 0) || (!aWon && tb.indexOf(p) >= 0)) im = Math.min(1, im + 0.15); imp[p] = im; });
          const dd = {};
          ta.forEach((p) => (dd[p] = Math.round(sqDelta(sqR[p], avgB, aWon, imp[p]) * 100) / 100));
          tb.forEach((p) => (dd[p] = Math.round(sqDelta(sqR[p], avgA, !aWon, imp[p]) * 100) / 100));
          chunk.forEach((p) => (sqR[p] += dd[p]));
          const rec = (team, opp, won) => team.forEach((p) => {
            const sm = pById[p].soloq;
            sm.push({ season: 0, round: rnd, won, delta: dd[p], ratings: atTime, deltas: dd, band: mband,
              teammates: team.filter((x) => x !== p).map((x) => pById[x].nickname),
              opponents: opp.map((x) => pById[x].nickname),
              teammate_ids: team.filter((x) => x !== p), opponent_ids: opp.slice() });
            if (sm.length > SQ_CAP) pById[p].soloq = sm.slice(-SQ_CAP);
          });
          rec(ta, tb, aWon); rec(tb, ta, !aWon);
        }
      });
    }
    return rnd;
  }
  // bootstrap: 8 preseasons × 10 = 80 strict-band SoloQ rounds. Snapshot the rating after EACH
  // preseason so the «Динамика SoloQ» (last 5 of 8) is already visible at season 0 (#12).
  const BOOT_PRESEASONS = 8;
  const sqRatings = {}; players.forEach((p) => (sqRatings[p.player_id] = p.base));
  const sqPids = players.filter((p) => !p.reserve).map((p) => p.player_id);
  const bootHist = {}; players.forEach((p) => (bootHist[p.player_id] = []));
  let sqRound = 0;
  for (let pre = 0; pre < BOOT_PRESEASONS; pre++) {
    sqRound = sqRunRounds(sqRatings, sqPids, 10, sqRound);
    players.forEach((p) => bootHist[p.player_id].push(Math.round(sqRatings[p.player_id] != null ? sqRatings[p.player_id] : p.base)));
  }
  players.forEach((p) => { p.ratingHist = (bootHist[p.player_id] || []).slice(); });   // 8 points before season 0
  const snaps = [];
  for (let s = 0; s < 3; s++) {
    const upper = teams.filter((t) => t.league === "upper").map((t) => t.team_id);
    const lowA = teams.filter((t) => t.league === "lower" && t.group === "A").map((t) => t.team_id);
    const lowZ = teams.filter((t) => t.league === "lower" && t.group === "Z").map((t) => t.team_id);
    const rec = {}; teams.forEach((t) => { rec[t.team_id] = { sw: 0, sl: 0, mw: 0, ml: 0 }; });
    const fixtures = []; const dayRef = { d: 1 };
    const rr = (ids, lg) => { for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++)
      fixtures.push(series(ids[i], ids[j], 2, "Регулярка", "league", lg, dayRef.d++, rec).row); };
    rr(upper, "upper"); rr(lowA, "lower"); rr(lowZ, "lower");
    fixtures.forEach((m, i) => (m.day = Math.floor(i / 2) + 1));   // regular: 2 series/day (matches the LIVE tab, #1)
    dayRef.d = Math.floor((fixtures.length + 1) / 2) + 1;          // playoffs then continue 1/day after the regular season
    const stand = (ids) => ids.slice().sort((x, y) => (rec[y].sw - rec[x].sw) ||
      ((rec[y].mw - rec[y].ml) - (rec[x].mw - rec[x].ml)) || (byId[y].power - byId[x].power));
    const upStand = stand(upper), aStand = stand(lowA), zStand = stand(lowZ);
    const upPlay = upperPlayoff(upStand, dayRef);
    const loPlay = lowerPlayoff(aStand, zStand, dayRef);
    const upperChamp = upPlay.champ, lowerChamp = loPlay.champ;
    const relegated = [upStand[8], upStand[9]];
    const promoted = [lowerChamp, loPlay.finalists.find((x) => x !== lowerChamp) || loPlay.finalists[0]];

    // ---- prize money: upper by league rank; lower PER GROUP (A and Z separate, equal); + GF bonus ----
    const prizeOf = {};
    upStand.forEach((id, i) => (prizeOf[id] = (PRIZE_U[i] || 120) + (id === upperChamp ? BONUS_U : 0)));
    [aStand, zStand].forEach((arr) => arr.forEach((id, i) =>
      (prizeOf[id] = (PRIZE_L_GROUP[i] || 80) + (id === lowerChamp ? BONUS_L : 0))));
    [...upper, ...lowA, ...lowZ].forEach((id) => {            // career earnings = equal split of the club's purse
      const rp = byId[id].roster, tot = prizeOf[id] || 0;
      if (rp.length && tot) { const share = Math.floor(tot / rp.length);
        rp.forEach((pid) => { pById[pid].earnings = (pById[pid].earnings || 0) + share; }); } });
    const poolRow = (id, i, table, champ, bonus) => ({ team_id: id, name: tname(id), abbrev: tabbr(id),
      place: i + 1, prize: table[i] || (table === PRIZE_U ? 120 : 80), bonus: id === champ ? bonus : 0, total: prizeOf[id] || 0 });
    const prizePool = {
      upper: upStand.map((id, i) => poolRow(id, i, PRIZE_U, upperChamp, BONUS_U)),
      lower_A: aStand.map((id, i) => poolRow(id, i, PRIZE_L_GROUP, lowerChamp, BONUS_L)),
      lower_Z: zStand.map((id, i) => poolRow(id, i, PRIZE_L_GROUP, lowerChamp, BONUS_L)),
    };

    const teamSnap = {};
    [...upper, ...lowA, ...lowZ].forEach((id) => {
      const arr = byId[id].league === "upper" ? upStand : (byId[id].group === "A" ? aStand : zStand);
      teamSnap[id] = { power: byId[id].power, place: arr.indexOf(id) + 1, league: byId[id].league,
        group: byId[id].group, titles: byId[id].titles, treasury: byId[id].treasury, certs: byId[id].certs,
        eloHist: byId[id].eloHist.slice(), trophies: byId[id].trophies.slice(),
        roster: byId[id].roster.slice(), rec: { ...rec[id] } };
    });

    const commentary = [
      { season: s, kind: "season_open", text: `Сезон ${s} начинается! По силе впереди: ${upStand.slice(0, 3).map(tname).join(", ")}. Поехали!` },
      { season: s, kind: "season_close", text: `Сезон ${s} завершён! Чемпион верхней лиги — ${tname(upperChamp)}. Из нижней поднимается ${tname(lowerChamp)}. Повышены: ${promoted.map(tname).join(", ")}; вылетают: ${relegated.map(tname).join(", ")}.` },
    ];
    const chronicle = []; let seq = 1;
    chronicle.push({ seq: seq++, kind: "stage", msg: "SoloQ: сыграно 10 раундов, рейтинги обновлены", source: null });
    const auctionWinners = shuf(teams).slice(0, 6);
    auctionWinners.forEach((t) => chronicle.push({ seq: seq++, kind: "train",
      msg: `Аукцион тренировок: ${t.name} выиграл слот за ${ri(40, 120)}`, source: null }));
    const trainPicks = auctionWinners.map((t) => { const rp = byId[t.team_id].roster;
      return { team: t.team_id, op: pk(OPS), player: rp.length ? pById[rp[0]].nickname : t.name }; });
    trainPicks.forEach((tp) => chronicle.push({ seq: seq++, kind: "train",
      msg: `${byId[tp.team].name}: тренировка ${tp.op} → ${tp.player} (1,000,000 шагов)`, source: "ЛЛМ" }));
    shuf(teams).slice(0, 4 + s).forEach((t) => { const fa = pk(freeAgents);
      chronicle.push({ seq: seq++, kind: "transfer", msg: `Трансфер: ${t.name} подписал ${fa.nickname} (бонус ${ri(30, 90)}, сгорает)`, source: null }); });
    chronicle.push({ seq: seq++, kind: "fallback", msg: `⚠️ фолбэк [playbook:${pk(teams).name}]: ЛЛМ не сработал — решает эвристика`, source: null });
    teams.forEach((t) => chronicle.push({ seq: seq++, kind: "playbook", msg: `${t.name}: плейбук (ЛЛМ), база «${pk(STRAT)}»`, source: "ЛЛМ" }));

    const training = trainPicks.map((tp, i) => ({ job_id: `s${s}j${i}`, world_id: WID, op: tp.op,
      player: tp.player, done: 1000000, total: 1000000, status: "done" }));

    // Manager comments are PRE-season statements shown in the off-season AFTER season s, i.e. they look
    // ahead to season s+1 — stamp/interpolate s+1 (matches the real backend, which reads the already-
    // incremented world season). Was season s -> showed "Сезон 0" after season 1 had begun (#9).
    const upcoming = s + 1;
    const managerComments = teams.map((t, i) => {
      const rp = byId[t.team_id].roster; const star = rp.length ? pById[rp[0]].nickname : "лидер";
      const rivals = teams.filter((x) => x.league === t.league && x.team_id !== t.team_id);
      const rival = rivals.length ? rivals[0].name : "соперники";
      return { season: upcoming, team_id: t.team_id, team: t.name, manager: `Менеджер ${t.abbrev}`,
        text: MGR_T[(i + s) % MGR_T.length].replace(/\{s\}/g, upcoming).replace(/\{p\}/g, star).replace(/\{rival\}/g, rival) };
    });
    // off-season transactions digest (hardcoded demo — every category populated)
    const poolFA = freeAgents.slice();
    const txAuctions = auctionWinners.map((t) => {            // each slot keeps its full ranked bid table
      const price = ri(60, 140);
      const others = shuf(teams.filter((x) => x.team_id !== t.team_id)).slice(0, 9);
      const bids = [{ team_id: t.team_id, team: t.name, amount: price }]
        .concat(others.map((o) => ({ team_id: o.team_id, team: o.name, amount: ri(10, price - 2) })))
        .sort((a, b) => b.amount - a.amount);
      return { team_id: t.team_id, team: t.name, price, steps: 1000000, bids };
    });
    const txTrains = trainPicks.map((tp) => ({ team_id: tp.team, team: byId[tp.team].name,
      op: tp.op, op_title: tp.op, player: tp.player, steps: 1000000 }));
    const txSign = shuf(teams).slice(0, 4 + s).map((t, i) => { const fa = poolFA[i % poolFA.length];
      return { team_id: t.team_id, team: t.name, player_id: fa.player_id, player: fa.nickname,
        bonus: ri(30, 90), kind: i % 4 === 0 ? "rookie" : "free_agent" }; });
    // Club-to-club buys: ACTUALLY move the bench player between rosters so the displayed transfer
    // matches reality next season (the seller no longer has him) — was display-only, so "продан X"
    // showed while X stayed on the roster (#13). Only a BENCH player (roster > 3) is tradeable.
    const txBuys = [];
    shuf(teams).slice(0, 3).forEach((t) => {
      const from = teams[(teams.indexOf(t) + 5) % teams.length];
      const rp = byId[from.team_id].roster;
      if (from.team_id === t.team_id || rp.length <= 3) return;     // keep a full starting trio
      const pid = rp[rp.length - 1], who = pById[pid];
      byId[from.team_id].roster = rp.filter((x) => x !== pid);      // remove from seller
      if (!byId[t.team_id].roster.includes(pid)) byId[t.team_id].roster.push(pid);   // add to buyer
      who.teamId = t.team_id;
      txBuys.push({ player_id: pid, player: who.nickname, from_team_id: from.team_id, from_team: from.name,
        to_team_id: t.team_id, to_team: t.name, price: ri(120, 480) });
    });
    const txRel = shuf(teams).slice(0, 2).map((t) => { const rp = byId[t.team_id].roster;
      const who = rp.length ? pById[rp[rp.length - 1]] : pById[players[0].player_id];
      return { player_id: who.player_id, player: who.nickname, team_id: t.team_id, team: t.name }; });
    const txReq = shuf(teams).slice(0, 8).map((t, i) => { const fa = poolFA[(i + 1) % poolFA.length];
      return { team_id: t.team_id, team: t.name, player_id: fa.player_id, player: fa.nickname,
        amount: ri(40, 200), won: i % 3 === 0 }; });
    const transactions = { auctions: txAuctions, training_requests: txTrains,
      signings: txSign, buys: txBuys, releases: txRel, requests: txReq,
      departures: txBuys.map((b) => ({ player_id: b.player_id, player: b.player,
        from_team: b.from_team, to_team: b.to_team, kind: "transfer" })),
      counts: { auctions: txAuctions.length, training_requests: txTrains.length,
        signings: txSign.length, buys: txBuys.length, releases: txRel.length, requests: txReq.length } };

    // rich off-season digest the inline межсезонье view renders per topic (mirrors offseason.py)
    const STRATS = ["standard", "core_rush", "all_in", "turtle", "macro", "baron"];
    // off-season SoloQ: 10 strict-band rounds — REAL movers, consistent with the ratings tab
    const sqBefore = {}; sqPids.forEach((pid) => (sqBefore[pid] = sqRatings[pid]));
    sqRound = sqRunRounds(sqRatings, sqPids, 10, sqRound);
    const sqMovers = sqPids
      .map((pid) => ({ player_id: pid, player: pById[pid].nickname, faction: pById[pid].faction,
        rating: Math.round(sqRatings[pid]), delta: Math.round((sqRatings[pid] - sqBefore[pid]) * 10) / 10 }))
      .filter((m) => Math.abs(m.delta) >= 0.05)
      .sort((a, b) => b.delta - a.delta);
    const trainReqs = txTrains.map((t) => ({ kind: "team", team_id: t.team_id, op: t.op, title: t.op_title,
      target: {}, steps: t.steps, est_minutes: Math.round(t.steps / 50000), below_floor: false }));
    const trainSteps = trainReqs.reduce((a, r) => a + r.steps, 0);
    // money ledger — FULL per-team flow (mirrors offseason._money_ledger). The period opens at the
    // PRE-prize balance so Δ (net) reflects EVERYTHING: + prize − salary − auction − signing ± transfers.
    // treasury_after = treasury_before + net ALWAYS holds (the owner can finally track prize money).
    const salaryOf = (pid) => Math.max(8, Math.round((pById[pid].base - 860) / 2.2));
    const mnRows = teams.map((t) => {
      const aS = txAuctions.filter((a) => a.team_id === t.team_id).reduce((x, a) => x + a.price, 0);
      const sS = txSign.filter((x) => x.team_id === t.team_id).reduce((x, q) => x + (q.bonus || 0), 0);
      const tOut = txBuys.filter((b) => b.to_team_id === t.team_id).reduce((x, b) => x + b.price, 0);
      const tIn = txBuys.filter((b) => b.from_team_id === t.team_id).reduce((x, b) => x + b.price, 0);
      const prize = prizeOf[t.team_id] || 0;
      const salary = byId[t.team_id].roster.reduce((x, pid) => x + salaryOf(pid), 0);
      const before = Math.round(t.treasury);                               // carried balance BEFORE the prize
      const after = before + prize - salary + tIn - tOut - aS - sS;        // no clamp (matches backend; can dip <0)
      const net = after - before;                                          // = prize − salary − spends ± transfers, EXACTLY
      t.treasury = after;                                                  // carry the COHERENT balance forward
      return { team_id: t.team_id, team: t.name, auction_spend: aS, signing_spend: sS, prize, salary,
        transfer_in: tIn, transfer_out: tOut, treasury_before: before, treasury_after: after, net };
    }).sort((a, b) => b.treasury_after - a.treasury_after);
    const money = { rows: mnRows, totals: { auction_spend: mnRows.reduce((x, r) => x + r.auction_spend, 0),
      transfer: mnRows.reduce((x, r) => x + r.transfer_out, 0), signing_spend: mnRows.reduce((x, r) => x + r.signing_spend, 0),
      prize: mnRows.reduce((x, r) => x + r.prize, 0), salary: mnRows.reduce((x, r) => x + r.salary, 0) } };
    // skill re-evaluations — a few trained players get card-characteristic updates (equal conditions)
    const skills = shuf(players.filter((p) => !p.reserve && p.attributes)).slice(0, 6).map((p) => {
      const before = { ...p.attributes }, changes = {};
      ["attack", "defense", "economy", "tempo", "consistency", "potential"].forEach((k) => {
        if (R() < 0.55) { const d = ri(-3, 7); if (d) { p.attributes[k] = Math.max(5, Math.min(99, p.attributes[k] + d)); changes[k] = p.attributes[k] - before[k]; } } });
      return { player_id: p.player_id, player: p.nickname, faction: p.faction, before, after: { ...p.attributes }, changes };
    }).filter((q) => Object.keys(q.changes).length);
    const offseason = {
      soloq: { gainers: sqMovers.slice(0, 10), losers: sqMovers.slice(-10).reverse(), n_moved: sqMovers.length },
      planning: teams.map((t, i) => ({ team_id: t.team_id, team: t.name, base: STRATS[(i + s) % STRATS.length] })),
      training: { auction: { slots: 6, awarded: txAuctions.length, spent: trainReqs.length, awards: txAuctions },
        preview: { n_requests: trainReqs.length, requests: trainReqs,
          by_kind: { team: { n: trainReqs.length, steps: trainSteps } },
          total_steps: trainSteps, total_minutes: Math.round(trainSteps / 50000) } },
      money, skills,
    };
    snaps.push({ num: s, teamSnap, upStand, aStand, zStand, fixtures, bracket: { upper: upPlay.cols, lower: loPlay.cols },
      champ: { upper_champion: upperChamp, lower_champion: lowerChamp, promoted, relegated },
      prizeOf, prizePool,
      commentary, chronicle, training, managerComments, transactions, offseason,
      draft: s === 0 ? { season: 0, kind: "bootstrap", picks: draftPicks.slice() } : null });

    // roll over
    byId[upperChamp].titles++; byId[upperChamp].trophies.push({ season: s, title: "Чемпион верхней" });
    byId[lowerChamp].trophies.push({ season: s, title: "Чемпион нижней" });
    relegated.forEach((id) => { byId[id].league = "lower"; });
    promoted.forEach((id) => { byId[id].league = "upper"; byId[id].group = null; });
    let gi = 0; teams.filter((t) => t.league === "lower").forEach((t) => { t.group = gi++ % 2 === 0 ? "A" : "Z"; });
    teams.filter((t) => t.league === "upper").forEach((t) => { t.group = null; });
    teams.forEach((t) => { const r = rec[t.team_id]; const d = (r.sw - r.sl) * ri(5, 12) + ri(-20, 20);
      t.power = Math.max(880, t.power + d); t.eloHist.push(Math.round(t.power)); t.certs = 0; });
      // treasury is NOT drifted randomly — the money ledger above already set the coherent balance.
    players.forEach((p) => { p.ratingHist.push(Math.round(sqRatings[p.player_id] != null ? sqRatings[p.player_id] : p.ratingHist[p.ratingHist.length - 1])); });
    if (s === 0) shuf(teams).slice(0, 9).forEach((t) => { const fa = freeAgents.find((p) => !p.teamId);
      if (fa && t.roster.length < 5) { fa.teamId = t.team_id; t.roster.push(fa.player_id); } });
  }
  // (SoloQ per-game history is recorded by sqRunRounds above — band-consistent, real Elo deltas)

  // ---- view builders (match server payload shapes) ---------------------- //
  let CUR = 0;
  const snap = () => snaps[CUR];
  function teamRow(id) {
    const ts = snap().teamSnap[id];
    const r = ts.rec;
    return { team_id: id, name: tname(id), abbrev: tabbr(id), league: ts.league, group: ts.group,
      power: Math.round(ts.power), titles: ts.titles, treasury: Math.round(ts.treasury), place: ts.place,
      points: r.sw * 3, series_wins: r.sw, series_losses: r.sl,
      maps_won: r.mw, maps_lost: r.ml, map_diff: r.mw - r.ml,
      roster: ts.roster.map((pid) => pById[pid].nickname), elo: Math.round(ts.power) };
  }
  function leaguesView() {
    if (LIVE) {       // a day-by-day reveal: standings AS OF the revealed games (0 points at day 0)
      const revset = lvRevSet();
      const st = lvStandRows(revset, lvProvPower(revset));
      const f = (gk, lg, grp) => st[gk].map((r, i) => ({ ...teamRow(r.team_id), place: i + 1,
        points: r.points, series_wins: r.series_wins, series_losses: r.series_losses,
        map_diff: r.map_diff, power: r.power, league: lg, group: grp }));
      return { upper: f("upper", "upper", null), lower_A: f("lower_A", "lower", "A"),
        lower_Z: f("lower_Z", "lower", "Z") };
    }
    const ids = Object.keys(snap().teamSnap);
    const f = (lg, grp) => ids.filter((id) => snap().teamSnap[id].league === lg &&
      (grp === undefined || snap().teamSnap[id].group === grp))
      .sort((a, b) => snap().teamSnap[a].place - snap().teamSnap[b].place)
      .map((id, i) => ({ ...teamRow(id), place: i + 1 }));
    return { upper: f("upper"), lower_A: f("lower", "A"), lower_Z: f("lower", "Z") };
  }
  function lvRevealedRegular() {     // revealed-so-far regular series (for matrix)
    const out = []; if (!LIVE) return out;
    const revset = lvRevSet();
    LIVE.days.forEach((d) => { if (d.phase === "regular") d.series.forEach((s) => { if (revset.has(s.gid)) out.push(s); }); });
    return out;
  }
  function matrixView() {
    const lv = leaguesView();
    const series = LIVE ? lvRevealedRegular() : snap().fixtures.filter((e) => e.stage === "league");
    const grid = (teamsRows, label) => {
      const ids = teamsRows.map((t) => t.team_id);
      const idset = new Set(ids);
      const h2h = {};
      series.forEach((e) => {
        if (!idset.has(e.home) || !idset.has(e.away)) return;
        h2h[`${e.home}|${e.away}`] = { for: e.maps_home, against: e.maps_away, won: e.winner === e.home };
        h2h[`${e.away}|${e.home}`] = { for: e.maps_away, against: e.maps_home, won: e.winner === e.away };
      });
      const cells = ids.map((r) => ids.map((c) => (r === c ? null : (h2h[`${r}|${c}`] || null))));
      const teams = teamsRows.map((t) => ({ team_id: t.team_id, abbrev: t.abbrev, name: t.name,
        place: t.place, points: t.points, map_diff: t.map_diff,
        series_wins: t.series_wins, series_losses: t.series_losses }));
      return { league: teamsRows[0] ? teamsRows[0].league : null, label, teams, cells };
    };
    return { season: CUR, groups: [
      grid(lv.upper, "Высшая лига"),
      grid(lv.lower_A, "Низшая · группа A"),
      grid(lv.lower_Z, "Низшая · группа Z"),
    ] };
  }
  function matchCommentary(home, away) {            // mirrors commentary.match_commentary (template)
    const hn = tname(home), an = tname(away), hp = powerOf(home), ap = powerOf(away);
    const ev = (snap().fixtures || []).find((m) =>
      (m.home === home && m.away === away) || (m.home === away && m.away === home));
    if (ev && ev.winner) {
      let mh = ev.maps_home, ma = ev.maps_away;
      if (ev.home !== home) { const t = mh; mh = ma; ma = t; }
      const winId = mh > ma ? home : away, win = tname(winId), los = tname(mh > ma ? away : home);
      const wp = winId === home ? hp : ap, lp = winId === home ? ap : hp;
      const margin = Math.min(mh, ma) === 0 ? "всухую" : Math.abs(mh - ma) === 1 ? "на тай-брейке" : "уверенно";
      const upset = wp < lp - 30;
      const text = `${hn} (${hp}) — ${an} (${ap}). ${win} забирают серию ${Math.max(mh, ma)}:${Math.min(mh, ma)} ${margin}. ` +
        (upset ? `Сенсация — ${win} (${wp}) свергли фаворита ${los} (${lp})!` : `Фаворит ${win} подтвердил силу.`);
      return { home: hn, away: an, home_power: hp, away_power: ap, played: true, text, score: `${Math.max(mh, ma)}:${Math.min(mh, ma)}` };
    }
    const fav = hp >= ap ? hn : an, gap = Math.abs(hp - ap);
    const odds = gap < 25 ? "игра равная" : gap < 80 ? `фаворит — ${fav}` : `${fav} — явный фаворит`;
    return { home: hn, away: an, home_power: hp, away_power: ap, played: false, score: null,
      text: `На очереди ${hn} (${hp}) против ${an} (${ap}). По силе ${odds}, но в этой игре всё решат нервы и порядок.` };
  }
  function soloqMatch(pid, season, rnd) {
    const p = pById[pid]; if (!p) throw new Error("игрок не найден");
    const g = (p.soloq || []).find((x) => String(x.season) === String(season) && String(x.round) === String(rnd));
    if (!g) throw new Error("матч не найден");
    const atR = g.ratings || {}, atD = g.deltas || {};   // ratings + deltas AS OF the match (band-consistent)
    const band = g.band != null ? g.band : Math.floor((atR[pid] != null ? atR[pid] : 1000) / 100) * 100;
    const info = (id) => { const q = pById[id] || {};
      const rating = atR[id] != null ? atR[id]
        : Math.round((q.ratingHist || [q.base])[(q.ratingHist || []).length - 1] || q.base || 1000);
      return { player_id: id, nickname: q.nickname, faction: q.faction, archetype: q.archetype, role: q.role,
        rating, band, delta: atD[id] != null ? atD[id] : g.delta,
        team_id: q.teamId, team: q.teamId ? tname(q.teamId) : null, team_abbrev: q.teamId ? tabbr(q.teamId) : null }; };
    const a = [pid].concat(g.teammate_ids || []);
    return { season: +season, round: +rnd, won_a: g.won, band,
      team_a: a.map(info), team_b: (g.opponent_ids || []).map(info) };
  }
  function playerView(p) {
    // show the rating AFTER the most-recent off-season (== last_offseason's SoloQ movers) so the
    // Рейтинг tab and the межсезонье digest agree (ratingHist = [boot, afterS0, afterS1, ...]).
    // ratingHist = [8 bootstrap points, then 1 per finished off-season]; show the 8 boot points + the
    // seasons up to CUR, so «Динамика SoloQ» has the last-5 window already at season 0 (#12).
    const hist = p.ratingHist.slice(0, BOOT_PRESEASONS + CUR).map(Math.round);
    const delta = hist.length >= 2 ? +(hist[hist.length - 1] - hist[hist.length - 2]).toFixed(1) : 0;
    const team = p.teamId ? tname(p.teamId)
      : (p.reserve ? "Резерв" : (p.role === "streamer" ? "Стример" : "Свободный агент"));
    const pos = {}; Object.entries(p.posStats).forEach(([k, v]) =>
      pos[k] = { games: v.games, wins: v.wins, winrate: v.games ? +(v.wins / v.games).toFixed(3) : null });
    return { player_id: p.player_id, nickname: p.nickname, faction: p.faction, archetype: p.archetype,
      role: p.role, status: p.reserve ? "reserve" : (p.teamId ? "rostered" : "free_agent"),
      rating: hist[hist.length - 1], delta, rating_history: hist, soloq_history: hist,
      attributes: p.attributes || {}, earnings: p.earnings || 0,
      salary: p.teamId ? Math.max(8, Math.round((p.base - 860) / 2.2)) : 0,
      soloq_total: p.soloq.length, net_id: p.net_id, team_id: p.teamId,
      team, team_abbrev: p.teamId ? tabbr(p.teamId) : null, games: p.games, generation: p.gen,
      position_stats: pos, soloq_games: p.soloq.slice(-20).reverse(), trophies: p.trophies };
  }
  function rankedPlayers() {
    const v = players.filter((p) => !p.reserve).map(playerView).sort((a, b) => b.rating - a.rating);
    v.forEach((x, i) => x.rank = i + 1); return v;
  }
  function draftView() {
    const d = snap().draft || snaps[0].draft;
    const picks = d.picks.map((p) => ({ round: p.round, overall: p.overall, team_id: p.team_id,
      team: tabbr(p.team_id), team_name: tname(p.team_id), player: pById[p.player_id].nickname,
      faction: pById[p.player_id].faction,
      archetype: pById[p.player_id].archetype, rating: Math.round(pById[p.player_id].ratingHist[0]) }));
    return { season: d.season, kind: d.kind, picks, n: picks.length };
  }
  function powerOf(id) {
    if (LIVE) { const p = lvProvPower(lvRevSet()); if (p[id] != null) return p[id]; }
    return Math.round(snap().teamSnap[id].power);
  }
  function teamRanking() {
    const rows = Object.keys(snap().teamSnap).map((id) => { const ts = snap().teamSnap[id];
      return { team_id: id, name: tname(id), abbrev: tabbr(id), league: ts.league, group: ts.group,
        power: powerOf(id), titles: ts.titles, treasury: Math.round(ts.treasury),
        roster: ts.roster.map((pid) => pById[pid].nickname) }; });
    rows.sort((a, b) => (b.power - a.power) || a.name.localeCompare(b.name));
    rows.forEach((r, i) => (r.rank = i + 1));
    return rows;
  }
  function teamMatches(id) {
    const out = [];
    const add = (s, played) => { const isHome = s.home === id, opp = isHome ? s.away : s.home;
      const mf = played ? (isHome ? s.maps_home : s.maps_away) : null;
      const ma = played ? (isHome ? s.maps_away : s.maps_home) : null;
      out.push({ season: LIVE ? LIVE.season : CUR, day: s.day, stage: s.stage,
        stage_label: STAGE_LABEL[s.stage] || s.stage, opponent: opp, opponent_abbrev: tabbr(opp),
        opponent_name: tname(opp), best_of: s.best_of, home: isHome, played, maps_for: mf, maps_against: ma,
        maps_seq: played ? Array(mf).fill("W").concat(Array(ma).fill("L")) : null,
        won: played ? (s.winner === id) : null }); };
    if (LIVE) {
      const revset = lvRevSet();
      LIVE.days.forEach((d) => { if (d.phase !== "regular") return; d.series.forEach((s) => {
        if (s.home === id || s.away === id) add(s, revset.has(s.gid)); }); });
      LIVE.days.forEach((d) => { if (d.phase !== "playoff") return; d.series.forEach((s) => {
        if ((s.home === id || s.away === id) && revset.has(s.gid)) add(s, true); }); });
    } else {
      snap().fixtures.forEach((m) => { if (m.home === id || m.away === id) add(m, true); });
      Object.values(snap().bracket).forEach((cols) => cols.forEach((col) => col.matches.forEach((m) => {
        if (m.home === id || m.away === id) add(m, true); })));
    }
    out.sort((a, b) => (a.day || 0) - (b.day || 0));
    return out;
  }
  function teamTransfers(id) {
    if (CUR === 0) return { in: [], out: [] };     // no off-season has happened yet -> no transfers
    const tx = (snaps[Math.max(0, CUR - 1)] || {}).transactions || {};
    const tin = (tx.signings || []).filter((s) => s.team_id === id).map((s) => ({ player: s.player, kind: s.kind, bonus: s.bonus, dir: "signed" }))
      .concat((tx.buys || []).filter((b) => b.to_team_id === id).map((b) => ({ player: b.player, from: b.from_team, price: b.price, dir: "bought" })));
    const tout = (tx.buys || []).filter((b) => b.from_team_id === id).map((b) => ({ player: b.player, to: b.to_team, price: b.price, dir: "sold" }))
      .concat((tx.releases || []).filter((r) => r.team_id === id).map((r) => ({ player: r.player, dir: "released" })));
    return { in: tin, out: tout };
  }
  function teamSeasonHistory(id) {
    const out = [];
    for (let s = 0; s < CUR; s++) { const snp = snaps[s], ts = snp.teamSnap[id]; if (!ts) continue;
      out.push({ season: s, league: ts.league, place: ts.place,
        champion: id === snp.champ.upper_champion || id === snp.champ.lower_champion,
        promoted: (snp.champ.promoted || []).includes(id), relegated: (snp.champ.relegated || []).includes(id) }); }
    return out;
  }
  function teamPlaybook(id) {        // mirrors views.team_playbook + playbook.build_default_plan (readable)
    const i = Object.keys(byId).indexOf(id);
    const base = STRAT[(i + CUR) % STRAT.length];
    const agg = "all_in", safe = "turtle";
    return { team_id: id, name: tname(id), base_strategy: STRAT_RU[base] || base, base_strategy_id: base,
      overrides: [
        { when: "соперник в основном Robots", strategy: STRAT_RU.core_rush, strategy_id: "core_rush",
          note: "раш до того как роботы раскачают экономику" },
        { when: "в плей-офф, серия из 2+ поражений", strategy: STRAT_RU[agg], strategy_id: agg,
          note: "must-win: идём ва-банк" },
        { when: "в плей-офф, ведём в серии", strategy: STRAT_RU[safe], strategy_id: safe,
          note: "ведём в серии — играем безопасно" },
      ] };
  }
  function teamView(id) {
    const ts = snap().teamSnap[id];
    const matches = teamMatches(id);
    const rk = teamRanking().find((r) => r.team_id === id);
    const rec = { series_wins: 0, series_losses: 0, maps_won: 0, maps_lost: 0 };
    matches.forEach((m) => { if (!m.played) return; rec.maps_won += m.maps_for;
      rec.maps_lost += m.maps_against; m.won ? rec.series_wins++ : rec.series_losses++; });
    return { team_id: id, name: tname(id), abbrev: tabbr(id), manager: "Менеджер", league: ts.league,
      group: ts.group, power: powerOf(id), power_rank: rk ? rk.rank : 0, n_teams: 22,
      titles: ts.titles, treasury: Math.round(ts.treasury),
      // Power curve: during a live reveal, drop the (pre-simulated) season's final point and extend
      // with the PROVISIONAL Power — so the «Динамика Power» sparkline never spoils the ending.
      elo_history: LIVE ? [...ts.eloHist.slice(0, -1), powerOf(id)] : ts.eloHist,
      trophies: ts.trophies, certificates: ts.certs, roster: ts.roster.map((pid) => playerView(pById[pid])),
      record: rec, matches_played: matches.filter((m) => m.played),
      matches_upcoming: matches.filter((m) => !m.played),
      transfers: teamTransfers(id), season_history: teamSeasonHistory(id),
      playbook: teamPlaybook(id) };
  }
  function playerPage(pid) {
    const p = pById[pid]; if (!p) return null;
    const base = playerView(p);
    base.team_history = p.teamId
      ? Array.from({ length: CUR + 1 }, (_, s) => ({ season: s, team_id: p.teamId,
          team: tname(p.teamId), role: p.role, result: "" }))
      : [];
    base.salary = p.teamId ? Math.round((p.base - 860) / 2.2) : 0;
    base.salary_history = [];
    base.lineage = { origin: p.role === "streamer" ? "streamer" : "rookie_fork",
      generation: p.gen, parent_id: null, parent: null };
    base.official_games = Math.round(p.games * 0.4);
    base.seasons_played = CUR + 1;
    return base;
  }
  function overview() {
    const ids = Object.keys(snap().teamSnap);
    const reserves = players.filter((p) => p.reserve).map(playerView);
    const rostered = ids.reduce((n, id) => n + snap().teamSnap[id].roster.length, 0);
    const history = snaps.slice(0, CUR).map((s) => {
      const lowerRank = (s.aStand || []).concat(s.zStand || []), nmO = (id) => ({ team_id: id, name: tname(id) });
      const prizeOf = s.prizeOf || {}, pbt = {};
      Object.keys(prizeOf).forEach((id) => (pbt[id] = prizeOf[id]));   // prize money 1:1 with the team page
      // finalists (grand-final pair) from the stored bracket's grand-final column (cols = [{matches:[row]}])
      const gf = (br) => { const col = (br || []).slice(-1)[0]; const m = col && (col.matches || [])[0];
        return m && m.home && m.away ? [nmO(m.home), nmO(m.away)] : []; };
      return { season: s.num, upper_champion: nmO(s.champ.upper_champion), lower_champion: nmO(s.champ.lower_champion),
        promoted: (s.champ.promoted || []).map(nmO), relegated: (s.champ.relegated || []).map(nmO),
        upper_top: (s.upStand || []).slice(0, 3).map((id, i) => ({ ...nmO(id), place: i + 1, total: prizeOf[id] || 0 })),
        lower_top: lowerRank.slice(0, 3).map((id, i) => ({ ...nmO(id), place: i + 1, total: prizeOf[id] || 0 })),
        upper_finalists: gf((s.bracket || {}).upper), lower_finalists: gf((s.bracket || {}).lower),
        prizes_by_team: pbt, prize_pool: s.prizePool || { upper: [], lower_A: [], lower_Z: [] },
        bracket: s.bracket || { upper: [], lower: [] }, n_matches: 85 };
    });
    const commentary = [];
    snaps.slice(0, CUR + 1).forEach((s) => { commentary.push(s.commentary[0]); if (s.num < CUR) commentary.push(s.commentary[1]); });
    const offSnap = snaps[Math.max(0, CUR - 1)] || snap();
    const managers = offSnap.managerComments || [];
    const livePhase = LIVE ? (LIVE.shownDay >= LIVE.nDays ? "offseason_due" : "live") : PHASE;
    return { world_id: WID, name: "Demo World", season: CUR, phase: livePhase,
      upper: [], lower: [], ladder: [], history, reserves, commentary, emblems: EMBLEMS, colors: COLORS,
      manager_comments: CUR > 0 ? managers : [],
      live: LIVE ? lvView() : null,
      last_offseason: CUR > 0 ? { season_ended: offSnap.num, next_season: CUR,
        transactions: offSnap.transactions, soloq: (offSnap.offseason || {}).soloq,
        planning: (offSnap.offseason || {}).planning, training: (offSnap.offseason || {}).training,
        money: (offSnap.offseason || {}).money, skills: (offSnap.offseason || {}).skills } : null,
      counts: { players: players.length,
        free_agents: players.filter((p) => p.role === "pro" && !p.teamId && !p.reserve).length,
        streamers: players.filter((p) => p.role === "streamer").length,
        rostered, reserves: reserves.length } };
  }
  function playersPool() {
    const poolv = players.filter((p) => p.role === "pro").map((p) => ({ net_id: p.net_id,
      faction: p.faction, archetype: p.archetype, overall: Math.round((p.base - 880) / 3), attrs: {} }))
      .sort((a, b) => b.overall - a.overall);
    return { pool: poolv, total: poolv.length, n_teams: 22, min_enter: 66 };
  }

  // ---- the fake API ----------------------------------------------------- //
  let lastSummary = {};
  let showTraining = false;       // the tracker appears only after an off-season (like the real UI)
  let PHASE = "regular";          // regular -> (play season) -> offseason_due -> (off-season) -> regular
  function doAdvance() {           // play the CURRENT season; the OFF-SEASON rolls to the next
    const s = snaps[CUR];
    showTraining = false; PHASE = "offseason_due";
    lastSummary = { season: s.num, upper_champion: tname(s.champ.upper_champion),
      lower_champion: tname(s.champ.lower_champion), promoted: s.champ.promoted.map(tname),
      relegated: s.champ.relegated.map(tname), commentary: s.commentary.map((c) => c.text) };
    return { job_id: "adv", status: "running" };
  }
  function doOffseason() {
    const s = snaps[CUR];
    showTraining = true; PHASE = "regular"; LIVE = null;   // the finished live season is now history
    lastSummary = { next_season: Math.min(CUR + 1, 2), rookies: 3, training_auctioned: 6,
      training_launched: Math.min(6, s.training.length), chronicle: s.chronicle,
      log: s.chronicle.map((c) => c.msg).slice(0, 50) };
    if (CUR < 2) CUR++;
    return { job_id: "off", status: "running" };
  }

  // ---- day-by-day LIVE demo (self-contained mini engine, mirrors live_season.py) ---- //
  let LIVE = null;
  function lvRounds(ids) {
    const a = ids.slice(); if (a.length % 2) a.push(null);
    const n = a.length, out = []; let arr = a.slice();
    for (let r = 0; r < n - 1; r++) {
      const pairs = [];
      for (let i = 0; i < n / 2; i++) { const x = arr[i], y = arr[n - 1 - i]; if (x && y) pairs.push([x, y]); }
      out.push(pairs); arr = [arr[0], arr[n - 1]].concat(arr.slice(1, n - 1));
    }
    return out;
  }
  function lvPlay(home, away, bo) {
    const need = (bo >> 1) + 1; let mh = 0, ma = 0; const seq = [];
    const p = 1 / (1 + Math.pow(10, (byId[away].power - byId[home].power) / 250));
    while (mh < need && ma < need) { if (R() < p) { mh++; seq.push(home); } else { ma++; seq.push(away); } }
    // demo: each map points at a bundled sample recording so «Смотреть настоящий матч» opens a real replay
    const map_replays = seq.map((_, i) => `../replays/replay_${(i % 5) + 1}.json`);
    return { home, away, maps_home: mh, maps_away: ma, winner: mh > ma ? home : away,
      best_of: bo, sequence: seq, map_replays, stage: "league" };
  }
  function lvStand(ids, results) {
    const rec = {}; ids.forEach((id) => (rec[id] = { team_id: id, sw: 0, sl: 0, mw: 0, ml: 0 }));
    results.forEach((r) => { if (!rec[r.home] || !rec[r.away]) return;
      rec[r.home].mw += r.maps_home; rec[r.home].ml += r.maps_away;
      rec[r.away].mw += r.maps_away; rec[r.away].ml += r.maps_home;
      if (r.maps_home > r.maps_away) { rec[r.home].sw++; rec[r.away].sl++; } else { rec[r.away].sw++; rec[r.home].sl++; } });
    const rows = Object.values(rec).map((x) => ({ ...x, points: x.sw * 3, map_diff: x.mw - x.ml }));
    rows.sort((p, q) => (q.points - p.points) || (q.map_diff - p.map_diff) || (q.mw - p.mw));
    rows.forEach((r, i) => (r.place = i + 1));
    return rows;
  }
  const lvRow = (r) => ({ team_id: r.team_id, abbrev: tabbr(r.team_id), name: tname(r.team_id),
    place: r.place, points: r.points, series_wins: r.sw, series_losses: r.sl, map_diff: r.map_diff });
  function lvResolve(ref, nodes, maxDay) {
    if (typeof ref === "string") return ref;
    const n = nodes[ref.w || ref.l], res = n && n.result; if (!res) return null;
    if (maxDay != null && (n.day || 0) > maxDay) return null;
    return ("w" in ref) ? res.winner : (res.winner === res.home ? res.away : res.home);
  }
  const lvReady = (br) => br.order.filter((k) => { const n = br.nodes[k];
    return !n.result && lvResolve(n.home, br.nodes) && lvResolve(n.away, br.nodes); });
  const lvDone = (br) => br.order.every((k) => br.nodes[k].result);
  function lvUpperBr(us) {
    return { order: ["ub", "lb", "lbf", "gf"], finalists: ["ub", "lbf"], nodes: {
      ub: { stage: "ub", home: us[0], away: us[1] }, lb: { stage: "lb", home: us[2], away: us[3] },
      lbf: { stage: "lb_final", home: { w: "lb" }, away: { l: "ub" } },
      gf: { stage: "grand_final", home: { w: "ub" }, away: { w: "lbf" } } } };
  }
  function lvLowerBr(as, zs) {
    return { order: ["ub", "lb1a", "lb1b", "lb2", "lbf", "gf"], finalists: ["ub", "lbf"], nodes: {
      ub: { stage: "ub", home: as[0], away: zs[0] }, lb1a: { stage: "lb_r1", home: as[1], away: zs[2] },
      lb1b: { stage: "lb_r1", home: zs[1], away: as[2] }, lb2: { stage: "lb_r2", home: { w: "lb1a" }, away: { w: "lb1b" } },
      lbf: { stage: "lb_final", home: { w: "lb2" }, away: { l: "ub" } },
      gf: { stage: "grand_final", home: { w: "ub" }, away: { w: "lbf" } } } };
  }
  // PLAY the whole demo season up front, store it PER GAME for reveal (mirrors live_season.py):
  // 2 regular games per day, 1 playoff game per day, UPPER bracket fully then LOWER. Reveal is per
  // gid (revealedGids), so a single match can be watched + committed on its own.
  const LV_REG_PER_DAY = 2;
  function lvBuild() {
    const up = teams.filter((t) => t.league === "upper").map((t) => t.team_id);
    const a = teams.filter((t) => t.league === "lower" && t.group === "A").map((t) => t.team_id);
    const z = teams.filter((t) => t.league === "lower" && t.group === "Z").map((t) => t.team_id);
    const rounds = { upper: lvRounds(up), lower_A: lvRounds(a), lower_Z: lvRounds(z) };
    const nreg = Math.max(rounds.upper.length, rounds.lower_A.length, rounds.lower_Z.length);
    const groups = { upper: up, lower_A: a, lower_Z: z };
    const results = { upper: [], lower_A: [], lower_Z: [] };
    let gid = 0;
    const tag = (s) => { s.gid = gid++; s.revealed = false; return s; };
    // regular: round-major then group, ALL pairings -> 2 games per day
    const regularGames = [];
    for (let r = 0; r < nreg; r++) {
      ["upper", "lower_A", "lower_Z"].forEach((gk) => { const rr = rounds[gk]; if (r >= rr.length) return;
        rr[r].forEach(([h, a2]) => { const s = tag(lvPlay(h, a2, 3)); s.group = gk; s.league = gk === "upper" ? "upper" : "lower";
          results[gk].push(s); regularGames.push(s); }); });
    }
    const days = [];
    for (let i = 0; i < regularGames.length; i += LV_REG_PER_DAY)
      days.push({ phase: "regular", series: regularGames.slice(i, i + LV_REG_PER_DAY) });
    const nregDays = days.length;
    const us = lvStand(up, results.upper).map((r) => r.team_id);
    const as = lvStand(a, results.lower_A).map((r) => r.team_id);
    const zs = lvStand(z, results.lower_Z).map((r) => r.team_id);
    const br = { upper: lvUpperBr(us), lower: lvLowerBr(as, zs) };
    // playoff: ONE decidable game per day; UPPER bracket fully, THEN lower (not interleaved)
    const playBracket = (b, lg) => {
      while (!lvDone(b)) { const ready = lvReady(b); if (!ready.length) break;
        ready.forEach((k) => { const n = b.nodes[k];
          const h = lvResolve(n.home, b.nodes), a2 = lvResolve(n.away, b.nodes);
          const s = tag(lvPlay(h, a2, 5)); s.stage = n.stage; s.league = lg; n.result = s; n.gid = s.gid;
          days.push({ phase: "playoff", series: [s] }); }); }
    };
    playBracket(br.upper, "upper"); playBracket(br.lower, "lower");
    days.forEach((d, di) => d.series.forEach((s) => (s.day = di + 1)));
    const baseElo = {}; teams.forEach((t) => (baseElo[t.team_id] = Math.round(byId[t.team_id].power)));
    LIVE = { season: CUR, nDays: days.length, nreg: nregDays, groups, days, playoff: br,
      revealedGids: [], shownDay: 0, baseElo,    // shownDay advances ONLY on «Следующий день»
      done: { season: CUR, upper_champion: tabbr(br.upper.nodes.gf.result.winner),
        lower_champion: tabbr(br.lower.nodes.gf.result.winner) } };
  }
  function lvRevSet() { return new Set(LIVE.revealedGids); }
  function lvProvPower(revset) {
    const elo = {}; Object.keys(LIVE.baseElo).forEach((k) => (elo[k] = LIVE.baseElo[k]));
    LIVE.days.forEach((d) => d.series.forEach((s) => { if (!revset.has(s.gid)) return;
      const need = (s.best_of >> 1) + 1, m = Math.abs(s.maps_home - s.maps_away) / need;
      elo[s.home] += 8 * m; elo[s.away] -= 8 * m; }));
    const o = {}; Object.keys(elo).forEach((k) => (o[k] = Math.round(elo[k]))); return o;
  }
  function lvLeagueLabel(s) {
    if (s.league === "upper") return "Высшая лига";
    return "Низшая лига" + (s.group === "lower_A" ? " · группа A" : s.group === "lower_Z" ? " · группа Z" : "");
  }
  function lvResolveRev(ref, nodes, revset) {       // resolve a feeder only once its source game is revealed
    if (typeof ref === "string") return ref;
    const n = nodes[ref.w || ref.l], res = n && n.result;
    if (!res || !revset.has(n.gid)) return null;
    return ("w" in ref) ? res.winner : (res.winner === res.home ? res.away : res.home);
  }
  function lvFixture(s) {
    return { home: s.home, away: s.away, home_abbrev: tabbr(s.home), away_abbrev: tabbr(s.away),
      stage: s.stage, group: s.group, league: s.league, best_of: s.best_of };
  }
  function lvResult(s, seq) {
    const o = lvFixture(s); o.gid = s.gid; o.maps_home = s.maps_home; o.maps_away = s.maps_away; o.winner = s.winner;
    if (seq) { o.sequence = s.sequence; o.map_replays = s.map_replays; } return o;
  }
  function lvReveal() {                              // «Следующий день»: reveal shown day's games, advance
    if (!LIVE || LIVE.shownDay >= LIVE.nDays) return { result: { already_done: true }, live: lvView() };
    const revset = lvRevSet(), sd = LIVE.shownDay;
    LIVE.days[sd].series.forEach((s) => { if (!revset.has(s.gid)) LIVE.revealedGids.push(s.gid); });
    LIVE.shownDay = sd + 1;
    const finished = LIVE.shownDay >= LIVE.nDays;    // advancing past the last day settles the season
    if (finished) PHASE = "offseason_due";
    return { result: { day: sd + 1, finished }, live: lvView() };
  }
  function lvWatch(home, away) {                     // watch + COMMIT one game of the SHOWN day; never advances
    if (!LIVE || LIVE.shownDay >= LIVE.nDays) throw new Error("нечего смотреть");
    const revset = lvRevSet(), sd = LIVE.shownDay;
    const s = LIVE.days[sd].series.find((x) => !revset.has(x.gid) && x.home === home && x.away === away);
    if (!s) throw new Error("это не матч текущего дня");
    LIVE.revealedGids.push(s.gid);                   // commit only — watching never finalises the season
    return { result: lvResult(s, true), finished: false, live: lvView() };
  }
  function lvStandRows(revset, power) {
    const rev = { upper: [], lower_A: [], lower_Z: [] };
    LIVE.days.forEach((d) => { if (d.phase !== "regular") return;
      d.series.forEach((s) => { if (revset.has(s.gid)) rev[s.group].push(s); }); });
    const st = {};
    ["upper", "lower_A", "lower_Z"].forEach((k) => (st[k] = lvStand(LIVE.groups[k], rev[k]).map((r) => {
      const row = lvRow(r); row.power = power[r.team_id]; return row; })));
    return st;
  }
  function lvBracketAt(br, revset) {
    if (!br) return [];
    return br.order.map((k) => { const n = br.nodes[k];
      const h = lvResolveRev(n.home, br.nodes, revset), a = lvResolveRev(n.away, br.nodes, revset), res = n.result;
      const shown = revset.has(n.gid);
      return { key: k, stage: n.stage,
        home: h ? { team_id: h, abbrev: tabbr(h), name: tname(h) } : null,
        away: a ? { team_id: a, abbrev: tabbr(a), name: tname(a) } : null,
        played: shown, maps_home: shown ? res.maps_home : null,
        maps_away: shown ? res.maps_away : null, winner: shown ? res.winner : null }; });
  }
  function lvSchedule() {        // ALL regular fixtures up front; a result only once its game is revealed
    const revset = lvRevSet();
    const toRow = (s) => { const played = revset.has(s.gid);
      return { day: s.day || 0, league: s.league, stage: s.stage, stage_label: STAGE_LABEL[s.stage] || s.stage,
        home: s.home, away: s.away, home_abbrev: tabbr(s.home), away_abbrev: tabbr(s.away),
        home_name: tname(s.home), away_name: tname(s.away), best_of: s.best_of, played,
        maps_home: played ? s.maps_home : null, maps_away: played ? s.maps_away : null,
        winner: played ? s.winner : null }; };
    const regular = []; LIVE.days.forEach((d) => { if (d.phase === "regular") d.series.forEach((s) => regular.push(toRow(s))); });
    const nPlayoff = LIVE.days.filter((d) => d.phase === "playoff" && d.series.every((s) => revset.has(s.gid))).length;
    return { season: LIVE.season, regular, bracket: lvBracketSched(), n_matches: regular.length + nPlayoff };
  }
  // Full bracket with TBD slots (seeds appear only after the regular is fully revealed) + feeder keys
  // for the connector graph — mirrors views._live_bracket (per-gid gating).
  function lvBracketSched() {
    if (!LIVE) return {};
    const revset = lvRevSet();
    // per-league: the LOWER bracket seeds as soon as the LOWER regular is done, even if upper is still
    // playing (and vice-versa) — don't make one league wait on the other (#1).
    const regDone = (lg) => LIVE.days.filter((d) => d.phase === "regular")
      .every((d) => d.series.every((s) => s.league !== lg || revset.has(s.gid)));
    const cols = (br, regularDone) => { const nodes = br.nodes;
      const slot = (ref) => (typeof ref === "string") ? (regularDone ? ref : null) : lvResolveRev(ref, nodes, revset);
      const byStage = {};
      br.order.forEach((k) => { const n = nodes[k], h = slot(n.home), a = slot(n.away), res = n.result;
        const shown = revset.has(n.gid);
        const feeders = [n.home, n.away].filter((r) => typeof r === "object" && r.w).map((r) => r.w);  // winner-advance only
        (byStage[n.stage] = byStage[n.stage] || []).push({ stage: n.stage, stage_label: STAGE_LABEL[n.stage] || n.stage,
          key: k, from: feeders, home: h, away: a, home_abbrev: h ? tabbr(h) : "TBD", away_abbrev: a ? tabbr(a) : "TBD",
          best_of: 5, played: shown, maps_home: shown ? res.maps_home : null, maps_away: shown ? res.maps_away : null,
          winner: shown ? res.winner : null }); });
      const order = ["lb_r1", "lb_r2", "ub", "lb", "lb_final", "grand_final"];
      return order.filter((st) => byStage[st]).map((st) => ({ stage: st, label: STAGE_LABEL[st] || st, matches: byStage[st] }));
    };
    return { upper: cols(LIVE.playoff.upper, regDone("upper")), lower: cols(LIVE.playoff.lower, regDone("lower")) };
  }
  const LV_STAGE = { league: "Регулярный сезон", ub: "Плей-офф · Верхняя сетка", lb: "Плей-офф · Нижняя сетка",
    lb_r1: "Плей-офф · Нижняя сетка, раунд 1", lb_r2: "Плей-офф · Нижняя сетка, раунд 2",
    lb_final: "Финал нижней сетки", grand_final: "ГРАНД-ФИНАЛ" };
  function lvCard(s, revset, power, rows) {
    const info = (tid) => { const row = rows[tid] || {}; const form = [];
      LIVE.days.forEach((d) => d.series.forEach((ss) => {
        if (revset.has(ss.gid) && (ss.home === tid || ss.away === tid)) form.push(ss.winner === tid ? "W" : "L"); }));
      return { team_id: tid, abbrev: tabbr(tid), name: tname(tid), power: power[tid] != null ? power[tid] : 0,
        place: row.place, points: row.points || 0, series_wins: row.series_wins || 0,
        series_losses: row.series_losses || 0, map_diff: row.map_diff || 0, form: form.slice(-6) }; };
    const revealed = revset.has(s.gid);
    const o = { gid: s.gid, stage: s.stage, stage_label: LV_STAGE[s.stage] || s.stage,
      league: s.league, league_label: lvLeagueLabel(s), group: s.group, best_of: s.best_of, day: s.day,
      home: info(s.home), away: info(s.away), revealed };
    if (revealed) { o.maps_home = s.maps_home; o.maps_away = s.maps_away; o.winner = s.winner; o.sequence = s.sequence; o.map_replays = s.map_replays; }
    return o;
  }
  function lvRevSetUpto(dayNum) {   // revealed gids whose (1-based) display day <= dayNum (frozen past day)
    const full = lvRevSet(), out = new Set();
    LIVE.days.forEach((d) => d.series.forEach((s) => { if (full.has(s.gid) && (s.day || 0) <= dayNum) out.add(s.gid); }));
    return out;
  }
  function lvView(day) {             // day (0-based) rewinds to an already-played day; null = live cursor / done
    if (!LIVE) return { active: false };
    const full = lvRevSet(), n = LIVE.nDays, cur = LIVE.shownDay, liveActive = cur < n;
    const maxView = liveActive ? cur : n - 1;
    const viewing = (day == null) ? maxView : Math.max(0, Math.min(day | 0, maxView));
    const isCurrentDay = liveActive && viewing === cur;
    const showingDone = !liveActive && day == null;          // legacy "season finished" entry screen
    const revset = (isCurrentDay || showingDone) ? full : lvRevSetUpto(viewing + 1);
    const power = lvProvPower(revset), standings = lvStandRows(revset, power);
    const rows = {}; ["upper", "lower_A", "lower_Z"].forEach((k) => standings[k].forEach((r) => (rows[r.team_id] = r)));
    const dayMatches = showingDone ? [] : LIVE.days[viewing].series.map((s) => lvCard(s, revset, power, rows));
    const playoffStarted = (showingDone ? n - 1 : viewing) >= LIVE.nreg;
    return { active: liveActive, season: LIVE.season,
      phase: showingDone ? "done" : LIVE.days[viewing].phase,
      day: viewing + 1, cursor: cur, viewing_day: viewing, total_days: n, n_regular_days: LIVE.nreg,
      is_current_day: isCurrentDay, can_prev: viewing > 0, can_next: viewing < maxView,
      day_matches: dayMatches, standings,
      bracket: { upper: playoffStarted ? lvBracketAt(LIVE.playoff.upper, revset) : [],
        lower: playoffStarted ? lvBracketAt(LIVE.playoff.lower, revset) : [] },
      done: showingDone ? LIVE.done : null };
  }

  window.__PREVIEW__ = async function (url, opts) {
    await sleep(160);
    const method = (opts && opts.method) || "GET";
    url = url.split("?")[0];
    if (url === "/api/world/info") return { ui: { locked: false, note: "" }, worlds: ["world-preview"] };
    if (url === "/api/world/franchises") {
      const TEMPO = ["balanced", "aggressive", "defensive"], WP = ["economy", "core_rush", "map_control"];
      return { franchises: teams.map((t, i) => ({ franchise_id: t.franchise_id, name: t.name, abbrev: t.abbrev,
        color: t.color, accent: t.accent || t.color, manager: `Менеджер ${t.abbrev}`,
        persona: `Клуб «${t.name}» — ${pk(["амбициозный", "хладнокровный", "дерзкий", "расчётливый"])}, ставит на ${pk(["молодёжь", "опыт", "агрессию", "экономику"])}.`,
        default_plan: { tempo: TEMPO[i % 3], win_path: WP[i % 3] }, emblem: EMBLEMS[t.team_id] || "" })) };
    }
    if (url === "/api/world/players-pool") return playersPool();
    if (url.startsWith("/api/world/job/")) return { status: "done", world_id: WID, phase: "Готово", summary: lastSummary };
    if (method === "POST" && url === "/api/world/create") { CUR = 0; lastSummary = {}; return { job_id: "cr", status: "running" }; }
    if (method === "POST" && url.endsWith("/live/start")) {
      if (!(LIVE && LIVE.shownDay < LIVE.nDays)) lvBuild();   // don't clobber an in-progress reveal
      lastSummary = { season: LIVE.season, total_days: LIVE.nDays };
      return { job_id: "live", status: "running" };
    }
    if (method === "POST" && url.endsWith("/live/day")) return lvReveal();
    if (method === "POST" && url.endsWith("/live/watch")) {
      const b = opts && opts.body ? JSON.parse(opts.body) : {};
      return lvWatch(b.home, b.away);
    }
    if (method === "POST" && url.endsWith("/training/clear")) { showTraining = false; return { cleared: 1 }; }
    if (method === "POST" && url.endsWith("/match/commentary")) {
      const b = opts && opts.body ? JSON.parse(opts.body) : {};
      return matchCommentary(b.home, b.away);
    }
    if (method === "POST" && url.endsWith("/advance")) return doAdvance();
    if (method === "POST" && url.endsWith("/offseason")) return doOffseason();
    if (method === "POST" && url.endsWith("/introduce")) return { introduced: 0, reserves_left: players.filter((p) => p.reserve).length };
    const path = url.replace(/^\/api\/world\/[^/]+/, "");
    if (path === "") return { world: overview() };       // real /api/world/<id> wraps in {world}
    if (path === "/leagues") return { season: CUR, leagues: leaguesView(),
      prizes: { upper: { places: PRIZE_U, champion_bonus: BONUS_U },
                lower: { places: PRIZE_L_GROUP, champion_bonus: BONUS_L, per_group: true } } };
    if (path === "/live") return { live: LIVE ? lvView() : { active: false } };
    const liveDayM = path.match(/^\/live\/(\d+)$/);
    if (liveDayM) return { live: LIVE ? lvView(+liveDayM[1]) : { active: false } };
    if (path === "/matrix") return matrixView();
    if (path === "/teams") return { season: CUR, ranking: teamRanking() };
    if (path === "/players") return { season: CUR, players: rankedPlayers() };
    if (path === "/schedule") return LIVE ? lvSchedule()
      : { season: CUR, regular: snap().fixtures, bracket: snap().bracket, n_matches: snap().fixtures.length };
    if (path === "/draft") return draftView();
    if (path === "/training") return { jobs: showTraining ? snaps[Math.max(0, CUR - 1)].training : [] };
    if (path.startsWith("/team/")) return { season: CUR, team: teamView(path.slice("/team/".length)) };
    if (path.startsWith("/player/")) return { player: playerPage(path.slice("/player/".length)) };
    if (path.startsWith("/soloq/")) { const parts = path.slice("/soloq/".length).split("/");
      return { match: soloqMatch(parts[0], parts[1], parts[2]) }; }
    return {};
  };
})();
