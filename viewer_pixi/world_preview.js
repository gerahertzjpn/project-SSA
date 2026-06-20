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
  const STRAT = ["standard", "core_rush", "all_in", "turtle", "macro", "baron"];
  const OPS = ["skill_up", "meta_adapt", "style_spec", "obedience_tune", "teamplay"];
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
  const PRE = ["Ash", "Bly", "Cor", "Dra", "Eko", "Fen", "Gax", "Hel", "Ira", "Jot", "Kor", "Lyx",
    "Mir", "Nyx", "Orb", "Pyr", "Quo", "Rho", "Syl", "Tau", "Umb", "Vex", "Wol", "Xan", "Ymi", "Zed"];
  const SUF = ["an", "ix", "or", "us", "ai", "en", "ra", "os", "yn", "el", "ar", "io", "ux", "is",
    "ov", "ek", "ym", "ad", "ul", "ez"];
  const TEAMS22 = [
    ["Iron Falcons", "IRF"], ["Void Reapers", "VDR"], ["Solar Kings", "SLK"], ["Drift Wolves", "DRW"],
    ["Neon Serpents", "NES"], ["Crimson Apex", "CRA"], ["Ember Vanguard", "EMV"], ["Storm Kraken", "STK"],
    ["Nova Drakes", "NVD"], ["Glacier Born", "GLB"], ["Echo Sentinels", "ECS"], ["Pulse Riders", "PLR"],
    ["Astral Wardens", "ASW"], ["Cobalt Giants", "CBG"], ["Molten Lynx", "MLX"], ["Vortex Hydra", "VXH"],
    ["Onyx Vipers", "ONV"], ["Prism Raptors", "PRR"], ["Tidal Forge", "TDF"], ["Frost Hollow", "FRH"],
    ["Ether Cobras", "ETC"], ["Quasar Bulls", "QSB"],
  ];
  const COLORS = ["#5aa9ff", "#9b6bff", "#ffb347", "#6ee7a8", "#ff7a7a", "#f15bb5", "#ffd166",
    "#48c9b0", "#7aa5ff", "#b388ff", "#ff8c69", "#4dd0e1"];

  const usedNick = new Set();
  function nick() { let n; do { n = pk(PRE) + pk(SUF); } while (usedNick.has(n)); usedNick.add(n); return n; }

  const WID = "world-preview";

  // ---- teams + leagues -------------------------------------------------- //
  let teams = TEAMS22.map((t, i) => ({
    team_id: `${WID}-t-fr${String(i).padStart(2, "0")}`, name: t[0], abbrev: t[1],
    franchise_id: `fr${i}`, color: COLORS[i % COLORS.length], base: ri(965, 1175),
  }));
  teams.sort((a, b) => b.base - a.base);
  teams.forEach((t, i) => {
    t.league = i < 10 ? "upper" : "lower";
    t.group = i < 10 ? null : ((i - 10) % 2 === 0 ? "A" : "Z");
    t.power = i < 10 ? t.base + 120 : t.base;     // upper seeded above lower (earned promotion)
    t.titles = 0; t.trophies = []; t.eloHist = [Math.round(t.power)]; t.certs = 0;
    t.treasury = t.league === "upper" ? 600 : 400; t.roster = [];
  });
  const byId = Object.fromEntries(teams.map((t) => [t.team_id, t]));
  const tname = (id) => (byId[id] ? byId[id].name : id);
  const tabbr = (id) => (byId[id] ? byId[id].abbrev : id);
  // simple generated emblems (the real World uses regionals emblem_svg server-side)
  function previewEmblem(t) {
    return `<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">` +
      `<rect width="120" height="120" rx="22" fill="#11161d"/>` +
      `<path d="M60 14 L102 30 V64 Q102 96 60 110 Q18 96 18 64 V30 Z" fill="${t.color}"/>` +
      `<path d="M60 14 L102 30 V64 Q102 96 60 110 Q18 96 18 64 V30 Z" fill="none" stroke="#0c1116" stroke-width="4"/>` +
      `<text x="60" y="74" font-family="Arial" font-size="36" font-weight="800" text-anchor="middle" fill="#0c1116">${t.abbrev.slice(0, 2)}</text></svg>`;
  }
  const EMBLEMS = Object.fromEntries(teams.map((t) => [t.team_id, previewEmblem(t)]));

  // ---- players (90: 78 debut, 12 reserve) + Season-0 draft --------------- //
  let players = [];
  for (let i = 0; i < 90; i++) {
    const fac = pk(FACTIONS), ar = pk(ARCH);
    const p = { player_id: `${WID}-p${String(i).padStart(3, "0")}`, nickname: nick(), faction: fac,
      archetype: ar, net_id: `${fac}_${ri(1, 6)}__${ar}`, role: "pro", base: ri(905, 1130),
      teamId: null, reserve: false, games: ri(24, 90), gen: 1, posStats: {},
      ratingHist: [], soloq: [], trophies: [] };
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
  const STAGE_LABEL = { ub: "Верхняя сетка", lb_r1: "Нижняя сетка — 1", lb_final: "Финал нижней сетки", grand_final: "Гранд-финал" };
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
  function playoff(s1, s2, s3, s4, league, dayRef) {
    const col = (stage, matches) => ({ stage, label: STAGE_LABEL[stage], matches });
    const ub = series(s1, s2, 3, STAGE_LABEL.ub, "ub", league, dayRef.d++);
    const lb1 = series(s3, s4, 3, STAGE_LABEL.lb_r1, "lb_r1", league, dayRef.d++);
    const ubLoser = ub.winner === s1 ? s2 : s1;
    const lbf = series(lb1.winner, ubLoser, 3, STAGE_LABEL.lb_final, "lb_final", league, dayRef.d++);
    const gf = series(ub.winner, lbf.winner, 3, STAGE_LABEL.grand_final, "grand_final", league, dayRef.d++);
    return { cols: [col("ub", [ub.row]), col("lb_r1", [lb1.row]), col("lb_final", [lbf.row]), col("grand_final", [gf.row])],
      champ: gf.winner, finalists: [ub.winner, lbf.winner] };
  }

  players.forEach((p) => { p.ratingHist = [p.base]; });   // S0 SoloQ seed
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
    const stand = (ids) => ids.slice().sort((x, y) => (rec[y].sw - rec[x].sw) ||
      ((rec[y].mw - rec[y].ml) - (rec[x].mw - rec[x].ml)) || (byId[y].power - byId[x].power));
    const upStand = stand(upper), aStand = stand(lowA), zStand = stand(lowZ);
    const upPlay = playoff(upStand[0], upStand[1], upStand[2], upStand[3], "upper", dayRef);
    const loPlay = playoff(aStand[0], zStand[0], aStand[1], zStand[1], "lower", dayRef);
    const upperChamp = upPlay.champ, lowerChamp = loPlay.champ;
    const relegated = [upStand[8], upStand[9]];
    const promoted = [lowerChamp, loPlay.finalists.find((x) => x !== lowerChamp) || loPlay.finalists[0]];

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

    const managerComments = teams.map((t, i) => {
      const rp = byId[t.team_id].roster; const star = rp.length ? pById[rp[0]].nickname : "лидер";
      const rivals = teams.filter((x) => x.league === t.league && x.team_id !== t.team_id);
      const rival = rivals.length ? rivals[0].name : "соперники";
      return { season: s, team_id: t.team_id, team: t.name, manager: `Менеджер ${t.abbrev}`,
        text: MGR_T[(i + s) % MGR_T.length].replace(/\{s\}/g, s).replace(/\{p\}/g, star).replace(/\{rival\}/g, rival) };
    });
    // off-season transactions digest (hardcoded demo — every category populated)
    const poolFA = freeAgents.slice();
    const txAuctions = auctionWinners.map((t) => ({ team_id: t.team_id, team: t.name, price: ri(60, 140) }));
    const txTrains = trainPicks.map((tp) => ({ team_id: tp.team, team: byId[tp.team].name,
      op: tp.op, op_title: tp.op, player: tp.player, steps: 1000000 }));
    const txSign = shuf(teams).slice(0, 4 + s).map((t, i) => { const fa = poolFA[i % poolFA.length];
      return { team_id: t.team_id, team: t.name, player_id: fa.player_id, player: fa.nickname,
        bonus: ri(30, 90), kind: i % 4 === 0 ? "rookie" : "free_agent" }; });
    const txBuys = shuf(teams).slice(0, 3).map((t, i) => { const from = teams[(teams.indexOf(t) + 5) % teams.length];
      const rp = byId[from.team_id].roster; const who = rp.length ? pById[rp[rp.length - 1]] : pById[players[i].player_id];
      return { player_id: who.player_id, player: who.nickname, from_team_id: from.team_id, from_team: from.name,
        to_team_id: t.team_id, to_team: t.name, price: ri(120, 480) }; });
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

    snaps.push({ num: s, teamSnap, upStand, aStand, zStand, fixtures, bracket: { upper: upPlay.cols, lower: loPlay.cols },
      champ: { upper_champion: upperChamp, lower_champion: lowerChamp, promoted, relegated },
      commentary, chronicle, training, managerComments, transactions,
      draft: s === 0 ? { season: 0, kind: "bootstrap", picks: draftPicks.slice() } : null });

    // roll over
    byId[upperChamp].titles++; byId[upperChamp].trophies.push({ season: s, title: "Чемпион верхней" });
    byId[lowerChamp].trophies.push({ season: s, title: "Чемпион нижней" });
    relegated.forEach((id) => { byId[id].league = "lower"; });
    promoted.forEach((id) => { byId[id].league = "upper"; byId[id].group = null; });
    let gi = 0; teams.filter((t) => t.league === "lower").forEach((t) => { t.group = gi++ % 2 === 0 ? "A" : "Z"; });
    teams.filter((t) => t.league === "upper").forEach((t) => { t.group = null; });
    teams.forEach((t) => { const r = rec[t.team_id]; const d = (r.sw - r.sl) * ri(5, 12) + ri(-20, 20);
      t.power = Math.max(880, t.power + d); t.eloHist.push(Math.round(t.power)); t.certs = 0;
      t.treasury = Math.max(120, t.treasury + ri(-120, 260)); });
    players.forEach((p) => { p.ratingHist.push(Math.max(880, p.ratingHist[p.ratingHist.length - 1] + ri(-45, 55))); });
    if (s === 0) shuf(teams).slice(0, 9).forEach((t) => { const fa = freeAgents.find((p) => !p.teamId);
      if (fa && t.roster.length < 5) { fa.teamId = t.team_id; t.roster.push(fa.player_id); } });
  }
  // SoloQ per-game history (player detail)
  players.forEach((p) => {
    for (let g = 0; g < ri(6, 12); g++) {
      const grp = shuf(players.filter((q) => q !== p)).slice(0, 5); const won = R() < 0.5;
      p.soloq.push({ season: ri(0, 2), round: ri(0, 9), won, delta: +(won ? ri(8, 24) : -ri(8, 22)),
        teammates: grp.slice(0, 2).map((q) => q.nickname), opponents: grp.slice(2, 5).map((q) => q.nickname) });
    }
  });

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
    const ids = Object.keys(snap().teamSnap);
    const f = (lg, grp) => ids.filter((id) => snap().teamSnap[id].league === lg &&
      (grp === undefined || snap().teamSnap[id].group === grp))
      .sort((a, b) => snap().teamSnap[a].place - snap().teamSnap[b].place)
      .map((id, i) => ({ ...teamRow(id), place: i + 1 }));
    return { upper: f("upper"), lower_A: f("lower", "A"), lower_Z: f("lower", "Z") };
  }
  function matrixView() {
    const lv = leaguesView();
    const grid = (teamsRows, label) => {
      const ids = teamsRows.map((t) => t.team_id);
      const idset = new Set(ids);
      const h2h = {};
      snap().fixtures.forEach((e) => {
        if (e.stage !== "league" || !idset.has(e.home) || !idset.has(e.away)) return;
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
  function playerView(p) {
    const hist = p.ratingHist.slice(0, CUR + 2).map(Math.round);
    const delta = hist.length >= 2 ? +(hist[hist.length - 1] - hist[hist.length - 2]).toFixed(1) : 0;
    const team = p.teamId ? tname(p.teamId)
      : (p.reserve ? "Резерв" : (p.role === "streamer" ? "Стример" : "Свободный агент"));
    const pos = {}; Object.entries(p.posStats).forEach(([k, v]) =>
      pos[k] = { games: v.games, wins: v.wins, winrate: v.games ? +(v.wins / v.games).toFixed(3) : null });
    return { player_id: p.player_id, nickname: p.nickname, faction: p.faction, archetype: p.archetype,
      role: p.role, status: p.reserve ? "reserve" : (p.teamId ? "rostered" : "free_agent"),
      rating: hist[hist.length - 1], delta, rating_history: hist, net_id: p.net_id, team_id: p.teamId,
      team, team_abbrev: p.teamId ? tabbr(p.teamId) : null, games: p.games, generation: p.gen,
      position_stats: pos, soloq_games: p.soloq.slice(0, 20), trophies: p.trophies };
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
  function teamView(id) {
    const ts = snap().teamSnap[id];
    return { team_id: id, name: tname(id), abbrev: tabbr(id), manager: "Менеджер", league: ts.league,
      group: ts.group, power: Math.round(ts.power), titles: ts.titles, treasury: Math.round(ts.treasury),
      elo_history: ts.eloHist, trophies: ts.trophies, certificates: ts.certs,
      roster: ts.roster.map((pid) => playerView(pById[pid])),
      record: { series_wins: ts.rec.sw, series_losses: ts.rec.sl, maps_won: ts.rec.mw, maps_lost: ts.rec.ml } };
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
    const history = snaps.slice(0, CUR).map((s) => ({ season: s.num,
      upper_champion: tname(s.champ.upper_champion), lower_champion: tname(s.champ.lower_champion) }));
    const commentary = [];
    snaps.slice(0, CUR + 1).forEach((s) => { commentary.push(s.commentary[0]); if (s.num < CUR) commentary.push(s.commentary[1]); });
    const offSnap = snaps[Math.max(0, CUR - 1)] || snap();
    const managers = offSnap.managerComments || [];
    const livePhase = LIVE ? (LIVE.phase === "done" ? "offseason_due" : "live") : PHASE;
    return { world_id: WID, name: "Demo World", season: CUR, phase: livePhase,
      upper: [], lower: [], ladder: [], history, reserves, commentary, emblems: EMBLEMS,
      manager_comments: CUR > 0 ? managers : [],
      live: LIVE ? lvView() : null,
      last_offseason: CUR > 0 ? { season_ended: offSnap.num, transactions: offSnap.transactions } : null,
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
    return { home, away, maps_home: mh, maps_away: ma, winner: mh > ma ? home : away,
      best_of: bo, sequence: seq, stage: "league" };
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
  function lvResolve(ref, nodes) {
    if (typeof ref === "string") return ref;
    const n = nodes[ref.w || ref.l], res = n && n.result; if (!res) return null;
    return ("w" in ref) ? res.winner : (res.winner === res.home ? res.away : res.home);
  }
  const lvReady = (br) => br.order.filter((k) => { const n = br.nodes[k];
    return !n.result && lvResolve(n.home, br.nodes) && lvResolve(n.away, br.nodes); });
  const lvDone = (br) => br.order.every((k) => br.nodes[k].result);
  function lvBracketPub(br) {
    if (!br) return [];
    return br.order.map((k) => { const n = br.nodes[k], h = lvResolve(n.home, br.nodes),
      a = lvResolve(n.away, br.nodes), res = n.result;
      return { key: k, stage: n.stage,
        home: h ? { team_id: h, abbrev: tabbr(h), name: tname(h) } : null,
        away: a ? { team_id: a, abbrev: tabbr(a), name: tname(a) } : null,
        played: !!res, maps_home: res ? res.maps_home : null,
        maps_away: res ? res.maps_away : null, winner: res ? res.winner : null }; });
  }
  function lvStart() {
    const up = teams.filter((t) => t.league === "upper").map((t) => t.team_id);
    const a = teams.filter((t) => t.league === "lower" && t.group === "A").map((t) => t.team_id);
    const z = teams.filter((t) => t.league === "lower" && t.group === "Z").map((t) => t.team_id);
    const rounds = { upper: lvRounds(up), lower_A: lvRounds(a), lower_Z: lvRounds(z) };
    const nreg = Math.max(rounds.upper.length, rounds.lower_A.length, rounds.lower_Z.length);
    LIVE = { season: CUR, phase: "regular", day: 1, nreg, rounds, cursor: 0,
      groups: { upper: up, lower_A: a, lower_Z: z },
      results: { upper: [], lower_A: [], lower_Z: [] }, playoff: null, done: null };
    return { live: lvView() };
  }
  function lvSeedPlayoffs() {
    const us = lvStand(LIVE.groups.upper, LIVE.results.upper).map((r) => r.team_id);
    const as = lvStand(LIVE.groups.lower_A, LIVE.results.lower_A).map((r) => r.team_id);
    const zs = lvStand(LIVE.groups.lower_Z, LIVE.results.lower_Z).map((r) => r.team_id);
    LIVE.playoff = {
      upper: { order: ["ub", "lb", "lbf", "gf"], nodes: {
        ub: { stage: "ub", home: us[0], away: us[1] }, lb: { stage: "lb", home: us[2], away: us[3] },
        lbf: { stage: "lb_final", home: { w: "lb" }, away: { l: "ub" } },
        gf: { stage: "grand_final", home: { w: "ub" }, away: { w: "lbf" } } } },
      lower: { order: ["ub", "lb1a", "lb1b", "lb2", "lbf", "gf"], nodes: {
        ub: { stage: "ub", home: as[0], away: zs[0] }, lb1a: { stage: "lb_r1", home: as[1], away: zs[2] },
        lb1b: { stage: "lb_r1", home: zs[1], away: as[2] }, lb2: { stage: "lb_r2", home: { w: "lb1a" }, away: { w: "lb1b" } },
        lbf: { stage: "lb_final", home: { w: "lb2" }, away: { l: "ub" } },
        gf: { stage: "grand_final", home: { w: "ub" }, away: { w: "lbf" } } } } };
    LIVE.phase = "playoff";
  }
  function lvDay() {
    if (!LIVE || LIVE.phase === "done") throw new Error("нет активного сезона");
    const played = [];
    if (LIVE.phase === "regular") {
      const rnd = LIVE.cursor;
      ["upper", "lower_A", "lower_Z"].forEach((gk) => {
        const rounds = LIVE.rounds[gk]; if (rnd >= rounds.length) return;
        const done = new Set(LIVE.results[gk].map((s) => [s.home, s.away].sort().join("|")));
        rounds[rnd].forEach(([h, a]) => {
          if (done.has([h, a].sort().join("|"))) return;
          const s = lvPlay(h, a, 3); LIVE.results[gk].push(s);
          played.push({ group: gk, home: h, away: a, home_abbrev: tabbr(h), away_abbrev: tabbr(a),
            maps_home: s.maps_home, maps_away: s.maps_away, winner: s.winner });
        });
      });
      LIVE.cursor++; const day = LIVE.day; LIVE.day++;
      if (LIVE.cursor >= LIVE.nreg) lvSeedPlayoffs();
      return { result: { day, phase: "regular", played }, live: lvView() };
    }
    [["upper", LIVE.playoff.upper], ["lower", LIVE.playoff.lower]].forEach(([lg, br]) => {
      lvReady(br).forEach((k) => { const n = br.nodes[k];
        const h = lvResolve(n.home, br.nodes), a = lvResolve(n.away, br.nodes);
        const s = lvPlay(h, a, 5); s.stage = n.stage; n.result = s;
        played.push({ league: lg, stage: n.stage, home: h, away: a, home_abbrev: tabbr(h),
          away_abbrev: tabbr(a), maps_home: s.maps_home, maps_away: s.maps_away, winner: s.winner });
      });
    });
    const day = LIVE.day; LIVE.day++;
    if (lvDone(LIVE.playoff.upper) && lvDone(LIVE.playoff.lower)) {
      LIVE.phase = "done";
      LIVE.done = { season: LIVE.season, upper_champion: LIVE.playoff.upper.nodes.gf.result.winner,
        lower_champion: LIVE.playoff.lower.nodes.gf.result.winner };
      PHASE = "offseason_due";
      return { result: { day, phase: "playoff", played, summary: LIVE.done }, live: lvView() };
    }
    return { result: { day, phase: "playoff", played }, live: lvView() };
  }
  function lvNextFixtures() {
    const out = [];
    if (!LIVE || LIVE.phase === "done") return out;
    if (LIVE.phase === "regular") {
      const rnd = LIVE.cursor;
      ["upper", "lower_A", "lower_Z"].forEach((gk) => { const rounds = LIVE.rounds[gk];
        if (rnd < rounds.length) rounds[rnd].forEach(([h, a]) =>
          out.push({ group: gk, stage: "league", best_of: 3, home: h, away: a,
            home_abbrev: tabbr(h), away_abbrev: tabbr(a) })); });
    } else {
      [["upper", LIVE.playoff.upper], ["lower", LIVE.playoff.lower]].forEach(([lg, br]) =>
        lvReady(br).forEach((k) => { const n = br.nodes[k];
          const h = lvResolve(n.home, br.nodes), a = lvResolve(n.away, br.nodes);
          out.push({ league: lg, stage: n.stage, best_of: 5, home: h, away: a,
            home_abbrev: tabbr(h), away_abbrev: tabbr(a) }); }));
    }
    return out;
  }
  function lvWatch(home, away) {
    const fx = lvNextFixtures().find((f) => f.home === home && f.away === away);
    if (!fx) throw new Error("это не ближайший матч");
    const s = lvPlay(home, away, fx.best_of); s.stage = fx.stage;
    if (LIVE.phase === "regular") { LIVE.results[fx.group].push(s); }
    else { const br = LIVE.playoff[fx.league];
      const k = lvReady(br).find((kk) => lvResolve(br.nodes[kk].home, br.nodes) === home &&
        lvResolve(br.nodes[kk].away, br.nodes) === away);
      if (k) br.nodes[k].result = s; }
    return { result: { home, away, home_abbrev: tabbr(home), away_abbrev: tabbr(away),
      maps_home: s.maps_home, maps_away: s.maps_away, winner: s.winner, best_of: s.best_of,
      sequence: s.sequence }, live: lvView() };
  }
  function lvView() {
    if (!LIVE) return { active: false };
    const st = {};
    ["upper", "lower_A", "lower_Z"].forEach((k) => (st[k] = lvStand(LIVE.groups[k], LIVE.results[k]).map(lvRow)));
    return { active: LIVE.phase !== "done", season: LIVE.season, phase: LIVE.phase,
      day: LIVE.day, total_days: LIVE.nreg + 4, n_regular_days: LIVE.nreg,
      next_fixtures: lvNextFixtures(), standings: st,
      bracket: { upper: lvBracketPub((LIVE.playoff || {}).upper), lower: lvBracketPub((LIVE.playoff || {}).lower) },
      done: LIVE.done };
  }

  window.__PREVIEW__ = async function (url, opts) {
    await sleep(160);
    const method = (opts && opts.method) || "GET";
    url = url.split("?")[0];
    if (url === "/api/world/info") return { ui: { locked: false, note: "" }, worlds: ["world-preview"] };
    if (url === "/api/world/franchises") {
      const TEMPO = ["balanced", "aggressive", "defensive"], WP = ["economy", "core_rush", "map_control"];
      return { franchises: teams.map((t, i) => ({ franchise_id: t.franchise_id, name: t.name, abbrev: t.abbrev,
        color: t.color, accent: t.color, manager: `Менеджер ${t.abbrev}`,
        persona: `Клуб «${t.name}» — ${pk(["амбициозный", "хладнокровный", "дерзкий", "расчётливый"])}, ставит на ${pk(["молодёжь", "опыт", "агрессию", "экономику"])}.`,
        default_plan: { tempo: TEMPO[i % 3], win_path: WP[i % 3] }, emblem: "" })) };
    }
    if (url === "/api/world/players-pool") return playersPool();
    if (url.startsWith("/api/world/job/")) return { status: "done", world_id: WID, phase: "Готово", summary: lastSummary };
    if (method === "POST" && url === "/api/world/create") { CUR = 0; lastSummary = {}; return { job_id: "cr", status: "running" }; }
    if (method === "POST" && url.endsWith("/live/start")) return lvStart();
    if (method === "POST" && url.endsWith("/live/day")) return lvDay();
    if (method === "POST" && url.endsWith("/live/watch")) {
      const b = opts && opts.body ? JSON.parse(opts.body) : {};
      return lvWatch(b.home, b.away);
    }
    if (method === "POST" && url.endsWith("/advance")) return doAdvance();
    if (method === "POST" && url.endsWith("/offseason")) return doOffseason();
    if (method === "POST" && url.endsWith("/introduce")) return { introduced: 0, reserves_left: players.filter((p) => p.reserve).length };
    const path = url.replace(/^\/api\/world\/[^/]+/, "");
    if (path === "") return { world: overview() };       // real /api/world/<id> wraps in {world}
    if (path === "/leagues") return { season: CUR, leagues: leaguesView() };
    if (path === "/live") return { live: LIVE ? lvView() : { active: false } };
    if (path === "/matrix") return matrixView();
    if (path === "/players") return { season: CUR, players: rankedPlayers() };
    if (path === "/schedule") return { season: CUR, regular: snap().fixtures, bracket: snap().bracket, n_matches: snap().fixtures.length };
    if (path === "/draft") return draftView();
    if (path === "/training") return { jobs: showTraining ? snaps[Math.max(0, CUR - 1)].training : [] };
    if (path.startsWith("/team/")) return { season: CUR, team: teamView(path.slice("/team/".length)) };
    if (path.startsWith("/player/")) return { player: playerPage(path.slice("/player/".length)) };
    return {};
  };
})();
