// League front-end: a small hash-routed SPA on top of the league API.
//
// Routes (hash):
//   #/                          lobby (list + create entry)
//   #/create                    team builder + season generation
//   #/league/<id>               standings (main page) + day controls
//   #/league/<id>/history       revealed match history
//   #/league/<id>/enc/<index>   encounter detail (games, stats, recordings)
//
// The whole season is simulated on the Python server at creation time; the UI
// only ever *reveals* already-computed results, gated by the league's day.

const root = document.getElementById("leagueApp");

const FACTION_LABELS = { humans: "Humans", infected: "Infected", robots: "Robots" };
const RESULT_LABEL = { home: "Home win", away: "Away win", draw: "Draw" };

const store = {
  meta: null, // { templates, models, factions }
};

// --------------------------------------------------------------------------- //
// Tiny DOM helper
// --------------------------------------------------------------------------- //
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      /* keep status */
    }
    throw new Error(message);
  }
  return response.json();
}

// --------------------------------------------------------------------------- //
// Routing
// --------------------------------------------------------------------------- //
function parseRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean);
  if (!parts.length) return { view: "lobby" };
  if (parts[0] === "create") return { view: "create" };
  if (parts[0] === "league" && parts[1]) {
    const id = decodeURIComponent(parts[1]);
    if (parts[2] === "history") return { view: "history", id };
    if (parts[2] === "enc" && parts[3] != null) return { view: "encounter", id, enc: Number(parts[3]) };
    return { view: "league", id };
  }
  return { view: "lobby" };
}

function navigate(hash) {
  if (location.hash === hash) {
    render();
  } else {
    location.hash = hash;
  }
}

function topbar(title, sub, backHash, extra = []) {
  const back = backHash === "menu"
    ? el("a", { class: "btn ghost", href: "./home.html" }, "← Menu")
    : el("button", { class: "btn ghost", onClick: () => navigate(backHash) }, "← Back");
  return el("div", { class: "lg-topbar" }, [
    back,
    el("div", {}, [el("h1", {}, title), sub ? el("div", { class: "lg-sub" }, sub) : null]),
    el("div", { class: "lg-spacer" }),
    ...extra,
  ]);
}

function showLoading(message = "Loading…") {
  root.replaceChildren(el("div", { class: "league-loading" }, message));
}

function showError(message, backHash = "#/") {
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Error", null, backHash),
    el("div", { class: "panel error-text" }, message),
  ]));
}

// --------------------------------------------------------------------------- //
// Models helper (mirror the viewer's "latest only" filter)
// --------------------------------------------------------------------------- //
function visibleModels() {
  const models = store.meta?.models || [];
  const latest = models.filter((m) => (
    m.kind === "baseline" || m.show_by_default === true || (m.show_by_default !== false && m.is_latest)
  ));
  return latest.length ? latest : models;
}

function defaultModelId() {
  const models = visibleModels();
  return models.find((m) => m.id === "scripted")?.id || models[0]?.id || "random";
}

function modelOptions(selectedId) {
  const select = el("select", {});
  for (const model of visibleModels()) {
    const option = el("option", { value: model.id }, model.label || model.id);
    if (model.id === selectedId) option.selected = true;
    select.append(option);
  }
  return select;
}

function factionOptions(selectedFaction) {
  const select = el("select", {});
  for (const faction of store.meta?.factions || ["humans", "infected", "robots"]) {
    const option = el("option", { value: faction }, FACTION_LABELS[faction] || faction);
    if (faction === selectedFaction) option.selected = true;
    select.append(option);
  }
  return select;
}

// --------------------------------------------------------------------------- //
// Views
// --------------------------------------------------------------------------- //
async function renderLobby() {
  showLoading("Loading leagues…");
  let data;
  try {
    data = await fetchJson("/api/league/list");
  } catch (error) {
    showError(`Could not reach the league API: ${error.message}. Start the local viewer server.`, "menu");
    return;
  }
  const create = el("button", { class: "btn primary", onClick: () => navigate("#/create") }, "+ Create league");
  const list = el("div", { class: "lg-list" });
  if (!data.leagues.length) {
    list.append(el("div", { class: "empty" }, "No leagues yet. Create one to simulate a full season."));
  }
  for (const item of data.leagues) {
    list.append(el("div", {
      class: "lg-list-item",
      onClick: () => navigate(`#/league/${encodeURIComponent(item.id)}`),
    }, [
      el("div", { class: "lg-spacer-none" }, [
        el("div", { class: "name" }, item.name),
        el("div", { class: "meta" }, `${item.template || "?"} · ${item.team_count} teams · day ${item.current_day}/${item.total_days}`),
      ]),
      el("div", { class: "lg-spacer" }),
      el("span", { class: "btn ghost" }, "Open →"),
    ]));
  }
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Leagues", "Round-robin Bo3 seasons", "menu", [create]),
    el("div", { class: "panel" }, [list]),
  ]));
}

function renderCreate() {
  const template = (store.meta?.templates || [])[0];
  if (!template) {
    showError("No league templates available.", "#/");
    return;
  }
  const teamCount = template.team_count;
  const perTeam = template.players_per_team;
  const defModel = defaultModelId();

  const nameInput = el("input", { type: "text", value: "My League", id: "lgName" });
  const templateSelect = el("select", { id: "lgTemplate" });
  for (const t of store.meta.templates) {
    templateSelect.append(el("option", { value: t.id }, `${t.name} — ${t.team_count}×${t.players_per_team}, Bo${t.best_of}`));
  }
  const seedInput = el("input", { type: "number", value: "0", id: "lgSeed", title: "0 = random each time" });

  const teamsGrid = el("div", { class: "teams-grid", id: "lgTeams" });
  for (let t = 0; t < teamCount; t += 1) {
    const playerRows = [];
    for (let p = 0; p < perTeam; p += 1) {
      playerRows.push(el("div", { class: "player-row" }, [
        el("input", { type: "text", value: `P${t + 1}-${p + 1}`, dataset: { field: "pname" } }),
        modelOptions(defModel),
        factionOptions("humans"),
      ]));
    }
    const quickModel = modelOptions(defModel);
    const quickFaction = factionOptions("humans");
    const card = el("div", { class: "team-card", dataset: { team: String(t) } }, [
      el("div", { class: "team-head" }, [
        el("span", { class: "muted" }, `#${t + 1}`),
        el("input", { type: "text", value: `Team ${t + 1}`, dataset: { field: "tname" } }),
      ]),
      ...playerRows,
      el("div", { class: "team-quick" }, [
        quickModel, quickFaction,
        el("button", {
          class: "btn ghost", onClick: () => {
            for (const row of card.querySelectorAll(".player-row")) {
              row.querySelectorAll("select")[0].value = quickModel.value;
              row.querySelectorAll("select")[1].value = quickFaction.value;
            }
          },
        }, "Apply to team"),
      ]),
    ]);
    teamsGrid.append(card);
  }

  const status = el("div", { class: "progress-wrap", id: "lgStatus" });
  const generate = el("button", { class: "btn primary", id: "lgGenerate", onClick: () => submitCreate(status, generate) },
    `Generate season (${teamCount} teams, Bo${template.best_of})`);

  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Create league", "All matches are simulated on the server up front", "#/"),
    el("div", { class: "panel" }, [
      el("div", { class: "create-grid" }, [
        el("div", { class: "field" }, [el("label", {}, "League name"), nameInput]),
        el("div", { class: "field" }, [el("label", {}, "Template"), templateSelect]),
        el("div", { class: "field" }, [el("label", {}, "Seed (0 = random)"), seedInput]),
      ]),
    ]),
    el("div", { class: "panel" }, [
      el("h2", {}, "Teams & players"),
      el("div", { class: "muted", style: "margin-bottom:12px;font-size:12.5px" },
        "Assign a model and faction to each player. (Manual now — built to be driven by an LLM manager later.)"),
      teamsGrid,
    ]),
    el("div", { class: "panel" }, [generate, status]),
  ]));
}

function collectTeams() {
  const teams = [];
  for (const card of document.querySelectorAll("#lgTeams .team-card")) {
    const name = card.querySelector('input[data-field="tname"]').value;
    const players = [];
    for (const row of card.querySelectorAll(".player-row")) {
      const pname = row.querySelector('input[data-field="pname"]').value;
      const selects = row.querySelectorAll("select");
      players.push({ name: pname, model: selects[0].value, faction: selects[1].value });
    }
    teams.push({ name, players });
  }
  return teams;
}

async function submitCreate(status, button) {
  const name = document.getElementById("lgName").value || "League";
  const template = document.getElementById("lgTemplate").value;
  const seed = Number(document.getElementById("lgSeed").value || 0);
  const teams = collectTeams();

  button.disabled = true;
  status.replaceChildren(
    el("div", { class: "progress-bar" }, [el("div", { class: "progress-fill", id: "lgFill" })]),
    el("div", { class: "progress-label", id: "lgLabel" }, "Starting season simulation…"),
  );

  let job;
  try {
    job = await fetchJson("/api/league/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, template, seed, teams }),
    });
  } catch (error) {
    button.disabled = false;
    status.append(el("div", { class: "error-text" }, error.message));
    return;
  }
  pollJob(job.job_id, status, button);
}

function pollJob(jobId, status, button) {
  const fill = document.getElementById("lgFill");
  const label = document.getElementById("lgLabel");
  const tick = async () => {
    let job;
    try {
      job = await fetchJson(`/api/league/job/${jobId}`);
    } catch (error) {
      button.disabled = false;
      status.append(el("div", { class: "error-text" }, error.message));
      return;
    }
    const total = job.total || 0;
    const done = job.done || 0;
    const pct = total ? Math.round((done / total) * 100) : 5;
    if (fill) fill.style.width = `${pct}%`;
    if (label) label.textContent = `Simulating encounters… ${done}/${total || "?"} (${pct}%)`;
    if (job.status === "done") {
      if (label) label.textContent = "Done! Opening league…";
      navigate(`#/league/${encodeURIComponent(job.league_id)}`);
      return;
    }
    if (job.status === "error") {
      button.disabled = false;
      status.append(el("div", { class: "error-text" }, job.error || "Generation failed"));
      return;
    }
    setTimeout(tick, 1200);
  };
  tick();
}

async function renderLeagueStandings(id) {
  showLoading("Loading standings…");
  let league;
  try {
    league = (await fetchJson(`/api/league/${encodeURIComponent(id)}`)).league;
  } catch (error) {
    showError(error.message, "#/");
    return;
  }
  const atEnd = league.current_day >= league.total_days;
  const advance = el("button", {
    class: "btn primary", disabled: atEnd ? "" : false,
    onClick: () => advanceDay(id),
  }, atEnd ? "Season complete" : "▶ Play next day");
  const history = el("button", { class: "btn", onClick: () => navigate(`#/league/${encodeURIComponent(id)}/history`) }, "History");

  const table = el("table", { class: "lg-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "rank" }, "#"),
      el("th", { class: "team" }, "Team"),
      el("th", {}, "P"), el("th", {}, "W"), el("th", {}, "D"), el("th", {}, "L"),
      el("th", {}, "GW"), el("th", {}, "GL"), el("th", { class: "points" }, "Pts"),
    ])),
    el("tbody", {}, league.standings.map((row) => el("tr", {}, [
      el("td", { class: "rank" }, row.rank),
      el("td", { class: "team" }, row.team_name),
      el("td", {}, row.played), el("td", {}, row.won), el("td", {}, row.drawn), el("td", {}, row.lost),
      el("td", {}, row.game_wins), el("td", {}, row.game_losses),
      el("td", { class: "points" }, row.points),
    ]))),
  ]);

  const dayInfo = el("div", { class: "lg-sub" }, `Day ${league.current_day} / ${league.total_days}`);
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar(league.name, league.template?.name || "", "#/", [history, advance]),
    el("div", { class: "panel" }, [
      el("div", { style: "display:flex;align-items:center;gap:12px;margin-bottom:12px" }, [
        el("h2", { style: "margin:0" }, "Standings"), el("div", { class: "lg-spacer" }), dayInfo,
      ]),
      el("div", { class: "table-scroll" }, [table]),
      league.current_day === 0
        ? el("div", { class: "muted", style: "margin-top:12px;font-size:12.5px" },
            "No matches revealed yet. Press “Play next day” to reveal the first encounter.")
        : null,
    ]),
  ]));
}

async function advanceDay(id) {
  try {
    await fetchJson(`/api/league/${encodeURIComponent(id)}/advance`, { method: "POST" });
  } catch (error) {
    showError(error.message, `#/league/${encodeURIComponent(id)}`);
    return;
  }
  renderLeagueStandings(id);
}

async function renderHistory(id) {
  showLoading("Loading history…");
  let league;
  try {
    league = (await fetchJson(`/api/league/${encodeURIComponent(id)}`)).league;
  } catch (error) {
    showError(error.message, "#/");
    return;
  }
  const list = el("div", { class: "match-list" });
  if (!league.history.length) {
    list.append(el("div", { class: "empty" }, "No matches revealed yet. Advance the day on the standings page."));
  }
  for (const enc of league.history) {
    const pillClass = enc.result === "home" ? "home" : enc.result === "away" ? "away" : "draw";
    list.append(el("div", {
      class: "match-row",
      onClick: () => navigate(`#/league/${encodeURIComponent(id)}/enc/${enc.encounter_index}`),
    }, [
      el("div", { class: "day" }, `Day ${enc.day}`),
      el("div", { class: "teams" }, [
        el("span", {}, enc.home_name), el("span", { class: "vs" }, "vs"), el("span", {}, enc.away_name),
      ]),
      el("div", { class: "score" }, `${enc.home_game_wins}–${enc.away_game_wins}`),
      el("span", { class: `pill ${pillClass}` }, RESULT_LABEL[enc.result] || enc.result),
    ]));
  }
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar(`${league.name} — History`, `Day ${league.current_day}/${league.total_days}`, `#/league/${encodeURIComponent(id)}`),
    el("div", { class: "panel" }, [list]),
  ]));
}

async function renderEncounter(id, encIndex) {
  showLoading("Loading match…");
  let detail;
  try {
    detail = (await fetchJson(`/api/league/${encodeURIComponent(id)}/encounter/${encIndex}`)).encounter;
  } catch (error) {
    showError(error.message, `#/league/${encodeURIComponent(id)}/history`);
    return;
  }

  const resultClass = detail.result === "home" ? "home" : detail.result === "away" ? "away" : "draw";
  const header = el("div", {}, [
    el("div", { class: "enc-head" }, [
      el("span", { class: "side home" }, detail.home_name),
      el("span", { class: "series-score" }, `${detail.home_game_wins} : ${detail.away_game_wins}`),
      el("span", { class: "side away" }, detail.away_name),
    ]),
    el("div", { class: "enc-sub" }, [
      el("span", { class: `pill ${resultClass}` }, RESULT_LABEL[detail.result] || detail.result),
      `  ·  Day ${detail.day}  ·  ${detail.games.length} game(s) played`,
    ]),
  ]);

  const body = el("div", { class: "panel" });
  const tabs = el("div", { class: "game-tabs" });
  const tabDefs = [{ key: "series", label: "Whole encounter" }].concat(
    detail.games.map((g, i) => ({ key: String(i), label: `Game ${i + 1}`, game: g })),
  );
  const renderTab = (key) => {
    for (const child of tabs.children) child.classList.toggle("active", child.dataset.key === key);
    if (key === "series") body.replaceChildren(renderSeriesView(detail));
    else body.replaceChildren(renderGameView(detail, Number(key), id, encIndex));
  };
  for (const def of tabDefs) {
    const dot = def.game ? el("span", { class: "dot" }, def.game.winner === "home" ? "◀" : def.game.winner === "away" ? "▶" : "=") : null;
    tabs.append(el("div", { class: "game-tab", dataset: { key: def.key }, onClick: () => renderTab(def.key) }, [def.label, dot]));
  }

  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Match detail", null, `#/league/${encodeURIComponent(id)}/history`),
    el("div", { class: "panel" }, [header]),
    tabs,
    body,
  ]));
  renderTab("series");
}

function renderSeriesView(detail) {
  // Aggregate per-slot stats across games of the encounter.
  const agg = {};
  for (const game of detail.games) {
    for (const [slot, st] of Object.entries(game.players)) {
      const a = agg[slot] || (agg[slot] = { ...st, core_hp: 0, core_damage_dealt: 0, alive_games: 0, games: 0 });
      a.core_hp += st.core_hp;
      a.core_damage_dealt += st.core_damage_dealt;
      a.alive_games += st.alive ? 1 : 0;
      a.games += 1;
    }
  }
  const gamesCount = detail.games.length || 1;
  const rows = Object.values(agg).map((a) => el("tr", {}, [
    el("td", { class: "player" }, `${a.player_name} (${a.slot})`),
    el("td", {}, `${a.team_name}`),
    el("td", {}, modelLabelFor(a.model)),
    el("td", {}, FACTION_LABELS[a.faction] || a.faction),
    el("td", {}, (a.core_hp / gamesCount).toFixed(1)),
    el("td", {}, a.core_damage_dealt.toFixed(1)),
    el("td", {}, `${a.alive_games}/${a.games}`),
  ]));
  const table = el("table", { class: "lg-table stat-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "you" }, "Player"), el("th", {}, "Team"), el("th", {}, "Model"),
      el("th", {}, "Faction"), el("th", {}, "Avg core HP"), el("th", {}, "Core dmg Σ"), el("th", {}, "Survived"),
    ])),
    el("tbody", {}, rows),
  ]);
  const gameStrip = el("div", { class: "game-meta" }, detail.games.map((g, i) =>
    el("span", {}, [`Game ${i + 1}: `, el("span", { class: `pill ${g.winner}` }, RESULT_LABEL[g.winner] || g.winner), ` (${g.ticks}t)`]),
  ));
  return el("div", {}, [
    el("h2", {}, "Encounter summary"),
    gameStrip,
    el("div", { class: "table-scroll" }, [table]),
  ]);
}

function renderGameView(detail, gameIndex, id, encIndex) {
  const game = detail.games[gameIndex];
  const watchUrl = buildWatchUrl(game, id, encIndex, `${detail.home_name} vs ${detail.away_name} — Game ${gameIndex + 1}`);
  const meta = el("div", { class: "game-meta" }, [
    el("span", {}, ["Result: ", el("span", { class: `pill ${game.winner}` }, RESULT_LABEL[game.winner] || game.winner)]),
    el("span", {}, [el("strong", {}, String(game.ticks)), " ticks"]),
    el("span", {}, ["Type: ", el("strong", {}, game.winner_type || "—")]),
    el("span", {}, ["Seed: ", el("strong", {}, String(game.seed))]),
    el("a", { class: "btn primary", href: watchUrl }, "▶ Watch recording"),
  ]);

  const slots = ["A1", "A2", "A3", "B1", "B2", "B3"];
  const rows = slots.map((slot) => {
    const st = game.players[slot];
    if (!st) return null;
    return el("tr", {}, [
      el("td", { class: "player" }, `${st.player_name} (${slot})`),
      el("td", {}, [el("span", { class: `pill ${st.side}` }, st.side === "home" ? "Home" : "Away")]),
      el("td", {}, st.team_name),
      el("td", {}, modelLabelFor(st.model)),
      el("td", {}, FACTION_LABELS[st.faction] || st.faction),
      el("td", {}, st.core_hp.toFixed(1)),
      el("td", {}, st.core_damage_dealt.toFixed(1)),
      el("td", { class: st.alive ? "alive" : "dead" }, st.alive ? "Alive" : "Dead"),
    ]);
  });
  const table = el("table", { class: "lg-table stat-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "you" }, "Player"), el("th", {}, "Side"), el("th", {}, "Team"),
      el("th", {}, "Model"), el("th", {}, "Faction"), el("th", {}, "Core HP"), el("th", {}, "Core dmg"), el("th", {}, "Status"),
    ])),
    el("tbody", {}, rows),
  ]);
  return el("div", {}, [
    el("h2", {}, `Game ${gameIndex + 1} details`),
    meta,
    el("div", { class: "table-scroll" }, [table]),
  ]);
}

function modelLabelFor(modelId) {
  const model = (store.meta?.models || []).find((m) => m.id === modelId);
  return model?.label || modelId;
}

function buildWatchUrl(game, id, encIndex, label) {
  const replayPath = `../${game.replay}`; // game.replay is repo-relative (leagues/...)
  const back = `league.html#/league/${encodeURIComponent(id)}/enc/${encIndex}`;
  const params = new URLSearchParams({ replay: replayPath, back, label });
  return `index.html?${params.toString()}`;
}

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //
async function render() {
  const route = parseRoute();
  if (route.view === "lobby") return renderLobby();
  if (route.view === "create") return renderCreate();
  if (route.view === "league") return renderLeagueStandings(route.id);
  if (route.view === "history") return renderHistory(route.id);
  if (route.view === "encounter") return renderEncounter(route.id, route.enc);
  return renderLobby();
}

async function boot() {
  try {
    store.meta = await fetchJson("/api/league/templates");
  } catch (error) {
    showError(`Could not reach the league API: ${error.message}. Start the local viewer server (start_viewer.ps1).`, "menu");
    return;
  }
  window.addEventListener("hashchange", render);
  render();
}

boot();
