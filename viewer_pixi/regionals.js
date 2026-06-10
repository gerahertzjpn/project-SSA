// Regionals front-end: hash-routed SPA on top of the regionals API.
//
// Routes:
//   #/                       lobby (list + create)
//   #/players                global player card registry
//   #/teams                  franchises: money, titles, head-to-head
//   #/r/<id>                 tournament (standings, bracket, commentator, schedule)
//   #/r/<id>/draft           draft recap
//   #/r/<id>/enc/<index>     encounter detail (games, ratings, recordings, commentary)
//
// The whole tournament (draft + league + playoff) is simulated server-side at
// creation; the UI reveals results day by day. Commentary is generated on
// button press only (token economy).

const root = document.getElementById("leagueApp");

const FACTION_LABELS = { humans: "Humans", infected: "Infected", robots: "Robots" };
const RESULT_LABEL = { home: "Победа хозяев", away: "Победа гостей", draw: "Ничья" };
const ATTR_LABELS = [
  ["attack", "Атака"], ["defense", "Оборона"], ["economy", "Экономика"],
  ["tempo", "Темп"], ["consistency", "Стабильн."], ["potential", "Потенциал"],
];

const store = { meta: null };

// --------------------------------------------------------------------------- //
// Helpers
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
    } catch { /* keep status */ }
    throw new Error(message);
  }
  return response.json();
}

function parseRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean);
  if (!parts.length) return { view: "lobby" };
  if (parts[0] === "players") return { view: "players" };
  if (parts[0] === "teams") return { view: "teams" };
  if (parts[0] === "r" && parts[1]) {
    const id = decodeURIComponent(parts[1]);
    if (parts[2] === "draft") return { view: "draft", id };
    if (parts[2] === "enc" && parts[3] != null) return { view: "encounter", id, enc: Number(parts[3]) };
    return { view: "tournament", id };
  }
  return { view: "lobby" };
}

function navigate(hash) {
  if (location.hash === hash) render();
  else location.hash = hash;
}

function topbar(title, sub, backHash, extra = []) {
  const back = backHash === "menu"
    ? el("a", { class: "btn ghost", href: "./home.html" }, "← Меню")
    : el("button", { class: "btn ghost", onClick: () => navigate(backHash) }, "← Назад");
  return el("div", { class: "lg-topbar" }, [
    back,
    el("div", {}, [el("h1", {}, title), sub ? el("div", { class: "lg-sub" }, sub) : null]),
    el("div", { class: "lg-spacer" }),
    ...extra,
  ]);
}

function showLoading(message = "Загрузка…") {
  root.replaceChildren(el("div", { class: "league-loading" }, message));
}

function showError(message, backHash = "#/") {
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Ошибка", null, backHash),
    el("div", { class: "panel error-text" }, message),
  ]));
}

function deltaChip(delta) {
  const value = Number(delta);
  const cls = value >= 0 ? "up" : "down";
  return el("span", { class: `rg-delta ${cls}` }, `${value >= 0 ? "+" : ""}${value.toFixed(1)}`);
}

// --------------------------------------------------------------------------- //
// Lobby
// --------------------------------------------------------------------------- //
async function renderLobby() {
  showLoading("Загрузка турниров…");
  let data;
  try {
    data = await fetchJson("/api/regionals/list");
  } catch (error) {
    showError(`API недоступен: ${error.message}. Запустите локальный viewer-сервер.`, "menu");
    return;
  }

  const templates = store.meta?.templates || [];
  const templateSelect = el("select", { id: "rgTemplate" });
  for (const t of templates) {
    templateSelect.append(el("option", { value: t.id }, t.name));
  }
  const nameInput = el("input", { type: "text", value: "Regionals Cup", id: "rgName" });
  const seedInput = el("input", { type: "number", value: "0", id: "rgSeed", title: "0 = случайный" });
  const status = el("div", { class: "progress-wrap", id: "rgStatus" });
  const createBtn = el("button", { class: "btn primary", onClick: () => submitCreate(status, createBtn) },
    "Запустить Regionals (драфт + сезон)");

  const llmWarning = store.meta?.llm_unavailable
    ? el("div", { class: "rg-warning" },
        `⚠ LLM недоступна (${store.meta.llm_unavailable}). Драфт пройдёт автоматически, комментатор будет молчать. ` +
        "Для полного режима: pip install anthropic и переменная ANTHROPIC_API_KEY.")
    : null;

  const list = el("div", { class: "lg-list" });
  if (!data.regionals.length) {
    list.append(el("div", { class: "empty" }, "Турниров пока нет. Запустите первый Regionals."));
  }
  for (const item of data.regionals) {
    list.append(el("div", {
      class: "lg-list-item",
      onClick: () => navigate(`#/r/${encodeURIComponent(item.id)}`),
    }, [
      el("div", {}, [
        el("div", { class: "name" }, item.name),
        el("div", { class: "meta" },
          `${item.template || "?"} · день ${item.current_day}/${item.total_days}` +
          (item.llm ? " · LLM" : " · авто-драфт")),
      ]),
      el("div", { class: "lg-spacer" }),
      el("span", { class: "btn ghost" }, "Открыть →"),
    ]));
  }

  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Regionals", "Лига 6 команд + плей-офф · LLM-менеджеры и комментатор", "menu", [
      el("button", { class: "btn", onClick: () => navigate("#/players") }, "Игроки"),
      el("button", { class: "btn", onClick: () => navigate("#/teams") }, "Команды"),
    ]),
    llmWarning,
    el("div", { class: "panel" }, [
      el("h2", {}, "Новый турнир"),
      el("div", { class: "create-grid" }, [
        el("div", { class: "field" }, [el("label", {}, "Название"), nameInput]),
        el("div", { class: "field" }, [el("label", {}, "Версия (шаблон)"), templateSelect]),
        el("div", { class: "field" }, [el("label", {}, "Seed (0 = случайный)"), seedInput]),
      ]),
      el("div", { class: "muted", style: "font-size:12.5px;margin:8px 0 12px" },
        "Составы решает драфт: 6 LLM-менеджеров с разными характерами выбирают по 1 игроку в раунд, 3 раунда, порядок пиков случайный и сбалансированный."),
      createBtn,
      status,
    ]),
    el("div", { class: "panel" }, [el("h2", {}, "Турниры"), list]),
  ]));
}

async function submitCreate(status, button) {
  const name = document.getElementById("rgName").value || "Regionals";
  const template = document.getElementById("rgTemplate").value;
  const seed = Number(document.getElementById("rgSeed").value || 0);
  button.disabled = true;
  status.replaceChildren(
    el("div", { class: "progress-bar" }, [el("div", { class: "progress-fill", id: "rgFill" })]),
    el("div", { class: "progress-label", id: "rgLabel" }, "Драфт игроков…"),
  );
  let job;
  try {
    job = await fetchJson("/api/regionals/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, template, seed }),
    });
  } catch (error) {
    button.disabled = false;
    status.append(el("div", { class: "error-text" }, error.message));
    return;
  }
  pollJob(job.job_id, status, button);
}

function pollJob(jobId, status, button) {
  const PHASES = { draft: "Драфт игроков (LLM-менеджеры)…", league: "Симуляция лиги…", playoff: "Плей-офф…" };
  const tick = async () => {
    let job;
    try {
      job = await fetchJson(`/api/regionals/job/${jobId}`);
    } catch (error) {
      button.disabled = false;
      status.append(el("div", { class: "error-text" }, error.message));
      return;
    }
    const fill = document.getElementById("rgFill");
    const label = document.getElementById("rgLabel");
    const total = job.total || 0;
    const done = job.done || 0;
    const pct = total ? Math.round((done / total) * 100) : 5;
    if (fill) fill.style.width = `${pct}%`;
    if (label) label.textContent = `${PHASES[job.phase] || "Подготовка…"} ${done}/${total || "?"}`;
    if (job.status === "done") {
      navigate(`#/r/${encodeURIComponent(job.regionals_id)}`);
      return;
    }
    if (job.status === "error") {
      button.disabled = false;
      status.append(el("div", { class: "error-text" }, job.error || "Ошибка генерации"));
      return;
    }
    setTimeout(tick, 1200);
  };
  tick();
}

// --------------------------------------------------------------------------- //
// Tournament view
// --------------------------------------------------------------------------- //
async function renderTournament(id) {
  showLoading("Загрузка турнира…");
  let rg;
  try {
    rg = (await fetchJson(`/api/regionals/${encodeURIComponent(id)}`)).regionals;
  } catch (error) {
    showError(error.message, "#/");
    return;
  }

  const atEnd = rg.current_day >= rg.total_days;
  const advance = el("button", {
    class: "btn primary", disabled: atEnd ? "" : false,
    onClick: async () => {
      try {
        await fetchJson(`/api/regionals/${encodeURIComponent(id)}/advance`, { method: "POST" });
        renderTournament(id);
      } catch (error) { showError(error.message, `#/r/${encodeURIComponent(id)}`); }
    },
  }, atEnd ? "Турнир завершён" : "▶ Сыграть следующий день");

  // ---- commentator panel ---- //
  const commentNodes = (rg.commentary || []).map((c) => el("div", { class: "rg-comment" }, [
    el("div", { class: "meta" },
      c.kind === "opening" ? "Открытие турнира" :
      c.kind === "final" ? "Итоги турнира" : `Матч, день ${c.day}`),
    c.text,
  ]));
  const finalBtnWrap = el("span", {});
  if (atEnd && !(rg.commentary || []).some((c) => c.kind === "final")) {
    const btn = el("button", {
      class: "btn",
      onClick: async () => {
        btn.disabled = true;
        btn.textContent = "Комментатор пишет…";
        try {
          await fetchJson(`/api/regionals/${encodeURIComponent(id)}/commentary`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "final" }),
          });
          renderTournament(id);
        } catch (error) {
          btn.disabled = false;
          btn.textContent = "🎙 Итоги турнира";
          finalBtnWrap.append(el("div", { class: "error-text" }, error.message));
        }
      },
    }, "🎙 Итоги турнира");
    finalBtnWrap.append(btn);
  }
  const commentator = el("div", { class: "panel" }, [
    el("div", { class: "rg-commentator" }, [
      el("div", { class: "avatar" }, "🎙"),
      el("div", { style: "flex:1" }, [
        el("h2", { style: "margin:0 0 4px" }, "Гарри «Голос» Вольский"),
        el("div", { class: "muted", style: "font-size:12px" },
          "Комментарий к матчу — кнопкой на странице встречи. " +
          (rg.llm?.used ? `Токены: ${rg.llm.usage?.input_tokens || 0} in / ${rg.llm.usage?.output_tokens || 0} out.` : "LLM не использовалась.")),
        ...commentNodes,
        finalBtnWrap,
      ]),
    ]),
  ]);

  // ---- standings ---- //
  const table = el("table", { class: "lg-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "rank" }, "#"), el("th", { class: "team" }, "Команда"),
      el("th", {}, "И"), el("th", {}, "В"), el("th", {}, "Н"), el("th", {}, "П"),
      el("th", {}, "ИВ"), el("th", {}, "ИП"), el("th", { class: "points" }, "Очки"),
    ])),
    el("tbody", {}, rg.standings.map((row) => el("tr", {}, [
      el("td", { class: "rank" }, row.rank),
      el("td", { class: "team" }, row.team_name),
      el("td", {}, row.played), el("td", {}, row.won), el("td", {}, row.drawn), el("td", {}, row.lost),
      el("td", {}, row.game_wins), el("td", {}, row.game_losses),
      el("td", { class: "points" }, row.points),
    ]))),
  ]);

  // ---- bracket ---- //
  const bracketNodes = rg.bracket.map((node) => {
    const clickable = node.revealed;
    const winnerSide = node.result === "home" ? "home" : node.result === "away" ? "away" : null;
    return el("div", {
      class: `rg-bracket-node${clickable ? "" : " locked"}`,
      onClick: clickable ? () => navigate(`#/r/${encodeURIComponent(id)}/enc/${node.encounter_index}`) : null,
    }, [
      el("div", { class: "stage" }, `${node.stage_label} · Bo${rg.template.playoff.best_of} · день ${node.day}`),
      el("div", { class: "pair" }, [
        el("span", { class: winnerSide === "home" ? "winner" : "" }, node.home_name),
        el("span", { class: "score" }, node.revealed ? `${node.home_game_wins}:${node.away_game_wins}` : "—"),
      ]),
      el("div", { class: "pair" }, [
        el("span", { class: winnerSide === "away" ? "winner" : "" }, node.away_name),
        el("span", { class: "score" }, ""),
      ]),
    ]);
  });
  const champion = rg.finished && rg.champion
    ? el("div", { class: "rg-champion" }, `🏆 Чемпион: ${rg.champion.team_name}`)
    : null;
  const prizeRows = rg.finished && rg.placements
    ? el("div", { class: "table-scroll", style: "margin-top:10px" }, [
        el("table", { class: "lg-table" }, [
          el("thead", {}, el("tr", {}, [
            el("th", {}, "Место"), el("th", { class: "team" }, "Команда"), el("th", {}, "Призовые"),
          ])),
          el("tbody", {}, rg.placements.map((p) => el("tr", {}, [
            el("td", {}, p.place),
            el("td", { class: "team" }, p.team_name),
            el("td", {}, el("span", { class: "rg-money" },
              `${p.prize}${p.champion_bonus ? ` + ${p.champion_bonus} (бонус)` : ""} cr`)),
          ]))),
        ]),
      ])
    : null;

  // ---- schedule / history ---- //
  const historyList = el("div", { class: "match-list" });
  const revealed = rg.schedule.filter((e) => e.revealed);
  if (!revealed.length) {
    historyList.append(el("div", { class: "empty" }, "Матчи ещё не сыграны. Нажмите «Сыграть следующий день»."));
  }
  for (const enc of revealed.slice().reverse()) {
    const pillClass = enc.result === "home" ? "home" : enc.result === "away" ? "away" : "draw";
    historyList.append(el("div", {
      class: "match-row",
      onClick: () => navigate(`#/r/${encodeURIComponent(id)}/enc/${enc.encounter_index}`),
    }, [
      el("div", { class: "day" }, `День ${enc.day}${enc.stage !== "league" ? ` · ${enc.stage_label}` : ""}`),
      el("div", { class: "teams" }, [
        el("span", {}, enc.home_name), el("span", { class: "vs" }, "vs"), el("span", {}, enc.away_name),
      ]),
      el("div", { class: "score" }, `${enc.home_game_wins}–${enc.away_game_wins}`),
      el("span", { class: `pill ${pillClass}` }, RESULT_LABEL[enc.result] || enc.result),
    ]));
  }

  // ---- tournament players (ratings as of revealed day) ---- //
  const playersTable = el("table", { class: "lg-table stat-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "you" }, "Игрок"), el("th", {}, "Команда"), el("th", {}, "Фракция"),
      el("th", {}, "Архетип"), el("th", {}, "Рейтинг"), el("th", {}, "Δ за турнир"),
    ])),
    el("tbody", {}, rg.players.map((p) => el("tr", {}, [
      el("td", { class: "player" }, p.name),
      el("td", {}, p.team_name),
      el("td", {}, FACTION_LABELS[p.faction] || p.faction),
      el("td", {}, p.archetype),
      el("td", {}, p.rating_now.toFixed(0)),
      el("td", {}, deltaChip(p.rating_now - p.rating_start)),
    ]))),
  ]);

  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar(rg.name, `${rg.template.name} · день ${rg.current_day}/${rg.total_days}`, "#/", [
      el("button", { class: "btn", onClick: () => navigate(`#/r/${encodeURIComponent(id)}/draft`) }, "Драфт"),
      advance,
    ]),
    commentator,
    el("div", { class: "panel" }, [
      el("h2", {}, "Плей-офф"),
      el("div", { class: "rg-bracket" }, bracketNodes),
      champion, prizeRows,
    ]),
    el("div", { class: "panel" }, [
      el("h2", {}, "Таблица лиги"),
      el("div", { class: "table-scroll" }, [table]),
    ]),
    el("div", { class: "panel" }, [el("h2", {}, "Сыгранные встречи"), historyList]),
    el("div", { class: "panel" }, [
      el("h2", {}, "Игроки турнира"),
      el("div", { class: "table-scroll" }, [playersTable]),
    ]),
  ]));
}

// --------------------------------------------------------------------------- //
// Draft recap
// --------------------------------------------------------------------------- //
async function renderDraft(id) {
  showLoading("Загрузка драфта…");
  let rg;
  try {
    rg = (await fetchJson(`/api/regionals/${encodeURIComponent(id)}`)).regionals;
  } catch (error) {
    showError(error.message, "#/");
    return;
  }
  const rounds = (rg.draft?.rounds || []).map((rnd) => el("div", { class: "rg-draft-round" }, [
    el("h3", {}, `Раунд ${rnd.round} · порядок: ${rnd.order.map((fid) => rg.teams.find((t) => t.id === fid)?.name || fid).join(" → ")}`),
    ...rnd.picks.map((p) => el("div", { class: "rg-pick" }, [
      el("span", { class: "num" }, `#${p.pick}`),
      el("span", { class: "team" }, p.team_name),
      el("span", {}, `${p.nickname} (${FACTION_LABELS[p.faction] || p.faction})`),
      p.source === "llm"
        ? el("span", { class: "reason" }, p.reason ? `«${p.reason}»` : "")
        : el("span", { class: "auto" }, "автовыбор"),
    ])),
  ]));
  const rosters = rg.teams.map((team) => el("div", { class: "rg-card" }, [
    el("div", { class: "head" }, [el("span", { class: "nick" }, team.name)]),
    el("div", { class: "sub" }, `Менеджер: ${team.manager}`),
    ...team.players.map((p) => el("div", { style: "font-size:13px;margin:3px 0" },
      `${p.name} — ${FACTION_LABELS[p.faction] || p.faction}, ${p.archetype}`)),
  ]));
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar(`${rg.name} — Драфт`, rg.draft?.llm ? "Выбор делали LLM-менеджеры" : "Автодрафт (LLM недоступна)",
      `#/r/${encodeURIComponent(id)}`),
    el("div", { class: "panel" }, [el("h2", {}, "Составы"), el("div", { class: "rg-cards" }, rosters)]),
    el("div", { class: "panel" }, [el("h2", {}, "Ход драфта"), ...rounds]),
  ]));
}

// --------------------------------------------------------------------------- //
// Encounter detail
// --------------------------------------------------------------------------- //
async function renderEncounter(id, encIndex) {
  showLoading("Загрузка встречи…");
  let detail;
  try {
    detail = (await fetchJson(`/api/regionals/${encodeURIComponent(id)}/encounter/${encIndex}`)).encounter;
  } catch (error) {
    showError(error.message, `#/r/${encodeURIComponent(id)}`);
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
      `  ·  ${detail.stage_label}  ·  День ${detail.day}  ·  игр: ${detail.games.length}`,
    ]),
  ]);

  // commentary block (button-triggered, cached server-side)
  const commentWrap = el("div", {});
  if (detail.commentary) {
    commentWrap.append(el("div", { class: "rg-comment" }, [
      el("div", { class: "meta" }, "Гарри «Голос» Вольский"),
      detail.commentary.text,
    ]));
  } else {
    const btn = el("button", {
      class: "btn",
      onClick: async () => {
        btn.disabled = true;
        btn.textContent = "Комментатор пишет…";
        try {
          const payload = await fetchJson(`/api/regionals/${encodeURIComponent(id)}/commentary`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "encounter", index: detail.encounter_index }),
          });
          btn.replaceWith(el("div", { class: "rg-comment" }, [
            el("div", { class: "meta" }, "Гарри «Голос» Вольский"),
            payload.commentary.text,
          ]));
        } catch (error) {
          btn.disabled = false;
          btn.textContent = "🎙 Комментарий к матчу";
          commentWrap.append(el("div", { class: "error-text" }, error.message));
        }
      },
    }, "🎙 Комментарий к матчу");
    commentWrap.append(btn);
  }

  // rating changes table
  const changeRows = Object.entries(detail.rating_changes || {}).map(([pid, change]) => {
    const player = [...detail.home_team.players, ...detail.away_team.players]
      .find((p) => p.player_id === pid);
    return el("tr", {}, [
      el("td", { class: "player" }, player?.name || pid),
      el("td", {}, player ? (detail.home_team.players.includes(player) ? detail.home_team.name : detail.away_team.name) : ""),
      el("td", {}, change.before.toFixed(0)),
      el("td", {}, deltaChip(change.delta)),
      el("td", {}, change.after.toFixed(0)),
      el("td", {}, change.games.map((g) => `${g.won ? "✓" : g.draw ? "=" : "✗"}${g.impact >= 0.6 ? "★" : ""}`).join(" ")),
    ]);
  });
  const ratingTable = el("table", { class: "lg-table stat-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "you" }, "Игрок"), el("th", {}, "Команда"), el("th", {}, "До"),
      el("th", {}, "Δ"), el("th", {}, "После"), el("th", {}, "Игры (★ = высокий импакт)"),
    ])),
    el("tbody", {}, changeRows),
  ]);

  // games with watch links
  const gameRows = detail.games.map((game, i) => {
    const back = `regionals.html#/r/${encodeURIComponent(id)}/enc/${detail.encounter_index}`;
    const params = new URLSearchParams({
      replay: `../${game.replay}`,
      back,
      label: `${detail.home_name} vs ${detail.away_name} — игра ${i + 1}`,
    });
    return el("div", { class: "game-meta", style: "margin:6px 0" }, [
      el("span", {}, [`Игра ${i + 1}: `, el("span", { class: `pill ${game.winner}` }, RESULT_LABEL[game.winner] || game.winner)]),
      el("span", {}, [el("strong", {}, String(game.ticks)), " тиков"]),
      el("span", {}, `seed ${game.seed}`),
      el("a", { class: "btn primary", href: `index.html?${params.toString()}` }, "▶ Запись"),
    ]);
  });

  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Встреча", null, `#/r/${encodeURIComponent(id)}`),
    el("div", { class: "panel" }, [header]),
    el("div", { class: "panel" }, [el("h2", {}, "Комментатор"), commentWrap]),
    el("div", { class: "panel" }, [el("h2", {}, "Рейтинг (Elo)"), el("div", { class: "table-scroll" }, [ratingTable])]),
    el("div", { class: "panel" }, [el("h2", {}, "Игры серии"), ...gameRows]),
  ]));
}

// --------------------------------------------------------------------------- //
// Global player registry
// --------------------------------------------------------------------------- //
function playerCard(card) {
  const attrs = ATTR_LABELS.map(([key, label]) => el("div", { class: "rg-attr" }, [
    el("span", { class: "label" }, label),
    el("div", { class: "bar" }, [el("div", { class: "fill", style: `width:${card.attributes[key]}%` })]),
    el("span", { class: "value" }, card.attributes[key]),
  ]));
  return el("div", { class: `rg-card faction-${card.faction}` }, [
    el("div", { class: "head" }, [
      el("span", { class: "nick" }, card.nickname),
      el("span", { class: "rating" }, Math.round(card.rating)),
    ]),
    el("div", { class: "sub" },
      `${FACTION_LABELS[card.faction] || card.faction} · ${card.archetype} · ${card.player_id}`),
    ...attrs,
    el("div", { class: "blurb" }, card.blurb || ""),
    el("div", { class: "career" }, [
      `Карьера: ${card.wins}W–${card.losses}L · турниров: ${card.tournaments} · титулов: ${card.titles} · `,
      el("span", { class: "rg-money" }, `${card.money} cr`),
    ]),
  ]);
}

async function renderPlayers() {
  showLoading("Загрузка игроков…");
  let data;
  try {
    data = await fetchJson("/api/regionals/players");
  } catch (error) {
    showError(error.message, "#/");
    return;
  }
  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Игроки", "18 карточек · 1 нейронка = 1 игрок · Elo с 1000", "#/"),
    el("div", { class: "panel" }, [
      el("div", { class: "rg-cards" }, data.players.map(playerCard)),
    ]),
  ]));
}

// --------------------------------------------------------------------------- //
// Franchises (money, titles, head-to-head)
// --------------------------------------------------------------------------- //
async function renderTeams() {
  showLoading("Загрузка команд…");
  let data;
  try {
    data = await fetchJson("/api/regionals/franchises");
  } catch (error) {
    showError(error.message, "#/");
    return;
  }
  const franchises = data.franchises;
  const byId = Object.fromEntries(franchises.map((f) => [f.franchise_id, f]));

  const summary = el("table", { class: "lg-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", { class: "team" }, "Команда"), el("th", {}, "Менеджер"),
      el("th", {}, "Сезонов"), el("th", {}, "Титулов"), el("th", {}, "Казна"),
    ])),
    el("tbody", {}, franchises.map((f) => el("tr", {}, [
      el("td", { class: "team" }, f.name),
      el("td", {}, f.manager),
      el("td", {}, f.seasons),
      el("td", {}, f.titles),
      el("td", {}, el("span", { class: "rg-money" }, `${f.money} cr`)),
    ]))),
  ]);

  // head-to-head matrix
  const headRow = el("tr", {}, [el("th", { class: "team" }, "История встреч"),
    ...franchises.map((f) => el("th", {}, f.name.split(" ")[0]))]);
  const h2hRows = franchises.map((f) => el("tr", {}, [
    el("td", { class: "team" }, f.name),
    ...franchises.map((opp) => {
      if (opp.franchise_id === f.franchise_id) return el("td", { class: "muted" }, "—");
      const rec = (f.head_to_head || {})[opp.franchise_id];
      return el("td", {}, rec ? `${rec.wins}–${rec.losses}${rec.draws ? `–${rec.draws}` : ""}` : "0–0");
    }),
  ]));
  const h2hTable = el("table", { class: "lg-table stat-table" }, [
    el("thead", {}, headRow), el("tbody", {}, h2hRows),
  ]);

  root.replaceChildren(el("div", { class: "league-app" }, [
    topbar("Команды", "Франшизы Regionals: казна, титулы и история встреч (победы–поражения)", "#/"),
    el("div", { class: "panel" }, [el("div", { class: "table-scroll" }, [summary])]),
    el("div", { class: "panel" }, [el("div", { class: "table-scroll" }, [h2hTable])]),
  ]));
}

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //
async function render() {
  const route = parseRoute();
  if (route.view === "lobby") return renderLobby();
  if (route.view === "players") return renderPlayers();
  if (route.view === "teams") return renderTeams();
  if (route.view === "tournament") return renderTournament(route.id);
  if (route.view === "draft") return renderDraft(route.id);
  if (route.view === "encounter") return renderEncounter(route.id, route.enc);
  return renderLobby();
}

async function boot() {
  try {
    store.meta = await fetchJson("/api/regionals/templates");
  } catch (error) {
    showError(`API недоступен: ${error.message}. Запустите локальный viewer-сервер.`, "menu");
    return;
  }
  window.addEventListener("hashchange", render);
  render();
}

boot();
