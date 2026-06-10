# WORKLOG.md

Записывать только крупные блоки работы.

## Template

```text
Дата:
Блок:
Цель:
Что сделано:
Файлы:
Проверки:
Результат:
Проблемы:
Следующий шаг:
```

## 2026-06-10 — Масштабный balance-тест популяции-18

Цель: определить фаворита среди фракций, активность использования механик,
причину дисбаланса и наличие отдельной «имба»-сети.

Что сделано:
- Новый `scripts/population_balance_test.py`: загружает 18 faction-locked сетей
  из `training_runs/population_18_v1/nets/<id>/latest.pt`, оборачивает каждого
  PolicyBot в RecordingBot (трекинг потока действий: produce/build/move/lord/hold,
  юниты, здания, трата ресурсов) + читает финальное состояние (лорд, ресурсы,
  стоящие здания). Два эксперимента: (A) чистые фракционные H2H для изоляции
  силы фракции, (B) смешанные 3v3 для per-net win-rate.
- Прогон 1340 матчей (540 pure + 800 mixed), headless, seed 20260610.
- Отчёт: `analysis/POPULATION_18_BALANCE_FINDINGS.md`,
  данные `analysis/population_balance_full.json`.

Результат:
- Явный фаворит — **humans** (pure win-rate 0.833), infected 0.542,
  robots **0.125**. Оба эксперимента совпадают: humans > infected > robots.
- «Имбы» среди отдельных сетей НЕТ: внутрифракционный разброс ~равный у всех
  (delta 0.21–0.23); per-net win-rate определяется фракцией. Топ — humans_6 0.698,
  дно — robots_1 0.294.
- Причина (data-driven): за равные деньги роботы получают вдвое меньше армии
  (36k юнитов vs 71k/82k при ~равной трате 306k золота) — дешёвый юнит DRONE=6
  против SWARMLING=1 / SKIRMISHER=2 при income 2/тик, плюс слабый старт 1 WALKER.
  Роботов перетемпливают по числу тел; они простаивают (hold 15.3% vs 2–7%).
  Это дисбаланс ростера/экономики, не провал обучения (все фракции тратят золото).
- Все механики используются (фракционные здания BOOSTER/POWER/LIFE активны);
  near-dead: ECONOMY_COLLECTOR у humans, TECH_2 у всех (бои кончаются раньше).

Проблемы: дисбаланс фракций сильный; рекомендации по фиксам в findings-доке
(не применял — это меняет геймплей-константы, нужен апрув + ре-трен после правки).

Следующий шаг: апрув правок экономики роботов → правка констант → короткий
ре-трен → повторный прогон balance-теста.

## 2026-06-09 — Плавное движение юнитов (visual-only)

Цель: убрать рывки/телепорты и скачки скорости при отображении передвижения
юнитов по полю.

Диагноз (data-driven, по visual_units из свежих реплеев):
- Движок продвигает группу посегментно: `progress += 1/travel_ticks`,
  клампится в 1.0, остаток сверх границы сегмента отбрасывается, а
  `_discrete_segment_ticks` через `ceil` даёт минимум 1 тик на сегмент. Из-за
  этого мировое смещение за тик неравномерно: юнит «крадётся» по коротким
  node_exit/node_transit и «спринтует» по длинным edge. На целочисленных тиках
  видно ускорение/замедление у каждого узла (Vanguard: 41→15→45 px/тик;
  Skirmisher edge-шаги 45↔90). Все per-unit x/y берутся из
  `moving_group_visual(group)` → один корень.

Что сделано (только визуал, геймплей не тронут):
- `src/core/visual2d.py`: `moving_group_visual_smooth(group, current_tick)` —
  строит полилинию всего пути (`center→exit→entry→center` на каждый хоп,
  зеркало `_movement_segments`) и ставит юнит в точку на доле `frac` ДЛИНЫ
  пути, где `frac` — доля ВРЕМЕНИ ноги `(tick-started)/(arrival-started)`.
  Время→дистанция линейно ⇒ постоянная скорость. Все метаданные сегмента
  (lane/approach/phase/transit) сохраняются; при отсутствии пути/тайминга —
  фолбэк на старую посегментную проекцию. Геймплейный `progress`, arrival и
  тайминги боёв не меняются; visual_units считаются только при record_replay,
  RL их не использует.
- `src/core/engine.py`: `_visual_unit_snapshots` (anchor) и
  `_moving_group_snapshot` (`visual`) используют smooth-версию.

Файлы: src/core/visual2d.py, src/core/engine.py, tests/test_visual2d.py.

Проверки:
- `tests/test_visual2d.py` +4 теста (старт=исток, финиш=цель, равномерность
  per-tick, сохранение метаданных, фолбэк). Полный набор: 291 passed (без
  браузерных) + 2 браузерных passed.
- Диагностика на свежем реплее: внутри-ножная неравномерность шага упала с
  ~300% (15↔45) до ≤12% CV; 0 ног с CV>0.15; концы пути точно в центрах узлов.

## 2026-06-09 — Плавный вход в бой (visual-only deploy-мост)

Цель: убрать «снап» юнитов в боевую формацию при входе в бой (продолжение
предыдущего блока).

Диагноз (по visual_units): при старте/входе в бой `place_player_units_2d`
ставит юнит в `entry_point − forward·depth` + латеральный разброс формации —
эти позиции геймплейные (range/targeting). Движущийся юнит был на подходе
(`frac<1`), поэтому первый боевой кадр «дёргал» его вперёд в формацию.
Замер: ENTER moving→battle p50 62px, p90 95px, **max 140px** (570 случаев).

Что сделано (visual-only, combat-резолюция не тронута):
- `src/core/units.py`: поля `UnitInstance` — `deploy_active`, `deploy_cur_x/y`,
  `deploy_seed_tick`.
- `src/core/engine.py`: при входе движущейся группы в бой
  (`_arm_visual_deploy_bridge`, вызывается из `_add_player_units_to_battle`,
  только для входящих из движения групп) визуальный курсор юнита сажается в
  smooth-якорь группы (где он рисовался этот тик в движении).
  `_visual_deploy_point` каждый тик двигает курсор к актуальному боевому слоту
  **не больше чем на маршевый шаг** `speed×30` (скорость-ограниченное
  преследование); на тике посева отдаёт сам якорь (вход = один маршевый шаг).
  Бой выводит eased-позицию только в visual-рядах; `unit.x/y` для боя не
  меняются. Активно лишь при record_replay (в обучении не выполняется).

Файлы: src/core/units.py, src/core/engine.py, tests/test_replay_schema.py,
scripts/browser_smoke_movement.py.

Проверки:
- `tests/test_replay_schema.py::test_battle_entry_has_no_visual_teleport`
  (вход в узловой бой: max per-tick шаг A1-вангардов < 60px).
- Полный набор: 292 passed (без браузерных).
- Замер после фикса: ENTER max 140→**75px** (= нормальный маршевый шаг),
  INTRA battle→battle 0 шагов выше 1.15× макс. скорости (мост не создаёт
  дрожания в бою). Детерминизм генерации сохранён (идентичные SHA по seed).
- Браузерный визуальный смок `scripts/browser_smoke_movement.py`: регенерация
  реплея → http.server → Pixi-viewer в Chromium (Playwright): реплей
  загружается (tickMax=135), воспроизведение анимируется (tickNow растёт),
  canvas рендерится, фатальных ошибок нет; скриншоты в viewer_logs/.

Известный остаток: «continuation»-телепорты на ВЫХОДЕ из боя (юнит после
короткого боя у узла мгновенно оказывается у соседнего узла, напр. CP1→A1) —
это отдельная геймплейная механика «продолжения после боя», не вход в бой;
viewer интерполирует за 1 тик (глайд, не жёсткий телепорт).

## 2026-06-09 — Тренировка Population-18 завершена

Цель: запустить и довести до конца тренировку 18 faction-locked сетей (population_18_v1).

Что сделано:
- Smoke-test перед запуском — зелёный.
- Полный прогон: 400 updates, 16 workers, GPU PPO, CPU rollout.
- Мониторинг без остановки от update 1 до 400. Вмешательств не потребовалось.

Ключевые метрики:
- Total steps: 3 617 538
- mean_episode_steps: 223.9 → 148.3 (−34%, −75 тиков) — сети научились
  заканчивать бои решительно, а не тянуть до max_ticks.
- Win rate: здоровое колебание ~0.5 на всей дистанции (frozen self-play).
- 18/18 чекпоинтов присутствуют; все 400 updates обучены.
- Disk: ~1.1 GB (18 latest.pt + 360 milestone .pt)

Файлы:
- training_runs/population_18_v1/ (gitignored)
- experiments/EXPERIMENTS.md (обновлён статус TRAINED)
- experiments/MODEL_REGISTRY.md (обновлён статус)

Результат: пул из 18 сетей готов к league-draft. Следующий шаг — teammate-swap
/ role-stress eval (без culling).

## 2026-06-09 — Подготовка тренировки Population-18 (free-agent draft pool)

Цель: подготовить (не запускать) систему тренировки 18 уникальных faction-locked
сетей — 6 на фракцию, «1 сид = 1 игрок», играющих на любой позиции с любым
союзником; self-play; без отбора. Запускать должна следующая нейронка.

Развилки зафиксированы вопросом пользователю: микс-фракционные команды,
frozen-snapshot self-play, уникальность = seed + рецепт (без вариаций
архитектуры).

Что сделано:
- `src/training/team_env.py`: аддитивно `player_factions` (фракция per-slot,
  проброс в `make_initial_state`) и `opponent_policy_state_ids` (per-slot
  frozen-оппонент через `PolicyBot(policy_state_id=...)`). Обратносовместимо.
- `src/training/population.py` (новый, изолированный): спецификация 18 сетей
  (`build_default_population`, genotype_hash, проверка уникальности), нативный
  population self-play тренер (per-episode микс-тройка обучаемых vs 3 frozen,
  рандом позиции, фракция следует за сетью), frozen-pool как один чекпоинт по
  net_id, per-net чекпоинты + реестр. Переиспользует `update_policy`/GAE/`PolicyBot`.
- `scripts/train_population.py` (новый): лаунчер для следующей нейронки —
  `--smoke`, `--init-only`, resume, frozen-pool ротация с пересборкой воркер-пула,
  метрики jsonl, реестр (json+md), без отбора.
- Документация: `docs/cloudcode/POPULATION_18_RUNBOOK.md` (запуск+мониторинг),
  PROJECT_STATE, EXPERIMENTS, MODEL_REGISTRY (18 хэшей), DECISIONS (ADR-0001).

Файлы: src/training/team_env.py, src/training/population.py,
scripts/train_population.py, tests/test_population.py,
docs/cloudcode/{POPULATION_18_RUNBOOK,PROJECT_STATE,DECISIONS,WORKLOG,TRAINING_RUNBOOK}.md,
experiments/{EXPERIMENTS,MODEL_REGISTRY}.md.

Проверки:
- `tests/test_population.py` — 9 passed; регрессия `tests/test_team_training_env.py
  test_training_spaces test_policy_bot test_team_ppo_trainer test_teammate_swap_eval
  test_training_env` — 49 passed.
- Smoke (workers=1 и workers=3 spawn), resume (start_update 1→3), full-length
  sanity (max_ticks 300 → реальные win 0.33 / loss 0.67, не только ничьи).
- Чекпоинты сетей и frozen pool проходят `checkpoint_is_compatible`; сети из
  pool грузятся по net_id.

Результат: система готова к запуску одной командой; обучение НЕ запускалось
(по требованию). Команда запуска и мониторинг — в POPULATION_18_RUNBOOK.md.

Следующий шаг: следующая нейронка запускает `scripts/train_population.py`
(полный прогон), мониторит metrics.jsonl, после — teammate-swap/role-stress eval
без отбора.

## 2026-06-09 — League v1

Цель: добавить режим «Лига» поверх существующего симулятора матчей.

Что сделано:
- Бэкенд-пакет `src/league/`: шаблоны лиг (`classic_6x3`: 6 команд×3,
  single round-robin, Bo3, 1 матч/день), генерация случайного расписания
  (circle method + перетасовка), полная симуляция ВСЕХ встреч на сервере при
  создании (через `run_match`), сохранение реплеев и результатов, таблица
  (3/1/0), персист в `leagues/<id>/league.json`, выдача с сокрытием
  нераскрытых результатов (`current_day`), пошаговое раскрытие (`advance_day`).
- API в `viewer/server.py`: templates/list/get/encounter (GET), create
  (фоновая джоба + progress), job-status, advance (POST).
- Фронтенд: главное окно `home.html`; SPA лиги `league.html`/`league.js`/
  `league.css` (lobby, конструктор команд, таблица, история, детали встречи
  с пер-матчевой и пер-игроцкой статистикой, кнопки Back, «Watch recording»);
  embed-режим плеера в `index.html`/`app.js`/`styles.css` (`?replay=&back=&label=`).
- Лаунчер открывает `home.html`; README дополнен разделом League.

Файлы: src/league/*, viewer/server.py, viewer_pixi/{home,league}.html,
viewer_pixi/league.{js,css}, viewer_pixi/{index.html,app.js,styles.css},
tests/test_league.py, tests/test_league_server.py, README.md, start_viewer.ps1,
.gitignore.

Проверки:
- `pytest tests/test_league.py` — 12 passed (вкл. реальный single-encounter smoke).
- `pytest tests/test_league_server.py tests/test_pixi_viewer_static.py
  tests/test_viewer_server.py` — 10 passed.
- Live HTTP smoke: list/get/advance/encounter, day-gating, отдача файла реплея
  (exact 200, через `../` 200, missing 404, контент валиден), home/league 200.

Результат: лига генерируется целиком на Python-сервере, результаты раскрываются
по дням, записи матчей открываются в существующем плеере. Все проверки зелёные.

Проблемы: создание лиги — это ~37 реальных симуляций (несколько минут);
сделано фоновой джобой с progress-баром, синхронной симуляции в HTTP нет.

Следующий шаг (на будущее): LLM-менеджер для составов; дополнительные шаблоны
(double round-robin, плей-офф); богаче пер-игроцкая статистика из state.


## 2026-06-10 — Визуал боя: ranged/melee линии атак

Цель: сверить, как бой реально считается в ядре, с тем, как он рисуется, и
поправить визуал, если он не передаёт смысл.

Что проверено в коде:
- Активный бой считается в `src/core/combat2d.py` (`tick_battle_units_2d`):
  реальные 2D-координаты (x,y), выбор цели по евклидову расстоянию, движение
  в свою дальность (range*24 map-units) с остановкой по первому контакту,
  фокус-огонь по раненым, одновременное применение урона. Движок отдаёт во
  вьюер `formation` (вкл. x,y и range_map) и `BATTLE_TICK.summary.attacks`.
- Вьюер (`viewer_pixi/app.js`) ставит юнитов боя по настоящим x,y — позиции,
  движение, HP-бары и смерти отображаются верно.

Найденное расхождение:
- Линии атак считали «дальнобойными» только тип `R` (RANGER). RAIDER, BMP,
  SPITTER, RAVAGER, BEHEMOTH, LASER, GUARDIAN_MECH, SIEGE, LORD и CORE
  (range 2.5–5.0) рисовались как ближний бой, хотя стреляют издалека.

Что сделал (`viewer_pixi/app.js`):
- Проброс реальной дальности юнита (`range`, `rangeMap`) из formation в спрайт.
- `drawAttackLines`: ranged/melee теперь по реальной дальности атакующего
  (порог 1.25 = движковый `_uses_melee_engagement`), с фолбэком по типу.
- Толщина/прозрачность линии масштабируются по урону выстрела — тяжёлые залпы
  читаются весомее.

Проверки:
- `pytest tests/test_pixi_viewer_static.py` — 3 passed.
- `python scripts/browser_smoke_movement.py --seed 7` — OK, плейбек анимируется,
  без JS/консольных ошибок.
- Скриншот боя на тике 28 (CP3) рендерится корректно: viewer_logs/battle_tick28_after.png.

Движок не менялся — правка чисто презентационная.


## 2026-06-10 — Визуал зданий/турелей в бою у ядра

Вопрос пользователя: здания в бою работают как «пластины»-слои, но визуально
это не читается; турели в точках (CP) рисуются в углу узла, хотя в бою стоят
по центру.

Что проверено в коде:
- Во время боя здания становятся боевыми юнитами (`_add_buildings_to_battle`),
  ставятся в ЦЕНТР узла (`place_static_units_2d`, role="building"), и остаются
  в `node.buildings`. Рендер идёт через `visual_units` → каждое здание/ядро
  выдаётся с реальными боевыми x,y (все в одной точке — центре).
- «Пластины» = правило таргетинга `_core_structure_target_role`: штурм ядра
  пробивает здания по слоям (slot index), ядро прикрыто, пока есть здания.
  Пространственно слои НЕ выражены — всё в центре.
- Турель — здание CP (hp150, atk3, range4), единственное стреляющее; здания
  ядра пассивны (atk0) — это HP-щиты.

Найденные расхождения:
1. Боевые спрайты зданий рисовались как generic-круг (нет ветки в drawUnitShape).
2. `node.buildings` всегда отдаётся → угловой маркер здания рисовался И во время
   боя, дублируя боевой спрайт в центре (турель «в углу» вместо центра).
3. Слои-«пластины» никак не показаны (всё свалено в центр).

Что сделал (`viewer_pixi/app.js`, чисто презентационно):
- `drawPlateShape`: боевые спрайты зданий = «пластина» (турель с дулом).
- `arrangeCoreBattlePlates`: здания/ядро боя расставляются слоями-щитом —
  внешний slot ближе к атакующим (он гибнет первым, как в движке), ядро сзади.
  Направление — к центру карты (стабильно, без джиттера). Линии атак следуют за
  спрайтами, поэтому удары визуально приходят по передней пластине.
- `buildingFightingInBattle`: если здание уже сражается боевым спрайтом, его
  угловой маркер скрывается (точное совпадение по `visual_units`). Недостроенные
  здания и мирное время — маркеры как раньше.

Проверки:
- `pytest tests/test_pixi_viewer_static.py` — 3 passed.
- `browser_smoke_movement.py --seed 11` — OK, без JS-ошибок.
- Синтетические сценарии штурма ядра/CP: подтверждены позиции и рендер.
  Скриншоты: viewer_logs/{core_plates_after_crop,cp_turret_after}.png.

Движок и боевая математика не менялись.

## 2026-06-10 — Regionals v1: лига + плей-офф с LLM-менеджерами и комментатором

Реализован новый режим **Regionals** (`src/regionals/`, ~8 модулей) поверх
существующей лиги:

- **Шаблоны-версии** (`templates.py`): `regionals_v1` = лига 6×3 round-robin Bo3
  + step-ladder плей-офф топ-3 (2-е vs 3-е, победитель vs 1-е) Bo5. Новые версии
  добавляются словарём и выбираются в UI.
- **Карточки игроков** (`players.py`): 18 уникальных игроков, строго 1 нейронка
  population_18_v1 = 1 игрок (инвариант проверяется). Атрибуты (atk/def/eco/
  tempo/consistency/potential, 0-100) выведены из реальных eval-данных
  `analysis/population_balance_full.json` (per-net winrate + механики) и
  снапшотятся в персистентный реестр `data/regionals/players.json`.
- **Elo** (`elo.py`): старт 1000, K=32, классическая ожидаемая оценка (победа
  над сильным даёт больше, поражение от слабого отнимает больше), импакт игрока
  (выживание/HP/ресурсы в игре) модулирует дельту максимум на ±20%.
- **Драфт** (`draft.py`): 3 раунда × 6 пиков, порядок случайный и
  сбалансированный (ни одна франшиза не пикает дважды с одной позиции).
  Каждый пик = 1 маленький LLM-вызов со structured output (enum доступных id);
  невалидный ответ/ошибка → авто-пик, драфт всегда завершается.
- **Франшизы** (`franchises.py`): 6 фиксированных команд с разными
  LLM-персонами менеджеров (агрессор/аналитик/хайроллер/оборона/синергия/
  охотница за звёздами); казна, титулы, сезоны и история встреч (h2h)
  персистентны в `data/regionals/franchises.json`.
- **LLM-слой** (`llm.py`): ленивый Anthropic SDK (model `claude-opus-4-8`,
  override `REGIONALS_LLM_MODEL`), учёт токенов, graceful degradation без
  ключа/пакета.
- **Комментатор** (`commentary.py`): персона Гарри «Голос» Вольский с памятью —
  прошлые реплики (открытие/матчи/итоги) подаются компактным списком в каждый
  вызов; комменты по кнопке, кэшируются, лимит слов в промпте.
- **Ядро** (`regionals.py`): create = драфт → лига → плей-офф (всё симулируется
  сразу), Elo по играм, призовые по местам (500/300/200/120/80/50 + бонус
  чемпиона 250, делятся на игроков), обновление реестров, day-gated public view
  с маскировкой плей-офф матчапов до завершения предыдущей стадии.

API (`viewer/server.py`): `/api/regionals/{templates,list,players,franchises,
job/<id>,<id>,<id>/encounter/<n>}` GET + `create`, `<id>/advance`,
`<id>/commentary` POST (фоновый job с фазами draft/league/playoff).

UI (`viewer_pixi/regionals.{html,js,css}` + карточка на home): лобби/создание,
турнир (комментатор, сетка плей-офф, таблица, история, рейтинги), драфт-рекап,
детали встречи (Elo-дельты, записи игр, кнопка комментария), глобальные
карточки игроков и страница франшиз (казна/титулы/h2h-матрица).

Проверки:
- `tests/test_regionals.py`: 20 новых тестов (карточки/инварианты, Elo, драфт
  с fake LLM + fallback, плей-офф (ничья → выше посев), экономика, гейтинг,
  комментатор с памятью и кэшем, персистентность) — зелёные.
- Полный suite: 328+1 passed (2 теста viewer_server обновлены под уже висевшую
  в worktree legacy-разметку discovered-чекпоинтов — падали до моих изменений).
- E2E smoke через HTTP: реальный движок, 17/17 встреч, ~40 c; policy:-id
  population-сеток корректно резолвятся (1 игра ≈ 0.8 c); артефакты смоука
  удалены (реестры сброшены к чистому старту, все 1000).

Зависимости: `anthropic>=0.109` добавлен в requirements (установлен).
`regionals/` и `data/regionals/` в .gitignore. ANTHROPIC_API_KEY в окружении
НЕТ — без него работает авто-драфт и заглушка открытия, комментарии по кнопке
вернут понятную ошибку. Для полного режима нужен ключ.
