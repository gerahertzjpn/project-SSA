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
