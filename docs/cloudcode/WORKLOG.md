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
