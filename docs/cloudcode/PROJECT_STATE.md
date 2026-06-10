# PROJECT_STATE.md

## Назначение проекта

2D авто-баттлер 3v3 с RL-агентами. Движок (`src/core/`), masked-PPO обучение
(`src/training/`), боты (`src/bots/`), серверная лига (`src/league/`), режим
Regionals с LLM-менеджерами (`src/regionals/`) и Pixi вьюеры (`viewer_pixi/`).
Три фракции: `humans`, `infected`, `robots`.

## Режим Regionals (новое, 2026-06-10)

Лига 6 команд Bo3 + плей-офф топ-3 Bo5 поверх `src/league`. 18 карточек
игроков (1 нейронка population_18_v1 = 1 игрок), LLM-драфт шестью разными
менеджерами-персонами, Elo (старт 1000, импакт ±20%), призовые, персистентные
реестры (`data/regionals/*.json`, gitignored), LLM-комментатор с памятью
(комментарии по кнопке). UI: `viewer_pixi/regionals.html`. API:
`/api/regionals/*`. LLM: пакет `anthropic`, ключ `ANTHROPIC_API_KEY` (без
ключа — авто-драфт и отключённый комментатор), модель `claude-opus-4-8`
(override `REGIONALS_LLM_MODEL`). Тесты: `tests/test_regionals.py`.

## Как установить зависимости

```powershell
pip install -r requirements.txt
```

## Как запускать тесты

```powershell
python -m pytest -q
# по областям, напр.:
python -m pytest tests/test_population.py tests/test_team_training_env.py -q
```

## Как запускать обучение

### Популяция 18 сетей (текущая основная задача — free-agent draft pool)

18 независимых faction-locked сетей (6/фракция), frozen-snapshot self-play,
микс-фракционные команды, рандомизация позиции, **без отбора**. Подробности и
мониторинг — `docs/cloudcode/POPULATION_18_RUNBOOK.md`.

```powershell
python scripts/train_population.py --smoke                 # быстрый sanity
python scripts/train_population.py --init-only             # спецификация + реестр без обучения
python scripts/train_population.py --run-dir training_runs/population_18_v1 \
    --updates 400 --episodes-per-net 3 --workers 16 --checkpoint-interval 20
```

### Командный тренер (одно трио A1/A2/A3, наследие Sprint 3)

```powershell
python scripts/train_team_policy.py --run-dir training_runs/<run> --updates N \
    --opponent-bot policy --opponent-policy-checkpoint <frozen> --role-assignment-mode shuffle
```

## Где данные

RL self-play — данных-датасета нет; опыт генерируется средой
`src/training/team_env.py` (теперь фракционно-осведомлённой).

## Где результаты

- runs: `training_runs/` (gitignored)
- per-net checkpoints: `training_runs/<run>/nets/<net_id>/latest.pt`
- frozen pool: `training_runs/<run>/frozen_pool/frozen_pool_v*.pt`
- metrics: `training_runs/<run>/metrics.jsonl`
- registry: `training_runs/<run>/registry.json` + `MODEL_REGISTRY.md`
- спецификация популяции: `src/training/population.py` (`build_default_population`)

## Последнее стабильное состояние

- Observation interface: `train_obs_v3.1`, `OBSERVATION_DIM = 5188`.
- Env поддерживает per-player фракции (`player_factions`) и per-slot frozen
  оппонентов (`opponent_policy_state_ids`) — аддитивно, обратносовместимо.
- `scripts/train_population.py` прошёл smoke (1 и multi-worker), resume, и
  full-length sanity (реальные win/loss). 9 новых тестов + 49 существующих
  зелёные.

## Известные ограничения

- Throughput self-play ~1.9M world steps/час (долг с прошлых спринтов).
- Milestone-чекпоинты занимают место (см. disk note в runbook).
- Нативный population-тренер контролирует только team A; оппоненты — frozen
  (B-side обрабатывается зеркалированием observation). Live-both-sides self-play
  не реализован (сознательно, по выбору пользователя).
