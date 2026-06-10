# TRAINING_RUNBOOK.md

> Population-18 (текущая основная задача): подробный runbook запуска и
> мониторинга — `docs/cloudcode/POPULATION_18_RUNBOOK.md`.
> Запуск: `python scripts/train_population.py --run-dir training_runs/population_18_v1
> --updates 400 --episodes-per-net 3 --workers 16 --checkpoint-interval 20`.

## Перед долгим обучением

- [ ] data checks
- [ ] split/leakage checks
- [ ] config saved
- [ ] run_id created
- [ ] smoke-test passed
- [ ] checkpoint save/load checked
- [ ] metrics logging checked
- [ ] registry update checked
- [ ] recovery command known

## Во время обучения

- monitor process
- monitor logs
- monitor metrics
- monitor checkpoints
- monitor NaN/Inf/OOM/divergence/overfit/underfit/diversity collapse

## При падении

1. сохранить tail лога и traceback;
2. определить тип ошибки;
3. исправить первопричину;
4. запустить smoke-test;
5. восстановиться из checkpoint, если безопасно;
6. обновить run summary и registry.
