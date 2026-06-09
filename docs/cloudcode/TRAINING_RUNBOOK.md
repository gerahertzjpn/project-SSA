# TRAINING_RUNBOOK.md

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
