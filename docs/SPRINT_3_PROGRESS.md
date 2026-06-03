# Sprint 3 Progress Log

Started: 2026-06-03T13:50:15+09:00

## Baseline Snapshot

- Start commit: `87a6b03411b190162040ae285c6ad409b6adf24e`
- Worktree at start: clean
- Active plan: `docs/SPRINT_3_PROJECT_PLAN.md`
- Accepted Sprint 2 checkpoint:
  `training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt`

## 0. Preflight

Commands:

```powershell
python -m pytest -q
python scripts/evaluate_checkpoints.py --run-dir training_runs/sprint2_scripted_clean_v1 --updates 100 --modes argmax,stochastic --seed-bases 20260603,20270603 --episodes-per-seed 25 --workers 16 --torch-threads 1 --ally-bot scripted --opponent-bot scripted --include-random-legal --out-dir training_runs/evals/sprint3_preflight_sprint2_checkpoint
```

Results:

- Tests: `124 passed in 16.43s`
- Scripted control over 50 episodes: `34%` win, `64%` loss, `2%` draw
- Random-legal A1 over 50 episodes: `26%` win, `38%` loss, `36%` draw
- Sprint 2 checkpoint update 100 argmax over 50 episodes:
  - `100%` win
  - `0%` loss
  - `0%` draw
  - `56%` normal wins
  - `44%` death-time wins
  - invalid action rate `0.0`
- Sprint 2 checkpoint update 100 stochastic over 50 episodes:
  - `78%` win
  - `10%` loss
  - `12%` draw
  - `18%` normal wins
  - `60%` death-time wins
  - invalid action rate `0.0`

Decision:

- Sprint 2 checkpoint and eval loop are healthy enough to use as Sprint 3 warm
  start/reference.
- Proceed to multi-agent synchronized 3v3 training infrastructure.
