# Sprint 3 8-Hour Run Log

Date: 2026-06-04

## Preflight

- `pytest -q`: `241 passed` before the marathon start.
- Scripted no-policy benchmark, 16 workers, 30 seconds:
  `4.42M` world steps/hour.
- `preflight_sprint3_training.py`: `ok=true`.
- Initial 100-match crash-test found two allied-lord invariants:
  - moving allied lord could keep a dead core as global target until arrival;
  - same-player reinforcement could overwrite a lord edge-battle side from
    `LORD_PUSH` to a normal movement intent.
- Fixed both before training:
  - moving lord groups now refresh/repair target and intent while moving;
  - battle metadata preserves existing lord-push route/intent when same-player
    support joins;
  - edge battle resolution has a safety net: surviving physical LORD continues
    as `LORD_PUSH`.
- Regression checks:
  - `pytest tests/test_lord_death_time.py -q`: `28 passed`;
  - `python scripts/crash_test_simulation.py --matches 100 --seed-base 20260604`:
    `seeds_with_issues=0`, `issue_summary={}`;
  - `pytest -q`: `243 passed`.

## Training

Fresh Sprint 3 scripted run is the next step. Self-play remains allowed below
`3M` world steps/hour for this run only, but only after scripted quality gates.

Started fresh run:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --overwrite --updates 300 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --opponent-bot scripted --ally-bot scripted --seed 20260604 --torch-threads 1 --random-baseline-episodes 100
```

Early argmax scripted eval checkpoints:

- update 10: `win_rate=0.28`, `normal_win_rate=0.00`,
  `death_time_win_rate=0.28`, `draw_rate=0.36`, `loss_rate=0.36`.
- update 20: `win_rate=0.56`, `normal_win_rate=0.00`,
  `death_time_win_rate=0.56`, `draw_rate=0.00`, `loss_rate=0.44`.
- update 30: `win_rate=0.46`, `normal_win_rate=0.04`,
  `death_time_win_rate=0.42`, `draw_rate=0.02`, `loss_rate=0.52`.
- update 40: `win_rate=0.54`, `normal_win_rate=0.10`,
  `death_time_win_rate=0.44`, `draw_rate=0.32`, `loss_rate=0.14`.
- update 50: `win_rate=1.00`, `normal_win_rate=0.74`,
  `death_time_win_rate=0.26`, `draw_rate=0.00`, `loss_rate=0.00`.
- update 60: `win_rate=0.90`, `normal_win_rate=0.42`,
  `death_time_win_rate=0.48`, `draw_rate=0.10`, `loss_rate=0.00`.
- update 70: `win_rate=0.68`, `normal_win_rate=0.38`,
  `death_time_win_rate=0.30`, `draw_rate=0.22`, `loss_rate=0.10`.
- update 80: `win_rate=1.00`, `normal_win_rate=0.96`,
  `death_time_win_rate=0.04`, `draw_rate=0.00`, `loss_rate=0.00`.
- update 90: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`.
- update 100: `win_rate=1.00`, `normal_win_rate=0.98`,
  `death_time_win_rate=0.02`, `draw_rate=0.00`, `loss_rate=0.00`.
- update 110: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`.
- update 120: `win_rate=0.94`, `normal_win_rate=0.90`,
  `death_time_win_rate=0.04`, `draw_rate=0.06`, `loss_rate=0.00`.
- update 130: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=58.50`.
- update 140: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=52.84`.
- update 150: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=49.20`.
- update 160: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=46.92`.
- update 170: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=46.36`.
- update 180: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=46.60`.
- update 190: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=45.02`.
- update 200: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=44.00`.
- update 210: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=43.54`.
- update 220: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=43.60`.

Interpretation at update 50: first strong argmax checkpoint under fixed rules,
but not accepted as a gate by itself. Need neighboring checkpoints and a
multi-seed sweep before self-play.

Interpretation at update 60: still clearly beats scripted on the eval seed and
has zero losses, but normal wins dropped and death-time rose. Continue training
and later select by multi-seed scripted sweep, not by this single eval.

Interpretation at update 70: weaker argmax checkpoint; training rollouts remain
strong, but checkpoint selection should prefer update 50 over update 70 unless
later evals recover and the sweep confirms them.

Interpretation at update 80: strongest argmax candidate so far. It wins by
normal victories, has zero draw/loss on this eval seed, and ends much earlier
than the death-time-heavy candidates.

Interpretation at update 90: strongest candidate so far after real resume.
Argmax wins all eval episodes by normal victories, with mean episode length
around `69` ticks on this eval seed.

Interpretation at update 100: still a strong argmax candidate after resume,
with almost all wins ending normally and no draw/loss on this eval seed.

Interpretation at update 110: another clean argmax checkpoint after resume,
confirming the post-80 policy region remains strong on the current eval seed.

Interpretation at update 120: still beats scripted with no losses, but weaker
than updates 90-110 because draws returned. Do not assume latest is best.

Interpretation at update 130: clean recovery after the update-120 dip.
Argmax again wins every eval episode by normal victory and ends faster than
the update-90/110 candidates.

Interpretation at update 140: another clean checkpoint, now under one minute
of simulation time on average. This is a strong scripted-gate candidate.

Interpretation at update 150: still clean, slightly faster than update 140.
No sign of death-time farming or argmax collapse on the current eval seed.

Interpretation at update 160: strongest current single-seed checkpoint by
speed, with `50/50` normal wins against scripted. It must still pass the
multi-seed argmax/stochastic sweep before self-play.

Interpretation at update 170: still clean, slightly faster than update 160 on
the current eval seed. No invalid-action or HOLD-collapse signal.

Interpretation at update 180: still clean. Speed is essentially tied with
update 170, so final selection should use the post-run multi-seed sweep rather
than single-seed mean length.

Interpretation at update 190: still clean and slightly faster. The argmax
policy changes its dominant actions, but not toward HOLD or invalid actions.

Interpretation at update 200: fastest clean checkpoint so far on the current
eval seed. Treat as a prime candidate, pending multi-seed argmax and
stochastic validation.

Interpretation at update 210: marginally faster than update 200 and still
clean. A3 argmax uses one transit action heavily, so stochastic validation is
important for league readiness.

Interpretation at update 220: effectively tied with update 210 and still
clean on argmax. Rollout update 221 showed a rare death-time stochastic win,
so the later stochastic sweep should separate "robust policy" from "fast but
narrow argmax."

Resume incident:

- After update 84 rollout collection crashed in `_finish_edge_battle` with
  `IndexError: list index out of range`.
- Root cause: edge-battle survivors were re-anchored to a shortened path but
  reused the old full-route `segment_index`. For a newly anchored edge
  continuation, the current edge segment must be index `1`.
- Fix:
  - edge-battle continuation now resumes at segment index `1`;
  - `_set_group_segment` now raises an explicit path/index diagnostic if this
    invariant is violated again.
- Regression checks after fix:
  - `pytest tests/test_lord_death_time.py -q`: `28 passed`;
  - `pytest tests -q`: `243 passed`;
  - `python scripts/crash_test_simulation.py --matches 100 --seed-base 20260604`:
    `seeds_with_issues=0`, `issue_summary={}`.
- `latest.pt` was update 80, so resume continues from update 81. This is the
  real checkpoint/resume test required by the Sprint 2/3 gate checklist.
- Resume command started with `start_update=81`, `updates_this_invocation=220`,
  and produced new rollout updates after restart. Checkpoint/resume is now
  validated by an actual restart, not just by a load/save smoke test.

Post-resume training tail:

- update 230: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=43.92`.
- update 240: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=43.34`.
- update 250: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=42.26`.
- update 260: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=42.18`.
- update 270: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=42.58`.
- update 280: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=41.76`.
- update 290: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=42.62`.
- update 300: `win_rate=1.00`, `normal_win_rate=1.00`,
  `death_time_win_rate=0.00`, `draw_rate=0.00`, `loss_rate=0.00`,
  `mean_episode_steps=42.58`.

Training completion:

- Fresh Sprint 3 scripted run finished at `total_steps=7,372,800`.
- Checkpoints exist through `checkpoint_update_300.pt`; `latest.pt` points to
  the final update.
- Post-training regression suite: `pytest -q` -> `243 passed`.
- Single eval seed from update 150 onward remained clean, but this is not the
  gate by itself. Gate selection proceeds through multi-seed argmax and
  stochastic sweeps.

Scripted sweep started:

- Command evaluates updates `160,180,200,220,240,250,260,270,280,290,300`.
- Modes: `argmax,stochastic`.
- Seeds: `20260604,20270604,20280604,20290604`.
- Episodes: `50` per seed block, `200` per checkpoint/mode.
- Control `scripted_control_team` on the same seeds produced `200/200` draws
  with `draw_no_core_damage`, so any accepted checkpoint must clearly beat this
  control rather than merely look good against a weak baseline metric.

Scripted sweep result:

- All tested checkpoints from update `160` through update `300` beat scripted
  in both `argmax` and `stochastic` modes:
  - `win_rate=1.00`;
  - `draw_rate=0.00`;
  - `loss_rate=0.00`;
  - invalid action rate was `0.00` in the reported player summaries.
- Nearly all scripted wins were normal. The only non-clean row was
  `update 300 stochastic`, with `normal_win_rate=0.995` and
  `death_time_win_rate=0.005`. This is one reason update `280`, not the latest
  checkpoint, was selected.
- Best pre-self-play checkpoint selected by speed and robustness:
  `training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt`.
- Update `280` scripted sweep:
  - argmax: `win_rate=1.00`, `normal_win_rate=1.00`,
    `mean_episode_steps=42.02`;
  - stochastic: `win_rate=1.00`, `normal_win_rate=1.00`,
    `mean_episode_steps=48.06`.

Random sanity after scripted training:

- Evaluated updates `260,280,300` against random/random controls.
- Random legal control: `win_rate=0.995`, `normal_win_rate=0.915`,
  `death_time_win_rate=0.080`, `loss_rate=0.005`.
- Update `280`:
  - argmax: `win_rate=1.00`, `normal_win_rate=1.00`,
    `death_time_win_rate=0.00`, `mean_episode_steps=51.34`;
  - stochastic: `win_rate=1.00`, `normal_win_rate=1.00`,
    `death_time_win_rate=0.00`, `mean_episode_steps=54.65`.

## Self-Play Preparation

Self-play pool:

```powershell
python scripts/build_self_play_pool.py --run-dir training_runs/sprint3_team_scripted_fresh_v1 --updates 160,200,240,260,280,300 --out training_runs/self_play_pools/sprint3_team_scripted_fresh_v1_pool.json
```

- Pool contains six scripted-fresh checkpoints.
- First self-play opponent for this run was fixed to update `280` in
  stochastic mode to isolate one variable: whether the new policy can improve
  against the accepted pre-self-play policy.

Self-play throughput benchmark:

```powershell
python scripts/benchmark_team_training_env_steps.py --duration-sec 30 --workers 16 --seed 20260604 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt --opponent-policy-mode stochastic --ally-bot scripted
```

- `world_steps_per_hour=1,898,960`.
- `agent_decisions_per_hour=5,696,881`.
- This is below the long-term `3M` world steps/hour target. The run was
  allowed anyway as an explicit Sprint 3 experiment, but self-play throughput
  remains a performance debt.

## Self-Play Run

Command:

```powershell
python scripts/train_team_policy.py --run-dir training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1 --overwrite --resume training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt --updates 100 --workers 16 --rollout-steps 512 --eval-episodes 50 --eval-interval 10 --checkpoint-interval 10 --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt --opponent-policy-mode stochastic --ally-bot scripted --seed 20260605 --torch-threads 1 --random-baseline-episodes 0
```

Completion:

- Run dir: `training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1`.
- Started from scripted-fresh update `280`.
- Finished at update `380`.
- `records=100`.
- `total_steps=9,338,880`.
- Final checkpoint: `checkpoint_update_380.pt`.
- `pytest -q` after the run: `243 passed`.

Inline eval against frozen stochastic update `280` opponent:

- update `300`: `win_rate=0.66`, `normal_win_rate=0.66`,
  `loss_rate=0.34`, `mean_episode_steps=94.44`.
- update `320`: `win_rate=0.84`, `normal_win_rate=0.84`,
  `loss_rate=0.16`, `mean_episode_steps=94.10`.
- update `350`: `win_rate=0.88`, `normal_win_rate=0.88`,
  `loss_rate=0.12`, `mean_episode_steps=79.48`.
- update `360`: `win_rate=0.96`, `normal_win_rate=0.94`,
  `death_time_win_rate=0.02`, `loss_rate=0.04`,
  `mean_episode_steps=79.50`.
- update `380`: `win_rate=0.94`, `normal_win_rate=0.94`,
  `death_time_win_rate=0.00`, `loss_rate=0.06`,
  `mean_episode_steps=76.78`.

Interpretation: self-play improved clearly against the frozen update `280`
opponent. Update `360` had the highest inline win rate but used a small
death-time fraction. Update `380` was cleaner and faster in the inline eval.

## Post-Self-Play Gates

Scripted preservation sweep:

- Updates: `300,320,340,350,360,370,380`.
- Modes: `argmax,stochastic`.
- Seeds: `20260604,20270604,20280604,20290604`.
- Episodes: `200` per checkpoint/mode.
- Every tested checkpoint and mode produced:
  - `win_rate=1.00`;
  - `normal_win_rate=1.00`;
  - `death_time_win_rate=0.00`;
  - `draw_rate=0.00`;
  - `loss_rate=0.00`.
- Scripted skill was preserved through self-play.

Policy-vs-policy sweep against frozen stochastic update `280`:

| Update | Mode | Win | Normal | Death | Loss | Mean ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 300 | argmax | 0.650 | 0.640 | 0.010 | 0.350 | 106.66 |
| 300 | stochastic | 0.495 | 0.490 | 0.005 | 0.505 | 99.08 |
| 320 | argmax | 0.825 | 0.825 | 0.000 | 0.175 | 99.71 |
| 320 | stochastic | 0.600 | 0.595 | 0.005 | 0.400 | 111.63 |
| 340 | argmax | 0.855 | 0.845 | 0.010 | 0.145 | 86.08 |
| 340 | stochastic | 0.645 | 0.640 | 0.005 | 0.355 | 101.22 |
| 350 | argmax | 0.895 | 0.895 | 0.000 | 0.105 | 77.20 |
| 350 | stochastic | 0.715 | 0.715 | 0.000 | 0.285 | 94.25 |
| 360 | argmax | 0.920 | 0.920 | 0.000 | 0.080 | 76.87 |
| 360 | stochastic | 0.710 | 0.705 | 0.005 | 0.290 | 94.17 |
| 370 | argmax | 0.885 | 0.885 | 0.000 | 0.115 | 78.35 |
| 370 | stochastic | 0.670 | 0.670 | 0.000 | 0.330 | 99.20 |
| 380 | argmax | 0.925 | 0.925 | 0.000 | 0.075 | 77.80 |
| 380 | stochastic | 0.795 | 0.790 | 0.005 | 0.205 | 91.49 |

Random sanity after self-play:

| Update | Mode | Win | Normal | Death | Loss | Mean ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 360 | argmax | 1.000 | 1.000 | 0.000 | 0.000 | 50.30 |
| 360 | stochastic | 1.000 | 1.000 | 0.000 | 0.000 | 57.27 |
| 380 | argmax | 1.000 | 1.000 | 0.000 | 0.000 | 51.91 |
| 380 | stochastic | 1.000 | 1.000 | 0.000 | 0.000 | 57.81 |

Lord impact analysis:

- Added reusable team-checkpoint analyzer:
  `scripts/analyze_team_policy_lord_stats.py`.
- This analyzer runs synchronized team policy episodes with replay/trace
  enabled and aggregates Lord kill/spawn/core impact through the same event
  parser used by random Lord analysis.
- Final checkpoint analyzed:
  `training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt`.
- Opponent: frozen stochastic update `280`.
- These are 100-match replay analyses, not replacements for the 200-episode
  gate evals above.

Argmax Lord analysis:

```powershell
python scripts/analyze_team_policy_lord_stats.py --matches 100 --seed 20260604 --workers 16 --checkpoint training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt --mode argmax --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt --opponent-policy-mode stochastic --ally-bot scripted --out analysis/team_policy_lord_stats_u380_vs_frozen_u280_argmax_100.json
```

- `TEAM_A` win rate: `0.90`.
- Winner types: `99` normal, `1` death-time.
- Neutral Lord killed: `100/100`.
- Lord spawned after kill: `88/100`.
- Killed but no owner: `12/100`.
- First Lord kill tick: mean `21.22`, median `17.5`, min `13`, max `52`.
- Owner teams after spawn: `TEAM_A=49`, `TEAM_B=39`.
- If `TEAM_A` owned the resurrected Lord, owner team won `49/49`.
- If `TEAM_B` owned the resurrected Lord, owner team won `9/39`;
  final policy still often recovered against enemy Lord ownership.
- Resurrected Lord destroyed a core in `55/88` spawns
  (`0.625` core-expended rate per spawn).
- Average Lord combat damage per spawn: `197.82`; average core damage:
  `39.01`; average enemy deaths during Lord-active ticks: `17.66`.

Stochastic Lord analysis:

```powershell
python scripts/analyze_team_policy_lord_stats.py --matches 100 --seed 20270604 --workers 16 --checkpoint training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt --mode stochastic --opponent-bot policy --opponent-policy-checkpoint training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt --opponent-policy-mode stochastic --ally-bot scripted --out analysis/team_policy_lord_stats_u380_vs_frozen_u280_stochastic_100.json
```

- `TEAM_A` win rate: `0.82`.
- Winner types: `100` normal, `0` death-time.
- Neutral Lord killed: `100/100`.
- Lord spawned after kill: `86/100`.
- Killed but no owner: `14/100`.
- First Lord kill tick: mean `20.50`, median `17`, min `12`, max `63`.
- Owner teams after spawn: `TEAM_A=42`, `TEAM_B=44`.
- If `TEAM_A` owned the resurrected Lord, owner team won `41/42`.
- If `TEAM_B` owned the resurrected Lord, owner team won `17/44`.
- Resurrected Lord destroyed a core in `68/86` spawns
  (`0.791` core-expended rate per spawn).
- Average Lord combat damage per spawn: `208.37`; average core damage:
  `50.37`; average enemy deaths during Lord-active ticks: `16.05`.

Final Sprint 3 candidate:

- `training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt`.
- Reason:
  - best post-self-play argmax win rate against frozen update `280`;
  - best stochastic win rate against frozen update `280`;
  - no scripted regression;
  - random sanity remains clean;
  - no invalid-action signal in the reported evals.

Open debt:

- Self-play speed is below target: `1.90M` world steps/hour vs the desired
  `3M+`. Fix later before longer league-scale training.
- The current self-play run used a fixed stochastic update `280` opponent, not
  a rotating league pool. This is intentional for the first isolated self-play
  proof, but the next self-play stage should use pool sampling and periodic
  promotion.
