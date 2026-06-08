# Sprint 3 Population Run Log

Date: 2026-06-05

## Hard Gates

- Teammate-swap eval is a required pre-population gate. Slot-preserving swaps passed before this run against scripted and a frozen stochastic policy opponent.
- Throughput optimization is accepted only when behavior is proven equivalent by tests. Safe optimizations were applied; deeper optimization is deferred because 3M world steps/hour was not reached without larger behavior-risking changes.
- Population work starts with at least 6 diverse seed lines before considering an 18-24 model expansion.

## Throughput

- Baseline policy-opponent benchmark before safe optimizations: about 1.756M world steps/hour.
- After safe optimizations and full regression tests: about 1.927M world steps/hour, about 5.782M controlled-agent decisions/hour.
- Status: below the 3M world steps/hour target. This remains a known debt, but it is not blocking this authorized run.

## Seed 01: `seed01_scripted_std`

- Run dir: `training_runs/population_s3_seed01_scripted_std_s20260651`
- Setup: fresh scripted opponent, seed `20260651`, 120 updates, entropy coef `0.01`.
- Random legal baseline: win `0.56`, normal `0.10`, death-time `0.46`, draw `0.14`, loss `0.30`.
- Update 20 argmax eval: win `1.00`, normal `0.08`, death-time `0.92`. This was death-time-heavy and not a clean candidate.
- Update 40 argmax eval: win `0.92`, normal `0.16`, death-time `0.76`, draw `0.08`. Still death-time-heavy.
- Update 80 argmax eval: win `1.00`, normal `0.98`, death-time `0.02`, invalid `0`. First clean strong checkpoint signal.
- Update 100 argmax eval: win `0.96`, normal `0.78`, death-time `0.18`, draw `0.04`. Not a strict improvement over update 80.
- Update 120 argmax eval during training: win `0.98`, normal `0.88`, death-time `0.10`, draw `0.02`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `0.92`, death-time `0.08`, loss `0.00`, draw `0.00`.
  - Stochastic: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`.
- Notes: valid working seed. The best checkpoint may be update 80 or latest depending on the selection objective; latest is fully usable, but update 80 looked cleaner in training argmax eval.

## Seed 02: `seed02_scripted_low_entropy`

- Run dir: `training_runs/population_s3_seed02_scripted_lowent_s20260652`
- Setup: fresh scripted opponent, seed `20260652`, 160 updates, entropy coef `0.005`.
- Random legal baseline: win `0.56`, normal `0.06`, death-time `0.50`, draw `0.12`, loss `0.32`.
- Update 20 argmax eval: win `1.00`, normal `0.00`, death-time `1.00`, invalid `0`. This checkpoint is rejected as timer-only behavior.
- Update 40 argmax eval: win `0.98`, normal `0.40`, death-time `0.58`, draw `0.02`, invalid `0`. Better than update 20, but still too death-time-heavy.
- Update 60 argmax eval: win `1.00`, normal `0.82`, death-time `0.18`, invalid `0`. First usable signal for this seed; still weaker than seed01 update 80 on normal-finish cleanliness.
- Update 80 argmax eval: win `0.94`, normal `0.76`, death-time `0.18`, loss `0.06`, invalid `0`. This checkpoint regressed versus rollout and should not be selected blindly.
- Update 100 argmax eval: win `0.98`, normal `0.94`, death-time `0.04`, loss `0.02`, invalid `0`, mean steps `94.08`. Strong checkpoint candidate.
- Update 120 argmax eval: win `1.00`, normal `1.00`, death-time `0.00`, invalid `0`, mean steps `78.38`. Best checkpoint signal so far for this seed.
- Update 160 argmax eval during training: win `1.00`, normal `1.00`, death-time `0.00`, invalid `0`, mean steps `58.38`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `58.38`, opening diversity `0.92`, sequence diversity `0.98`.
  - Stochastic: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `64.20`, opening diversity `1.00`, sequence diversity `1.00`.
- Notes: valid working seed. This is a fast low-entropy line; checkpoint latest is usable and currently stronger than the earlier candidates.
- Status: complete.

## Seed 03: `seed03_scripted_high_entropy`

- Run dir: `training_runs/population_s3_seed03_scripted_highent_s20260653`
- Setup: fresh scripted opponent, seed `20260653`, 140 updates, entropy coef `0.02`.
- Random legal baseline: win `0.58`, normal `0.16`, death-time `0.42`, draw `0.16`, loss `0.26`.
- Update 20 argmax eval: win `0.46`, normal `0.00`, death-time `0.46`, draw `0.54`, invalid `0`. Rejected; argmax is still degenerate/timer-draw heavy.
- Update 21 rollout immediately after eval: win `0.92`, normal `0.64`, death-time `0.28`, loss `0.08`, draw `0.00`; stochastic training behavior is improving faster than argmax.
- Update 40 argmax eval: win `1.00`, normal `0.66`, death-time `0.34`, loss `0.00`, draw `0.00`, invalid `0`. This is a working checkpoint, but still not clean enough because a third of wins rely on death-time.
- Update 60 argmax eval: win `1.00`, normal `0.38`, death-time `0.62`, loss `0.00`, draw `0.00`, invalid `0`. This regressed as a candidate despite stable win-rate; keep training, but do not select this checkpoint.
- Update 80 argmax eval: win `0.98`, normal `0.68`, death-time `0.30`, draw `0.02`, loss `0.00`, invalid `0`. Better than update 60, but still weaker than Seed01/Seed02 quality gates.
- Update 100 argmax eval: win `0.98`, normal `0.98`, death-time `0.00`, loss `0.02`, draw `0.00`, invalid `0`, mean steps `93.90`. First strong clean checkpoint for this high-entropy line.
- Update 120 argmax eval: win `0.96`, normal `0.90`, death-time `0.06`, loss `0.04`, draw `0.00`, invalid `0`, mean steps `110.70`. This regressed versus update 100; keep update 100 as best-so-far.
- Update 140 argmax eval during training: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, invalid `0`, mean steps `68.40`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `67.96`, opening diversity `1.00`, sequence diversity `1.00`.
  - Stochastic: win `1.00`, normal `0.98`, death-time `0.02`, loss `0.00`, draw `0.00`, mean steps `119.62`, opening diversity `1.00`, sequence diversity `1.00`.
- Notes: valid working seed. The line was noisy in the middle because of higher entropy, but latest recovered into a clean argmax checkpoint; latest is a usable population candidate.
- Status: complete.

## Seed 04: `seed04_selfplay_fresh_vs_u280`

- Run dir: `training_runs/population_s3_seed04_selfplay_fresh_vs_u280_s20260654`
- Setup: fresh policy against frozen stochastic self-play checkpoint update 280, seed `20260654`, 140 updates, entropy coef `0.01`.
- Early status: running. The first updates are losing hard against the frozen policy, which is expected for a fresh line in this harder regime. Do not judge this seed before the scheduled argmax gates.
- Update 20 argmax eval: win `0.00`, normal `0.00`, death-time `0.00`, draw `0.28`, loss `0.72`, invalid `0`, mean steps `254.04`.
- Notes: first argmax gate is not usable. The policy is partly degenerate under argmax, with dominant actions around repeated skirmisher capture/transit patterns. Continue to later gates, but do not count this as a working population seed yet.
- Update 40 argmax eval: win `0.00`, normal `0.00`, death-time `0.00`, draw `0.32`, loss `0.68`, invalid `0`, mean steps `300.00`.
- Notes: second argmax gate failed harder from an action-collapse perspective. A1 and A2 are nearly deterministic on repeated `CP6` capture actions. This line is currently classified as failed hard-start self-play and should not be counted toward the minimum 6 working seeds unless a later gate recovers dramatically.
- Update 60 argmax eval: win `0.00`, normal `0.00`, death-time `0.00`, draw `0.44`, loss `0.56`, invalid `0`, mean steps `299.98`.
- Stop decision: manually stopped after update 60 because the hard-start self-play gate failed three times in a row. Rollout stochastic win-rate briefly touched `0.10-0.13`, but argmax stayed at `0.00` and collapsed into repeated `CP6` actions for all three agents. This line is rejected as a working population seed.
- Replacement decision: use a staged/warm-start self-play variant instead of continuing fresh-vs-frozen-u280 to update 140.
- Status: stopped and rejected.

## Recovery Batch: `sprint3_population_recovery_v1`

- Reason: the original Seed04 hard-start proved that fresh-vs-strong-frozen is too harsh and collapses argmax. To preserve the minimum 6 working seed target, the remaining run is switched to a recovery/curriculum batch.
- Selected runs:
  - `seed04_warmstart_seed03_vs_u280`: resume Seed03 latest, continue against frozen scripted-u280 stochastic for 80 updates, entropy `0.015`.
  - `seed05_resume_u280_vs_u380`: resume scripted-u280, continue against frozen self-play-u380 stochastic for 120 updates.
  - `seed06_resume_u380_vs_u240_argmax`: resume self-play-u380, continue against scripted-u240 argmax for 100 updates.
  - `seed07_scripted_alt_entropy`: backup fresh scripted anchor with alternate seed and entropy `0.015` for 120 updates.
- Dry-run validation: passed. Referenced checkpoints exist and are compatible.

### Recovery Seed04: `seed04_warmstart_seed03_vs_u280`

- Run dir: `training_runs/population_s3_seed04_warmstart_seed03_vs_u280_s20260664`
- Setup: warm-start from Seed03 latest, opponent frozen scripted-u280 stochastic, seed `20260664`, 80 continuation updates, entropy coef `0.015`.
- Update 160 argmax eval vs frozen: win `0.54`, normal `0.54`, death-time `0.00`, draw `0.02`, loss `0.44`, invalid `0`, mean steps `134.08`.
- Update 180 argmax eval vs frozen: win `0.56`, normal `0.54`, death-time `0.02`, draw `0.00`, loss `0.44`, invalid `0`, mean steps `104.26`.
- Update 200 argmax eval vs frozen: win `0.88`, normal `0.88`, death-time `0.00`, draw `0.00`, loss `0.12`, invalid `0`, mean steps `85.36`.
- Update 220 argmax eval vs frozen during training: win `0.68`, normal `0.68`, death-time `0.00`, draw `0.00`, loss `0.32`, invalid `0`, mean steps `108.78`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `78.94`, opening diversity `0.94`, sequence diversity `1.00`.
  - Stochastic: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `117.66`, opening diversity `1.00`, sequence diversity `1.00`.
- Notes: valid working seed. Against the harder frozen opponent, update 200 was stronger than final/latest, so checkpoint selection may prefer update 200 if we need an anti-frozen candidate. Latest is still fully usable as a population seed because scripted argmax/stochastic post-eval is clean.
- Status: complete.

### Recovery Seed05: `seed05_resume_u280_vs_u380`

- Run dir: `training_runs/population_s3_seed05_resume_u280_vs_u380_s20260655`
- Setup: resume scripted-u280, continue against frozen self-play-u380 stochastic, seed `20260655`, 120 continuation updates.
- Early status: running. This is a harder opponent than scripted, and rollout is initially losing hard after resume; do not classify until the scheduled argmax gates.
- Update 300 argmax eval vs frozen: win `0.54`, normal `0.52`, death-time `0.02`, draw `0.00`, loss `0.46`, invalid `0`, mean steps `125.98`.
- Update 320 argmax eval vs frozen: win `0.32`, normal `0.32`, death-time `0.00`, draw `0.00`, loss `0.68`, invalid `0`, mean steps `110.24`.
- Update 340 argmax eval vs frozen: win `0.48`, normal `0.48`, death-time `0.00`, draw `0.00`, loss `0.52`, invalid `0`, mean steps `117.92`.
- Update 360 argmax eval vs frozen: win `0.82`, normal `0.82`, death-time `0.00`, draw `0.00`, loss `0.18`, invalid `0`, mean steps `107.04`.
- Update 380 argmax eval vs frozen: win `0.80`, normal `0.80`, death-time `0.00`, draw `0.00`, loss `0.20`, invalid `0`, mean steps `108.26`.
- Update 400 argmax eval vs frozen: win `0.82`, normal `0.80`, death-time `0.02`, draw `0.00`, loss `0.18`, invalid `0`, mean steps `91.60`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `51.64`, opening diversity `0.72`, sequence diversity `1.00`.
  - Stochastic: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `58.46`, opening diversity `1.00`, sequence diversity `1.00`.
- Notes: valid working seed. First gate was weak but alive; update 320 regressed sharply, update 340 only partially recovered, then updates 360, 380, and 400 became strong frozen-opponent checkpoints. Latest is selected unless later teammate-swap shows this line is too overfit.
- Status: complete.

### Recovery Seed06: `seed06_resume_u380_vs_u240_argmax`

- Run dir: `training_runs/population_s3_seed06_resume_u380_vs_u240_argmax_s20260656`
- Setup: resume self-play update 380, continue against frozen scripted-u240 argmax for 100 continuation updates, seed `20260656`.
- Update 400 argmax eval vs frozen: win `0.56`, normal `0.38`, death-time `0.18`, draw `0.38`, loss `0.06`, invalid `0`, mean steps `222.26`.
- Update 420 argmax eval vs frozen: win `0.62`, normal `0.54`, death-time `0.08`, draw `0.18`, loss `0.20`, invalid `0`, mean steps `175.62`.
- Update 440 argmax eval vs frozen: win `0.72`, normal `0.68`, death-time `0.04`, draw `0.14`, loss `0.14`, invalid `0`, mean steps `146.56`.
- Update 460 argmax eval vs frozen: win `0.98`, normal `0.92`, death-time `0.06`, draw `0.00`, loss `0.02`, invalid `0`, mean steps `104.46`. This is the strongest frozen-opponent gate for this seed.
- Update 480 argmax eval vs frozen during training: win `0.98`, normal `0.90`, death-time `0.08`, draw `0.00`, loss `0.02`, invalid `0`, mean steps `88.86`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `54.98`, opening diversity `0.92`, sequence diversity `0.98`.
  - Stochastic: win `1.00`, normal `1.00`, death-time `0.00`, loss `0.00`, draw `0.00`, mean steps `71.28`, opening diversity `1.00`, sequence diversity `1.00`.
- Notes: valid working seed. Update 460 is the best anti-frozen checkpoint, while latest is a clean scripted candidate and does not collapse under stochastic sampling.
- Status: complete.

### Recovery Seed07: `seed07_scripted_alt_entropy`

- Run dir: `training_runs/population_s3_seed07_scripted_altent_s20260657`
- Setup: fresh scripted anchor with alternate seed `20260657`, 120 updates, entropy coef `0.015`.
- Update 20 argmax eval vs scripted: win `0.94`, normal `0.18`, death-time `0.76`, draw `0.06`, loss `0.00`, invalid `0`, mean steps `279.10`. This checkpoint is timer-heavy and should not be selected.
- Update 40 argmax eval vs scripted: win `1.00`, normal `0.06`, death-time `0.94`, draw `0.00`, loss `0.00`, invalid `0`, mean steps `291.16`. This checkpoint is also rejected as death-time-heavy and shows partial action dominance.
- Update 60 argmax eval vs scripted: win `0.94`, normal `0.72`, death-time `0.22`, draw `0.04`, loss `0.02`, invalid `0`, mean steps `198.14`. Recovering, but still not clean enough.
- Update 80 argmax eval vs scripted: win `1.00`, normal `0.94`, death-time `0.06`, draw `0.00`, loss `0.00`, invalid `0`, mean steps `146.72`. First clean candidate.
- Update 100 argmax eval vs scripted: win `1.00`, normal `0.96`, death-time `0.04`, draw `0.00`, loss `0.00`, invalid `0`, mean steps `122.12`.
- Update 120 argmax eval during training: win `1.00`, normal `1.00`, death-time `0.00`, draw `0.00`, loss `0.00`, invalid `0`, mean steps `126.16`.
- Post-eval latest vs scripted:
  - Argmax: win `1.00`, normal `0.96`, death-time `0.04`, loss `0.00`, draw `0.00`, mean steps `99.34`, opening diversity `0.98`, sequence diversity `1.00`.
  - Stochastic: win `1.00`, normal `0.96`, death-time `0.04`, loss `0.00`, draw `0.00`, mean steps `138.58`, opening diversity `1.00`, sequence diversity `1.00`.
- Notes: valid working seed and useful as an extra alternate scripted line. Early checkpoints were timer-heavy, but latest is stable enough for population/swap evaluation.
- Status: complete.

## Pre-Population Candidate Set

- Minimum working target: at least 6 valid lines.
- Current status: 7 valid lines available, with the original hard-start Seed04 explicitly rejected and replaced by recovery Seed04.
- Candidate latest checkpoints:
  - Seed01: `training_runs/population_s3_seed01_scripted_std_s20260651/checkpoints/latest.pt`
  - Seed02: `training_runs/population_s3_seed02_scripted_lowent_s20260652/checkpoints/latest.pt`
  - Seed03: `training_runs/population_s3_seed03_scripted_highent_s20260653/checkpoints/latest.pt`
  - Seed04 recovery: `training_runs/population_s3_seed04_warmstart_seed03_vs_u280_s20260664/checkpoints/latest.pt`
  - Seed05 recovery: `training_runs/population_s3_seed05_resume_u280_vs_u380_s20260655/checkpoints/latest.pt`
  - Seed06 recovery: `training_runs/population_s3_seed06_resume_u380_vs_u240_argmax_s20260656/checkpoints/latest.pt`
  - Seed07 recovery: `training_runs/population_s3_seed07_scripted_altent_s20260657/checkpoints/latest.pt`
- Remaining gate before calling this population-ready: final teammate-swap eval on mixed teams from the candidate set.

## Final Teammate-Swap Eval

### Against Scripted

- Eval dir: `training_runs/evals/pre_population_teammate_swap_population_v1`
- Checkpoints: 7 latest population candidates.
- Suite: `native,slot_mixed`
- Modes: `argmax,stochastic`
- Seed bases: `20260605,20270605,20280605,20290605`
- Episodes: 25 per seed block, 100 per team/mode.

Summary:

| team kind | mode | win min | win avg | win max | normal avg | death avg | loss avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 1.000 | 1.000 | 1.000 | 0.983 | 0.017 | 0.000 |
| native | stochastic | 0.980 | 0.997 | 1.000 | 0.984 | 0.013 | 0.003 |
| slot_mixed | argmax | 0.990 | 0.998 | 1.000 | 0.992 | 0.006 | 0.002 |
| slot_mixed | stochastic | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

Interpretation:

- Scripted teammate-swap gate passes.
- Slot-preserving mixed teams do not collapse.
- Stochastic league-mode sampling is not worse than argmax in this scripted gate.
- No invalid-action collapse was observed.

### Against Frozen Policy `sprint3_team_scripted_fresh_v1/checkpoint_update_280.pt`

- Eval dir: `training_runs/evals/pre_population_teammate_swap_population_vs_frozen_u280_v1`
- Opponent: frozen stochastic policy checkpoint update 280.
- Seed bases: `20260615,20270615`
- Episodes: 20 per seed block, 40 per team/mode.

Summary:

| team kind | mode | win min | win avg | win max | normal avg | death avg | loss avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.150 | 0.607 | 0.775 | 0.600 | 0.007 | 0.393 |
| native | stochastic | 0.050 | 0.314 | 0.625 | 0.300 | 0.014 | 0.686 |
| slot_mixed | argmax | 0.450 | 0.655 | 0.875 | 0.655 | 0.000 | 0.340 |
| slot_mixed | stochastic | 0.225 | 0.425 | 0.650 | 0.420 | 0.005 | 0.575 |

Interpretation:

- Frozen u280 is a much harder opponent than scripted and exposes a real strength spectrum.
- Some native lines and some mixed teams are weak against this frozen checkpoint, especially under stochastic sampling.
- Mixed teams are not worse on average than native teams in this batch; the best mixed argmax team is stronger than the best native argmax team here.
- This supports accepting the 7-line foundation, but the next expansion should include mixed-team/self-play curriculum rather than only producing more scripted-anchor clones.

## Role-Stress Eval

Purpose:

- Test whether a policy trained for one slot can be assigned to a different slot.
- Stress mapping: target `A1 <- source A2`, target `A2 <- source A3`, target `A3 <- source A1`.
- This is stricter than normal league teammate-swap. It is relevant only if the future league wants agents to be role-flexible across A1/A2/A3 instead of slot-specialized.

### Against Scripted

- Eval dir: `training_runs/evals/pre_population_role_stress_scripted_v1`
- Checkpoints: 7 latest population candidates.
- Suite: `role_stress`
- Modes: `argmax,stochastic`
- Seed bases: `20260625,20270625,20280625,20290625`
- Episodes: 25 per seed block, 100 per team/mode.

Summary:

| mode | win avg | normal avg | death avg | draw avg | loss avg | mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax | 0.956 | 0.716 | 0.240 | 0.029 | 0.016 | 176.4 |
| stochastic | 0.991 | 0.940 | 0.051 | 0.000 | 0.009 | 142.0 |

Interpretation:

- Against weak scripted opposition, cross-slot role-stress mostly survives.
- Argmax is noticeably less clean than stochastic: more death-time wins, more draws, and lower normal-win rate.
- This is not enough to approve cross-slot role-flexibility because scripted is too weak to expose the issue fully.

### Against Frozen Policy `sprint3_team_scripted_fresh_v1/checkpoint_update_280.pt`

- Eval dir: `training_runs/evals/pre_population_role_stress_vs_frozen_u280_v1`
- Opponent: frozen stochastic policy checkpoint update 280.
- Seed bases: `20260635,20270635`
- Episodes: 20 per seed block, 40 per team/mode.

Summary:

| mode | win avg | normal avg | death avg | draw avg | loss avg | mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax | 0.071 | 0.054 | 0.018 | 0.093 | 0.836 | 134.1 |
| stochastic | 0.057 | 0.057 | 0.000 | 0.000 | 0.943 | 73.6 |

Interpretation:

- Cross-slot role-stress fails hard against trained opposition.
- Current candidates are suitable as slot-specialized policies, not as arbitrary role-swappable policies.
- The next large run should not assume that A1/A2/A3 source policies are interchangeable.
- If role-flexibility is desired, add a dedicated role-randomized curriculum and likely bump the observation schema to expose an explicit role/slot signal.

## Final Decision

- Minimum working seed target is met: 7 valid candidates for a target of 6.
- Original hard-start Seed04 remains rejected and must not be counted.
- Scripted teammate-swap gate is accepted.
- Frozen-policy teammate-swap does not block the foundation, but marks mixed-team/self-play curriculum as important for the next expansion.
- Role-stress does block arbitrary cross-slot policy mixing. The foundation remains accepted only under slot-preserving use: A1-style policies in A1/B1 slots, A2-style policies in A2/B2 slots, and A3-style policies in A3/B3 slots.
- Throughput remains below the 3M world-steps/hour target. The accepted safe value is still about `1.927M` world steps/hour, so performance optimization remains an explicit debt and should only be revisited with exact replay/state equivalence checks.
- Population foundation status: accepted.

## 2026-06-07 Fresh Universal Roots v3.1

Purpose:

- Restart the population foundation after the new v3.1 action/observation and
  balance changes.
- Build at least 6 fresh universal roots.
- Do not train roots into explicit tactics or special counters. Diversity must
  come from seed and hyperparameter variation, not from scripted-core,
  raider-core, BMP-core, or other narrow opponent curricula.

Correction:

- An earlier batch template included special opponent variants for later roots.
  This was rejected before those special-opponent roots were trained.
- The batch script was corrected so all 6 fresh roots train only against the
  generic `scripted` opponent with role shuffle enabled.
- The first two completed roots were already generic scripted roots and remain
  valid. The remaining four roots were then trained from scratch with the same
  universal rule.

Training artifacts:

| root | run dir | selected checkpoint |
| --- | --- | --- |
| fresh01 | `training_runs/fresh_population_v31_seed01_scripted_std_s20261201` | `checkpoint_update_300.pt` |
| fresh02 | `training_runs/fresh_population_v31_seed02_scripted_lowent_s20261202` | `checkpoint_update_320.pt` |
| fresh03 | `training_runs/fresh_population_v31_seed03_scripted_highent_s20261203` | `checkpoint_update_240.pt` |
| fresh04 | `training_runs/fresh_population_v31_seed04_scripted_fastlr_s20261204` | `checkpoint_update_240.pt` |
| fresh05 | `training_runs/fresh_population_v31_seed05_scripted_slow_highent_s20261205` | `checkpoint_update_280.pt` |
| fresh06 | `training_runs/fresh_population_v31_seed06_scripted_wide_minibatch_s20261206` | `checkpoint_update_320.pt` |

Selection eval:

- Eval dirs: `training_runs/evals/fresh_population_v31_selection_fresh01`
  through `fresh_population_v31_selection_fresh06`.
- Opponent: generic `scripted`.
- Modes: `argmax,stochastic`.
- Seed bases: `20261331,20271331,20281331,20291331`.
- Episodes: 50 per seed block, 200 per checkpoint/mode.

Best selected checkpoint summary against scripted:

| root | checkpoint | argmax win | argmax normal | stochastic win | stochastic normal | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fresh01 | u300 | 1.000 | 1.000 | 0.960 | 0.960 | strong |
| fresh02 | u320 | 1.000 | 1.000 | 0.905 | 0.905 | strong argmax, usable stochastic |
| fresh03 | u240 | 0.985 | 0.985 | 0.805 | 0.800 | weakest stochastic root |
| fresh04 | u240 | 1.000 | 1.000 | 0.860 | 0.855 | u320 is more stochastic-friendly but weaker argmax |
| fresh05 | u280 | 0.995 | 0.995 | 0.785 | 0.780 | reject u340 despite better stochastic; u340 argmax collapses |
| fresh06 | u320 | 1.000 | 1.000 | 0.940 | 0.940 | strongest overall, u220 has best stochastic |

Teammate-swap gate:

- Eval dir: `training_runs/evals/fresh_population_v31_teammate_swap_universal_roots_v1`.
- Checkpoints: the selected 6 fresh roots above.
- Suite: `native,slot_mixed,role_stress`.
- Modes: `argmax,stochastic`.
- Seed bases: `20261441,20271441,20281441,20291441`.
- Episodes: 50 per seed block, 200 per team/mode.
- Observation/action metadata: `train_obs_v3.1`, observation dim `5188`,
  action dim `447`.
- Records: 48/48 completed, no stderr errors.

Summary:

| team kind | mode | win avg | win min | win max |
| --- | ---: | ---: | ---: | ---: |
| native | argmax | 0.994 | 0.965 | 1.000 |
| native | stochastic | 0.894 | 0.815 | 0.965 |
| slot_mixed | argmax | 0.970 | 0.745 | 1.000 |
| slot_mixed | stochastic | 0.907 | 0.845 | 0.960 |
| role_stress | argmax | 0.943 | 0.725 | 1.000 |
| role_stress | stochastic | 0.912 | 0.825 | 0.975 |

Weak samples below 0.85 win-rate:

```text
0.725 role_stress argmax: fresh03 u240 with A1<-A2, A2<-A3, A3<-A1
0.745 slot_mixed argmax: A1 fresh03 u240 + A2 fresh05 u280 + A3 fresh04 u240
0.815 native stochastic: fresh03 u240
0.825 role_stress stochastic: fresh05 u280 with A1<-A2, A2<-A3, A3<-A1
0.835 role_stress stochastic: fresh03 u240 with A1<-A2, A2<-A3, A3<-A1
0.845 native stochastic: fresh05 u280
0.845 slot_mixed stochastic: A1 fresh03 u240 + A2 fresh04 u240 + A3 fresh05 u280
```

Decision:

- The 6-root universal foundation is valid against generic scripted.
- Slot-mixed teammate-swap mostly passes; mixed teams do not collapse on
  average.
- Role-stress is no longer a hard failure against scripted in v3.1, but
  `fresh03` and `fresh05` are weaker and should not be treated as equal-quality
  league roots without stabilization or peer/self-play exposure.
- The next large run should keep these as universal roots and proceed into a
  mixed-team/self-play bridge instead of adding tactical-specialized roots.
- Any later branch specialization must happen after these universal roots, not
  in the root-training phase.

## 2026-06-07 Fresh v3.1 Self-Play Bridge

Purpose:

- Continue the 6 fresh universal roots against other universal roots.
- Keep roots universal. This is not a tactical-specialized curriculum.
- Stabilize the weaker fresh roots (`fresh03`, `fresh05`) without training them
  toward a hand-authored style.

Batch:

- Script: `scripts/run_fresh_population_v31_selfplay_bridge.py`.
- Batch id: `fresh_population_v31_selfplay_bridge_v1`.
- Batch dir:
  `training_runs/fresh_population_v31_bridge_batches/fresh_population_v31_selfplay_bridge_v1`.
- Workers: 16.
- Rollout steps: 512.
- Role assignment: `shuffle`.
- Opponent type: frozen universal policy peers.
- Result: 6/6 runs complete, 6 compatible checkpoints in manifest.

Runs:

| bridge | resume | opponent | updates | final/best internal peer eval |
| --- | --- | --- | ---: | --- |
| bridge01 | fresh01 u300 | fresh06 u320 stochastic | 140 | u440: win 0.96, normal 0.96 |
| bridge02 | fresh02 u320 | fresh04 u240 stochastic | 140 | u460: win 1.00, normal 1.00 |
| bridge03 | fresh03 u240 | fresh06 u320 stochastic | 180 | u400/u420: win 0.98, normal 0.98 |
| bridge04 | fresh04 u240 | fresh02 u320 stochastic | 140 | u360/u380: win 1.00, normal 1.00 |
| bridge05 | fresh05 u280 | fresh01 u300 stochastic | 180 | u300/u340/u360/u380/u400/u420: win 0.98, normal 0.98 |
| bridge06 | fresh06 u320 | fresh03 u240 stochastic | 120 | u440: win 1.00, normal 1.00 |

Selection eval against generic scripted:

- Eval dirs: `training_runs/evals/fresh_population_v31_bridge_selection_01`
  through `fresh_population_v31_bridge_selection_06`.
- Modes: `argmax,stochastic`.
- Seed bases: `20261661,20271661,20281661,20291661`.
- Episodes: 50 per seed block, 200 per checkpoint/mode.

Selected bridge checkpoint summary:

| root | checkpoint | argmax win | argmax normal | stochastic win | stochastic normal | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bridge01 | u440 | 0.895 | 0.895 | 0.970 | 0.965 | stochastic improved, argmax scripted weaker |
| bridge02 | u460 | 1.000 | 0.990 | 0.995 | 0.995 | strong |
| bridge03 | u400 | 1.000 | 1.000 | 0.980 | 0.980 | weak fresh root recovered |
| bridge04 | u380 | 0.920 | 0.920 | 0.990 | 0.990 | stochastic strong, argmax scripted weaker |
| bridge05 | u420 | 0.990 | 0.960 | 0.995 | 0.990 | strong stochastic; u300 has cleanest argmax |
| bridge06 | u440 | 0.905 | 0.905 | 0.995 | 0.995 | stochastic strong, argmax scripted weaker |

Interpretation:

- Self-play bridge improves league-mode stochastic performance.
- It can reduce argmax performance against the weak scripted baseline for some
  roots (`bridge01`, `bridge04`, `bridge06`), so selection cannot use a single
  metric.
- For future league use, stochastic robustness is more important than scripted
  argmax perfection, but scripted argmax remains a useful regression guard.
- `fresh03` and `fresh05` were not lost; both improved under universal peer
  bridge without tactical specialization.

Bridge teammate-swap gate:

- Eval dir: `training_runs/evals/fresh_population_v31_bridge_teammate_swap_v1`.
- Suite: `native,slot_mixed,role_stress`.
- Modes: `argmax,stochastic`.
- Seed bases: `20261771,20271771,20281771,20291771`.
- Episodes: 50 per seed block, 200 per team/mode.
- Records: 48/48 complete.

Summary:

| team kind | mode | win avg | win min | win max | normal avg | loss avg | death avg | draw avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.943 | 0.880 | 1.000 | 0.934 | 0.058 | 0.008 | 0.000 |
| native | stochastic | 0.981 | 0.970 | 0.990 | 0.981 | 0.019 | 0.000 | 0.000 |
| slot_mixed | argmax | 0.981 | 0.880 | 1.000 | 0.981 | 0.019 | 0.000 | 0.000 |
| slot_mixed | stochastic | 0.973 | 0.930 | 1.000 | 0.973 | 0.028 | 0.000 | 0.000 |
| role_stress | argmax | 0.974 | 0.905 | 1.000 | 0.974 | 0.026 | 0.000 | 0.000 |
| role_stress | stochastic | 0.975 | 0.955 | 0.990 | 0.975 | 0.025 | 0.000 | 0.000 |

Weak samples below 0.90:

```text
0.880 native argmax: bridge06 u440
0.880 slot_mixed argmax: bridge04 u380 + bridge06 u440 + bridge05 u420
0.885 native argmax: bridge01 u440
```

Decision:

- Bridge population is accepted as a stronger universal stochastic pool than
  the raw fresh roots.
- It is not accepted as "argmax-perfect against scripted"; that is not the
  right league objective, but it remains a regression note.
- Next step should be a peer/self-play eval or bridge-vs-bridge league gate,
  not another scripted-only training run.
- Do not expand to 18-24 equal roots until the bridge pool is checked against
  trained peers, not just scripted.

### Peer Gate Against Frozen `bridge03 u400`

Purpose:

- Check whether the bridge pool only beats scripted or can also stand against a
  trained peer.
- Opponent: `bridge03 u400` as frozen stochastic policy.
- First full-size attempt: `fresh_population_v31_bridge_teammate_swap_vs_bridge03_u400_v1`.
  This reached 35/36 records before the shell timeout stopped the final record.
- Clean artifact: `fresh_population_v31_bridge_teammate_swap_vs_bridge03_u400_quick_v1`.
  This used 2 seed blocks and 25 episodes per block, 50 episodes per team/mode,
  and completed 36/36 records.

Quick peer-gate summary:

| team kind | mode | win avg | win min | win max | normal avg | loss avg | death avg | draw avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.673 | 0.400 | 0.840 | 0.617 | 0.310 | 0.057 | 0.017 |
| native | stochastic | 0.323 | 0.120 | 0.460 | 0.323 | 0.677 | 0.000 | 0.000 |
| slot_mixed | argmax | 0.595 | 0.240 | 0.860 | 0.578 | 0.400 | 0.017 | 0.005 |
| slot_mixed | stochastic | 0.375 | 0.200 | 0.580 | 0.375 | 0.625 | 0.000 | 0.000 |

Interpretation:

- Scripted gates are not enough. Against a trained peer, the bridge pool is not
  yet league-ready.
- Argmax has some useful teams, but stochastic league-mode performance is weak.
- The immediate next training phase should be a peer/self-play curriculum from
  the bridge pool, not expansion to 18-24 roots.
- This is still a good result for the current phase: we now have a working
  universal foundation and a clear harder opponent signal.

## Fresh Population v3.1 Peer Curriculum Batch

Date: 2026-06-08.

Purpose:

- Continue from the six universal bridge roots without tactical
  specialization.
- Train each root against a frozen trained peer, with role shuffle still
  enabled.
- Validate that the v3.1/factorized action-space population can learn against
  trained policy opponents before expanding toward 18-24 league roots.

Batch:

- Batch id: `fresh_population_v31_peer_curriculum_v1`.
- Batch dir:
  `training_runs/fresh_population_v31_peer_batches/fresh_population_v31_peer_curriculum_v1`.
- Script: `scripts/run_fresh_population_v31_peer_curriculum.py`.
- Workers: 16.
- Rollout steps: 512.
- Eval interval: 20 updates.
- Eval episodes per internal eval: 60.
- Role assignment: `shuffle`.
- Opponents: frozen stochastic bridge policies.
- Result: 6/6 runs complete, all returncode `0`, empty `stderr.log`.

Final internal evals against each run's frozen opponent:

| peer run | resume | opponent | final update | final eval win | final eval loss | final eval draw |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| peer01 | bridge01 u440 | bridge03 u400 stochastic | 660 | 0.983 | 0.017 | 0.000 |
| peer02 | bridge02 u460 | bridge03 u400 stochastic | 700 | 0.917 | 0.083 | 0.000 |
| peer03 | bridge03 u400 | bridge06 u440 stochastic | 580 | 0.983 | 0.017 | 0.000 |
| peer04 | bridge04 u380 | bridge03 u400 stochastic | 600 | 0.933 | 0.067 | 0.000 |
| peer05 | bridge05 u420 | bridge03 u400 stochastic | 660 | 1.000 | 0.000 | 0.000 |
| peer06 | bridge06 u440 | bridge02 u460 stochastic | 620 | 0.900 | 0.100 | 0.000 |

Selected checkpoints:

```text
peer01: training_runs/fresh_population_v31_peer_seed01_vs_bridge03_s20262001/checkpoints/checkpoint_update_660.pt
peer02: training_runs/fresh_population_v31_peer_seed02_vs_bridge03_s20262002/checkpoints/checkpoint_update_700.pt
peer03: training_runs/fresh_population_v31_peer_seed03_vs_bridge06_s20262003/checkpoints/checkpoint_update_580.pt
peer04: training_runs/fresh_population_v31_peer_seed04_vs_bridge03_s20262004/checkpoints/checkpoint_update_600.pt
peer05: training_runs/fresh_population_v31_peer_seed05_vs_bridge03_s20262005/checkpoints/checkpoint_update_660.pt
peer06: training_runs/fresh_population_v31_peer_seed06_vs_bridge02_s20262006/checkpoints/checkpoint_update_620.pt
```

Interpretation:

- All six lines learned against trained peer opponents.
- No line collapsed into draw-heavy play during internal eval.
- The lines are not clones: starts differed strongly (`peer02` and `peer05`
  began weak and recovered; `peer01` and `peer03` were strong earlier).
- Internal opponent eval is not enough for league acceptance, so the batch was
  followed by scripted, teammate-swap, and cross-peer gates.

### Gate: Scripted-Core Teammate Swap

Eval dir:
`training_runs/fresh_population_v31_peer_batches/fresh_population_v31_peer_curriculum_v1/eval_scripted_core_swap`.

Setup:

- Suite: `native,slot_mixed,role_stress`.
- Modes: `argmax,stochastic`.
- Opponent: `scripted_core`.
- Seed bases: `20300101,20310101,20320101,20330101`.
- Episodes: 20 per seed block, 80 per team/mode.
- Records: 48/48 complete.

Summary:

| team kind | mode | win avg | win min | win max |
| --- | --- | ---: | ---: | ---: |
| native | argmax | 0.965 | 0.900 | 1.000 |
| native | stochastic | 0.988 | 0.975 | 1.000 |
| role_stress | argmax | 0.971 | 0.925 | 1.000 |
| role_stress | stochastic | 0.994 | 0.988 | 1.000 |
| slot_mixed | argmax | 0.798 | 0.288 | 1.000 |
| slot_mixed | stochastic | 0.982 | 0.913 | 1.000 |

Important note:

- Native and role-stress scripted-core performance is strong.
- Stochastic slot-mixed is strong.
- Argmax slot-mixed has a severe outlier at `0.288`, so deterministic
  mixed-team play is not yet reliable. This matters less for the intended
  stochastic league mode, but it is a useful regression signal.

### Gate: Frozen `bridge03 u400` Policy Teammate Swap

Eval dir:
`training_runs/fresh_population_v31_peer_batches/fresh_population_v31_peer_curriculum_v1/eval_bridge03_policy_swap`.

Setup:

- Suite: `native,slot_mixed,role_stress`.
- Modes: `argmax,stochastic`.
- Opponent: `bridge03 u400` as frozen stochastic policy.
- Seed bases: `20340101,20350101,20360101,20370101`.
- Episodes: 15 per seed block, 60 per team/mode.
- Records: 48/48 complete.

Summary:

| team kind | mode | win avg | win min | win max |
| --- | --- | ---: | ---: | ---: |
| native | argmax | 0.964 | 0.950 | 1.000 |
| native | stochastic | 0.892 | 0.717 | 0.983 |
| role_stress | argmax | 0.964 | 0.900 | 1.000 |
| role_stress | stochastic | 0.875 | 0.617 | 1.000 |
| slot_mixed | argmax | 0.875 | 0.567 | 1.000 |
| slot_mixed | stochastic | 0.815 | 0.533 | 0.983 |

Weak samples:

```text
0.533 slot_mixed stochastic:
  peer05 A1 + peer01 A2 + peer06 A3
0.567 slot_mixed argmax:
  peer04 A1 + peer05 A2 + peer06 A3
0.617 role_stress stochastic:
  peer06 role rotation
0.717 native stochastic:
  peer06 native
```

Interpretation:

- Native teams now beat the old frozen `bridge03` peer clearly in argmax and
  mostly in stochastic mode.
- Role-stress is acceptable on average, but `peer06` is weak under stochastic
  role rotation.
- Mixed stochastic teams are not yet robust enough for free league drafting.
  The pool is usable as training material, not as a final open-draft league.

### Gate: Native Cross-Peer Against Strong `peer01`

Eval dir:
`training_runs/fresh_population_v31_peer_batches/fresh_population_v31_peer_curriculum_v1/eval_peer01_native_cross`.

Setup:

- Suite: `native`.
- Modes: `argmax,stochastic`.
- Opponent: `peer01 u660` as frozen stochastic policy.
- Seed bases: `20380101,20390101,20400101,20410101`.
- Episodes: 20 per seed block, 80 per checkpoint/mode.
- Records: 12/12 complete.

Summary:

| evaluated native team | argmax win | stochastic win | notes |
| --- | ---: | ---: | --- |
| peer01 u660 | 0.713 | 0.425 | mirror-ish self check still favors opponent side/stochastic opponent |
| peer02 u700 | 0.688 | 0.488 | close but not dominant |
| peer03 u580 | 0.238 | 0.038 | weak into peer01 |
| peer04 u600 | 0.625 | 0.225 | argmax useful, stochastic weak |
| peer05 u660 | 0.625 | 0.550 | best stochastic non-peer01 line |
| peer06 u620 | 0.325 | 0.150 | weak into peer01 |

Interpretation:

- Cross-peer is the current hard gate and it is not passed.
- `peer01` is a strong opponent; `peer05` is the best stochastic challenger,
  while `peer03` and `peer06` are weak in this matchup.
- The current six checkpoints should not be multiplied into 18-24 population
  lines yet.
- Next recommended training phase: rotating peer/self-play curriculum that
  samples opponents from this six-line pool and uses teammate mixing during
  training. The goal is to preserve universal roots while improving cross-peer
  and mixed-team robustness.

Decision:

- Accept the peer curriculum batch as a successful training step.
- Do not accept the population as league-ready.
- Do not proceed to 18-24 expansion yet.
- Proceed to a rotating peer/self-play mix run with:
  - the six selected peer checkpoints as the initial pool;
  - stochastic opponents sampled from the pool;
  - role shuffle enabled;
  - teammate mixing included if the trainer supports it, or implemented as the
    next training adapter change;
  - gates after the run: scripted-core, bridge03, peer01, and full teammate
    swap.

## Fresh Population v3.1 Rotating Peer Mix v1

Date: 2026-06-08.

Goal:

- Continue the six peer-curriculum roots against stronger frozen policy peers.
- Keep role shuffle enabled so roots stay position-universal.
- Check whether simple peer/opponent rotation is enough for league-style
  teammate swaps.

Batch:

- Batch id: `fresh_population_v31_rotating_peer_mix_v1`.
- Batch dir:
  `training_runs/fresh_population_v31_rotmix_batches/fresh_population_v31_rotating_peer_mix_v1`.
- Script: `scripts/run_fresh_population_v31_rotating_peer_mix.py`.
- Workers: 16.
- Rollout steps: 512.
- Role assignment: `shuffle`.
- Stage 1: six roots, 120 updates each, resumed from peer-curriculum
  checkpoints.
- Stage 2: six roots, 80 updates each, resumed from Stage 1 and rotated
  against updated peer lines.
- Result: 12/12 runs complete, all returncode `0`, no batch stderr.

Important limitation:

- This was rotating peer/opponent training, not true in-episode teammate
  mixing. The current trainer still resumes a whole three-slot team checkpoint.
  Mixed teammate compatibility must therefore be validated by eval and, if it
  fails, fixed by a training-adapter change.

Internal eval summary:

| run | best update | best W/L/D | final update | final W/L/D |
| --- | ---: | --- | ---: | --- |
| mix01 stage1 | 720 | 0.983 / 0.017 / 0.000 | 780 | 0.933 / 0.050 / 0.017 |
| mix01 stage2 | 840 | 0.950 / 0.050 / 0.000 | 860 | 0.933 / 0.067 / 0.000 |
| mix02 stage1 | 780 | 0.950 / 0.050 / 0.000 | 820 | 0.867 / 0.133 / 0.000 |
| mix02 stage2 | 880 | 0.883 / 0.117 / 0.000 | 900 | 0.883 / 0.117 / 0.000 |
| mix03 stage1 | 680 | 0.600 / 0.383 / 0.017 | 700 | 0.533 / 0.350 / 0.117 |
| mix03 stage2 | 780 | 0.817 / 0.167 / 0.017 | 780 | 0.817 / 0.167 / 0.017 |
| mix04 stage1 | 660 | 0.950 / 0.033 / 0.017 | 720 | 0.950 / 0.050 / 0.000 |
| mix04 stage2 | 800 | 0.767 / 0.167 / 0.067 | 800 | 0.767 / 0.167 / 0.067 |
| mix05 stage1 | 760 | 0.967 / 0.017 / 0.017 | 780 | 0.817 / 0.150 / 0.033 |
| mix05 stage2 | 840 | 0.800 / 0.133 / 0.067 | 860 | 0.783 / 0.117 / 0.100 |
| mix06 stage1 | 720 | 0.617 / 0.317 / 0.067 | 740 | 0.617 / 0.367 / 0.017 |
| mix06 stage2 | 800 | 0.767 / 0.200 / 0.033 | 820 | 0.767 / 0.217 / 0.017 |

Selected candidate checkpoints:

```text
mix01: training_runs/fresh_population_v31_rotmix_mix01_peer01_stage2_s20263101/checkpoints/checkpoint_update_840.pt
mix02: training_runs/fresh_population_v31_rotmix_mix02_peer02_stage2_s20263102/checkpoints/checkpoint_update_900.pt
mix03: training_runs/fresh_population_v31_rotmix_mix03_peer03_stage2_s20263103/checkpoints/checkpoint_update_780.pt
mix04: training_runs/fresh_population_v31_rotmix_mix04_peer04_stage1_s20263004/checkpoints/checkpoint_update_720.pt
mix05: training_runs/fresh_population_v31_rotmix_mix05_peer05_stage1_s20263005/checkpoints/checkpoint_update_760.pt
mix06: training_runs/fresh_population_v31_rotmix_mix06_peer06_stage2_s20263106/checkpoints/checkpoint_update_820.pt
```

Why Stage 1 checkpoints were kept for `mix04` and `mix05`:

- Their Stage 2 best/final evals were worse than Stage 1.
- Keeping the stronger checkpoint preserves line diversity without accepting a
  clearly regressed continuation.

### Gate: Scripted-Core Rotmix v1 Swap

Eval dir: `training_runs/evals/rotmix_v1_swap_scripted_core`.

Setup:

- Suite: `native,slot_mixed,role_stress`.
- Modes: `argmax,stochastic`.
- Opponent: `scripted_core`.
- Ally bot: `scripted_core`.
- Seed bases: `20260609,20270609,20280609`.
- Episodes: 20 per seed block, 60 per team/mode.
- Records: 48/48 complete.

Summary:

| team kind | mode | win avg | win min | win max | loss avg | draw avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.650 | 0.317 | 1.000 | 0.350 | 0.000 |
| native | stochastic | 0.981 | 0.950 | 1.000 | 0.019 | 0.000 |
| role_stress | argmax | 0.711 | 0.133 | 1.000 | 0.283 | 0.006 |
| role_stress | stochastic | 0.961 | 0.917 | 1.000 | 0.039 | 0.000 |
| slot_mixed | argmax | 0.886 | 0.300 | 1.000 | 0.114 | 0.000 |
| slot_mixed | stochastic | 0.994 | 0.983 | 1.000 | 0.006 | 0.000 |

Weak argmax samples:

```text
0.133 role_stress: mix06 u820, A1<-A2 A2<-A3 A3<-A1
0.300 slot_mixed: mix04 u720 + mix05 u760 + mix06 u820
0.317 native: mix06 u820
0.333 slot_mixed_alt: mix04 u720 + mix06 u820 + mix05 u760
0.350 role_stress: mix02 u900, A1<-A2 A2<-A3 A3<-A1
```

Interpretation:

- Stochastic scripted-core performance is strong, including mixed teams.
- Argmax is not clean. `mix06` and some role/slot combinations can collapse.
- This does not block stochastic league experiments by itself, but it is a
  warning that deterministic eval should not be the only selection criterion.

### Gate: Frozen `peer01 u660` Stochastic Rotmix v1 Swap

Eval dir: `training_runs/evals/rotmix_v1_swap_vs_peer01_stoch`.

Setup:

- Suite: `native,slot_mixed,role_stress`.
- Modes: `argmax,stochastic`.
- Opponent: `peer01 u660` as frozen stochastic policy.
- Ally bot: `scripted_core`.
- Seed bases: `20260610,20270610,20280610`.
- Episodes: 20 per seed block, 60 per team/mode.
- Records: 48/48 complete.

Summary:

| team kind | mode | win avg | win min | win max | loss avg | draw avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.839 | 0.567 | 1.000 | 0.133 | 0.028 |
| native | stochastic | 0.689 | 0.283 | 0.933 | 0.308 | 0.003 |
| role_stress | argmax | 0.803 | 0.450 | 1.000 | 0.156 | 0.042 |
| role_stress | stochastic | 0.672 | 0.150 | 0.950 | 0.317 | 0.011 |
| slot_mixed | argmax | 0.715 | 0.167 | 0.967 | 0.279 | 0.006 |
| slot_mixed | stochastic | 0.454 | 0.133 | 0.683 | 0.546 | 0.000 |

Weak samples:

```text
0.133 stochastic slot_mixed: mix04 u720 + mix05 u760 + mix06 u820
0.150 stochastic slot_mixed_alt: mix04 u720 + mix06 u820 + mix05 u760
0.150 stochastic role_stress: mix03 u780, A1<-A2 A2<-A3 A3<-A1
0.167 argmax slot_mixed: mix04 u720 + mix05 u760 + mix06 u820
0.233 argmax slot_mixed_alt: mix04 u720 + mix06 u820 + mix05 u760
0.283 stochastic native: mix06 u820
0.450 stochastic native: mix03 u780
```

Interpretation:

- Native argmax mostly improved against the hard `peer01` opponent.
- Stochastic native is uneven: `mix03` and `mix06` remain weak.
- Slot-mixed stochastic is the hard failure: average win rate is below 50% and
  several mixed teams are nearly unusable.
- Rotating peer/opponent training alone is not enough for open teammate draft.

Decision:

- Accept `fresh_population_v31_rotating_peer_mix_v1` as a useful training
  step.
- Do not accept the pool as league-ready.
- Do not expand to 18-24 population lines yet.
- Next step must target teammate compatibility directly:
  - implement a training adapter that can assemble A1/A2/A3 from different
    checkpoint player states during training, or otherwise sample mixed
    teammates in the trainer;
  - keep role shuffle enabled;
  - train against stochastic peers sampled from the pool;
  - re-run gates against `scripted_core`, `bridge03`, `peer01`, and full
    teammate swap.

## Fresh Population v3.1 Teammate-Mix Adaptation

Goal: close the make-or-break league risk where policies only worked with
their native A1/A2/A3 teammates. Training used mixed team checkpoints, role
shuffle, and frozen stochastic `peer01 u660` pressure. This is not a tactical
specialization pass; it is a compatibility pass for open teammate draft.

### Teammate-Mix v1

Run family: `training_runs/fresh_population_v31_teammate_mix_tmix01..06_*`.

Selected candidates:

```text
tmix01: training_runs/fresh_population_v31_teammate_mix_tmix01_worst456_s20264001/checkpoints/latest.pt
tmix02: training_runs/fresh_population_v31_teammate_mix_tmix02_worst465_s20264002/checkpoints/latest.pt
tmix03: training_runs/fresh_population_v31_teammate_mix_tmix03_132_s20264003/checkpoints/latest.pt
tmix04: training_runs/fresh_population_v31_teammate_mix_tmix04_354_s20264004/checkpoints/checkpoint_update_40.pt
tmix05: training_runs/fresh_population_v31_teammate_mix_tmix05_621_s20264005/checkpoints/latest.pt
tmix06: training_runs/fresh_population_v31_teammate_mix_tmix06_243_s20264006/checkpoints/checkpoint_update_40.pt
```

Gate against frozen stochastic `peer01 u660`:
`training_runs/evals/teammate_mix_adapt_v1_swap_vs_peer01_stoch`.

| team kind | mode | win avg | win min | win max | loss avg | draw avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.889 | 0.733 | 0.983 | 0.100 | 0.011 |
| native | stochastic | 0.750 | 0.617 | 0.850 | 0.244 | 0.006 |
| role_stress | argmax | 0.947 | 0.867 | 1.000 | 0.044 | 0.008 |
| role_stress | stochastic | 0.819 | 0.650 | 0.883 | 0.181 | 0.000 |
| slot_mixed | argmax | 0.731 | 0.050 | 0.983 | 0.242 | 0.028 |
| slot_mixed | stochastic | 0.544 | 0.083 | 0.850 | 0.454 | 0.001 |

Decision: v1 improved averages but did not close the gate. Worst stochastic
slot-mixed teams were still unusable.

### Hard-Negative v2

Targeted exact weak combinations from v1:

```text
tmix07_hard612: A1 tmix06 u40, A2 tmix01 latest, A3 tmix02 latest
tmix08_hard345: A1 tmix03 latest, A2 tmix04 u40, A3 tmix05 latest
tmix09_hard561: A1 tmix05 latest, A2 tmix06 u40, A3 tmix01 latest
```

Internal best checkpoints:

| run | best update | best eval W/L/D | final update | final eval W/L/D |
| --- | ---: | ---: | ---: | ---: |
| tmix07_hard612 | 60 | 0.983 / 0.017 / 0.000 | 120 | 0.867 / 0.133 / 0.000 |
| tmix08_hard345 | 100 | 1.000 / 0.000 / 0.000 | 120 | 1.000 / 0.000 / 0.000 |
| tmix09_hard561 | 60 | 0.983 / 0.017 / 0.000 | 120 | 0.983 / 0.017 / 0.000 |

Selected v2 pool gate:
`training_runs/evals/teammate_mix_hard_v2_selected_full_vs_peer01_stoch`.

| team kind | mode | win avg | win min | win max | loss avg | draw avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.889 | 0.733 | 0.987 | 0.111 | 0.000 |
| native | stochastic | 0.836 | 0.760 | 0.880 | 0.164 | 0.000 |
| role_stress | argmax | 0.942 | 0.907 | 0.987 | 0.044 | 0.013 |
| role_stress | stochastic | 0.880 | 0.867 | 0.893 | 0.120 | 0.000 |
| slot_mixed | argmax | 0.900 | 0.787 | 0.987 | 0.091 | 0.009 |
| slot_mixed | stochastic | 0.742 | 0.400 | 0.920 | 0.253 | 0.004 |

Decision: v2 was much better, but still failed the league gate because one
slot-mixed stochastic team fell to 40%.

### Hard-Negative v3

Targeted remaining weak slot-mixed stochastic teams:

```text
tmix10_hard987: A1 tmix09 u60, A2 tmix08 u100, A3 tmix07 u100
tmix11_hard978: A1 tmix09 u60, A2 tmix07 u100, A3 tmix08 u100
```

Training checkpoints:

| run | best update | best eval W/L/D | final update | final train W/L/D | final eval W/L/D |
| --- | ---: | ---: | ---: | ---: | ---: |
| tmix10_hard987 | 80 | 1.000 / 0.000 / 0.000 | 140 | 0.940 / 0.060 / 0.000 | 0.967 / 0.033 / 0.000 |
| tmix11_hard978 | 80 | 1.000 / 0.000 / 0.000 | 120 | 0.892 / 0.108 / 0.000 | 0.983 / 0.017 / 0.000 |

Selected v3 pool:

```text
tmix10: training_runs/fresh_population_v31_teammate_mix_tmix10_hard987_s20264110/checkpoints/checkpoint_update_80.pt
tmix11: training_runs/fresh_population_v31_teammate_mix_tmix11_hard978_s20264111/checkpoints/checkpoint_update_80.pt
tmix08: training_runs/fresh_population_v31_teammate_mix_tmix08_hard345_s20264108/checkpoints/checkpoint_update_100.pt
tmix09: training_runs/fresh_population_v31_teammate_mix_tmix09_hard561_s20264109/checkpoints/checkpoint_update_60.pt
```

Gate: `training_runs/evals/teammate_mix_hard_v3_selected_full_vs_peer01_stoch`.

| team kind | mode | win avg | win min | win max | loss avg | draw avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.975 | 0.950 | 0.983 | 0.021 | 0.004 |
| native | stochastic | 0.850 | 0.667 | 0.950 | 0.150 | 0.000 |
| role_stress | argmax | 0.962 | 0.900 | 1.000 | 0.029 | 0.008 |
| role_stress | stochastic | 0.887 | 0.833 | 0.933 | 0.113 | 0.000 |
| slot_mixed | argmax | 0.973 | 0.900 | 1.000 | 0.021 | 0.006 |
| slot_mixed | stochastic | 0.858 | 0.750 | 0.950 | 0.142 | 0.000 |

Decision:

- Teammate-swap blocker is reduced enough to begin the next self-play bridge.
- `slot_mixed stochastic` improved from min 0.400 in v2 to min 0.750 in v3.
- Keep `tmix09` under watch: native stochastic was the lowest remaining sample
  at 0.667, although its mixed/stress results are acceptable.
- Next step: launch `fresh_population_v31_selfplay_v3_pool_v1`, six stochastic
  peer-pressure continuations from the selected v3 pool.

## 2026-06-08 - Self-Play v3 Pool

Batch: `fresh_population_v31_selfplay_v3_pool_v1`.

Goal: continue the selected hard-negative teammate-mix pool into self-play style
training without specializing agents to a hand-authored tactic. All runs used
role shuffle, stochastic frozen peer opponents, factorized actions, and
`train_obs_v3.1`.

Internal results:

| run | source vs peer | best update | best eval W/L/D | final update | final train W/L/D | final eval W/L/D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| spv3_01 | tmix10 vs tmix11 | 200 | 0.983 / 0.000 / 0.017 | 240 | 0.889 / 0.056 / 0.056 | 0.967 / 0.033 / 0.000 |
| spv3_02 | tmix11 vs tmix08 | 240 | 0.983 / 0.017 / 0.000 | 240 | 0.913 / 0.065 / 0.022 | 0.983 / 0.017 / 0.000 |
| spv3_03 | tmix08 vs tmix09 | 240 | 0.550 / 0.317 / 0.133 | 240 | 0.788 / 0.212 / 0.000 | 0.550 / 0.317 / 0.133 |
| spv3_04 | tmix09 vs tmix10 | 240 | 0.983 / 0.017 / 0.000 | 240 | 0.769 / 0.205 / 0.026 | 0.983 / 0.017 / 0.000 |
| spv3_05 | tmix10 vs tmix08 | 220 | 0.867 / 0.067 / 0.067 | 220 | 0.743 / 0.229 / 0.029 | 0.867 / 0.067 / 0.067 |
| spv3_06 | tmix11 vs tmix09 | 200 | 0.900 / 0.033 / 0.067 | 220 | 0.825 / 0.175 / 0.000 | 0.850 / 0.067 / 0.083 |

Manifest:
`training_runs/self_play_pools/fresh_population_v31_selfplay_v3_pool_v1_manifest.json`.
Compatible entries: 6/6.

External gate against frozen stochastic `tmix10 u80`:
`training_runs/evals/fresh_population_v31_selfplay_v3_pool_vs_peer10_stoch_v1`.

| team kind | mode | win avg | win min | win max | loss avg | draw avg | normal avg | death-time avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.781 | 0.233 | 1.000 | 0.108 | 0.111 | 0.703 | 0.078 |
| native | stochastic | 0.703 | 0.600 | 0.783 | 0.253 | 0.044 | 0.647 | 0.056 |
| role_stress | argmax | 0.842 | 0.700 | 0.950 | 0.075 | 0.083 | 0.725 | 0.117 |
| role_stress | stochastic | 0.708 | 0.583 | 0.817 | 0.231 | 0.061 | 0.633 | 0.075 |
| slot_mixed | argmax | 0.806 | 0.550 | 0.933 | 0.086 | 0.108 | 0.696 | 0.110 |
| slot_mixed | stochastic | 0.662 | 0.467 | 0.833 | 0.278 | 0.060 | 0.585 | 0.078 |

Worst findings:

- `spv3_03` is not accepted as a strong line yet: native argmax fell to 0.233
  vs frozen stochastic `tmix10 u80`.
- `slot_mixed stochastic` minimum is 0.467, mostly involving `spv3_03`.
- `spv3_05` is usable but soft: native stochastic 0.600 and role-stress
  stochastic 0.633.

Decision:

- Do not accept this as a clean six-line open-draft pool yet.
- Keep `spv3_01`, `spv3_02`, `spv3_04`, and `spv3_06` as strong candidates.
- Launch a recovery/stabilization pass for `spv3_03` and `spv3_05` before the
  next population gate. This is still universal training: no scripted tactic
  specialization, just stochastic peer pressure with role shuffle.

## 2026-06-08 - Self-Play v3 Recovery

Recovery runs were launched after the v3 pool gate identified `spv3_03` as
the main failed line and `spv3_05` as a soft line. Both runs used the same
universal setup: role shuffle, stochastic frozen `tmix10 u80` peer, no
hand-authored tactical specialization.

| run | resume | best update | best eval W/L/D | final update | final train W/L/D | final eval W/L/D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| spv303_recovery | spv3_03 u240 | 420 | 0.850 / 0.117 / 0.033 | 420 | 0.723 / 0.255 / 0.021 | 0.850 / 0.117 / 0.033 |
| spv305_recovery | spv3_05 u220 | 340 | 0.950 / 0.017 / 0.033 | 360 | 0.842 / 0.132 / 0.026 | 0.933 / 0.000 / 0.067 |

Recovery candidate pool:

```text
spv3_01: training_runs/fresh_population_v31_selfplay_v3_tmix10_vs_tmix11_s20264201/checkpoints/checkpoint_update_200.pt
spv3_02: training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix08_s20264202/checkpoints/checkpoint_update_240.pt
spv3_03R: training_runs/fresh_population_v31_selfplay_v3_recovery_spv303_vs_peer10_s20264303/checkpoints/checkpoint_update_420.pt
spv3_04: training_runs/fresh_population_v31_selfplay_v3_tmix09_vs_tmix10_s20264204/checkpoints/checkpoint_update_240.pt
spv3_05R: training_runs/fresh_population_v31_selfplay_v3_recovery_spv305_vs_peer10_s20264305/checkpoints/checkpoint_update_340.pt
spv3_06: training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix09_s20264206/checkpoints/checkpoint_update_200.pt
```

Recovery pool gate:
`training_runs/evals/fresh_population_v31_selfplay_v3_recovery_pool_vs_peer10_stoch_v1`.

| team kind | mode | win avg | win min | win max | loss avg | draw avg | normal avg | death-time avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | argmax | 0.903 | 0.767 | 0.983 | 0.036 | 0.061 | 0.811 | 0.092 |
| native | stochastic | 0.789 | 0.600 | 0.867 | 0.150 | 0.061 | 0.711 | 0.078 |
| role_stress | argmax | 0.878 | 0.750 | 0.967 | 0.058 | 0.064 | 0.769 | 0.108 |
| role_stress | stochastic | 0.775 | 0.700 | 0.833 | 0.181 | 0.044 | 0.725 | 0.050 |
| slot_mixed | argmax | 0.849 | 0.600 | 0.967 | 0.071 | 0.081 | 0.747 | 0.101 |
| slot_mixed | stochastic | 0.713 | 0.567 | 0.833 | 0.221 | 0.067 | 0.646 | 0.067 |

Decision:

- Recovery fixed the catastrophic native argmax hole: min improved from 0.233
  to 0.767.
- Open-draft stochastic mixed-team compatibility is still not clean: worst
  `slot_mixed stochastic` improved only from 0.467 to 0.567.
- The current pool is much better than v3, but not yet acceptable as a final
  six-line open-draft league pool.

## 2026-06-08 - Hard-Negative v4 Probe

Created v4 hard-negative mixed init checkpoints from the worst recovery-pool
mixed teams:

```text
tmix12_spv4_465: A1 spv3_04 u240, A2 spv3_06 u200, A3 spv3_05R u340
tmix13_spv4_243: A1 spv3_02 u240, A2 spv3_04 u240, A3 spv3_03R u420
tmix14_spv4_345: A1 spv3_03R u420, A2 spv3_04 u240, A3 spv3_05R u340
```

Ran the first and most important v4 probe:
`training_runs/fresh_population_v31_teammate_mix_v4_tmix12_465_s20264412`.

| run | best update | best eval W/L/D | final update | final train W/L/D | final eval W/L/D |
| --- | ---: | ---: | ---: | ---: | ---: |
| tmix12_spv4_465 | 100 | 0.817 / 0.017 / 0.167 | 120 | 0.531 / 0.250 / 0.219 | 0.817 / 0.067 / 0.117 |

Decision:

- v4 hard-negative init is not obviously enough by itself: `tmix12` improved
  over the raw worst mix but did not reach the strong-line region, and draws
  stayed high.
- Do not blindly spend the next hours on `tmix13`/`tmix14` with the exact same
  setup unless we accept this as exploration.
- Better next engineering/training step: add proper varied-teammate sampling
  during training, or run a broader curriculum that rotates teammates/opponents
  inside the trainer instead of producing one fixed mixed checkpoint per weak
  case.
