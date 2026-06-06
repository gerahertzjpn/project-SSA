# Pre-Population League Readiness Gate

Date: 2026-06-05

Status: active. This document is a hard gate before spending compute on the
18-24 model population phase.

## Goal

Before training a population of league agents, prove two assumptions:

1. Trained agents can coordinate with unfamiliar teammates.
2. Self-play throughput is good enough, or any optimization attempt is rejected
   unless it preserves game behavior exactly.

Do not start the expensive population phase just because the last native team
checkpoint was strong.

## League Draft Model

The league target is a free-agent draft, not a set of fixed native teams.

In the future league, an LLM drafter may assemble a team from individual
policies across different seed lineages and source slots. A team-checkpoint is
therefore only a source container; its `A1`, `A2`, and `A3` policies are draft
candidates, not an inseparable trio.

Implications:

- Count both levels:
  - seed lineage diversity, e.g. 7 independent roots;
  - individual slot-policy candidates, e.g. up to 21 draftable policies.
- A policy is not considered fully draft-ready just because its native
  checkpoint is strong.
- Each draft candidate needs a scouting record:
  - source lineage;
  - source slot;
  - preferred target roles;
  - scripted/random baseline performance;
  - peer-stress performance;
  - teammate compatibility;
  - role-stress performance;
  - stochastic-vs-argmax stability;
  - invalid action rate and action collapse checks.
- Fixed native teams are useful controls, not the final product.

For early safety, the LLM drafter may prefer slot-preserving drafts unless
role-stress gates prove that cross-slot placement is reliable. The long-term
goal is freer drafting, but the evidence must decide how much freedom is safe.

## Gate 1 - Teammate Swap

Question:

```text
Do agents coordinate with unfamiliar partners, or only with the exact native
A1/A2/A3 trio they were trained with?
```

This is make-or-break for league play. A league will mix, trade, and recombine
agents. If each policy only works with its original two teammates, then a pool
of 18-24 models is not useful as a composable population.

Required eval:

- Build non-native teams from trained agents.
- Primary swap is slot-preserving:
  - target `A1` gets an `A1` policy from one checkpoint/lineage;
  - target `A2` gets an `A2` policy from another checkpoint/lineage;
  - target `A3` gets an `A3` policy from a third checkpoint/lineage.
- Role-swap stress is mandatory for free-agent draft readiness:
  - target `A1` may receive source `A2` or `A3`;
  - target `A2` may receive source `A1` or `A3`;
  - target `A3` may receive source `A1` or `A2`.
  It may remain a separate harder gate from slot-preserving swap, but it must
  be measured before the LLM drafter is allowed to draft cross-slot freely.
- Compare swapped teams against native-team controls.
- Evaluate argmax and stochastic modes separately.
- Track:
  - win-rate;
  - normal/death-time/draw/loss rates;
  - invalid action rate;
  - action distribution collapse;
  - performance by seed block.

Interpretation:

- If swapped teams hold up, population work can proceed with more confidence.
- If swapped teams break, population training must include varied teammates
  before producing many lineages.

## Gate 2 - Throughput Optimization

Known debt:

- Latest self-play benchmark: about `1.90M` world steps/hour.
- Target before population phase: `3M+` world steps/hour.

Rule:

```text
Optimization is accepted only if it preserves current game behavior exactly.
```

Allowed optimization path:

- Optimize frozen-policy inference, environment stepping, batching, or worker
  layout only if exact-behavior tests prove no gameplay change.
- Any optimization that changes simulation outcome, legal action masks, chosen
  actions under the same deterministic seed, winner, tick count, combat result,
  or replay-critical state is rejected.

If exact preservation cannot be proven:

- Do not merge/accept the optimization.
- Record the failed attempt and reason.
- Continue to the next planning step with throughput debt explicitly open.

## Gate 3 - Population Seeds

Only after the swap gate is understood and throughput is either improved or
explicitly accepted as debt:

- Prepare a population plan for `18-24` agents.
- It is acceptable to start smaller, but the first working seed set should have
  at least `6` useful seeds.
- Seeds must be genuinely diverse:
  - different random seeds;
  - different ancestry/lineages;
  - different training lengths;
  - different opponent mixes or teammate mixes where appropriate.
- Avoid producing 18-24 clones of the same style.

## Current Starting Point

Accepted Sprint 3 checkpoint:

```text
training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt
```

Accepted pre-self-play checkpoint:

```text
training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt
```

The next implementation step is a real teammate-swap eval tool and report.

## 2026-06-05 Gate Results

### Teammate-Swap Eval

Tool:

```powershell
python scripts/evaluate_teammate_swaps.py
```

Checkpoints tested:

```text
training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_240.pt
training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt
training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_300.pt
training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_340.pt
training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_360.pt
training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt
```

Primary suite:

```text
native + slot_mixed
argmax + stochastic
4 seed blocks * 25 episodes
16 workers
```

Against scripted:

```text
native:     100% win-rate across all tested checkpoints and modes
slot_mixed: 100% win-rate across all tested mixes and modes
invalid A1: 0.0
```

Against frozen stochastic `checkpoint_update_280.pt`:

```text
native argmax mean win-rate:      0.718, min 0.490, max 0.880
native stochastic mean win-rate:  0.517, min 0.280, max 0.730
slot_mixed argmax mean win-rate:  0.755, min 0.580, max 0.930
slot_mixed stochastic mean:       0.553, min 0.420, max 0.680
```

Interpretation:

```text
Slot-preserving teammate-swap passes for the currently available checkpoints.
There is no current evidence that the policies only coordinate with their
native A1/A2/A3 trio.
```

This does not prove future independent lineages will always compose. Re-run the
same gate after the first real 6-seed population exists.

### Throughput Optimization Attempt

Baseline policy-opponent no-learning benchmark:

```text
~1.756M world steps/hour
```

Accepted exact-behavior micro-optimizations:

- frozen `PolicyBot` action-only inference, skipping critic/logprob/entropy;
- skipping full observations for `ExternalActionBot`, which already receives
  externally decoded actions;
- route/movable cache inside one observation/player;
- `TeamTrainingEnv.step()` reuses its own observation cache for returned
  observations.

Behavior guards:

```text
pytest tests/test_training_spaces.py tests/test_team_training_env.py tests/test_policy_trainer.py tests/test_policy_bot.py tests/test_teammate_swap_eval.py -q
41 passed
```

Latest same-profile benchmark:

```text
~1.927M world steps/hour
~5.782M controlled-agent decisions/hour
```

Decision:

```text
The safe micro-optimizations are accepted, but the 3M+ world-step target is not
met. Further optimization likely requires a deeper observation/mask refactor or
batched policy-opponent inference. That work is not accepted inside this run
unless replay/state equivalence is proven separately.
```

Per the gate rule, the population-prep step may continue with this throughput
debt explicitly open.

### Population Seed Gate

Minimum first population foundation:

```text
6 working seeds
```

Because slot-preserving teammate-swap passed, first seed diversity should focus
on:

- different random seeds;
- different ancestry;
- different frozen opponents;
- different training lengths;
- scripted-anchor retention checks after self-play.

Varied-teammate training is not mandatory for the next run, but the swap test
must be repeated after the first 6 seed lineages exist.

## 2026-06-05 Population Foundation Re-Test

First foundation result:

```text
valid working lines: 7
minimum target:      6
rejected lines:      original seed04 hard-start fresh-vs-frozen-u280
```

Candidate latest checkpoints:

```text
training_runs/population_s3_seed01_scripted_std_s20260651/checkpoints/latest.pt
training_runs/population_s3_seed02_scripted_lowent_s20260652/checkpoints/latest.pt
training_runs/population_s3_seed03_scripted_highent_s20260653/checkpoints/latest.pt
training_runs/population_s3_seed04_warmstart_seed03_vs_u280_s20260664/checkpoints/latest.pt
training_runs/population_s3_seed05_resume_u280_vs_u380_s20260655/checkpoints/latest.pt
training_runs/population_s3_seed06_resume_u380_vs_u240_argmax_s20260656/checkpoints/latest.pt
training_runs/population_s3_seed07_scripted_altent_s20260657/checkpoints/latest.pt
```

Post-foundation teammate-swap against scripted:

```text
eval dir: training_runs/evals/pre_population_teammate_swap_population_v1
native argmax:          min 1.000, avg 1.000, max 1.000
native stochastic:      min 0.980, avg 0.997, max 1.000
slot_mixed argmax:      min 0.990, avg 0.998, max 1.000
slot_mixed stochastic:  min 1.000, avg 1.000, max 1.000
slot_mixed stochastic normal-win: 1.000 average
invalid-action collapse: not observed
```

Post-foundation teammate-swap against frozen stochastic u280:

```text
eval dir: training_runs/evals/pre_population_teammate_swap_population_vs_frozen_u280_v1
native argmax:          min 0.150, avg 0.607, max 0.775
native stochastic:      min 0.050, avg 0.314, max 0.625
slot_mixed argmax:      min 0.450, avg 0.655, max 0.875
slot_mixed stochastic:  min 0.225, avg 0.425, max 0.650
```

Interpretation:

```text
The scripted teammate-swap gate is passed for the new population foundation.
Agents do not appear locked to only their native training trio in slot-preserving
mixes.

The frozen-policy eval is intentionally harder and exposes a real strength
spectrum. Some native and mixed teams lose to frozen u280, but mixed teams are
not worse on average and the best mixed team is stronger than any native team in
this small frozen batch. This supports proceeding with the population foundation,
but the next expansion should include mixed-team/self-play curriculum rather
than only producing more scripted-anchor clones.
```

### Role-Stress Re-Test

Reason:

```text
Slot-preserving swap only proves that A1 policies compose with other A1
policies, A2 with A2, and A3 with A3. It does not prove that a policy trained
for one role can be dropped into a different role.
```

Stress mapping:

```text
target A1 gets source A2
target A2 gets source A3
target A3 gets source A1
```

Against scripted:

```text
eval dir: training_runs/evals/pre_population_role_stress_scripted_v1
argmax:     avg win 0.956, normal 0.716, death-time 0.240, draw 0.029, loss 0.016
stochastic: avg win 0.991, normal 0.940, death-time 0.051, draw 0.000, loss 0.009
```

Against frozen stochastic u280:

```text
eval dir: training_runs/evals/pre_population_role_stress_vs_frozen_u280_v1
argmax:     avg win 0.071, normal 0.054, death-time 0.018, draw 0.093, loss 0.836
stochastic: avg win 0.057, normal 0.057, death-time 0.000, draw 0.000, loss 0.943
```

Interpretation:

```text
Role-stress passes only against weak scripted opposition and fails hard against
the trained frozen u280 opponent. Current checkpoints are slot-specialized.
They can be mixed with unfamiliar teammates in matching slots, but they should
not be treated as role-flexible A1/A2/A3 policies.
```

Observation note:

```text
Observation version train_obs_v1.0 does not expose an explicit A1/A2/A3 role
one-hot. B-side observations are mirrored into the A-side canonical view:
B1->A1, B2->A2, B3->A3, with mirrored nodes/actions. Therefore B-side matching
slots are handled, but cross-slot role transfer is only inferred indirectly from
map/relation/legal-mask features.

Adding an explicit role feature would change the observation vector and require
a new observation version/checkpoint family. Existing v1.0 checkpoints should
not be expected to load unchanged after such a schema change.
```

Throughput status remains unchanged:

```text
accepted safe speed: ~1.927M world steps/hour
target debt:         3M+ world steps/hour
decision:            debt accepted for this run; deeper optimization remains a
                     separate exact-equivalence project.
```
