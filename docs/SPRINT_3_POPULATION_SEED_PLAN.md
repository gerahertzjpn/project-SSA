# Sprint 3 Population Seed Plan

Date: 2026-06-05

Status: first 7-line population foundation completed; next expansion should use
mixed-team/self-play curriculum. Role-flexible cross-slot use is not approved
for the current checkpoint family.

## Hard Rules

- Do not train 18-24 clones.
- Keep the current RL interface unchanged:
  - `OBSERVATION_VERSION = train_obs_v1.0`
  - `OBSERVATION_DIM = 1117`
  - `ACTION_DIM = 1356`
- First population target is at least 6 working seeds.
- Teammate-swap must be rerun after the first 6 seeds exist.
- Slot-preserving use is allowed: A1-style policies in A1/B1 slots, A2-style
  policies in A2/B2 slots, A3-style policies in A3/B3 slots.
- Cross-slot use is not allowed without a dedicated role-randomized curriculum.
- Throughput debt remains open: current safe speed is about `1.93M` world
  steps/hour, below the `3M+` target.

## Gate Summary

Teammate-swap passed on current checkpoints:

- scripted opponent: native and slot-mixed teams both reached `100%` win-rate;
- frozen u280 opponent: slot-mixed teams did not collapse and were slightly
  above native average in this eval batch.

Therefore varied-teammate training is not required before this first 6-seed
foundation, but must be retested after the new seeds are produced.

Role-stress was also tested after the 7-line foundation:

- mapping: target `A1 <- source A2`, target `A2 <- source A3`,
  target `A3 <- source A1`;
- scripted opponent: survived, but argmax became death-time heavier;
- frozen u280 opponent: failed hard, with average win-rate about `0.071`
  argmax and `0.057` stochastic.

Therefore the current foundation is accepted only as slot-specialized. If the
league should allow an agent to play any position, the next training route must
include role-randomized/position-randomized training, and probably a new
observation schema with an explicit role signal.

Safe throughput work accepted:

- policy bot action-only inference;
- skipped observations for external-action controlled bots;
- per-observation route/movable cache;
- training env return-observation cache reuse.

Risky deeper optimization is deferred unless exact replay/state equivalence is
proven separately.

## Foundation Result

The initial six-line target was met and exceeded:

```text
valid working lines: 7
minimum target:      6
rejected lines:      original seed04 hard-start fresh-vs-frozen-u280
```

Valid candidate latest checkpoints:

```text
training_runs/population_s3_seed01_scripted_std_s20260651/checkpoints/latest.pt
training_runs/population_s3_seed02_scripted_lowent_s20260652/checkpoints/latest.pt
training_runs/population_s3_seed03_scripted_highent_s20260653/checkpoints/latest.pt
training_runs/population_s3_seed04_warmstart_seed03_vs_u280_s20260664/checkpoints/latest.pt
training_runs/population_s3_seed05_resume_u280_vs_u380_s20260655/checkpoints/latest.pt
training_runs/population_s3_seed06_resume_u380_vs_u240_argmax_s20260656/checkpoints/latest.pt
training_runs/population_s3_seed07_scripted_altent_s20260657/checkpoints/latest.pt
```

Post-foundation teammate-swap:

```text
scripted opponent:
  slot_mixed argmax      min 0.990, avg 0.998, max 1.000
  slot_mixed stochastic  min 1.000, avg 1.000, max 1.000

frozen u280 opponent:
  slot_mixed argmax      min 0.450, avg 0.655, max 0.875
  slot_mixed stochastic  min 0.225, avg 0.425, max 0.650
```

Decision:

```text
Proceed with this foundation. Do not expand by simply making more scripted
clones. The next larger population step should deliberately include mixed
teammates and frozen/self-play opponents, because the harder frozen eval shows
a real strength spectrum and some weak mixed combinations.

Do not use this foundation as arbitrary cross-slot policies. The role-stress
gate failed against trained opposition.
```

## First 6 Seed Lines

Runner:

```powershell
python scripts/run_population_seed_batch.py --batch-id sprint3_population_seed_batch_v1
```

Shared run settings:

```text
workers=16
rollout_steps=512
eval_episodes=50
eval_interval=20
checkpoint_interval=20
torch_threads=1
post_eval=scripted argmax+stochastic, 2 seed blocks * 25 episodes
```

Seed lines:

| id | ancestry | opponent | purpose |
| --- | --- | --- | --- |
| seed01_scripted_std | fresh | scripted | clean scripted anchor |
| seed02_scripted_low_entropy | fresh | scripted | lower-entropy scripted style |
| seed03_scripted_high_entropy | fresh | scripted | higher-exploration scripted style |
| seed04_selfplay_fresh_vs_u280 | fresh | frozen scripted-u280 stochastic | fresh policy-opponent style |
| seed05_resume_u280_vs_u380 | scripted-u280 | frozen selfplay-u380 stochastic | continued ancestry under stronger frozen opponent |
| seed06_resume_u380_vs_u240_argmax | selfplay-u380 | frozen scripted-u240 argmax | hardened branch against deterministic frozen opponent |

## Outputs

The runner writes:

```text
training_runs/population_batches/sprint3_population_seed_batch_v1/plan.json
training_runs/population_batches/sprint3_population_seed_batch_v1/status.json
training_runs/population_batches/sprint3_population_seed_batch_v1/logs/*.log
training_runs/self_play_pools/sprint3_population_seed_batch_v1_manifest.json
```

The manifest is not a league sampler by itself. It is the first population
inventory. League sampling still needs a later trainer integration step.

## Acceptance After Batch

Accept a seed as working if:

- `latest.pt` exists and is compatible;
- training command returned `0`;
- post scripted eval returned `0`;
- eval has no invalid-action collapse;
- replay/gameplay sanity is not obviously broken.

After all 6:

1. Run teammate-swap again on the six latest checkpoints.
2. Compare native vs mixed teams against scripted and at least one frozen-policy
   opponent.
3. Decide whether the next population step needs varied-teammate training or
   can proceed to a larger league pool.

Result:

```text
Accepted as the first population foundation. Varied-teammate/self-play training
is recommended for the next expansion. Role-randomized training is required if
future league rules need A1/A2/A3 policies to be interchangeable.
```
