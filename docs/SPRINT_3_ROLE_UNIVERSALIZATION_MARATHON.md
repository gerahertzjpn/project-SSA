# Sprint 3 Role Universalization Marathon

Date: 2026-06-05

Status: active.

## User Directive

The root seed models must become approximately universal across A1/A2/A3
roles before the next large run. Later branches may specialize, but the current
population roots should not be locked to one role.

Run order is mandatory:

1. Universalize the current models for any role.
2. After role-stress gates confirm approximate universality, start the big run.

Do not skip directly to the big run while role-stress against trained opposition
is failing.

## Current Diagnosis

The 7 current population candidates are valid, diverse, and useful, but they are
slot-specialized:

```text
slot-preserving teammate-swap: accepted
cross-slot role-stress vs scripted: mostly survives
cross-slot role-stress vs frozen u280: fails hard
```

Frozen u280 role-stress result:

```text
argmax average win-rate:     0.071
stochastic average win-rate: 0.057
```

This means A1/B1 policies can be mixed with A1/B1 policies, A2/B2 with A2/B2,
and A3/B3 with A3/B3, but current roots are not safe as arbitrary A1/A2/A3
policies.

## Implementation Rule

Keep the RL interface stable for this marathon:

```text
OBSERVATION_VERSION = train_obs_v1.0
OBSERVATION_DIM = 1117
ACTION_DIM = 1356
```

Do not add an explicit role one-hot unless the v1.0 role-randomized curriculum
fails. Adding role one-hot requires a new observation version and a new
checkpoint family.

## Curriculum Added

Training now supports:

```text
--role-assignment-mode fixed
--role-assignment-mode shuffle
```

`fixed` is the old behavior.

`shuffle` randomly maps policy ids to environment slots at episode boundaries:

```text
policy A1 can play env slot A1/A2/A3
policy A2 can play env slot A1/A2/A3
policy A3 can play env slot A1/A2/A3
```

Experience is written back to the policy id that actually played the slot, not
to the environment slot. This trains each policy on all roles while preserving
the current observation/action interface.

## Validation Before Marathon Training

Completed:

```text
pytest tests/test_team_ppo_trainer.py tests/test_team_training_env.py tests/test_training_spaces.py tests/test_teammate_swap_eval.py -q
39 passed
```

Smoke completed:

```text
python scripts/train_team_policy.py --run-dir training_runs/tmp_role_shuffle_smoke --overwrite --resume training_runs/population_s3_seed01_scripted_std_s20260651/checkpoints/latest.pt --updates 1 --workers 1 --rollout-steps 16 --eval-episodes 2 --eval-interval 1 --checkpoint-interval 1 --random-baseline-episodes 0 --torch-threads 1 --role-assignment-mode shuffle --opponent-bot scripted
```

Result: CLI resume/train/eval/save path works with role shuffle.

## Universalization Plan

Use the 7 accepted latest checkpoints as roots.

For each root:

- resume from the current latest checkpoint;
- train with `--role-assignment-mode shuffle`;
- use scripted first to restore role-flexible basics;
- if role-stress vs frozen remains weak, continue selected candidates against
  frozen u280/self-play opponents with shuffle still enabled.

Initial gate after universalization:

```text
role_stress vs scripted:
  target argmax win >= 0.95
  target stochastic win >= 0.95
  normal-win should be high, not mostly death-time

role_stress vs frozen u280:
  target argmax average should be meaningfully above previous 0.071
  target stochastic average should be meaningfully above previous 0.057
  practical pass target for moving on: average >= 0.40 and no total collapse
```

If frozen role-stress remains near the old 5-7% range, the marathon must switch
to observation v1.1 planning with an explicit role signal instead of launching
the big population/self-play run.

## Big Run Gate

Only after the universal roots pass role-stress:

- build a new manifest of universalized roots;
- launch the larger population/self-play run;
- keep mixed teammates enabled;
- keep role-shuffle enabled for root/universal branches unless a branch is
  deliberately marked role-specialized.

## Open Risk

Role-shuffle may reduce peak native-slot strength while improving universality.
This is acceptable for roots. Later branches can specialize again after the
universal foundation exists.

## 2026-06-05 Progress Log

### V1: scripted role-shuffle universalization

Batch:

```text
training_runs/role_universalization_batches/sprint3_role_universalization_v1
```

Result:

```text
7 / 7 runs completed
7 / 7 latest checkpoints compatible
role_assignment_mode = shuffle
```

Scripted role-stress gate:

```text
argmax:
  average win-rate: 0.994
  minimum win-rate: 0.970
  max loss-rate:    0.000
  max draw-rate:    0.030

stochastic:
  average win-rate: 1.000
  minimum win-rate: 1.000
  max loss-rate:    0.000
  max draw-rate:    0.000
```

Conclusion: role universality against scripted is accepted.

Frozen u280 role-stress gate after V1:

```text
argmax:
  average win-rate: 0.436
  minimum win-rate: 0.075

stochastic:
  average win-rate: 0.232
  minimum win-rate: 0.150
```

Conclusion: V1 improved the old frozen role-stress baseline, but not enough to
launch the big population/self-play run. The roots are now role-flexible against
scripted, but still too weak or unstable against a trained stochastic policy
when role-stressed.

Important discovery: V1 trained all roots against `--opponent-bot scripted`.
The run names preserved root lineage such as `vs_u280`, but the actual V1
training opponent was scripted. This explains why scripted role-stress passed
cleanly while frozen role-stress remained weak.

### V2: policy-pressure role-shuffle continuation

Added:

```text
scripts/run_role_universalization_policy_batch.py
```

Purpose:

- resume from the V1 role-universalized latest checkpoints;
- keep `--role-assignment-mode shuffle`;
- train against frozen policy opponents instead of scripted;
- vary the frozen pressure across u240/u280/u300 and argmax/stochastic modes
  to avoid producing one narrow exploit.

Batch:

```text
training_runs/role_universalization_batches/sprint3_role_universalization_policy_v1
```

Validation before launch:

```text
python scripts/run_role_universalization_policy_batch.py --dry-run
python -m pytest tests/test_team_ppo_trainer.py tests/test_teammate_swap_eval.py -q

9 passed
```

Gate after V2:

1. Re-run scripted role-stress to verify we did not forget the basics.
2. Re-run frozen u280 role-stress.
3. Big run may start only if frozen role-stress no longer shows total slot
   collapse and stochastic league-mode is acceptable.

### V2 result: policy-pressure helped, but stochastic gate is not closed

Batch:

```text
training_runs/role_universalization_batches/sprint3_role_universalization_policy_v1
```

Result:

```text
7 / 7 runs completed
7 / 7 latest checkpoints compatible
```

Best internal eval checkpoints:

```text
seed01 latest/u300: win 0.68
seed02 latest/u340: win 0.72
seed03 latest/u320: win 0.90
seed04 latest/u400: win 0.76
seed05 latest/u580: win 0.56
seed06 checkpoint_update_640: win 0.66
seed07 checkpoint_update_220: win 0.62
```

Scripted role-stress on V2 best candidates:

```text
argmax:
  average win-rate: 0.984
  minimum win-rate: 0.910
  max loss-rate:    0.020
  max draw-rate:    0.090

stochastic:
  average win-rate: 0.997
  minimum win-rate: 0.990
  max loss-rate:    0.010
  max draw-rate:    0.000
```

Conclusion: scripted gate still passes.

Frozen u280 role-stress on V2 best candidates:

```text
argmax:
  average win-rate: 0.621
  minimum win-rate: 0.267

stochastic:
  average win-rate: 0.318
  minimum win-rate: 0.067
```

Conclusion: V2 improved argmax substantially, but stochastic league-mode still
fails. Big population/self-play run must not start yet.

Weak roots after V2 frozen stress:

```text
seed05: stochastic win 0.067
seed06: stochastic win 0.173
seed07: stochastic win 0.200
```

### V3: stochastic stabilization

Added:

```text
scripts/run_role_universalization_stabilize_batch.py
```

Purpose:

- resume from V2 best candidates;
- train all roots against frozen u280 stochastic;
- keep role shuffle enabled;
- reduce entropy and learning rate to sharpen stochastic policy behavior rather
  than only improving argmax.

Validation before launch:

```text
python scripts/run_role_universalization_stabilize_batch.py --dry-run
python -m pytest tests/test_team_ppo_trainer.py tests/test_teammate_swap_eval.py -q

9 passed
```

V3 gate:

1. scripted role-stress must remain acceptable;
2. frozen u280 stochastic role-stress must improve above V2;
3. if weak roots remain, either exclude them from the big root pool or run a
   targeted follow-up only for those roots.

### V3 result: close, but weak stochastic tails remained

Batch:

```text
training_runs/role_universalization_batches/sprint3_role_universalization_stabilize_v1
```

Best candidates used for V3 gate:

```text
seed01 latest
seed02 checkpoint_update_400
seed03 latest
seed04 latest
seed05 checkpoint_update_640
seed06 latest
seed07 latest
```

Scripted role-stress:

```text
argmax:
  average win-rate: 0.994
  minimum win-rate: 0.980
  max loss-rate:    0.010
  max draw-rate:    0.020

stochastic:
  average win-rate: 0.999
  minimum win-rate: 0.990
  max loss-rate:    0.010
  max draw-rate:    0.000
```

Frozen u280 role-stress:

```text
argmax:
  average win-rate: 0.746
  minimum win-rate: 0.570

stochastic:
  average win-rate: 0.581
  minimum win-rate: 0.410
```

Weak stochastic tails:

```text
seed02: 0.500
seed06: 0.470
seed07: 0.410
```

Conclusion: V3 made the roots much more universal, but the stochastic league
gate was not clean enough for all roots. A targeted V4 follow-up was required
instead of launching the big population/self-play run.

### V4: targeted low-entropy stabilization

Added:

```text
scripts/run_role_universalization_targeted_stabilize_batch.py
```

Purpose:

- resume only weak or borderline V3 roots;
- keep role shuffle enabled;
- train against frozen u280 stochastic;
- use lower entropy and learning rate to make stochastic sampling closer to
  the learned skill without changing `train_obs_v1.0`.

Validation before launch:

```text
python scripts/run_role_universalization_targeted_stabilize_batch.py --dry-run
python -m pytest tests/test_team_ppo_trainer.py tests/test_teammate_swap_eval.py -q

9 passed
```

Batch:

```text
training_runs/role_universalization_batches/sprint3_role_universalization_targeted_v4
```

Internal best checkpoints:

```text
seed02: checkpoint_update_520, win 0.917
seed04: checkpoint_update_560, win 0.883
seed06: checkpoint_update_820, win 0.783
seed07: checkpoint_update_320, win 0.833
```

Final accepted universal-root set:

```text
training_runs/role_universal_stabilize_seed01_u280_stoch_s20260901/checkpoints/latest.pt
training_runs/role_universal_v4_seed02_u280_stoch_lowent_s20261002/checkpoints/checkpoint_update_520.pt
training_runs/role_universal_stabilize_seed03_u280_stoch_s20260903/checkpoints/latest.pt
training_runs/role_universal_v4_seed04_u280_stoch_lowent_s20261004/checkpoints/checkpoint_update_560.pt
training_runs/role_universal_stabilize_seed05_u280_stoch_s20260905/checkpoints/checkpoint_update_640.pt
training_runs/role_universal_v4_seed06_u280_stoch_lowent_s20261006/checkpoints/checkpoint_update_820.pt
training_runs/role_universal_v4_seed07_u280_stoch_lowent_s20261007/checkpoints/checkpoint_update_320.pt
```

Final scripted role-stress:

```text
argmax:
  average win-rate: 0.991
  minimum win-rate: 0.940
  max loss-rate:    0.000
  max draw-rate:    0.060

stochastic:
  average win-rate: 1.000
  minimum win-rate: 1.000
  max loss-rate:    0.000
  max draw-rate:    0.000
```

Final frozen u280 role-stress:

```text
argmax:
  average win-rate: 0.830
  minimum win-rate: 0.730

stochastic:
  average win-rate: 0.671
  minimum win-rate: 0.570
```

Decision: role-universal root gate is accepted. The roots are not perfect, but
they are no longer slot-locked and all seven stochastic role-stress results are
meaningfully above the old collapse. It is safe to start the larger
population/self-play continuation, keeping role shuffle enabled.

### Big continuation: role-universal frozen self-play population

Added:

```text
scripts/run_role_universal_population_big_batch.py
```

Purpose:

- continue all seven accepted universal roots;
- keep `--role-assignment-mode shuffle`;
- use frozen peer-policy opponents instead of only scripted anchors;
- vary ancestry, opponent, entropy, duration, and stochastic/argmax pressure so
  the next pool is not a set of clones.

Validation before launch:

```text
python scripts/run_role_universal_population_big_batch.py --dry-run
python -m pytest tests/test_team_ppo_trainer.py tests/test_teammate_swap_eval.py -q

9 passed
```

Batch:

```text
training_runs/role_universal_population_batches/sprint3_role_universal_population_big_v1
```

Run lines:

```text
unipop_seed01_vs_seed03_stoch
unipop_seed02_vs_seed05_stoch
unipop_seed03_vs_seed01_argmax
unipop_seed04_vs_seed02_stoch
unipop_seed05_vs_seed07_stoch
unipop_seed06_vs_seed03_stoch
unipop_seed07_vs_seed05_stoch
```

This is the first post-gate population/self-play continuation. It is still a
safe frozen-opponent bridge, not full manifest-sampled league training.

Live progress notes:

```text
unipop_seed01_vs_seed03_stoch:
  status: finished
  last update: 560
  best eval: u440/u460/u500/u560 all reached 1.000 argmax win-rate
  preferred checkpoint so far: latest.pt / checkpoint_update_560.pt
  note: strong stable branch, no draw problem.

unipop_seed02_vs_seed05_stoch:
  status: finished
  last update: 700
  trajectory: weak start -> u600 0.800 -> u620/u640 dip -> u680 0.950
  last eval: u700 win 0.883, normal 0.800, loss 0.117, draw 0.000
  preferred checkpoint so far: checkpoint_update_680.pt
  note: useful branch, but do not select latest blindly.

unipop_seed03_vs_seed01_argmax:
  status: finished
  last update: 560
  trajectory: u420/u440 0.883 -> u460 dip -> u500 0.950 -> u540 0.967
  last eval: u560 win 0.933, normal 0.883, loss 0.000, draw 0.067
  preferred checkpoint so far: checkpoint_update_540.pt
  note: deterministic peer pressure eventually helped, but latest has too much draw.

unipop_seed04_vs_seed02_stoch:
  status: finished
  last update: 740
  trajectory: severe weak start u580 0.267 -> u620 0.467 -> u640 0.950 -> u720/u740 1.000
  last eval: u740 win 1.000, normal 1.000, loss 0.000, draw 0.000
  preferred checkpoint so far: checkpoint_update_660.pt after post-eval
  note: late branch recovered internally, but u660 is cleaner than u740 against scripted.

unipop_seed05_vs_seed07_stoch:
  status: finished
  last update: 820
  trajectory: strong from first eval; u660 0.983 -> u720 1.000 -> u820 1.000
  last eval: u820 win 1.000, normal 1.000, loss 0.000, draw 0.000
  preferred checkpoint so far: latest.pt / checkpoint_update_820.pt
  note: stable strong branch against the weakest accepted stochastic peer.

unipop_seed06_vs_seed03_stoch:
  status: finished
  last update: 1020
  trajectory: u840 0.900 -> u860 0.983 -> u880 through u1020 all 1.000
  last eval: u1020 win 1.000, normal 1.000, loss 0.000, draw 0.000
  preferred checkpoint so far: latest.pt / checkpoint_update_1020.pt
  note: strongest stable branch so far; long perfect eval streak.

unipop_seed07_vs_seed05_stoch:
  status: finished
  last update: 520
  trajectory: u340 0.583 -> u400 0.633 -> u460 0.733 -> u500 0.783 -> u520 0.883
  last eval: u520 win 0.883, normal 0.850, loss 0.083, draw 0.033
  preferred checkpoint so far: latest.pt / checkpoint_update_520.pt
  note: partial recovery, weakest competitive lineage.
```

Post-run focused tests:

```text
python -m pytest tests/test_team_ppo_trainer.py tests/test_teammate_swap_eval.py tests/test_evaluate_checkpoints.py -q

11 passed
```

Selected population pool v2:

```text
seed01: training_runs/universal_population_seed01_vs_seed03_stoch_s20261101/checkpoints/latest.pt
seed02: training_runs/universal_population_seed02_vs_seed05_stoch_s20261102/checkpoints/checkpoint_update_680.pt
seed03: training_runs/universal_population_seed03_vs_seed01_argmax_s20261103/checkpoints/checkpoint_update_540.pt
seed04: training_runs/universal_population_seed04_vs_seed02_stoch_s20261104/checkpoints/checkpoint_update_660.pt
seed05: training_runs/universal_population_seed05_vs_seed07_stoch_s20261105/checkpoints/latest.pt
seed06: training_runs/universal_population_seed06_vs_seed03_stoch_s20261106/checkpoints/latest.pt
seed07: training_runs/universal_population_seed07_vs_seed05_stoch_s20261107/checkpoints/latest.pt
```

Reason for seed04 selection: internal latest was perfect against its frozen
opponent, but scripted post-eval showed `checkpoint_update_660.pt` was cleaner:

```text
u660 argmax/stochastic: win 1.000, normal 1.000, loss 0.000, draw 0.000
u740 argmax:            win 0.850, normal 0.700, draw 0.150, death_time 0.150
u740 stochastic:        win 1.000, normal 1.000, loss 0.000, draw 0.000
```

Selected v2 scripted teammate-swap gate:

```text
records: 56

native/argmax:        avg win 1.000, min win 1.000, avg normal 0.979, min normal 0.900
native/stochastic:    avg win 1.000, min win 1.000, avg normal 0.993, min normal 0.950
role_stress/argmax:   avg win 1.000, min win 1.000, avg normal 0.993, min normal 0.950
role_stress/stoch:    avg win 1.000, min win 1.000, avg normal 1.000, min normal 1.000
slot_mixed/argmax:    avg win 1.000, min win 1.000, avg normal 0.996, min normal 0.950
slot_mixed/stoch:     avg win 1.000, min win 1.000, avg normal 1.000, min normal 1.000

invalid actions: 0
```

Decision: the scripted role-universal teammate-swap gate is cleanly passed.
The selected v2 pool can play native, mixed teammate, and role-stressed
scripted eval without losses.

Selected v2 peer stress against frozen `seed06` stochastic:

```text
records: 56

native/argmax:        avg win 0.371, min win 0.000, avg loss 0.429, max draw 0.700
native/stochastic:    avg win 0.300, min win 0.000, avg loss 0.671
role_stress/argmax:   avg win 0.529, min win 0.200, avg loss 0.157, avg draw 0.314
role_stress/stoch:    avg win 0.286, min win 0.000, avg loss 0.657
slot_mixed/argmax:    avg win 0.457, min win 0.000, avg loss 0.364, max draw 0.800
slot_mixed/stoch:     avg win 0.386, min win 0.000, avg loss 0.600

invalid actions: 0
```

Strong peer-stress samples:

```text
slot_mixed_alt seed04+seed06+seed05 argmax: win 1.000, normal 0.900
role_stress seed05 argmax:                  win 1.000, normal 0.900
slot_mixed_alt seed05+seed07+seed06 argmax: win 0.900, normal 0.900
role_stress seed04 argmax:                  win 0.900, normal 0.900
native seed04 argmax:                       win 0.800, normal 0.800
native seed05 argmax:                       win 0.800, normal 0.700
```

Weak peer-stress samples:

```text
native seed02 stochastic: loss 1.000
role_stress seed03 stochastic: loss 1.000
role_stress seed07 stochastic: loss 1.000
slot_mixed seed07+seed01+seed02 argmax/stoch: loss 1.000 in stochastic, weak in argmax
slot_mixed seed06+seed01+seed07 argmax/stoch: loss 1.000
native seed03 argmax: win 0.000, draw 0.700, loss 0.300
```

Peer decision: this does not invalidate the scripted role-universal gate. It
does show that selected v2 is not yet a balanced league population. Seed06 is a
strong frozen anchor, seed04/seed05 are the cleanest peers, and seed01/seed02/
seed03/seed07 need recovery or curriculum exposure before being treated as
equal-strength league roots.

Next recommended training shape:

```text
1. Keep role shuffle enabled.
2. Use selected v2 as the initial pool, but weight opponents by curriculum:
   - scripted/random sanity remains a low-frequency guardrail;
   - seed06 and seed05 are strong anchors;
   - seed04 is a clean high-skill branch;
   - seed01/seed02/seed03/seed07 get recovery exposure against mixed anchors.
3. Do not launch 18-24 equal population lines yet.
4. Launch a smaller recovery+league bridge first, then rerun teammate-swap and
   peer-stress before expanding the population.
```

Role/position observation note:

```text
No explicit new slot-id feature was added. Adding one would change
OBSERVATION_DIM and break existing checkpoints.

The policy still sees its practical role implicitly:
- owner relation marks the controlled player's own core/node as `self`;
- route reachability, route edge count, and route ETA are computed from that
  player's available sources;
- per-node stationed, moving, and battle features are ego-centric
  self/ally/enemy/neutral;
- B-side players are mirrored into A-side policy perspective before encoding
  and decoded back after action selection.

Result: role shuffle can train A1/A2/A3 interchangeability without changing
the neural network input shape. The post-run scripted role-stress gate confirms
this is enough for scripted play; peer stress shows strength/generalization is
the remaining issue, not checkpoint incompatibility.
```
