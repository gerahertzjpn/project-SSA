# Player Pool Lineage Audit

Date: 2026-06-09

## Executive Verdict

The current active pool is not a clean 18-player league population.

It contains 6 team checkpoints. Each team checkpoint has 3 policy states
(`A1`, `A2`, `A3`), so there are 18 physical policy states, but many of them are
close relatives or near-duplicates.

Also, every serious training run checked so far used the default faction setup:
all players are `humans`. The code supports `humans`, `infected`, and `robots`,
and the observation/action schema is faction-aware, but the current population
has not learned the infected or robot rosters.

Accepted usage of this pool:

- human-only diagnostics;
- frozen opponents for sanity checks;
- proof that the training/eval/checkpoint pipeline works;
- behavior references for movement, construction, and self-play tooling.

Rejected usage of this pool:

- final three-faction league population;
- claim of 18 independent players;
- claim that the current agents know how to play infected or robots.

## Faction Training Audit

`make_initial_state()` defaults every player to `humans` unless
`player_factions` is explicitly passed.

Training env reset currently calls:

```text
make_initial_state(seed=self.config.seed)
```

The searched training configs do not contain `player_factions`, `infected`, or
`robots`. Therefore the existing policy checkpoints should be treated as
human-only policies.

Why this matters:

- movement and targeting behavior may partially transfer;
- human production/building logits were trained;
- infected/robot production logits were not trained under legal masks;
- infected/robot unique-building strategies were not learned;
- cross-faction combat decisions were not learned from real policy play.

## Current Active Pool

Active recovery candidate pool:

```text
Team 01 / spv3_01:
training_runs/fresh_population_v31_selfplay_v3_tmix10_vs_tmix11_s20264201/checkpoints/checkpoint_update_200.pt

Team 02 / spv3_02:
training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix08_s20264202/checkpoints/checkpoint_update_240.pt

Team 03 / spv3_03R:
training_runs/fresh_population_v31_selfplay_v3_recovery_spv303_vs_peer10_s20264303/checkpoints/checkpoint_update_420.pt

Team 04 / spv3_04:
training_runs/fresh_population_v31_selfplay_v3_tmix09_vs_tmix10_s20264204/checkpoints/checkpoint_update_240.pt

Team 05 / spv3_05R:
training_runs/fresh_population_v31_selfplay_v3_recovery_spv305_vs_peer10_s20264305/checkpoints/checkpoint_update_340.pt

Team 06 / spv3_06:
training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix09_s20264206/checkpoints/checkpoint_update_200.pt
```

## Original Fresh Roots

These roots were also human-only under the current audit.

```text
fresh01 = training_runs/fresh_population_v31_seed01_scripted_std_s20261201/checkpoints/checkpoint_update_300.pt
fresh02 = training_runs/fresh_population_v31_seed02_scripted_lowent_s20261202/checkpoints/checkpoint_update_320.pt
fresh03 = training_runs/fresh_population_v31_seed03_scripted_highent_s20261203/checkpoints/checkpoint_update_240.pt
fresh04 = training_runs/fresh_population_v31_seed04_scripted_fastlr_s20261204/checkpoints/checkpoint_update_240.pt
fresh05 = training_runs/fresh_population_v31_seed05_scripted_slow_highent_s20261205/checkpoints/checkpoint_update_280.pt
fresh06 = training_runs/fresh_population_v31_seed06_scripted_wide_minibatch_s20261206/checkpoints/checkpoint_update_320.pt
```

They became bridge/peer/mix lines:

```text
bridge01 <- fresh01, peer01 <- bridge01, mix01 <- peer01
bridge02 <- fresh02, peer02 <- bridge02, mix02 <- peer02
bridge03 <- fresh03, peer03 <- bridge03, mix03 <- peer03
bridge04 <- fresh04, peer04 <- bridge04, mix04 <- peer04
bridge05 <- fresh05, peer05 <- bridge05, mix05 <- peer05
bridge06 <- fresh06, peer06 <- bridge06, mix06 <- peer06
```

Opponent checkpoints influenced behavior during training, but the direct genetic
parent is the resumed/init checkpoint.

## Teammate Mix Lineage

Teammate-mix v1:

```text
tmix01:
  A1 <- mix04 <- fresh04
  A2 <- mix05 <- fresh05
  A3 <- mix06 <- fresh06

tmix02:
  A1 <- mix04 <- fresh04
  A2 <- mix06 <- fresh06
  A3 <- mix05 <- fresh05

tmix03:
  A1 <- mix01 <- fresh01
  A2 <- mix03 <- fresh03
  A3 <- mix02 <- fresh02

tmix04:
  A1 <- mix03 <- fresh03
  A2 <- mix05 <- fresh05
  A3 <- mix04 <- fresh04

tmix05:
  A1 <- mix06 <- fresh06
  A2 <- mix02 <- fresh02
  A3 <- mix01 <- fresh01

tmix06:
  A1 <- mix02 <- fresh02
  A2 <- mix04 <- fresh04
  A3 <- mix03 <- fresh03
```

Hard-negative v2:

```text
tmix07:
  A1 <- tmix06:A1 <- fresh02
  A2 <- tmix01:A2 <- fresh05
  A3 <- tmix02:A3 <- fresh05

tmix08:
  A1 <- tmix03:A1 <- fresh01
  A2 <- tmix04:A2 <- fresh05
  A3 <- tmix05:A3 <- fresh01

tmix09:
  A1 <- tmix05:A1 <- fresh06
  A2 <- tmix06:A2 <- fresh04
  A3 <- tmix01:A3 <- fresh06
```

Hard-negative v3:

```text
tmix10:
  A1 <- tmix09:A1 <- fresh06
  A2 <- tmix08:A2 <- fresh05
  A3 <- tmix07:A3 <- fresh05

tmix11:
  A1 <- tmix09:A1 <- fresh06
  A2 <- tmix07:A2 <- fresh05
  A3 <- tmix08:A3 <- fresh01
```

`tmix10` and `tmix11` are not identical, but they are close relatives:

```text
tmix10 = [fresh06, fresh05, fresh05]
tmix11 = [fresh06, fresh05, fresh01]
```

## Current 18 Physical Players

```text
Player 01 = Team 01 spv3_01:A1 <- tmix10:A1 <- fresh06
Player 02 = Team 01 spv3_01:A2 <- tmix10:A2 <- fresh05
Player 03 = Team 01 spv3_01:A3 <- tmix10:A3 <- fresh05

Player 04 = Team 02 spv3_02:A1 <- tmix11:A1 <- fresh06
Player 05 = Team 02 spv3_02:A2 <- tmix11:A2 <- fresh05
Player 06 = Team 02 spv3_02:A3 <- tmix11:A3 <- fresh01

Player 07 = Team 03 spv3_03R:A1 <- tmix08:A1 <- fresh01
Player 08 = Team 03 spv3_03R:A2 <- tmix08:A2 <- fresh05
Player 09 = Team 03 spv3_03R:A3 <- tmix08:A3 <- fresh01

Player 10 = Team 04 spv3_04:A1 <- tmix09:A1 <- fresh06
Player 11 = Team 04 spv3_04:A2 <- tmix09:A2 <- fresh04
Player 12 = Team 04 spv3_04:A3 <- tmix09:A3 <- fresh06

Player 13 = Team 05 spv3_05R:A1 <- tmix10:A1 <- fresh06
Player 14 = Team 05 spv3_05R:A2 <- tmix10:A2 <- fresh05
Player 15 = Team 05 spv3_05R:A3 <- tmix10:A3 <- fresh05

Player 16 = Team 06 spv3_06:A1 <- tmix11:A1 <- fresh06
Player 17 = Team 06 spv3_06:A2 <- tmix11:A2 <- fresh05
Player 18 = Team 06 spv3_06:A3 <- tmix11:A3 <- fresh01
```

## Ancestry Count

```text
fresh01: Players 06, 07, 09, 18 = 4
fresh02: none = 0
fresh03: none = 0
fresh04: Player 11 = 1
fresh05: Players 02, 03, 05, 08, 14, 15, 17 = 7
fresh06: Players 01, 04, 10, 12, 13, 16 = 6
```

This is the biggest pool-quality problem: `fresh02` and `fresh03` disappeared
from the active pool, `fresh04` is almost gone, and most active players are
descendants of `fresh05`/`fresh06`.

## Closest Relatives / Likely Duplicates

These are not byte-identical checkpoints, but they are sibling continuations
from the same source slot and should not be counted as independent players:

```text
Player 01 and Player 13: both tmix10:A1 / fresh06
Player 02 and Player 14: both tmix10:A2 / fresh05
Player 03 and Player 15: both tmix10:A3 / fresh05
Player 04 and Player 16: both tmix11:A1 / fresh06
Player 05 and Player 17: both tmix11:A2 / fresh05
Player 06 and Player 18: both tmix11:A3 / fresh01
```

Observed similarity examples:

```text
spv3_02:A1 <-> spv3_06:A1 cosine ~= 0.987
spv3_02:A2 <-> spv3_06:A2 cosine ~= 0.982
spv3_02:A3 <-> spv3_06:A3 cosine ~= 0.986
spv3_01:A1 <-> spv3_05R:A1 cosine ~= 0.981
spv3_01:A2 <-> spv3_05R:A2 cosine ~= 0.979
spv3_01:A3 <-> spv3_05R:A3 cosine ~= 0.978
```

Practical meaning of cosine `0.98`:

- very close weight direction;
- likely a sibling variant rather than a fresh independent player;
- not guaranteed to choose identical actions every tick;
- still bad for population diversity if counted as a separate root.

## What Was Wasted vs What Was Salvaged

Wasted for final league population:

- counting these as 18 useful independent agents;
- treating human-only policies as faction-ready;
- training repeated close relatives instead of maintaining ancestry diversity.

Salvaged:

- training pipeline works under `train_obs_v3.1`;
- factorized produce/move/build action interface is usable;
- role-stress, teammate-swap, and stochastic eval tooling exists;
- checkpoint/resume/eval flows are tested;
- the pool can serve as human-only opponents and diagnostics.

## Required Next Correction

Before any expensive league run:

1. Add explicit faction assignment to training configs/envs.
2. Train fresh roots under the intended faction curriculum.
3. Preserve a manifest with player IDs, faction, source seed, direct parent, and
   accepted/rejected status.
4. Never count sibling continuations as independent population roots unless a
   diversity eval says they behave differently enough.
5. Reintroduce missing ancestry if needed: `fresh02`, `fresh03`, and `fresh04`
   should not silently disappear from the active pool.

Recommended next population structure:

```text
minimum:
  3 factions x 6 fresh roots = 18 faction-specialized roots

better:
  3 factions x 6 roots + mixed-faction eval gates before self-play expansion

strict:
  each accepted player gets a stable Player ID and cannot be renamed by run dir
```

