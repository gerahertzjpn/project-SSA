# Balance Patch: Raider 7 / Collector 24

Date: 2026-06-06

Status: implemented and tested.

## Changes

```text
RAIDER:
  cost 7
  hp 32
  atk 12
  range 2.0
  speed 3.0

ECONOMY_COLLECTOR:
  cost 24
  build 16t
  income 2/tick unchanged
```

Collector payback from start is now 28 ticks: 16 build ticks plus 12 ticks to
recover 24 resources at +2/tick.

## Scripted Eval Variants

Added model ids:

```text
scripted_core
scripted_core_raider
scripted_core_bmp
```

These are intended for eval gates where the baseline must pressure cores. The
old `scripted` baseline remains useful as a weak/stability control, but it is
too passive and too draw-heavy to be the only Sprint 3 gate.

## Balance Results

Source files:

```text
analysis/current_balance_suite_raider7_collector24_20260606.json
analysis/construction_balance_deep_dive_raider7_collector24_20260606_80.json
```

Random macro, 120 matches:

```text
TEAM_A 57 / TEAM_B 59 / DRAW 4
normal 45 / death_time 71 / draw_no_core_damage 4
avg ticks 261.65, median 286
```

Lord trace, 60 matches:

```text
neutral lord killed: 60 / 60
spawned after kill: 42 / 60
killed but no valid owner: 18 / 60
first kill tick: min 27, median 68, mean 67.883, max 108
spawned lord core-expended: 22 / 42
spawned lord died before core: 20 / 42
```

Equal-budget 50 micro:

```text
RAIDER beats VANGUARD: 7 raiders win with 6 alive
RAIDER loses to RANGER: 10 rangers win with 2 alive
RAIDER beats SKIRMISHER: 7 raiders win with 3 alive
RAIDER loses to BMP: 3 BMP win with 2 alive
```

Lord objective at 50 budget:

```text
VANGUARD loses; Lord survives at 57 hp
RANGER loses; Lord survives at 100 hp
SKIRMISHER loses; Lord survives at 42 hp
RAIDER wins; 2 Raiders survive
BMP wins; 2 BMP survive
```

Core-focus strategy checks, 80 matches each:

```text
core_basic_vs_scripted: TEAM_A 80, normal 78, median 90t
core_raider_fast_vs_scripted: TEAM_A 76 / DRAW 4, normal 62, median 129.5t
core_basic_vs_basic: TEAM_A 60 / TEAM_B 20, normal 60, median 97t
core_raider_fast_vs_basic: TEAM_A 70 / TEAM_B 10, normal 70, median 160t
core_bmp_rush_vs_basic: TEAM_A 79 / TEAM_B 1, normal 79, median 170.5t
```

Non-core strategy checks show why core-focus eval matters:

```text
raider_fast_vs_basic: TEAM_B 80, death_time 80
bmp_rush_vs_basic: TEAM_B 80, death_time 80
eco_turtle_vs_basic: TEAM_B 56 / TEAM_A 3 / DRAW 21
ranger_spam_vs_basic: TEAM_A 25 / TEAM_B 7 / DRAW 48
```

The same tech units can be strong in fights but fail macro if the bot refuses to
pressure cores. Balance gates should therefore include explicit core-pressure
baselines, not only generic mirror/random outcomes.

## Crash Tests

```text
python scripts/crash_test_simulation.py --matches 100 --seed-base 2026060600 \
  --bot-types random,scripted_core,scripted_core_raider,scripted_core_bmp \
  --visual-audit --visual-ticks 180 --visual-jump 160 --visual-speed-ratio 6

result: seeds_with_issues 0, issue_summary {}

python scripts/crash_test_simulation.py --matches 25 --seed-base 2026060990 \
  --bot-types random,scripted_core_raider \
  --visual-audit --visual-ticks 0 --visual-jump 140 --visual-speed-ratio 5

result: seeds_with_issues 0, issue_summary {}
```

## Current Interpretation

- No obvious team-side bias showed up in random macro: 57 vs 59 wins.
- RAIDER is no longer a cheap all-purpose unit. It beats Vanguard/Skirmisher,
  loses narrowly to Rangers, and loses to BMP.
- BMP remains the most suspicious power unit in pure micro. Its current limiter
  is tech2 cost/build/slot pressure, not combat weakness.
- Collector is less free than before, but still powerful in long games. It is
  currently balanced mainly by base-slot conflict.
- Old scripted and naive spam mirrors are still too draw-heavy. This is a bot
  strategy/eval issue more than a hard rules issue.

