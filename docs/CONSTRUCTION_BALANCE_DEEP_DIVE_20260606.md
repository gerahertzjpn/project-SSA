# Construction Balance Deep Dive

Date: 2026-06-06

Status: diagnostic pass, no numeric balance constants changed in this pass.

Primary files:

```text
analysis/construction_balance_deep_dive_20260606_50.json
analysis/construction_balance_core_focus_20260607_50.json
scripts/construction_balance_deep_dive.py
```

## Question

We needed to answer:

- is death-time common because random bots cannot finish;
- or is the ruleset itself too slow / too defensive;
- whether new economy, turrets, Tech1/Tech2, Raider, and BMP create an obvious
  dominant strategy.

## Key Finding

Death-time is mostly a bot/target-priority problem, not a hard game-speed
problem.

Non-core-focused strategies often reach tick 300, even when one team is clearly
ahead. Core-focused strategies can close games normally:

```text
core_raider_fast_vs_scripted:
  normal wins: 50 / 50
  median ticks: 84
  mean ticks: 95.82

core_raider_fast_vs_basic:
  normal wins: 43 / 50
  death-time: 1 / 50
  draw_no_core_damage: 6 / 50
  median ticks: 101

core_bmp_rush_vs_basic:
  normal wins: 48 / 50
  death-time: 2 / 50
  median ticks: 172.5
```

So the engine can produce normal finishes. The current random/scripted/simple
heuristics usually fail to prioritize sequential core kills.

## Macro Results

50 matches per scenario:

```text
random_mirror:
  winners: TEAM_A 22 / TEAM_B 24 / DRAW 4
  winner types: normal 8 / death_time 38 / draw_no_core_damage 4
  mean ticks: 283.36
  mean core combat damage: 170.84

scripted_mirror:
  winners: DRAW 50
  winner types: draw_no_core_damage 50
  core combat damage: 0

basic_mix_mirror:
  winner types: death_time 36 / draw_no_core_damage 14
  core combat damage mean: 57.6

raider_fast_vs_basic:
  TEAM_A 38 / DRAW 12
  winner types: death_time 38 / draw_no_core_damage 12
  mean destroyed B cores at tick 269: 0.76

raider_fast_vs_scripted:
  TEAM_A 50 / 50
  winner types: death_time 50
  mean destroyed B cores at tick 269: 0.98

bmp_rush_vs_basic:
  TEAM_B 50 / 50
  winner types: death_time 50
  mean destroyed A cores at tick 269: 1.0
```

Interpretation:

- simple non-core-focused strategies frequently kill one core, then fail to
  finish all three before death-time;
- current scripted baseline is not a useful balance gate because it can produce
  many units and still deal zero core damage in mirror;
- core-focus immediately converts similar unit/economy setups into normal wins.

## Economy

Current economy:

```text
base income: +2 per player per tick
collector: cost 14, build 10t, income +2 per tick
collector payback after completion: 7t
collector payback from build start: 17t
```

Because current non-core-focused games often last to 300, Collector has extreme
long-game ROI:

```text
scripted_mirror collector income:
  TEAM_A 1731.36 per match
  TEAM_B 1731.36 per match

random_mirror collector income:
  TEAM_A 1180.8 per match
  TEAM_B 1188.72 per match
```

This does not automatically make Collector the best strategy because base core
slots are limited to 2:

```text
Collector + Tech1 blocks Tech2
Tech1 + Tech2 blocks Collector
Collector + Enhanced Turret blocks tech
```

However, if the game stays long, Collector is probably too efficient. It doubles
a player's base income for very low cost and pays back very early.

## Tech Timing

Fast Tech1:

```text
raider_fast:
  first TECH_1 complete median: 15t
  first Raider produced median: 16t
```

Eco Tech1:

```text
raider_eco:
  first Collector complete median: 11-12t
  first TECH_1 complete median: 22-23t
  first Raider produced median: 22-23t
```

Tech2/BMP:

```text
bmp_rush:
  first TECH_1 complete median: 15-16t
  first TECH_2 complete median: 32-33t
  first BMP produced median: 32-33t
```

Current random rarely reaches Tech2:

```text
random_mirror:
  TECH_2 completed in 11 / 50 matches
  BMP produced in 11 / 50 matches
  median first BMP tick among those: 147
```

Current scripted never reaches Tech2 because it fills its two core slots with
Collector + Tech1.

## Unit Balance

Basic RPS is not perfectly stable at every budget, but the intended shape is
mostly present:

```text
Ranger tends to beat Vanguard at 50+
Vanguard beats Skirmisher
Skirmisher can beat Ranger at mid budgets, but not cleanly at every budget
```

Raider is the suspicious Tech1 unit:

```text
budget 20:
  6 Raider beat 6 Vanguard with 6 Raiders alive
  6 Raider beat 4 Ranger with 2 Raiders alive
  6 Raider beat 10 Skirmisher with 5 Raiders alive

budget 50:
  16 Raider beat 16 Vanguard with 14 Raiders alive
  16 Raider beat 10 Ranger with 5 Raiders alive
  16 Raider beat 25 Skirmisher with 16 Raiders alive
```

Raider currently counters all basics very hard and unlocks around tick 15-23.
This is the most likely current imbalance.

BMP is very strong after unlock, but not obviously dominant in macro because
Tech2 has high opportunity cost:

```text
budget 50:
  3 BMP beat Vanguard/Ranger/Skirmisher/Raider

budget 20:
  1 BMP loses or trades badly into Raider groups
```

Core-focused BMP can close games, but slower than core-focused Raider.

## Turrets And Base Defense

Standalone turrets are not overpowered:

```text
TURRET:
  20 budget of any tested unit type can clear it.

ENHANCED_TURRET:
  20 budget of any tested unit type can clear it.
```

Enhanced turret plus core is much more meaningful:

```text
ENHANCED_TURRET + CORE first-win thresholds:
  BMP:       20 budget, 1 BMP, 23t, survives at 15 hp
  RAIDER:    30 budget, 10 Raiders, 9t
  SKIRMISHER:40 budget, 20 Skirmishers, 5t
  VANGUARD:  50 budget, 16 Vanguards, 19t
  RANGER:    80 budget, 16 Rangers, 12t
```

So turrets are not an obvious imba by themselves. Base defense mostly matters
when combined with core range/damage.

## Current Balance Risks

1. Current scripted baseline is broken as a balance signal.

It mirrors into `draw_no_core_damage 50 / 50`. It builds Collector + Tech1,
spams Raiders, and still never damages cores in mirror.

2. Raider is probably too efficient.

It wins every basic matchup very hard while being cheap and fast.

3. Collector is probably too efficient if games remain long.

Payback from build start is only 17 ticks. In long games this becomes a massive
resource advantage.

4. BMP is strong but gated enough for now.

BMP micro is scary, but Tech2 rush loses to basic non-core-focused pressure if
it does not target cores correctly. Core-focused BMP wins, but slower than
core-focused Raider.

5. Death-time should not be patched first.

Core-focus proves normal finishes are available. The first patch should fix
baseline target priorities and evaluate policy learning under core-visible
objectives before changing tick limit or death-time damage.

## Recommended Patch Direction

Do first:

- add/fix scripted baseline variants:
  - `scripted_core_basic`
  - `scripted_core_raider`
  - optional `scripted_core_bmp`
- make scripted eval use at least one core-focused baseline;
- keep death-time as-is until RL is tested against core-focused opponents.

Likely balance changes to test next:

```text
RAIDER:
  raise cost 3 -> 4
  or reduce hp 12 -> 9/10
  or reduce damage 8 -> 6/7

ECONOMY_COLLECTOR:
  raise cost 14 -> 20-24
  or build ticks 10 -> 14-16
  or income 2 -> 1
```

Do not nerf BMP yet without a second pass. Its first-unit effective cost includes
Tech1 + Tech2, and current macro does not show it as the dominant fastest route.

## Training Implication

For the next training run, reward/eval must include real core-pressure metrics:

- normal win rate separately from death-time win rate;
- cores destroyed before tick 269;
- core damage before death-time;
- target-priority diagnostics after first enemy core dies.

If a policy learns core-focus, death-time rate should drop naturally without
touching the timer.
