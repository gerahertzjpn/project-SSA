# Balance audit before next large training

Date: 2026-06-03

## Goal

Rebalance the game before the next large training run and separate two questions:

- is Ranger intrinsically overpowered;
- or did earlier policies exploit weak baseline behavior and soft objectives.

## Method

Script:

```bash
python scripts/analyze_balance.py --matches 300 --workers 16 --bot-types random,scripted --out analysis/balance_audit_rebalanced_80hp_300.json
```

The audit measures:

- equal-budget RPS fights;
- minimal armies for neutral point, Lord, and core objectives;
- repeated failed raids into persistent Lord/core HP;
- random-vs-random and scripted-vs-scripted match/lord/economy stats.

## Findings

Ranger:

- Ranger is a strong DPS component against Lord and cores.
- The cheapest full Lord kill remains a mixed army around `47` resources:
  `8 Ranger + 5 Skirmisher` or `8 Ranger + 5 Vanguard`, with heavy losses.
- Ranger does not dominate the whole equal-budget RPS table. At larger budgets,
  Skirmishers beat Rangers, and Vanguards can beat Ranger-heavy armies depending
  on side/formation.
- Decision: do not nerf Ranger in this pass. The clearer balance issues were
  objective durability and neutral-point ROI.

Core:

- Before rebalance, an empty full-health core could be destroyed by about `12`
  resources.
- After rebalance, the cheapest empty-core kill is about `19` resources:
  `1 Ranger + 5 Vanguard`.
- A core with `5 Vanguard` defenders now requires about `31` attacking resources.

Neutral economy:

- Before rebalance, high-strength neutral points were too profitable:
  strength `7` could be cleared by about `6` resources and paid `7/tick`.
- After rebalance, strength `7` needs about `12` resources and pays `4/tick`.
- First viable full-force payback for strengths `1..7` is now roughly:
  `3, 3, 1.5, 3, 3, 3, 3` ticks.

Lord:

- Lord kill rate remains high enough for games to resolve through the objective.
- Random-vs-random, 300 matches:
  - Lord killed: `87.0%`;
  - Lord spawned after kill: `79.0%`;
  - first kill tick mean/median: `139.8 / 130`;
  - spawned Lord destroys a core: `67.9%` per spawn.
- Scripted-vs-scripted still kills Lord every game and mostly resolves through
  death time; this is a baseline limitation, not a Ranger-only signal.

Match stats after rebalance:

- Random-vs-random:
  - winners: `TEAM_A 150`, `TEAM_B 144`, `DRAW 6`;
  - winner types: `normal 58`, `death_time 236`, `draw_no_core_damage 6`;
  - mean tick: `280.3`.
- Scripted-vs-scripted:
  - winners: `TEAM_A 146`, `TEAM_B 129`, `DRAW 25`;
  - winner types: `death_time 275`, `draw_no_core_damage 25`;
  - mean tick: `300`.

## Applied ruleset

- Core: `80hp`, `6 atk`, range `12.0`.
- Neutral Lord: `120hp`, `16 atk`, attack rate `3`, range `11.0`.
- Neutral guard total attack: `1.5 * guard_strength`.
- Neutral income: `ceil(guard_strength / 2)`, so strength `1..7` pays
  `1,1,2,2,3,3,4`.

## Training note

This is a new ruleset. Older checkpoints are useful only as historical proof of
learning/exploitation and should not be treated as behavior-compatible warm
starts for the next large run unless explicitly retrained on this ruleset.

The next training run should watch:

- normal win-rate separately from death-time win-rate;
- Ranger-heavy policies, but without assuming Ranger is the root problem;
- neutral-point farming versus actual core pressure;
- Lord contest timing and whether policies learn to deny or escort Lord.
