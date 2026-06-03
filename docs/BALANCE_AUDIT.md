# Balance audit before next large training

Date: 2026-06-03

## Goal

Rebalance the game before the next large training run and separate two questions:

- is Ranger intrinsically overpowered;
- or did earlier policies exploit weak baseline behavior and soft objectives.

## Method

Current script:

```bash
python scripts/analyze_balance.py --matches 300 --workers 16 --bot-types random,scripted --out analysis/balance_audit_neutral_hp2_300.json
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

Open issue after compact deployment fix:

- Direct scenario `10 Skirmisher` entering from the left versus `10 Ranger`
  entering from the right currently ends with `9 Ranger` surviving.
- `14 Skirmisher` (`42` resources) versus `10 Ranger` (`40` resources) still
  loses, leaving roughly `7-8 Ranger`.
- `20 Skirmisher` (`60` resources) versus `10 Ranger` (`40` resources) also
  loses, leaving roughly `3 Ranger`.
- This result was measured after fixing compact combat deployment, where large
  armies no longer spawn stretched through the whole `0..20` combat line. It
  means the previous Skirmisher-over-Ranger result was contaminated by a
  formation bug, and Ranger is again the main balance suspect before a large
  training run.

Core:

- Before rebalance, an empty full-health core could be destroyed by about `12`
  resources.
- After rebalance, the cheapest empty-core kill is about `19` resources:
  `1 Ranger + 5 Vanguard`.
- A core with `5 Vanguard` defenders now requires about `31` attacking resources.

Neutral economy:

- Before rebalance, high-strength neutral points were too profitable:
  strength `7` could be cleared by about `6` resources and paid `7/tick`.
- After rebalance, strength `7` needs about `12` resources, usually loses
  about `9` resources of attackers, and pays `4/tick`.
- First viable full-force payback for strengths `1..7` is now roughly:
  `3, 3, 3, 3.5, 3, 4, 3` ticks.

Lord:

- Lord kill rate remains high enough for games to resolve through the objective.
- Random-vs-random, 300 matches:
  - Lord killed: `88.3%`;
  - Lord spawned after kill: `77.0%`;
  - first kill tick mean/median: `143.8 / 133`;
  - spawned Lord destroys a core: `71.0%` per spawn.
- Scripted-vs-scripted still kills Lord every game and mostly resolves through
  death time; this is a baseline limitation, not a Ranger-only signal.

Match stats after rebalance:

- Random-vs-random:
  - winners: `TEAM_A 162`, `TEAM_B 132`, `DRAW 6`;
  - winner types: `normal 51`, `death_time 243`, `draw_no_core_damage 6`;
  - mean tick: `282.1`.
- Scripted-vs-scripted:
  - winners: `TEAM_A 120`, `TEAM_B 117`, `DRAW 63`;
  - winner types: `death_time 237`, `draw_no_core_damage 63`;
  - mean tick: `300`.

## Applied ruleset

- Core: `80hp`, `6 atk`, range `12.0`.
- Skirmisher: `6hp`, `3 atk`, range `1.5`, speed `1.5`.
- Combat deployments are compact: player units spawn inside a bounded side
  deployment band, and replay rows expose separate `sim_pos` and visual-only
  `render_pos`.
- Neutral Lord: `120hp`, `16 atk`, attack rate `3`, range `11.0`.
- Neutral guards: `2hp` each.
- Neutral guard total attack: `2 * guard_strength`.
- Neutral income: `ceil(guard_strength / 2)`, so strength `1..7` pays
  `1,1,2,2,3,3,4`.
- Death-time core damage is integer-scheduled over ticks `270..299`: the total
  scheduled damage is always `CORE_HP - 1`, distributed with per-tick variance at
  most `1`, so a full-health core reaches exactly `1hp` before the final tick-300
  lethal hit.

## Training note

This is a new ruleset. Older checkpoints are useful only as historical proof of
learning/exploitation and should not be treated as behavior-compatible warm
starts for the next large run unless explicitly retrained on this ruleset.

The next training run should watch:

- normal win-rate separately from death-time win-rate;
- Ranger-heavy policies, but without assuming Ranger is the root problem;
- neutral-point farming versus actual core pressure;
- Lord contest timing and whether policies learn to deny or escort Lord.
