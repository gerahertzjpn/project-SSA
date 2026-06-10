# Population-18 Self-Play Runbook

Audience: the next agent who **launches and monitors** training. The system is
already prepared, smoke-tested, and ready to run. Do not re-architect it; launch
it, watch it, and intervene only per the rules below.

## What this trains

A free-agent **draft pool of 18 independent neural nets** — "1 seed = 1 player":

- 6 nets per faction: `humans_1..6`, `infected_1..6`, `robots_1..6` (18 total).
- Each net is faction-locked and **uniquely** defined (unique seed + unique
  training recipe → 18 distinct genotype hashes). No two nets are clones.
- **No selection / no culling.** All 18 nets are trained and checkpointed every
  round. Nothing is pruned.

Training mechanic — **frozen-snapshot self-play** with **mixed-faction teams**
and **randomized positions**:

- Each episode samples 3 live nets (any factions) as the controlled team and 3
  nets from a periodically-refreshed frozen snapshot pool as the opponent team.
- Position is shuffled every episode (a net plays A1/A2/A3 across episodes), so
  nets learn to play **any position with any ally**.
- Each player's faction follows its net into whatever slot it occupies, so a
  faction-locked net always plays its own faction.
- The controlled nets learn via masked PPO; frozen opponents do not learn.
- The frozen pool is refreshed from the live nets every `--snapshot-interval`
  updates — this is the self-play loop.

Interface (must not change): `OBSERVATION_VERSION = train_obs_v3.1`,
`OBSERVATION_DIM = 5188`, `ACTION_DIM` per `src/training/spaces.py`. All
checkpoints embed and validate this.

## How to launch

Validate first (a few CPU-minutes, end-to-end):

```powershell
python scripts/train_population.py --smoke
```

Inspect the population + registry without training:

```powershell
python scripts/train_population.py --init-only --run-dir training_runs/population_18_v1
```

Full run (sized for your machine — has a CUDA GPU; PPO updates run on GPU,
rollout workers on CPU):

```powershell
python scripts/train_population.py `
  --run-dir training_runs/population_18_v1 `
  --updates 400 `
  --episodes-per-net 3 `
  --workers 16 `
  --snapshot-interval 5 `
  --checkpoint-interval 20 `
  --torch-threads 1
```

Resume after a stop/crash — re-run the **exact same command**. Each net's
`latest.pt` loads automatically and training continues from the next update. (If
only some nets have checkpoints, resume aborts rather than corrupt the pool.)

### Key flags

| flag | meaning | guidance |
| --- | --- | --- |
| `--updates` | PPO updates over the whole pool | 300–800 for a foundation run |
| `--episodes-per-net` | episodes each net learns from per update | 2–4; higher = more data/update, slower |
| `--workers` | parallel rollout processes (spawn) | ~physical cores; rollouts are CPU-bound |
| `--snapshot-interval` | refresh frozen opponents every N updates | 5–10; smaller = faster-moving self-play |
| `--checkpoint-interval` | save `latest.pt` + milestone every N updates | 20 for long runs (see disk note) |
| `--opponent-mode` | `stochastic` (default) or `argmax` frozen opponents | keep `stochastic` |
| `--torch-threads` | torch threads per worker | 1 (workers already parallelize) |

The worker pool is rebuilt on each frozen-pool refresh to keep per-worker
opponent-weight caches bounded — expected, not a leak.

## Outputs (`--run-dir`)

```text
population.json                 # the 18 net specs + genotype hashes
config.json                     # resolved config + git state
registry.json                   # live model registry (all 18 nets, no culling)
MODEL_REGISTRY.md               # human-readable registry table
metrics.jsonl                   # one row per update (see monitoring)
frozen_pool/frozen_pool_v*.pt   # current frozen opponent pool (older deleted)
nets/<net_id>/latest.pt         # each net's latest weights (draft artifact)
nets/<net_id>/update_<n>.pt     # milestones
```

Each `nets/<net_id>/latest.pt` is a standalone, `PolicyBot`-loadable draft
candidate. The frozen pool holds all 18 in one file keyed by `net_id`
(`PolicyBot(..., policy_state_id="robots_3")`).

## Monitoring (read `metrics.jsonl`, do not spam the chat)

Per-update fields to track:

- `controlled_win_rate` / `loss_rate` / `draw_rate` — controlled (live) team vs
  frozen opponents. In healthy self-play this oscillates around ~0.5 as the
  frozen pool refreshes; a flat 0.0 or 1.0 for many updates is a red flag.
- `mean_episode_steps` — should be well under `max_ticks` once nets stop
  stalling into death-time/truncation.
- `min_net_appearances` — every net should appear every update (coverage).
- `min_distinct_allies_seen` — grows over the run; confirms ally diversity.
- `loss_by_net.<net_id>` — `policy_loss`, `value_loss`, `entropy`, `approx_kl`
  per net. Watch for `approx_kl` spikes (instability) or `entropy` collapsing to
  ~0 (premature determinism).

Aggregate a report every ~45–90 min or on a notable event, not per update.

## Health checks and safe interventions

Stop and diagnose (do not blindly restart) if you see:

- `NaN`/`Inf` in `loss_by_net` → stop; this is a bug, not a tuning issue.
- `approx_kl` repeatedly large (e.g. > 0.1) and growing → lower `--episodes-per-net`
  or learning is unstable; reduce per-recipe LR (a code change → new run, do not
  hot-edit a live run).
- entropy → ~0 across many nets early → too-low entropy; investigate before a
  long run.
- `controlled_win_rate` pinned at 1.0 for many updates → frozen pool too weak;
  reduce `--snapshot-interval` so opponents refresh sooner.
- a worker crash / hang → stop, read the traceback, fix root cause, resume.

Allowed autonomously (record the reason): adjusting `--snapshot-interval`,
`--episodes-per-net`, `--workers`, `--checkpoint-interval` and resuming;
stopping a clearly-broken run.

Not allowed without explicit approval: deleting net checkpoints, removing nets
from the pool (culling), changing the observation/action interface, or claiming
an improvement without an honest eval.

## After the run — diversity & composability checks (do not skip)

The foundation run is not "done" until you confirm the pool is usable and
diverse (CLAUDE.md §9). Suggested next steps (eval only, no culling):

1. Confirm `registry.json` shows 18 nets, 18 unique genotype hashes, 6 per
   faction, every net with a `latest.pt`.
2. Teammate-swap / role-stress evals across nets (reuse
   `scripts/evaluate_teammate_swaps.py` patterns; compose teams with
   `scripts/create_mixed_team_checkpoint.py` using `policy_state_id`).
3. Track per-net win-contribution and entropy spread to verify the population
   did not collapse into one style.

## Disk note

Milestones are `18 nets × (updates / checkpoint-interval) × ~2.8 MB`. For
`--updates 400 --checkpoint-interval 20` that is ~1 GB plus `latest.pt`. Raise
`--checkpoint-interval` to reduce milestone churn; `latest.pt` always keeps
resume safe. `training_runs/` is gitignored.
