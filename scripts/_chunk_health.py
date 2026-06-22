"""Throwaway: summarize population_v1 training health from metrics.jsonl."""
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = sys.argv[1] if len(sys.argv) > 1 else "training_runs/population_v1/metrics.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("kind") == "population_update"]
n = len(rows)
first, last = rows[0], rows[-1]
secs = [r["elapsed_sec"] for r in rows]
steps = sum(r["steps_this_update"] for r in rows)
sps = steps / sum(secs)
wr = [r["controlled_win_rate"] for r in rows]
mes = [r["mean_episode_steps"] for r in rows]

keys = set()
nan = False
kls = []
ents = []
losses = []
for r in rows:
    lbn = r.get("loss_by_net") or {}
    for nid, d in lbn.items():
        if isinstance(d, dict):
            keys |= set(d.keys())
            for k, v in d.items():
                if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                    nan = True
                if k in ("approx_kl", "kl"):
                    kls.append(v)
                if "entropy" in k:
                    ents.append(v)
                if k in ("loss", "total_loss", "policy_loss"):
                    losses.append(v)

print(f"updates={n}  total_steps={last['total_steps']}  avg_sec/upd={sum(secs)/n:.2f}  throughput={sps:.0f} st/s")
print(f"win_rate first/last={wr[0]:.2f}/{wr[-1]:.2f}  range=[{min(wr):.2f},{max(wr):.2f}]")
print(f"mean_episode_steps first/last={mes[0]:.1f}/{mes[-1]:.1f}  (down = learning to finish)")
print(f"min_net_appearances last={last.get('min_net_appearances')}  min_distinct_allies={last.get('min_distinct_allies_seen')}")
print(f"NaN/Inf in loss_by_net: {nan}")
print(f"loss_by_net keys: {sorted(keys)}")
if kls:
    print(f"approx_kl range=[{min(kls):.4f},{max(kls):.4f}] mean={sum(kls)/len(kls):.4f}")
if ents:
    print(f"entropy range=[{min(ents):.3f},{max(ents):.3f}] first/last={ents[0]:.3f}/{ents[-1]:.3f}")
if losses:
    print(f"loss range=[{min(losses):.4f},{max(losses):.4f}]")

# per-net entropy over the last WINDOW updates (diversity / collapse watch)
WINDOW = 40
recent = rows[-WINDOW:]
per_net_ent: dict = {}
for r in recent:
    for nid, d in (r.get("loss_by_net") or {}).items():
        if isinstance(d, dict) and "entropy" in d:
            per_net_ent.setdefault(nid, []).append(d["entropy"])
avg_ent = {nid: sum(v) / len(v) for nid, v in per_net_ent.items() if v}
if avg_ent:
    vals = sorted(avg_ent.values())
    lo = sorted(avg_ent.items(), key=lambda kv: kv[1])[:4]
    hi = sorted(avg_ent.items(), key=lambda kv: -kv[1])[:3]
    spread = max(vals) - min(vals)
    print(f"per-net entropy (last {WINDOW} upd, {len(avg_ent)} nets): "
          f"min={min(vals):.3f} median={vals[len(vals)//2]:.3f} max={max(vals):.3f} spread={spread:.3f}")
    print("  lowest: " + ", ".join(f"{n}={e:.3f}" for n, e in lo))
    print("  highest: " + ", ".join(f"{n}={e:.3f}" for n, e in hi))
    collapsed = [n for n, e in avg_ent.items() if e < 0.15]
    if collapsed:
        print(f"  ⚠ near-collapse (<0.15): {collapsed}")

# sec/update trend in 100-update buckets (diagnose slowdown)
print("sec/upd by 100-update bucket (update_lo: sec/upd, steps/upd, st/s):")
for lo in range(0, n, 100):
    b = rows[lo:lo + 100]
    sc = sum(r["elapsed_sec"] for r in b) / len(b)
    st = sum(r["steps_this_update"] for r in b) / len(b)
    print(f"  upd {b[0]['update']:>4}-{b[-1]['update']:<4}: {sc:6.2f}s  {st:6.0f} steps  {st/sc:6.0f} st/s")
