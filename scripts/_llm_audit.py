"""Scratch: exercise EVERY World LLM decision surface against the live client; report pass/fail."""
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.world import llm_manager, manager, training_auction
from src.world.world import get_player

w = json.load(open("data/world/world-ede07a7e/world.json", encoding="utf-8"))
llm = llm_manager.get_client()
team = next(t for t in w["teams"].values() if len(t.get("roster", [])) >= 3)
prof = manager.natural_profile(team["franchise_id"])
results = []


def check(name, fn):
    try:
        out = fn()
        results.append((name, "PASS", str(out)[:80]))
    except Exception as exc:
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


# 1. draft
dfn = llm_manager.draft_pick_fn(w, llm=llm)
avail = [w["players"][pid] for pid in list(w["players"])[:40]]
check("draft", lambda: dfn(dict(team, roster=team["roster"][:1]), avail))
# 2. playbook
check("playbook", lambda: llm_manager._llm_playbook(w, team, llm))
# 3. market plan
check("market", lambda: llm_manager._llm_market(w, team, llm, prof))
# 4. auction max-bid
check("auction_bid", lambda: llm_manager._llm_auction_bid(w, team, llm, 1_000_000, prof))
# 5. training spend (fabricate a wallet with enough steps to unlock every op incl. fork)
netted = [p for p in (get_player(w, pid) for pid in team["roster"]) if p.get("net_id")]
wallet = [{"cert_id": "c1", "steps": training_auction.FORK_MIN_STEPS}]
check("train_spend", lambda: llm_manager._llm_spend(w, team, netted, wallet, llm))
# 6. transfer bid (free agents, or a couple of unrostered players as stand-ins)
fa_ids = list(w.get("free_agents") or [])
fas = [get_player(w, pid) for pid in fa_ids][:20]
if not fas:
    rostered = {pid for t in w["teams"].values() for pid in t.get("roster", [])}
    fas = [w["players"][pid] for pid in w["players"] if pid not in rostered][:20]
check("transfer_bid", lambda: llm_manager._llm_transfer_bid(w, dict(team, treasury=500), fas, llm))
# 7. free-text commentary (llm.say)
check("commentary(say)", lambda: llm.say("Ты комментатор.", "Одна фраза про матч.", max_tokens=120))

print("\n==== LLM SURFACE AUDIT ====")
for name, status, info in results:
    print(f"  {status:4}  {name:18} {info}")
npass = sum(1 for _, s, _ in results if s == "PASS")
print(f"\n{npass}/{len(results)} surfaces OK")
print("total LLM output tokens used:", llm.usage.get("output_tokens"))
