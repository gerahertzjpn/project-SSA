"""Can the RIGHT order let a genuinely WEAK trio beat a genuinely STRONG one?
Phase 1: rank candidates per faction by focal winrate vs a fixed cast -> pick the strongest +
weakest per faction. Phase 2: strong trio (A) vs weak trio (B), faction-mirrored, PAIRED seeds,
baseline + each order on the WEAK team."""
import json, sys, statistics
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.core.runner import run_match
from src.training.spaces import strategy_order_for_slot, STRATEGY_PRESETS

m = json.load(open('population/frozen_population_v1.json'))
idx = {e['net_id']: ('policy:' + e['checkpoint'], e['faction'])
       for g in ('descendants', 'ancestors') for e in m[g]}
reg = {e['net_id']: e for e in json.load(open('training_runs/league_population_v1/registry.json'))['entries']}

fixed_team = ['humans_2', 'infected_2']
fixed_opp = ['humans_1', 'infected_1', 'robots_1']
byfac = {'humans': [], 'infected': [], 'robots': []}
for e in m['ancestors']:
    if e['net_id'] not in fixed_team + fixed_opp:
        byfac[e['faction']].append(e['net_id'])               # ancestors (likely strong)
g2 = {'humans': [], 'infected': [], 'robots': []}
for nid, e in reg.items():
    if e['generation'] == 2:
        g2[e['faction']].append(nid)                          # fresh respawns (maybe weak)
cands = {f: byfac[f][:3] + g2[f][:2] for f in byfac}          # 5 candidates/faction

def focal_winrate(net, N):
    wins = 0
    for s in range(N):
        models, factions = {}, {}
        for slot, n in zip(('A1', 'A2', 'A3'), [net] + fixed_team): models[slot], factions[slot] = idx[n]
        for slot, n in zip(('B1', 'B2', 'B3'), fixed_opp): models[slot], factions[slot] = idx[n]
        r = run_match(seed=30000 + s, bot_type='league', mode='training', replay_path=None,
                      player_models=models, player_factions=factions)
        wins += 1 if r.winner == 'TEAM_A' else 0
    return wins / N

print('Phase 1: rank candidates by focal winrate vs fixed cast (N=10)...')
rank = {}
for f in cands:
    for net in cands[f]:
        rank[net] = focal_winrate(net, 10)
        print(f'  {net:<28} {f:<9} winrate {rank[net]:.2f}  ({reg.get(net, {}).get("steps_trained", "anc"):>9} steps)')
strong = {f: max(cands[f], key=lambda n: rank[n]) for f in cands}
weak = {f: min(cands[f], key=lambda n: rank[n]) for f in cands}
S = [strong['humans'], strong['infected'], strong['robots']]
W = [weak['humans'], weak['infected'], weak['robots']]
print(f'\nSTRONG trio (A): {S}')
print(f'WEAK   trio (B): {W}')

def play(order_id, seed):
    models, factions = {}, {}
    for slot, n in zip(('A1', 'A2', 'A3'), S): models[slot], factions[slot] = idx[n]
    for slot, n in zip(('B1', 'B2', 'B3'), W): models[slot], factions[slot] = idx[n]
    orders = None
    if order_id:
        orders = {}
        for i, slot in enumerate(('B1', 'B2', 'B3')):
            o = strategy_order_for_slot(order_id, i)
            orders[slot] = {'tempo': o['tempo'], 'win_path': o['win_path'], 'role': o['role']}
    r = run_match(seed=seed, bot_type='league', mode='training', replay_path=None,
                  player_models=models, player_factions=factions, orders=orders)
    return 1.0 if r.winner == 'TEAM_B' else 0.0   # did the WEAK team win?

N = 40
seeds = list(range(40000, 40000 + N))
base = statistics.mean(play(None, s) for s in seeds)
print(f'\nPhase 2 (paired N={N}): WEAK winrate vs STRONG')
print(f'  no order: {base:.2f}  (lower = bigger real skill gap)')
best = (base, '—')
for p in STRATEGY_PRESETS:
    wr = statistics.mean(play(p['id'], s) for s in seeds)
    if wr > best[0]: best = (wr, p['label'])
    print(f"  «{p['label']:<14}» {wr:.2f}  (Δ{wr-base:+.2f})")
print(f'\nbest order for weak: «{best[1]}» -> {best[0]:.2f} from {base:.2f}; '
      f'{"FLIPS to a win" if best[0] > 0.5 else "still loses"}.')
