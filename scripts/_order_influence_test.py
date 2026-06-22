"""Can the RIGHT order let a WEAK team beat a STRONG one? Strong = ancestors (~85M steps),
weak = fresh gen-2 respawns (~1M). Faction-MIRRORED (each side has one humans/infected/robots),
so the only differences are skill + the order we put on the WEAK team."""
import json, statistics, sys
sys.path.insert(0, '.')
from src.core.runner import run_match
from src.training.spaces import strategy_order_for_slot, STRATEGY_PRESETS

m = json.load(open('population/frozen_population_v1.json'))
idx = {e['net_id']: ('policy:' + e['checkpoint'], e['faction'])
       for g in ('descendants', 'ancestors') for e in m[g]}
reg = json.load(open('training_runs/league_population_v1/registry.json'))['entries']

# weak = one gen-2 respawn per faction
g2 = {}
for e in reg:
    if e['generation'] == 2 and e['faction'] not in g2:
        g2[e['faction']] = e['net_id']
strong = ['humans_1', 'infected_1', 'robots_1']                       # ancestors
weak = [g2['humans'], g2['infected'], g2['robots']]                   # g2 respawns
print('STRONG (A):', strong)
print('WEAK   (B):', weak)

def play(order_id, seed):
    models, factions = {}, {}
    for s, n in zip(('A1', 'A2', 'A3'), strong): models[s], factions[s] = idx[n]
    for s, n in zip(('B1', 'B2', 'B3'), weak):   models[s], factions[s] = idx[n]
    orders = None
    if order_id:
        orders = {}
        for i, s in enumerate(('B1', 'B2', 'B3')):
            o = strategy_order_for_slot(order_id, i)
            orders[s] = {'tempo': o['tempo'], 'win_path': o['win_path'], 'role': o['role']}
    r = run_match(seed=seed, bot_type='league', mode='training', replay_path=None,
                  player_models=models, player_factions=factions, orders=orders)
    return 1.0 if r.winner == 'TEAM_B' else 0.0   # did the WEAK team win?

N = 20
base = statistics.mean(play(None, 5000 + g) for g in range(N))
print(f'\nBASELINE — weak winrate vs strong, NO order: {base:.2f}  (>.5 means weak already wins)')
print('weak team + each order (strong plays free):')
rows = []
for p in STRATEGY_PRESETS:
    wr = statistics.mean(play(p['id'], 6000 + g) for g in range(N))
    rows.append((wr, p['label']))
    print(f"  «{p['label']:<14}» winrate {wr:.2f}   swing {wr-base:+.2f}")
best = max(rows)
print(f'\nBEST order for weak: «{best[1]}» -> {best[0]:.2f} (from {base:.2f}); '
      f'{"FLIPS to a win" if best[0] > 0.5 else "still loses"}.')

# --- confirmation: bigger N, order on EITHER team, to prove it's the ORDER not the team ---
def play2(order_id, on_team, seed):
    models, factions = {}, {}
    for s, n in zip(('A1','A2','A3'), strong): models[s], factions[s] = idx[n]
    for s, n in zip(('B1','B2','B3'), weak):   models[s], factions[s] = idx[n]
    slots = ('A1','A2','A3') if on_team=='A' else ('B1','B2','B3')
    orders=None
    if order_id:
        orders={}
        for i,s in enumerate(slots):
            o=strategy_order_for_slot(order_id,i); orders[s]={'tempo':o['tempo'],'win_path':o['win_path'],'role':o['role']}
    r=run_match(seed=seed,bot_type='league',mode='training',replay_path=None,player_models=models,player_factions=factions,orders=orders)
    return r.winner==('TEAM_A' if on_team=='A' else 'TEAM_B')  # did the ORDERED team win?
print('\n=== CONFIRMATION (N=30): winrate of the ORDERED team, order on A(strong) and B(weak) ===')
import statistics as st
for p in STRATEGY_PRESETS:
    a=st.mean(play2(p['id'],'A',7000+g) for g in range(30))
    b=st.mean(play2(p['id'],'B',8000+g) for g in range(30))
    print(f"  «{p['label']:<14}» ordered-A wins {a:.2f}  | ordered-B wins {b:.2f}  (free=0.50)")
