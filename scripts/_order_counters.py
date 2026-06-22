"""Do COUNTERS matter? Equal teams, A commits to a style, B tries every order (paired seeds).
If orders are a real rock-paper-scissors lever, B's BEST response should CHANGE with A's order
and the spread (best - worst B response) should be sizeable. If everything sits ~0.50, counters
are pointless."""
import json, sys, statistics
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.core.runner import run_match
from src.training.spaces import strategy_order_for_slot, STRATEGY_PRESET_BY_ID

m = json.load(open('population/frozen_population_v1.json'))
idx = {e['net_id']: ('policy:' + e['checkpoint'], e['faction'])
       for g in ('descendants', 'ancestors') for e in m[g]}
reg = json.load(open('training_runs/league_population_v1/registry.json'))['entries']
g2 = {}
for e in reg:
    if e['generation'] == 2 and e['faction'] not in g2:
        g2[e['faction']] = e['net_id']
A = ['humans_1', 'infected_1', 'robots_1']                       # equal matchup (baseline 0.50)
B = [g2['humans'], g2['infected'], g2['robots']]

def order_for(slots, order_id):
    if not order_id:
        return {}
    out = {}
    for i, s in enumerate(slots):
        d = strategy_order_for_slot(order_id, i)
        out[s] = {'tempo': d['tempo'], 'win_path': d['win_path'], 'role': d['role']}
    return out

def play(a_order, b_order, seed):
    models, factions = {}, {}
    for s, n in zip(('A1', 'A2', 'A3'), A): models[s], factions[s] = idx[n]
    for s, n in zip(('B1', 'B2', 'B3'), B): models[s], factions[s] = idx[n]
    orders = {**order_for(('A1', 'A2', 'A3'), a_order), **order_for(('B1', 'B2', 'B3'), b_order)}
    r = run_match(seed=seed, bot_type='league', mode='training', replay_path=None,
                  player_models=models, player_factions=factions, orders=orders or None)
    return 1.0 if r.winner == 'TEAM_B' else 0.0

N = 25
seeds = list(range(50000, 50000 + N))
A_commit = ['all_in', 'turtle', 'macro']                        # A commits to a clear style
B_opts = [None, 'standard', 'core_rush', 'all_in', 'turtle', 'macro', 'baron']
lab = {None: 'нет', 'standard': 'Стандарт', 'core_rush': 'Раш', 'all_in': 'Олл-ин',
       'turtle': 'Черепаха', 'macro': 'Макро', 'baron': 'Лорд'}
print(f'Counter test (equal teams, paired N={N}). B winrate vs a COMMITTED A-order.')
print('Looking for: does B\'s best response CHANGE with A, and is the spread big?')
best_responses = {}
for a in A_commit:
    res = {b: statistics.mean(play(a, b, s) for s in seeds) for b in B_opts}
    best = max(res, key=res.get); worst = min(res, key=res.get)
    best_responses[a] = lab[best]
    print(f"\nA commits «{STRATEGY_PRESET_BY_ID[a]['label']}»:")
    for b in B_opts:
        tag = '  <- BEST' if b == best else ('  <- worst' if b == worst else '')
        print(f"   B={lab[b]:<9} {res[b]:.2f}{tag}")
    print(f"   spread {res[best]-res[worst]:.2f}")
print(f"\nB's best response per A-commit: {best_responses}")
print("If best response differs across A AND spread>~0.20 -> counters matter; else orders are weak.")
