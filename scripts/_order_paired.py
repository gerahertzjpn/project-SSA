"""PAIRED order test: same seed with vs without the order -> the ONLY difference is the order.
Reports the order's effect on B's winrate AND on behaviour (core damage dealt, own core_hp),
plus how many games the order actually changed."""
import json, sys, statistics
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.core.runner import run_match
from src.training.spaces import strategy_order_for_slot, STRATEGY_PRESETS

m = json.load(open('population/frozen_population_v1.json'))
idx = {e['net_id']: ('policy:' + e['checkpoint'], e['faction'])
       for g in ('descendants', 'ancestors') for e in m[g]}
reg = json.load(open('training_runs/league_population_v1/registry.json'))['entries']
g2 = {}
for e in reg:
    if e['generation'] == 2 and e['faction'] not in g2:
        g2[e['faction']] = e['net_id']
A = ['humans_1', 'infected_1', 'robots_1']
B = [g2['humans'], g2['infected'], g2['robots']]

def play(order_id, seed):
    models, factions = {}, {}
    for s, n in zip(('A1', 'A2', 'A3'), A): models[s], factions[s] = idx[n]
    for s, n in zip(('B1', 'B2', 'B3'), B): models[s], factions[s] = idx[n]
    orders = None
    if order_id:
        orders = {}
        for i, s in enumerate(('B1', 'B2', 'B3')):
            o = strategy_order_for_slot(order_id, i)
            orders[s] = {'tempo': o['tempo'], 'win_path': o['win_path'], 'role': o['role']}
    r = run_match(seed=seed, bot_type='league', mode='training', replay_path=None,
                  player_models=models, player_factions=factions, orders=orders)
    st = r.state
    return (1.0 if r.winner == 'TEAM_B' else 0.0,
            float(st.core_damage_by_team.get('TEAM_B', 0.0)),          # B's dmg to enemy cores
            max(0.0, float(getattr(st.players.get('B1'), 'core_hp', 0.0) or 0.0)))  # B1 own core_hp

N = 50
seeds = list(range(20000, 20000 + N))
print(f'STRONG(A)={A}\nWEAK(B)={B}\nPAIRED N={N}, same seeds with vs without order on B.')
base = {s: play(None, s) for s in seeds}
b_wr = statistics.mean(v[0] for v in base.values())
b_dmg = statistics.mean(v[1] for v in base.values())
b_hp = statistics.mean(v[2] for v in base.values())
print(f'\nNO ORDER: B winrate {b_wr:.2f} | B core-dmg dealt {b_dmg:.0f} | B own core_hp {b_hp:.0f}')
print('order on B (paired vs the SAME no-order games):')
for p in STRATEGY_PRESETS:
    res = {s: play(p['id'], s) for s in seeds}
    wr = statistics.mean(v[0] for v in res.values())
    dmg = statistics.mean(v[1] for v in res.values())
    hp = statistics.mean(v[2] for v in res.values())
    changed = sum(1 for s in seeds if res[s][0] != base[s][0])
    print(f"  «{p['label']:<14}» wr {wr:.2f} (Δ{wr-b_wr:+.2f}) | dmg {dmg:.0f} (Δ{dmg-b_dmg:+.0f}) | "
          f"core_hp {hp:.0f} (Δ{hp-b_hp:+.0f}) | outcome changed {changed}/{N}")
