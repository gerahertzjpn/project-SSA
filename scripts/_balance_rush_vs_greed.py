import sys; sys.path.insert(0,'.')
from src.core.runner import run_match
from src.core.constants import TEAM_A_PLAYERS, TEAM_B_PLAYERS
import statistics
FAC = {"A1":"humans","A2":"infected","A3":"robots","B1":"humans","B2":"infected","B3":"robots"}
def models(a,b):
    m={}
    for p in TEAM_A_PLAYERS: m[p]=a
    for p in TEAM_B_PLAYERS: m[p]=b
    return m
def h2h(a,b,n):
    aw=bw=dr=0; tk=[]
    for s in range(n):
        r=run_match(seed=1000+s, bot_type="scripted", mode="training",
                    player_models=models(a,b), player_factions=FAC)
        if r.winner=="TEAM_A": aw+=1
        elif r.winner=="TEAM_B": bw+=1
        else: dr+=1
        tk.append(r.total_ticks)
    return aw,bw,dr,tk
def symmetric(x,y,n):
    # run both orientations, return X win-rate vs Y (side bias cancelled)
    xw1,yw1,d1,t1=h2h(x,y,n)
    yw2,xw2,d2,t2=h2h(y,x,n)   # x now on B
    xwins=xw1+xw2; ywins=yw1+yw2; draws=d1+d2; tot=2*n
    return xwins,ywins,draws,tot,statistics.mean(t1+t2)
N=30
for label,(x,y) in [("RUSH vs GREED(BMP-tech)",("scripted_core","scripted_core_bmp")),
                    ("RUSH vs RAIDER-tech",("scripted_core","scripted_core_raider")),
                    ("RUSH vs RUSH (control)",("scripted_core","scripted_core"))]:
    xw,yw,dr,tot,mt=symmetric(x,y,N)
    print(f"{label:<26} {x} {xw/tot:.0%}  vs  {y} {yw/tot:.0%}  (draw {dr/tot:.0%}, n={tot}, mean_ticks {mt:.0f})")
