"""e_select s21: 被否定规则的 80 种子复核（A/B 相对基线的 CAGR 差）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
T = D.tier
rules = {
    'no_board60': Config(mask=f.board.values != '60'),
    'no_price>20': Config(mask=f.praw.values <= 20),
    'no_bars<500': Config(mask=f.bars.values >= 500),
    'no_r250<-0.5': Config(mask=f.r250.values >= -0.5),
    'skip_idxdd>-0.20': Config(mask=f.idx_dd250.values <= -0.20),
    'skip_t0_idxdd>-0.20': Config(mask=(T != 0) | (f.idx_dd250.values <= -0.20)),
    'lag2': Config(lag=2),
    'lag3': Config(lag=3),
    'tierw_pickA(6/12/3,pmNone)': Config(weights=(0.06, 0.12, 0.03), panic_max_pos=None),
    'tierw_pickB(0/4/3,pmNone)': Config(weights=(0.0, 0.04, 0.03), panic_max_pos=None),
}
b = {w: ev(Config(), w, range(80)) for w in ('A', 'B')}
for n, c in rules.items():
    s = []
    for w in ('A', 'B'):
        m = ev(c, w, range(80))
        s.append(f"{w}: {m['cagr']*100:5.2f}% ({(m['cagr']-b[w]['cagr'])*100:+.2f}) DD {m['max_dd']*100:5.1f}")
    print(f'{n:28s}', ' | '.join(s), flush=True)
