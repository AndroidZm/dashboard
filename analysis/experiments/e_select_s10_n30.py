"""e_select s10: 按信号密度 n30 连续加权 w = clip(w_ref*(n30/10)^g, 0.01, wmax)，A/B 两窗口 20 种子。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
n30 = D.n30.astype(float)
rows = []
for g in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
    for wref in (0.04, 0.06, 0.08, 0.10):
        for wmax in (0.10, 0.15):
            w = np.clip(wref * (n30 / 10.0) ** g, 0.01, wmax)
            fn = (lambda arr: (lambda s, t, st: arr[s]))(w)
            r = dict(g=g, wref=wref, wmax=wmax)
            for win in ('A', 'B'):
                m = ev(Config(weight_fn=fn), win, range(20))
                r[win + '_cagr'] = m['cagr'] * 100
                r[win + '_dd'] = m['max_dd'] * 100
            rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(df.round(2).to_string(index=False))
print('spearman A vs B:', round(df.A_cagr.corr(df.B_cagr, method='spearman'), 3))
print(df.groupby('g')[['A_cagr', 'B_cagr', 'A_dd', 'B_dd']].mean().round(2))
