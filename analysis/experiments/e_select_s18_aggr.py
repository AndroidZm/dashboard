"""e_select s18: 更激进（仍不加杠杆）：总仓位上限 90%/100%，低波动剔除阈值 x k，A/B 两窗口 20 种子；
另测只对平常/调整档剔除低波动（恐慌档全保留）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
T = D.tier
rows = []
for cap in (0.9, 1.0):
    for thr in (None, 0.38, 0.40):
        for k in (1.0, 1.25, 1.5, 2.0, 3.0):
            mk = None if thr is None else V[1] >= thr
            r = dict(cap=cap, thr=thr or 0, k=k)
            for w in ('A', 'B'):
                m = ev(Config(mask=mk, weights=(0.02 * k, 0.08 * k, 0.06 * k), cap=cap), w, range(20))
                r[w + '_cagr'] = m['cagr'] * 100; r[w + '_dd'] = m['max_dd'] * 100; r[w + '_final'] = m['final'] / 1e4
            rows.append(r)
for thr in (0.38, 0.40):
    mk = (V[1] >= thr) | (T == 2)
    r = dict(cap=0.9, thr=f'{thr}_keep_panic', k=1.0)
    for w in ('A', 'B'):
        m = ev(Config(mask=mk), w, range(20))
        r[w + '_cagr'] = m['cagr'] * 100; r[w + '_dd'] = m['max_dd'] * 100; r[w + '_final'] = m['final'] / 1e4
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(df.round(2).to_string(index=False))
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s18_aggr.csv', index=False)
