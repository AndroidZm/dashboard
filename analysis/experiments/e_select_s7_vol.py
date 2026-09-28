"""e_select s7: “剔除低波动信号”的阈值平台 + 不同波动率窗口（20/60/120/250 日）+ 降权替代剔除。20 种子。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
e = D.entry.astype(int)
lr = np.diff(np.log(D.close), axis=1)
S = len(f)
vols = {}
for k in (20, 60, 120, 250):
    v = np.empty(S)
    for s in range(S):
        seg = lr[s, max(e[s] - k, 0):e[s]]
        tr = D.traded[s, max(e[s] - k, 0) + 1:e[s] + 1]
        seg = seg[tr] if tr.sum() > 5 else seg
        v[s] = seg.std() * np.sqrt(244)
    vols[k] = v
np.save('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy', np.vstack([vols[k] for k in (20, 60, 120, 250)]))
for k in vols:
    print(k, 'quantiles', np.round(np.quantile(vols[k], [0.1, 0.25, 0.5, 0.75, 0.9]), 3))
rows = []
for k in (20, 60, 120, 250):
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        thr = np.quantile(vols[k], q)
        mk = vols[k] >= thr
        r = dict(win=k, q=q, thr=round(thr, 3), nA=int(mk[f.win == 'A'].sum()), nB=int(mk[f.win == 'B'].sum()))
        for w, d in [('A', None), ('B', None), ('B', '2024-02')]:
            m = ev(Config(mask=mk), w, range(20), d)
            key = w + ('x' if d else '')
            b0 = ev(Config(), w, range(20), d)
            r[key + '_dC'] = (m['cagr'] - b0['cagr']) * 100
            r[key + '_dDD'] = (m['max_dd'] - b0['max_dd']) * 100
        rows.append(r)
        print(r, flush=True)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(2).to_string(index=False))
