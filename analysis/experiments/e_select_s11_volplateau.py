"""e_select s11: 低波动剔除的阈值平台（绝对阈值 + 相对指数波动的阈值），剔除 vs 降权，A/B/Bx，20 种子。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')  # rows: 20,60,120,250（只用有成交日）
e = D.entry.astype(int)
ix = np.log(D.index['sh000852'].values)
ilr = np.diff(ix)
ivol = {k: np.array([ilr[max(t - k, 0):t].std() * np.sqrt(244) for t in e]) for k in (60, 120)}
feats = {'vol60': V[1], 'vol120': V[2], 'rv60': V[1] / ivol[60], 'rv120': V[2] / ivol[120]}
for k, v in feats.items():
    print(k, np.round(np.quantile(v, [0.05, 0.1, 0.2, 0.3, 0.5]), 3))
wins = [('A', None), ('B', None), ('B', '2024-02')]
b0 = {w + (d or ''): ev(Config(), w, range(20), d) for w, d in wins}
winA = f.win.values == 'A'
def row(name, thr, cfg, mk):
    r = dict(feat=name, thr=thr, remA=int((~mk & winA).sum()), remB=int((~mk & ~winA).sum()))
    for w, d in wins:
        m = ev(cfg, w, range(20), d)
        k = w + ('x' if d else '')
        bb = b0[w + (d or '')]
        r[k + '_dC'] = (m['cagr'] - bb['cagr']) * 100
        r[k + '_dDD'] = (m['max_dd'] - bb['max_dd']) * 100
    return r
rows = []
grids = {'vol60': [0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.43],
         'vol120': [0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.43],
         'rv60': [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
         'rv120': [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}
for name, thrs in grids.items():
    for thr in thrs:
        mk = feats[name] >= thr
        rows.append(row(name, thr, Config(mask=mk), mk))
# 降权而非剔除：低波动信号权重乘 0.5
for name in ('vol60', 'vol120'):
    for thr in (0.30, 0.34, 0.38):
        lo = feats[name] < thr
        base_w = np.array([0.02, 0.08, 0.06])[D.tier]
        w = np.where(lo, 0.5 * base_w, base_w)
        fn = (lambda arr: (lambda s, t, st: arr[s]))(w)
        rows.append(row(name + '_half', thr, Config(weight_fn=fn), ~lo))
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s11_volplateau.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(df.round(2).to_string(index=False))
