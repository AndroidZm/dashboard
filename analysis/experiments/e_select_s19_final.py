"""e_select s19: 最终候选 80 种子——阈值 x k 细平台（A/B/Bx），以及 cap=1.0 激进版的完整报告（LOCO、执行成本）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from dataclasses import replace
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
W = lambda k: (0.02 * k, 0.08 * k, 0.06 * k)
base = {('A', None): ev(Config(), 'A', range(80)), ('B', None): ev(Config(), 'B', range(80)), ('B', '2024-02'): ev(Config(), 'B', range(80), '2024-02')}
rows = []
for thr in (0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42):
    for k in (1.0, 1.25, 1.5):
        r = dict(thr=thr, k=k)
        for (w, d), m0 in base.items():
            m = ev(Config(mask=V[1] >= thr, weights=W(k)), w, range(80), d)
            key = w + ('x' if d else '')
            r[key + '_cagr'] = m['cagr'] * 100; r[key + '_dC'] = (m['cagr'] - m0['cagr']) * 100; r[key + '_dd'] = m['max_dd'] * 100
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(df.round(2).to_string(index=False), flush=True)
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s19_plateau80.csv', index=False)
for name, cfg in {'vol40_k1.5_cap1.0': Config(mask=V[1] >= 0.40, weights=W(1.5), cap=1.0),
                  'vol40_k1.25_cap1.0': Config(mask=V[1] >= 0.40, weights=W(1.25), cap=1.0),
                  'cap1.0_only': Config(cap=1.0)}.items():
    full_report(cfg, Config(), seeds=range(80), label=name)
    for w in ('A', 'B'):
        m1 = ev(replace(cfg, lag=1, slip=0.005), w, range(80)); m0 = ev(Config(lag=1, slip=0.005), w, range(80))
        print(f'   lag1+slip0.5% {w}: {line(m1)} vs base {line(m0)} dC {(m1["cagr"]-m0["cagr"])*100:+.2f}', flush=True)
