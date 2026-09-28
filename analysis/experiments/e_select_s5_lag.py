"""e_select s5: 入场时点——信号后第 k 个交易日收盘买入（k=0..10）。逐笔（A/B/Bx）与组合（A/B，20 种子）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis')
import numpy as np, pandas as pd
from engine import Data, Config, run_seeds
D = Data.load()
sig = D.sig
mon = sig.signal_date.str[:7].values
winA = sig.signal_date.values < '2016-01-01'
for lag in [0, 1, 2, 3, 5, 10, 20]:
    e, x, r = D.exits(0.4, 0.4, None, None, lag)
    ret = r - 1
    tp = r >= 1.4
    held = x - e
    parts = []
    for nm, m in [('A', winA), ('B', ~winA), ('Bx', (~winA) & (mon != '2024-02'))]:
        parts.append(f"{nm}: ret {ret[m].mean():.3f} tp {tp[m].mean():.3f} held {np.median(held[m]):.0f} eff {ret[m].sum()/(held[m].sum()/244):.3f}")
    print('lag', lag, ' | '.join(parts), flush=True)
print('--- portfolio (20 seeds)')
for lag in [0, 1, 2, 3, 5, 10]:
    out = []
    for wn, (a, b) in [('A', ('2006-01-04', '2016-01-28')), ('B', ('2016-01-01', '2026-08-21'))]:
        m = run_seeds(D, Config(start=a, end=b, lag=lag), seeds=range(20))
        out.append(f"{wn}: {m['final']/1e4:6.1f}万 {m['cagr']*100:5.2f}% DD {m['max_dd']*100:5.1f}%")
    print('lag', lag, ' | '.join(out), flush=True)
