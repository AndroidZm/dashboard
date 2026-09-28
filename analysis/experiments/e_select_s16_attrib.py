"""e_select s16: 归因——低波动剔除(+k) 与基线在各年度的收益差、各档成交笔数（20 种子中位）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from dataclasses import replace
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
W = lambda k: (0.02 * k, 0.08 * k, 0.06 * k)
cfgs = {'base': Config(), 'vol40': Config(mask=V[1] >= 0.40), 'vol40_k1.5': Config(mask=V[1] >= 0.40, weights=W(1.5)), 'flat_k1.5': Config(weights=W(1.5))}
for win in ('A', 'B'):
    a, b = WIN[win]
    yr = {}
    tiers = {}
    for n, c in cfgs.items():
        ys, ts = [], []
        for s in range(20):
            r = run(D, replace(c, start=a, end=b, record=True), seed=s)
            eq = r['equity']
            eq.index = pd.to_datetime(eq.index)
            ys.append(eq.resample('YE').last().pct_change().fillna(eq.resample('YE').last().iloc[0] / 600000 - 1))
            ts.append(r['n_by_tier'])
        yr[n] = pd.concat(ys, axis=1).median(axis=1) * 100
        tiers[n] = np.median(np.array(ts), axis=0)
    df = pd.DataFrame(yr)
    df.index = df.index.year
    df['vol40-base'] = df['vol40'] - df['base']
    df['vol40k1.5-base'] = df['vol40_k1.5'] - df['base']
    print('=== window', win, 'yearly return % (median of 20 seeds)')
    print(df.round(1).to_string())
    print('trades by tier (median):', {k: v.tolist() for k, v in tiers.items()})
