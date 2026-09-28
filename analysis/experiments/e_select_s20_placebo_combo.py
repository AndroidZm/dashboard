"""e_select s20: 组合候选的分层安慰剂——同月随机剔除同样数量信号 + 相同的 k / cap，
与真正按低波动剔除的结果比较（20 种子，每类 100 个随机掩码）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from dataclasses import replace
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
W = lambda k: (0.02 * k, 0.08 * k, 0.06 * k)
seeds = range(20)
months = f.month.values
rng = np.random.default_rng(99)
cands = {'C2 vol38_k1.25': (0.38, Config(weights=W(1.25))), 'C3 vol40_k1.5_cap1.0': (0.40, Config(weights=W(1.5), cap=1.0))}
for name, (thr, cfg) in cands.items():
    rule = V[1] >= thr
    obs = {w: ev(replace(cfg, mask=rule), w, seeds)['cagr'] * 100 for w in ('A', 'B')}
    ctl = {w: ev(cfg, w, seeds)['cagr'] * 100 for w in ('A', 'B')}
    res = []
    for i in range(100):
        mk = np.ones(len(f), bool)
        for m in np.unique(months):
            ii = np.nonzero(months == m)[0]
            k = int((~rule[ii]).sum())
            if k:
                mk[rng.choice(ii, k, replace=False)] = False
        res.append({w: ev(replace(cfg, mask=mk), w, seeds)['cagr'] * 100 for w in ('A', 'B')})
    df = pd.DataFrame(res)
    print(name, 'rule', {k: round(v, 2) for k, v in obs.items()}, ' same k/cap no filter', {k: round(v, 2) for k, v in ctl.items()})
    for w in ('A', 'B'):
        print('   ', w, 'placebo 5/25/50/75/95:', np.round(np.quantile(df[w], [0.05, 0.25, 0.5, 0.75, 0.95]), 2), ' P(placebo>=rule)=', round((df[w] >= obs[w]).mean(), 3))
    print('    joint P =', round(((df.A >= obs['A']) & (df.B >= obs['B'])).mean(), 3), flush=True)
