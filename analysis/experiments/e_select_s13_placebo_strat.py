"""e_select s13: 分层安慰剂——对 vol60>=0.38 规则，按 (窗口, 档位, 月份) 分层随机剔除同样数量的信号，
看组合 CAGR 变化分布；同时给“事后看持有期最长的信号”（不可实现，只检验机制）作参照。20 种子。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
rule = V[1] >= 0.38
seeds = range(20)
b = {w: ev(Config(), w, seeds) for w in ('A', 'B')}
def dc(mk):
    return {w: (ev(Config(mask=mk), w, seeds)['cagr'] - b[w]['cagr']) * 100 for w in ('A', 'B')}
obs = dc(rule)
print('rule vol60>=0.38', {k: round(v, 2) for k, v in obs.items()})
# 机制参照：按事后持有期剔除（不可实现）
for H in (250, 350, 450):
    print('oracle remove held>', H, {k: round(v, 2) for k, v in dc(f.held.values <= H).items()}, 'removed', int((f.held.values > H).sum()))
print('spearman(vol60, held): A', round(pd.Series(V[1][f.win == 'A']).corr(pd.Series(f.held.values[f.win == 'A']), method='spearman'), 3),
      ' B', round(pd.Series(V[1][f.win == 'B']).corr(pd.Series(f.held.values[f.win == 'B']), method='spearman'), 3))
rng = np.random.default_rng(7)
strata = f.month.values  # 月份内分层（也就隐含了窗口和大部分档位）
rows = []
for i in range(150):
    mk = np.ones(len(f), bool)
    for m in np.unique(strata):
        ii = np.nonzero(strata == m)[0]
        k = int((~rule[ii]).sum())
        if k:
            mk[rng.choice(ii, k, replace=False)] = False
    rows.append(dc(mk))
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s13_placebo_strat.csv', index=False)
for w in ('A', 'B'):
    print(w, 'placebo quantiles 5/25/50/75/95:', np.round(np.quantile(df[w], [0.05, 0.25, 0.5, 0.75, 0.95]), 2),
          ' P(placebo >= rule) =', round((df[w] >= obs[w]).mean(), 3))
print('P(placebo A>=obsA and B>=obsB) =', round(((df.A >= obs['A']) & (df.B >= obs['B'])).mean(), 3))
