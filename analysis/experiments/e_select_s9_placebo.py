"""e_select s9: 安慰剂检验——随机剔除同样比例的信号，组合 CAGR 变化的分布。
用来判断“剔除某类信号后收益变高”是否只是剔除本身（腾出仓位/换了路径）造成的。10 种子/掩码。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
S = len(D.tier)
rng = np.random.default_rng(12345)
seeds = range(10)
b = {w: ev(Config(), w, seeds) for w in ('A', 'B')}
bx = ev(Config(), 'B', seeds, '2024-02')
rows = []
for p in (0.15, 0.2, 0.3, 0.4):
    for i in range(100):
        mk = rng.random(S) >= p
        r = dict(p=p, i=i)
        for w in ('A', 'B'):
            m = ev(Config(mask=mk), w, seeds)
            r[w + '_dC'] = (m['cagr'] - b[w]['cagr']) * 100
            r[w + '_dDD'] = (m['max_dd'] - b[w]['max_dd']) * 100
        m = ev(Config(mask=mk), 'B', seeds, '2024-02')
        r['Bx_dC'] = (m['cagr'] - bx['cagr']) * 100
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s9_placebo.csv', index=False)
for p, g in df.groupby('p'):
    q = lambda c: np.round(np.quantile(g[c], [0.05, 0.25, 0.5, 0.75, 0.95]), 2)
    print(f'p={p}: A_dC {q("A_dC")}  B_dC {q("B_dC")}  Bx_dC {q("Bx_dC")}  B_dDD {q("B_dDD")}')
    print('   corr(A_dC,B_dC)=', round(g.A_dC.corr(g.B_dC), 2), ' P(A>0 & B>0)=', round(((g.A_dC > 0) & (g.B_dC > 0)).mean(), 2))
