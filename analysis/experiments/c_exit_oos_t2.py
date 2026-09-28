"""两段式移动止盈家族的双向样本外选择（wide80 的 72 个 trail2 配置，80 种子）。"""
import pandas as pd, numpy as np
df = pd.read_csv('c_exit_out/wide80.csv')
P = df.pivot_table(index='name', columns='win', values=['final', 'cagr', 'max_dd'])
P.columns = [f'{v}_{w}' for v, w in P.columns]
b = P.loc['base']
T2 = P[P.index.str.startswith('t2_')]
for slf, lab in [(lambda n: True, '全部 sl'), (lambda n: n.endswith('sl0.4'), '仅 sl0.4'), (lambda n: n.endswith('sl0.7'), '仅 sl0.7')]:
    Q = T2[[slf(n) for n in T2.index]]
    for sel, oth in [('A', 'B'), ('B', 'A')]:
        best = Q[f'cagr_{sel}'].idxmax(); r = Q.loc[best]
        print(f'{lab} 选于{sel}: {best}  {sel} dC={r["cagr_"+sel]-b["cagr_"+sel]:+.2f}  -> {oth} dC={r["cagr_"+oth]-b["cagr_"+oth]:+.2f} (final {r["final_"+oth]:.0f})  F dC={r.cagr_F-b.cagr_F:+.2f} F dd={r.max_dd_F:.1f}')
    d = Q[['cagr_A', 'cagr_B', 'cagr_F']] - b[['cagr_A', 'cagr_B', 'cagr_F']].values
    print(f'  {lab}: {len(Q)} 个配置中 A>基线 {int((d.cagr_A>0).sum())}, B>基线 {int((d.cagr_B>0).sum())}, 两窗都>基线 {int(((d.cagr_A>0)&(d.cagr_B>0)).sum())}; B dC 中位 {d.cagr_B.median():+.2f}, A dC 中位 {d.cagr_A.median():+.2f}')
