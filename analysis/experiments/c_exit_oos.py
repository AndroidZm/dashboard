"""样本外选择：把所有 80 种子网格（fine80 / wide80 / slgrid 无压力部分）合并，
在一个窗口上按风险档（回撤约束）挑最优，再看另一个窗口和全期。另做 3x3 邻域平滑后的选择（trail 网格）。"""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
a = pd.read_csv('c_exit_out/grid_fine80.csv')
b = pd.read_csv('c_exit_out/wide80.csv')
c = pd.read_csv('c_exit_out/slgrid.csv')
c = c[c.name.str.endswith('|q0')].copy(); c['name'] = c.name.str.replace('|q0', '', regex=False)
# 统一命名：trail 规则的 key
def norm(n):
    return n.replace('trail_', 'tr_')
for d in (a, b, c):
    d['name'] = d.name.map(norm)
allr = pd.concat([a[['name', 'win', 'final', 'cagr', 'max_dd']], b[['name', 'win', 'final', 'cagr', 'max_dd']], c[['name', 'win', 'final', 'cagr', 'max_dd']]])
allr = allr.drop_duplicates(['name', 'win'])
P = allr.pivot_table(index='name', columns='win', values=['final', 'cagr', 'max_dd'])
P.columns = [f'{v}_{w}' for v, w in P.columns]
base = P.loc['base'] if 'base' in P.index else P.loc['fixed_tp0.4_sl0.4']
P = P.drop(index=[i for i in ['fixed_tp0.4_sl0.4'] if i in P.index and 'base' in P.index])
print('候选总数', len(P))
print('基线', base.round(2).to_dict())
rows = []
fams = {'全部': lambda n: True, '仅保留-40%止损': lambda n: n.endswith('sl0.4') or n == 'base',
        '固定止盈止损': lambda n: n.startswith('fixed') or n.startswith('fx') or n == 'base'}
for fam, f in fams.items():
    Q = P[[f(n) for n in P.index]]
    for sel in 'AB':
        oth = 'B' if sel == 'A' else 'A'
        for lvl, lim in [('L1 dd>=-30', -30.5), ('L2 dd>=-40', -40.5), ('L3 无约束', -999)]:
            q = Q[Q[f'max_dd_{sel}'] >= lim]
            best = q[f'final_{sel}'].idxmax()
            r = P.loc[best]
            rows.append(dict(family=fam, select_on=sel, level=lvl, pick=best,
                             sel_final=r[f'final_{sel}'], sel_dC=r[f'cagr_{sel}'] - base[f'cagr_{sel}'],
                             oth_final=r[f'final_{oth}'], oth_dC=r[f'cagr_{oth}'] - base[f'cagr_{oth}'], oth_dd=r[f'max_dd_{oth}'],
                             F_final=r['final_F'], F_dC=r['cagr_F'] - base['cagr_F'], F_dd=r['max_dd_F']))
out = pd.DataFrame(rows)
print(out.round(2).to_string(index=False))
# 3x3 邻域平滑（fine80 trail，act x trail，sl=None / 0.4）
f = pd.read_csv('c_exit_out/grid_fine80.csv'); f = f[f.fam == 'trail'].copy(); f['sl'] = f.sl.fillna(9)
for sl in [0.4, 9]:
    for sel in 'AB':
        oth = 'B' if sel == 'A' else 'A'
        g = f[(f.sl == sl) & (f.win == sel)].pivot_table(index='act', columns='tr', values='cagr')
        sm = g.rolling(3, center=True, min_periods=2).mean().T.rolling(3, center=True, min_periods=2).mean().T
        act, tr = sm.stack().idxmax()
        o = f[(f.sl == sl) & (f.win == oth) & (f.act == act) & (f.tr == tr)].cagr.iloc[0]
        ff = f[(f.sl == sl) & (f.win == 'F') & (f.act == act) & (f.tr == tr)].cagr.iloc[0]
        print(f'平滑选择 sl={sl} 选于{sel}: act={act} trail={tr}  -> {oth} dC={o - base["cagr_" + oth]:+.2f}pp, F dC={ff - base["cagr_F"]:+.2f}pp')
