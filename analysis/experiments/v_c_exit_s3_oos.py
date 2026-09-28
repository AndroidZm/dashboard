"""v_c_exit 第 3 步：基于 20 种子网格的双向样本外选择（按回撤档）与候选邻域。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
df = pd.read_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s2_grid20.csv')
P = df.pivot_table(index='name', columns='win', values=['final', 'cagr', 'dd'])
P.columns = [f'{v}_{w}' for v, w in P.columns]
b = P.loc['base']
print('base20', b.round(2).to_dict())
for w in 'ABF':
    P[f'dC_{w}'] = P[f'cagr_{w}'] - b[f'cagr_{w}']
P['fam'] = P.index.str.split('|').str[0]
P['sl'] = P.index.str.split('|').str[-1]
fams = {'all': lambda q: q, 'trail': lambda q: q[q.fam == 'tr'], 'trail2': lambda q: q[q.fam == 't2'], 'fixed': lambda q: q[q.fam == 'fx'],
        'sl0.4 only': lambda q: q[q.sl == '0.4'], 'trail sl0.7': lambda q: q[(q.fam == 'tr') & (q.sl == '0.7')], 'trail2 sl0.7': lambda q: q[(q.fam == 't2') & (q.sl == '0.7')]}
rows = []
for fn, ff in fams.items():
    Q = ff(P)
    for sel, oth in (('A', 'B'), ('B', 'A')):
        for lvl, lim in (('L1 dd>=-30', -30.5), ('L2 dd>=-40', -40.5), ('L3', -999)):
            q = Q[Q[f'dd_{sel}'] >= lim]
            if not len(q):
                continue
            best = q[f'cagr_{sel}'].idxmax(); r = P.loc[best]
            # 平滑选择：同族内每个配置取自身与 top 邻居的平均过于复杂，这里另给前 5 名在另一窗口的中位
            top5 = q[f'cagr_{sel}'].nlargest(5).index
            rows.append(dict(fam=fn, sel=sel, lvl=lvl, pick=best, sel_dC=r[f'dC_{sel}'], oth_dC=r[f'dC_{oth}'], oth_dd=r[f'dd_{oth}'],
                             F_dC=r['dC_F'], F_dd=r['dd_F'], top5_oth_dC_med=P.loc[top5, f'dC_{oth}'].median()))
print(pd.DataFrame(rows).round(2).to_string(index=False))

# 家族整体：两窗同时 > 基线的比例
for fn, ff in fams.items():
    Q = ff(P)
    print(f'{fn:14s} n={len(Q):4d}  A>0 {int((Q.dC_A>0).sum()):4d}  B>0 {int((Q.dC_B>0).sum()):4d}  both {int(((Q.dC_A>0)&(Q.dC_B>0)).sum()):4d}  corr(dC_A,dC_B)={Q.dC_A.corr(Q.dC_B):+.2f}  medB {Q.dC_B.median():+.2f} medA {Q.dC_A.median():+.2f}')

# 单段 trail：各 sl 下 act x trail 的 B / A 年化差热图
for sl in ['0.4', '0.6', '0.7', '0.8', 'None']:
    Q = P[(P.fam == 'tr') & (P.sl == sl)].copy()
    Q['act'] = Q.index.str.split('|').str[1].astype(float); Q['tr'] = Q.index.str.split('|').str[2].astype(float)
    for w in 'BA':
        print(f'--- trail sl={sl} dC_{w} (rows act, cols trail)')
        print(Q.pivot_table(index='act', columns='tr', values=f'dC_{w}').round(2).to_string())
# 固定止盈 x 止损
Q = P[P.fam == 'fx'].copy(); Q['tp'] = Q.index.str.split('|').str[1]
for w in 'BAF':
    print(f'--- fixed dC_{w} (rows tp, cols sl)')
    print(Q.pivot_table(index='tp', columns='sl', values=f'dC_{w}').round(2).to_string())
