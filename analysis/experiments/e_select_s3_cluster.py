"""e_select s3: 以“月簇”为单位看时点特征（指数回撤、档位）与逐笔结果的关系；档位 x 指数回撤交叉表。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
f['ddb'] = pd.cut(f.idx_dd250, [-1, -0.3, -0.2, -0.15, -0.1, 0.01])
print(f.pivot_table(index=['win', 'tier'], columns='ddb', values='ret', aggfunc=['count', 'mean'], observed=False).round(3))
fx = f[f.month != '2024-02']
print('--- without 2024-02')
print(fx.pivot_table(index=['win', 'tier'], columns='ddb', values='ret', aggfunc=['count', 'mean'], observed=False).round(3))
# 月簇层面
g = f.groupby('month').agg(n=('ret', 'size'), ret=('ret', 'mean'), tp=('tp', 'mean'), sl=('sl', 'mean'), held=('held', 'median'),
                           idx_dd=('idx_dd250', 'mean'), hs_dd=('hs_dd250', 'mean'), tier=('tier', 'max'), n30=('n30', 'max'),
                           board60=('board', lambda b: (b == '60').mean()))
print(g.round(3).to_string())
for w, h in [('A', g[g.index < '2016-01']), ('B', g[g.index >= '2016-01'])]:
    print(w, 'clusters', len(h), 'spearman(idx_dd, ret) =', round(h.idx_dd.corr(h.ret, method='spearman'), 3),
          ' n-weighted corr', round(np.corrcoef(h.idx_dd, h.ret)[0, 1], 3))
    for thr in (-0.1, -0.15, -0.2, -0.25):
        a, b = h[h.idx_dd <= thr], h[h.idx_dd > thr]
        print(f'   thr {thr}: deep clusters {len(a)} sig {a.n.sum()} mean ret(cluster-avg) {a.ret.mean():.3f} tp {a.tp.mean():.3f} | shallow {len(b)} sig {b.n.sum()} ret {b.ret.mean():.3f} tp {b.tp.mean():.3f}')
