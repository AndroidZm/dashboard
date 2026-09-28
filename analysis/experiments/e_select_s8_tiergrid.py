"""e_select s8: 档位权重 (平常, 调整, 恐慌) 网格 x 恐慌持仓上限 {15, 无}，A/B 两窗口 20 种子。
目的：把“按档位挑信号”与“整体放大仓位”分开——和等比例放大基线权重（flat k）的收益-回撤前沿比较。"""
import sys, itertools
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
rows = []
grid = []
for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
    grid.append(('flat', (0.02 * k, 0.08 * k, 0.06 * k), 15))
for w0, w1, w2, pm in itertools.product((0.0, 0.02, 0.04, 0.06), (0.04, 0.06, 0.08, 0.10, 0.12, 0.15), (0.03, 0.06, 0.09), (15, None)):
    grid.append(('grid', (w0, w1, w2), pm))
for fam, w, pm in grid:
    r = dict(fam=fam, w0=w[0], w1=w[1], w2=w[2], pm=pm if pm else 0)
    for win in ('A', 'B'):
        m = ev(Config(weights=w, panic_max_pos=pm), win, range(20))
        r[win + '_cagr'] = m['cagr'] * 100
        r[win + '_dd'] = m['max_dd'] * 100
        r[win + '_final'] = m['final'] / 1e4
        r[win + '_n2'] = m.get('n_panic', np.nan)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s8_tiergrid.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(df[df.fam == 'flat'].round(2).to_string(index=False))
g = df[df.fam == 'grid']
print('top by A cagr'); print(g.sort_values('A_cagr', ascending=False).head(15).round(2).to_string(index=False))
print('top by B cagr'); print(g.sort_values('B_cagr', ascending=False).head(15).round(2).to_string(index=False))
