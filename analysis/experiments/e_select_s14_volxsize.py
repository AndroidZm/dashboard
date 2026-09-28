"""e_select s14: 低波动剔除 x 等比例放大档位权重 k（不加杠杆，总仓位上限 90% 不变），A/B 两窗口 20 种子。
看剔除慢信号腾出仓位后，放大单笔是否能在回撤约束下换来更多利润。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
rows = []
for thr in (None, 0.34, 0.36, 0.38, 0.40, 0.43):
    mk = None if thr is None else V[1] >= thr
    for k in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        w = (0.02 * k, 0.08 * k, 0.06 * k)
        r = dict(thr=thr or 0, k=k)
        for win in ('A', 'B'):
            m = ev(Config(mask=mk, weights=w), win, range(20))
            r[win + '_cagr'] = m['cagr'] * 100
            r[win + '_dd'] = m['max_dd'] * 100
            r[win + '_exp'] = m['avg_exposure'] * 100
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s14_volxsize.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(df.round(2).to_string(index=False))
