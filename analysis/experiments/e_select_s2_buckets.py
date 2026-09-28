"""e_select s2: 逐笔分桶统计（A / B / B 去掉 2024-02），看哪些信号日可得特征在两个窗口方向一致。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_feat import *
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
D = Data.load()
f = features(D)
f.to_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
fx = f[f.month != '2024-02'].copy(); fx['win'] = fx.win.map({'A': 'A', 'B': 'Bx'})
F = pd.concat([f, fx[fx.win == 'Bx']])
specs = [
    ('tier', None, None),
    ('n30', [0, 5, 20, 50, 100, 200, 1000], None),
    ('bars', [0, 500, 1000, 2500, 10000], None),
    ('praw', [0, 3, 6, 10, 20, 1000], None),
    ('board', None, None),
    ('r250', [-1, -0.5, -0.35, -0.2, 0.0, 5], None),
    ('r60', [-1, -0.4, -0.3, -0.2, -0.1, 5], None),
    ('r20', [-1, -0.25, -0.15, -0.08, 5], None),
    ('dd250', [-1, -0.6, -0.5, -0.4, -0.3, 0], None),
    ('idx_dd250', [-1, -0.4, -0.3, -0.2, -0.1, 0.01], None),
    ('idx_r20', [-1, -0.15, -0.08, -0.03, 1], None),
    ('hs_dd250', [-1, -0.3, -0.2, -0.1, 0.01], None),
    ('rel250', [-2, -0.3, -0.15, 0, 5], None),
    ('vol60', [0, 0.3, 0.45, 0.6, 5], None),
]
for col, bins, lab in specs:
    t = bucket_stats(F, col, bins, lab)
    print('=====', col)
    print(t.round(3).to_string(index=False))
