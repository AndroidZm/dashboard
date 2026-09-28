"""e_select s17: 低波动剔除与其它旋钮的交互——恐慌持仓上限放开/加大、最长持有期、总仓位 100%。20 种子。
若别的旋钮已解决“慢仓占位”，剔除低波动的增益应缩小。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from dataclasses import replace
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
mk = V[1] >= 0.40
ctx = {'default': Config(), 'pm25': Config(panic_max_pos=25), 'pmNone': Config(panic_max_pos=None),
       'maxhold250': Config(max_hold=250), 'maxhold500': Config(max_hold=500), 'cap1.0': Config(cap=1.0),
       'k1.5': Config(weights=(0.03, 0.12, 0.09)), 'pmNone_k1.5': Config(panic_max_pos=None, weights=(0.03, 0.12, 0.09))}
for n, c in ctx.items():
    s = []
    for w in ('A', 'B'):
        m0 = ev(c, w, range(20))
        m1 = ev(replace(c, mask=mk), w, range(20))
        s.append(f"{w}: ctx {m0['cagr']*100:5.2f}%/{m0['max_dd']*100:5.1f}  +vol40 {m1['cagr']*100:5.2f}%/{m1['max_dd']*100:5.1f}  d {(m1['cagr']-m0['cagr'])*100:+.2f}")
    print(f'{n:12s}', ' | '.join(s), flush=True)
