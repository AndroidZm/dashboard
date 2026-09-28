"""e_select s12: 低波动剔除候选的 80 种子确认：A/B/F、B去2024-02、F 上逐个去掉大簇（LOCO）；另测次日成交+0.5%滑点。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
T = D.tier
cands = {
    'vol60>=0.36': V[1] >= 0.36,
    'vol60>=0.38': V[1] >= 0.38,
    'vol60>=0.40': V[1] >= 0.40,
    'vol120>=0.35': V[2] >= 0.35,
    'vol60>=0.38_t12only': (V[1] >= 0.38) | (T == 0),
}
out = []
for name, mk in cands.items():
    df = full_report(Config(mask=mk), Config(), seeds=range(80), label=name)
    df['rule'] = name
    out.append(df)
    # 执行成本
    for w in ('A', 'B'):
        m1 = ev(Config(mask=mk, lag=1, slip=0.005), w, range(80))
        m0 = ev(Config(lag=1, slip=0.005), w, range(80))
        print(f'   lag1+slip0.5% {w}: {line(m1)}  vs base {line(m0)}  dC {(m1["cagr"]-m0["cagr"])*100:+.2f}')
pd.concat(out).to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s12_volconfirm.csv', index=False)
