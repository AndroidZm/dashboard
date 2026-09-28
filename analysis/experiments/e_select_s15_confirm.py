"""e_select s15: 决选 80 种子——低波动剔除 + 权重放大 k 的组合，A/B/F、B去2024-02、LOCO、次日成交+滑点。
对照：同样 k 但不剔除。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
def W(k):
    return (0.02 * k, 0.08 * k, 0.06 * k)
cands = {
    'flat_k1.25': Config(weights=W(1.25)),
    'flat_k1.5': Config(weights=W(1.5)),
    'vol38_k1.25': Config(mask=V[1] >= 0.38, weights=W(1.25)),
    'vol38_k1.5': Config(mask=V[1] >= 0.38, weights=W(1.5)),
    'vol40_k1.25': Config(mask=V[1] >= 0.40, weights=W(1.25)),
    'vol40_k1.5': Config(mask=V[1] >= 0.40, weights=W(1.5)),
    'vol38_k2': Config(mask=V[1] >= 0.38, weights=W(2.0)),
}
out = []
for name, cfg in cands.items():
    df = full_report(cfg, Config(), seeds=range(80), label=name)
    df['rule'] = name
    out.append(df)
    for w in ('A', 'B'):
        m1 = ev(__import__('dataclasses').replace(cfg, lag=1, slip=0.005), w, range(80))
        m0 = ev(Config(lag=1, slip=0.005), w, range(80))
        print(f'   lag1+slip0.5% {w}: {line(m1)} vs base {line(m0)} dC {(m1["cagr"]-m0["cagr"])*100:+.2f}', flush=True)
pd.concat(out).to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s15_confirm.csv', index=False)
