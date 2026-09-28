"""补充：均线与信号密度 regime 组合；次日成交下为控回撤所需的现金保留；80 种子。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
ma = I.reg_trend_h(D, "sh000300", 120)
ma1 = I.reg_trend_h(D, "sh000300", 120, 0, 1)
q = I.reg_quiet(D, thr=5)
ap = I.reg_after(D, thr=20, hold=120)
CF = {
    "ma120_AND_quiet5_r0": dict(regime=ma & q),
    "ma120_OR_afterpanic120_r0": dict(regime=ma | ap),
    "ma120_r30": dict(regime=ma, reserve=0.3),
    "ma120_r30_lag1": dict(regime=ma1, reserve=0.3),
    "ma120_r10_lag1": dict(regime=ma1, reserve=0.1),
    "quiet5_r20": dict(regime=q, reserve=0.2),
}
for name, kw in CF.items():
    out = []
    for w, (st, en) in {"A": I.WIN_A, "B": I.WIN_B, "F": I.WIN_F}.items():
        m = I.run_seeds(D, I.IConfig(start=st, end=en, idx="sh000300", **kw))
        out.append(f"{w} {m['final']/1e4:6.0f}万 {m['cagr']*100:5.2f}% dd {m['max_dd']*100:5.1f} (p10 {m['dd_p10']*100:5.1f})")
    print(f"{name:28s}", " | ".join(out), flush=True)
