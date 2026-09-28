"""h_critic_: 换仓不许卖停牌股时的影响（NEED_TRADED 开/关），以及关时与 g_combined_engine 逐种子一致性。"""
import os, sys
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import g_combined_engine as G
import h_critic_engine as H
from g_combined_plans import make, D
PL = {"base": make(), "P1": make(S=True), "P2": make(S=True, C=True)}
# 一致性（开关关闭）
for nm in ("P1", "P2"):
    for w in ("A", "B"):
        a = [G.run(D, G.win_cfg(PL[nm], w), s)["final"] for s in range(10)]
        b = [H.run(D, H.win_cfg(PL[nm], w), s)["final"] for s in range(10)]
        print("equal", nm, w, np.max(np.abs(np.array(a) - np.array(b))))
H.NEED_TRADED = True
for real in (False, True):
    for w in ("A", "B", "F"):
        _, b = G.run_seeds(D, G.win_cfg(PL["base"], w, realistic=real), raw=True)
        for nm in ("P1", "P2"):
            _, g = G.run_seeds(D, G.win_cfg(PL[nm], w, realistic=real), raw=True)
            _, h = H.run_seeds(D, H.win_cfg(PL[nm], w, realistic=real), raw=True)
            print(f"{'实盘' if real else '当日'} {w} {nm}: 原 {g.cagr.median()*100:.2f} ({np.median(g.cagr-b.cagr)*100:+.2f})  "
                  f"不卖停牌 {h.cagr.median()*100:.2f} ({np.median(h.cagr-b.cagr)*100:+.2f}) 回撤 {h.max_dd.median()*100:.1f}")
