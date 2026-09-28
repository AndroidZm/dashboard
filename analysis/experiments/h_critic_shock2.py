"""h_critic_: 瞬时冲击（股票 -37%）补充：L11 与 P1×1.1。口径同 h_critic_bench.py。"""
import os, sys
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import g_combined_engine as G
from g_combined_plans import make, D, SWAP
PL = {"L11": G.GConfig(weights=(0.022, 0.088, 0.066), cap=0.99),
      "P1x1.1": G.GConfig(weights=(0.022, 0.088, 0.044), cap=0.99, **SWAP)}
for w in ("B", "F"):
    for nm, cfg in PL.items():
        v = []
        for seed in range(20):
            r = G.run(D, replace(G.win_cfg(cfg, w), record=True), seed)
            eq, pos = r["equity"].values, r["pos_value"].values
            v.append(((eq - 0.37 * pos) / np.maximum.accumulate(eq) - 1).min())
        print(w, nm, f"shock_dd median {np.median(v)*100:.1f} worst {np.min(v)*100:.1f}")
