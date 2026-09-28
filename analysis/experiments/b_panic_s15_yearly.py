"""阶段 15：逐年收益（20 种子中位）基线 vs 恐慌换仓，看增益落在哪些年份。全期 2006-2026。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from dataclasses import replace
from b_panic_engine import Data, PConfig, run, WIN_F

D = Data.load()
SW5 = dict(swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
C = {"base": PConfig(), "L1_swap": PConfig(swap_policy="oldest", weights=(0.02, 0.08, 0.04), **SW5),
     "L2_cap1_wa12": PConfig(cap=1.0, swap_policy="oldest", weights=(0.03, 0.12, 0.04), **SW5)}
out = {}
for n, c in C.items():
    ys = []
    for sd in range(20):
        eq = run(D, replace(c, start=WIN_F[0], end=WIN_F[1], record=True), sd)["equity"]
        eq.index = pd.to_datetime(eq.index)
        ys.append(eq.resample("YE").last().pct_change().fillna(eq.resample("YE").last().iloc[0] / 600000 - 1))
    out[n] = pd.concat(ys, axis=1).median(axis=1) * 100
df = pd.DataFrame(out)
df.index = df.index.year
df["L1-base"] = df.L1_swap - df.base
df["L2-base"] = df.L2_cap1_wa12 - df.base
print(df.round(1).to_string())
