"""阶段 11：换仓规则叠加在更高仓位（cap<=1，不融资）/更高单笔权重的底座上，是否仍然增益；用于第 2/3 档风险。20 种子。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
rows = []
for cap, (wn, wa) in itertools.product([0.9, 1.0], [(0.02, 0.08), (0.03, 0.10), (0.03, 0.12), (0.04, 0.12), (0.04, 0.16)]):
    for sw, w2 in [(None, 0.06), ("oldest", 0.03), ("oldest", 0.04), ("oldest", 0.06), ("flat", 0.04)]:
        row = dict(cap=cap, wn=wn, wa=wa, swap=sw or "-", w2=w2)
        for tag, (st, en), mk in WINS:
            kw = dict(swap_policy=sw, swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5) if sw else {}
            m = run_seeds(D, PConfig(start=st, end=en, mask=mk, cap=cap, weights=(wn, wa, w2), **kw), SE)
            row[tag] = round(m["cagr"] * 100, 2)
            row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
        row["F_final"] = round(m["final"] / 1e4, 1)
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s11.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
