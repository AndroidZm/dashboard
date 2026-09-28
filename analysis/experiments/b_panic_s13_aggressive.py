"""阶段 13：不融资 (cap=1.0) 下更激进的单笔权重 + 恐慌换仓，看收益是否继续上升、回撤到哪（档 3）。20 种子。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
rows = []
for wn, wa, w2, sw in itertools.product([0.03, 0.05, 0.08], [0.12, 0.16, 0.20, 0.25], [0.04, 0.06, 0.08], ["oldest", None]):
    if sw is None and w2 != 0.06:
        continue
    row = dict(wn=wn, wa=wa, w2=w2, swap=sw or "-")
    for tag, (st, en), mk in WINS:
        kw = dict(swap_policy=sw, swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5) if sw else {}
        m = run_seeds(D, PConfig(start=st, end=en, mask=mk, cap=1.0, weights=(wn, wa, w2), **kw), SE)
        row[tag] = round(m["cagr"] * 100, 2); row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
    row["F_fin"] = round(m["final"] / 1e4, 1)
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
