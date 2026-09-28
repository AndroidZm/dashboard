"""阶段 1：只放开恐慌档持仓数上限 / 总仓位上限 / 恐慌权重（不换仓、不留干火药）。20 种子筛选。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
rows = []
for pmp, cap, w2 in itertools.product([15, 20, 25, 30, 40, None], [0.9, 1.0], [0.02, 0.04, 0.06, 0.08, 0.10]):
    row = dict(pmp=pmp, cap=cap, w2=w2)
    for tag, (st, en), mk in (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None)):
        m = run_seeds(D, PConfig(start=st, end=en, panic_max_pos=pmp, cap=cap, weights=(0.02, 0.08, w2), mask=mk), SE)
        row[tag + "_cagr"] = round(m["cagr"] * 100, 2); row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
        row[tag + "_np"] = m["n_panic"]
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string())
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s1.csv", index=False)
