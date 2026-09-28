"""阶段 2：给恐慌档预留“干火药”——平常/调整档只能用到 cap_np，恐慌档可用到 cap；可同时降低调整档权重。20 种子筛选。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
rows = []
for cap_np, cap, wa, w2 in itertools.product([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.9, 1.0], [0.04, 0.06, 0.08], [0.02, 0.04, 0.06, 0.08, 0.10]):
    if cap_np > cap:
        continue
    row = dict(cap_np=cap_np, cap=cap, wa=wa, w2=w2)
    for tag, (st, en), mk in (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None)):
        m = run_seeds(D, PConfig(start=st, end=en, panic_max_pos=None, cap=cap, cap_np=cap_np, weights=(0.02, wa, w2), mask=mk), SE)
        row[tag + "_cagr"] = round(m["cagr"] * 100, 2); row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
        row[tag + "_np"] = m["n_panic"]
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s2.csv", index=False)
