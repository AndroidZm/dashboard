"""阶段 8：资金回收推广——换仓不只用于恐慌档，也用于调整/平常档被仓位上限挡住时；并与总仓位上限 cap(<=1, 不融资) 组合。20 种子。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
base = {tag: run_seeds(D, PConfig(start=st, end=en, mask=mk), SE) for tag, (st, en), mk in WINS}
rows = []
for tiers, cap, pol, w2 in itertools.product([(), (2,), (1, 2), (0, 1, 2)], [0.9, 0.95, 1.0], ["oldest", "flat", "underwater"], [0.02, 0.04, 0.06]):
    if tiers == () and pol != "oldest":
        continue
    row = dict(tiers="".join(map(str, tiers)) or "none", cap=cap, pol=pol, w2=w2)
    for tag, (st, en), mk in WINS:
        m = run_seeds(D, PConfig(start=st, end=en, mask=mk, swap_policy=pol if tiers else None, swap_on="both", panic_max_pos=None,
                                 weights=(0.02, 0.08, w2), swap_keep_panic=True, swap_max_n=3, swap_tiers=tiers, cap=cap), SE)
        row["d" + tag] = round((m["cagr"] - base[tag]["cagr"]) * 100, 2)
        row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
    row["B_sw"] = m["n_swap"]
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s8.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print({k: (round(v["cagr"] * 100, 2), round(v["max_dd"] * 100, 1)) for k, v in base.items()})
print(df.to_string())
