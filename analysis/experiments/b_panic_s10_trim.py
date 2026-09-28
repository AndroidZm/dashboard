"""阶段 10：整只卖出 vs 只减仓到刚好够买（swap_partial），以及一次最多卖几只 (mx)。20 种子。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
base = {tag: run_seeds(D, PConfig(start=st, end=en, mask=mk), SE) for tag, (st, en), mk in WINS}
rows = []
for pol, w2, (mode, mx) in itertools.product(["oldest", "flat", "underwater", "gain_pct"], [0.02, 0.03, 0.04, 0.06],
                                             [("full", 2), ("full", 3), ("full", 5), ("full", 10), ("part", 10)]):
    row = dict(pol=pol, w2=w2, mode=mode, mx=mx)
    for tag, (st, en), mk in WINS:
        m = run_seeds(D, PConfig(start=st, end=en, mask=mk, swap_policy=pol, swap_on="both", panic_max_pos=None, swap_keep_panic=True,
                                 weights=(0.02, 0.08, w2), swap_max_n=mx, swap_partial=(mode == "part")), SE)
        row["d" + tag] = round((m["cagr"] - base[tag]["cagr"]) * 100, 2)
        row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
        if tag in ("A", "B"):
            row[tag + "_np"] = m["n_panic"]; row[tag + "_fail"] = m["n_swap_fail"]
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s10.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print({k: (round(v["cagr"] * 100, 2), round(v["max_dd"] * 100, 1)) for k, v in base.items()})
print(df.to_string())
print(df.groupby(["mode", "mx"])[["dA", "dB", "dBx", "dF", "A_dd", "B_dd"]].mean().round(2))
