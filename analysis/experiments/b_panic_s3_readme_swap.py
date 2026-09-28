"""阶段 3：直接检验 README 的“恐慌期换仓（卖持有最久的一只）”——用 engine 原口径 panic_swap，
在持仓数阈值 panic_max_pos 网格上看 A / B / B去2024-02 / 全期 四个口径的增量是否变号。80 种子。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, Config
from engine import run_seeds
from b_panic_engine import WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
rows = []
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
for lag in (0, 1):
    base = {tag: run_seeds(D, Config(start=st, end=en, mask=mk, lag=lag)) for tag, (st, en), mk in WINS}
    for pmp in (8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 30):
        for w2 in (0.04, 0.06, 0.08):
            row = dict(lag=lag, pmp=pmp, w2=w2)
            for tag, (st, en), mk in WINS:
                m = run_seeds(D, Config(start=st, end=en, mask=mk, lag=lag, panic_swap=True, panic_max_pos=pmp, weights=(0.02, 0.08, w2)))
                row["d" + tag] = round((m["cagr"] - base[tag]["cagr"]) * 100, 2)
                row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
                row[tag + "_np"] = m["n_by_tier"] if "n_by_tier" in m else None
            rows.append(row)
    print("lag", lag, "base", {k: (round(v["cagr"] * 100, 2), round(v["max_dd"] * 100, 1)) for k, v in base.items()})
df = pd.DataFrame(rows).drop(columns=[c for c in ("A_np", "B_np", "Bx_np", "F_np")])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s3.csv", index=False)
