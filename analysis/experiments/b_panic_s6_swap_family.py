"""阶段 6：“仓位满时换仓买恐慌股”规则族的平滑性检查（swap_on=both、修正口径、不限恐慌持仓数）。20 种子。
维度：卖谁 policy × 恐慌单笔权重 w2 × 是否不卖恐慌档持仓 kp × 一次最多卖几只 mx。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
base = {tag: run_seeds(D, PConfig(start=st, end=en, mask=mk), SE) for tag, (st, en), mk in WINS}
rows = []
for pol, w2, kp, mx in itertools.product(["oldest", "newest", "gain_pct", "gain_abs", "flat", "underwater"],
                                         [0.02, 0.03, 0.04, 0.05, 0.06, 0.08], [True, False], [1, 3]):
    row = dict(pol=pol, w2=w2, kp=kp, mx=mx)
    for tag, (st, en), mk in WINS:
        m = run_seeds(D, PConfig(start=st, end=en, mask=mk, swap_policy=pol, swap_on="both", panic_max_pos=None,
                                 weights=(0.02, 0.08, w2), swap_keep_panic=kp, swap_max_n=mx), SE)
        row["d" + tag] = round((m["cagr"] - base[tag]["cagr"]) * 100, 2)
        row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s6.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print({k: (round(v["cagr"] * 100, 2), round(v["max_dd"] * 100, 1)) for k, v in base.items()})
for kp in (True, False):
    for mx in (1, 3):
        sub = df[(df.kp == kp) & (df.mx == mx)]
        print(f"--- kp={kp} mx={mx}")
        for col in ("dA", "dB", "dBx", "dF", "B_dd", "A_dd"):
            print(col); print(sub.pivot(index="pol", columns="w2", values=col).to_string())
