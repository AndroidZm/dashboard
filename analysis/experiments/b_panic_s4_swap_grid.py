"""阶段 4：换仓规则变体网格（修正口径：卖了必须能买）。20 种子筛选。
policy: 卖谁；on: maxpos=只在持仓数挡住时 / both=仓位上限挡住也换；pmp: 持仓数阈值；w2: 恐慌权重；kp: 不卖恐慌档持仓。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
base = {tag: run_seeds(D, PConfig(start=st, end=en, mask=mk), SE) for tag, (st, en), mk in WINS}
print({k: (round(v["cagr"] * 100, 2), round(v["max_dd"] * 100, 1)) for k, v in base.items()})
rows = []
for pol, on, pmp, w2, kp in itertools.product(["oldest", "newest", "gain_pct", "gain_abs", "flat", "underwater"],
                                             ["maxpos", "both"], [10, 15, 20, None], [0.04, 0.06, 0.08], [False, True]):
    if on == "maxpos" and pmp is None:
        continue
    row = dict(pol=pol, on=on, pmp=pmp, w2=w2, kp=kp)
    for tag, (st, en), mk in WINS:
        m = run_seeds(D, PConfig(start=st, end=en, mask=mk, swap_policy=pol, swap_on=on, panic_max_pos=pmp,
                                 weights=(0.02, 0.08, w2), swap_keep_panic=kp), SE)
        row["d" + tag] = round((m["cagr"] - base[tag]["cagr"]) * 100, 2)
        row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
        if tag in ("A", "B"):
            row[tag + "_np"] = m["n_panic"]; row[tag + "_sw"] = m["n_swap"]
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s4.csv", index=False)
