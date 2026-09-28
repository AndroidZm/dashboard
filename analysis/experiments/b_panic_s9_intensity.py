"""阶段 9：簇强度 (n_signals_30d) 与持有时长门槛。20 种子。
(a) 只在 n30 >= k 时换仓；(b) 只换掉持有 >= age 个交易日的持仓；(c) 恐慌权重按 (n30/40)^a 缩放（夹在 [0.25, 4] 倍）。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
base = {tag: run_seeds(D, PConfig(start=st, end=en, mask=mk), SE) for tag, (st, en), mk in WINS}
SW = dict(swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=3)


def ev(**kw):
    row = {}
    for tag, (st, en), mk in WINS:
        m = run_seeds(D, PConfig(start=st, end=en, mask=mk, **SW, **kw), SE)
        row["d" + tag] = round((m["cagr"] - base[tag]["cagr"]) * 100, 2)
        row[tag + "_dd"] = round(m["max_dd"] * 100, 1)
        if tag == "B":
            row["B_np"] = m["n_panic"]
    return row


rows = []
for pol, w2, k, age in itertools.product(["oldest", "flat"], [0.02, 0.04], [0, 25, 30, 40, 60], [0, 10, 40]):
    rows.append(dict(grid="n30/age", pol=pol, w2=w2, k=k, age=age, a=None, **ev(swap_policy=pol, weights=(0.02, 0.08, w2), swap_min_n30=k, swap_min_age=age)))
for pol, w2, a in itertools.product(["oldest", "flat"], [0.02, 0.04], [-1.0, -0.5, 0.0, 0.5, 1.0]):
    f = (lambda n, a=a: float(np.clip((n / 40.0) ** a, 0.25, 4.0)))
    rows.append(dict(grid="mult", pol=pol, w2=w2, k=0, age=0, a=a, **ev(swap_policy=pol, weights=(0.02, 0.08, w2), panic_mult=f)))
# 不换仓，只做强度缩放（恐慌持仓数不限）
for w2, a in itertools.product([0.02, 0.04, 0.06], [-1.0, -0.5, 0.5, 1.0]):
    f = (lambda n, a=a: float(np.clip((n / 40.0) ** a, 0.25, 4.0)))
    rows.append(dict(grid="mult_noswap", pol=None, w2=w2, k=0, age=0, a=a, **ev(swap_policy=None, weights=(0.02, 0.08, w2), panic_mult=f)))
df = pd.DataFrame(rows)
df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s9.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print({k: (round(v["cagr"] * 100, 2), round(v["max_dd"] * 100, 1)) for k, v in base.items()})
print(df.to_string())
