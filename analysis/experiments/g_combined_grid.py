"""核心旋钮（全部是复核判为 robust 的）组合网格，20 种子筛选。

维度: 恐慌换仓 on/off × 总仓位上限 cap × 平常/调整/恐慌 单笔权重。
窗口: A、B、全期 F、B 去 2024-02、F 去 2024-02。
输出 g_combined_out/grid.csv
"""
import itertools
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import g_combined_engine as G

OUT = os.path.join(HERE, "g_combined_out")
os.makedirs(OUT, exist_ok=True)
SW = dict(swap_policy="oldest", swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
D = G.Data.load()

CAPS = [0.85, 0.9, 0.95, 1.0]
WN = [0.01, 0.02, 0.03, 0.04]
WA = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
WP = [0.03, 0.04, 0.06]
EVALS = [("A", ()), ("B", ()), ("F", ()), ("B", ("2024-02",)), ("F", ("2024-02",))]


def job(args):
    swap, cap, wn, wa, wp = args
    kw = dict(SW) if swap else dict(panic_max_pos=None)
    base = G.GConfig(cap=cap, weights=(wn, wa, wp), **kw)
    row = dict(swap=swap, cap=cap, wn=wn, wa=wa, wp=wp)
    for w, drop in EVALS:
        m = G.run_seeds(D, G.win_cfg(base, w, drop, D=D), seeds=range(20))
        tag = w + ("_no2402" if drop else "")
        row[tag + "_final"] = m["final"]
        row[tag + "_cagr"] = m["cagr"]
        row[tag + "_dd"] = m["max_dd"]
        row[tag + "_ddp10"] = m["dd_p10"]
        if not drop:
            row[tag + "_npanic"] = m["n_panic"]
            row[tag + "_exp"] = m["avg_exposure"]
    return row


if __name__ == "__main__":
    grid = list(itertools.product([True, False], CAPS, WN, WA, WP))
    with Pool(3) as p:
        rows = p.map(job, grid, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    print(len(df), "configs written")
