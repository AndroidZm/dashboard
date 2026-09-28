"""s4: 候选点细网格确认（80 种子）。

(1) L 比例放大细网格 L=0.9..2.5，三个窗口 × 剔簇 (无/2024-02/LOCO) × 融资利率 5%/8%。
(2) 只放大权重、不放大上限（cap=0.9 固定，k=1..2）与只放大上限（k=1，cap 0.9..2.0）两条对照线。
输出 f_lever_out/s4_fine.csv
"""
import itertools
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import Data, cfg_win, month_mask, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "f_lever_out")
LS = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.2, 2.5]
DROPS = ["", "2024-02", "2012-01", "2018-02", "2022-05", "2022-10"]
_D = None


def job(p):
    global _D
    if _D is None:
        _D = Data.load()
    fam, k, cap, br, win, drop = p
    kw = dict(k=k, cap=cap, borrow_rate=br)
    if drop:
        kw["mask"] = month_mask(_D, [drop])
    m = run_seeds(_D, cfg_win(win, **kw), range(80))
    return dict(fam=fam, k=k, cap=cap, borrow=br, win=win, drop=drop, **m)


if __name__ == "__main__":
    params = []
    for L, br, win, drop in itertools.product(LS, [0.05, 0.08], "ABF", DROPS):
        if win == "A" and drop not in ("", "2012-01"):
            continue
        if win == "B" and drop == "2012-01":
            continue
        if br == 0.08 and drop:
            continue
        params.append(("L", L, round(0.9 * L, 4), br, win, drop))
    for k, win in itertools.product([1.0, 1.25, 1.5, 1.75, 2.0], "ABF"):
        params.append(("k_only", k, 0.9, 0.05, win, ""))
    for cap, win in itertools.product([0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0], "ABF"):
        params.append(("cap_only", 1.0, cap, 0.05, win, ""))
    print(len(params))
    with Pool(3) as pool:
        rows = pool.map(job, params, chunksize=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "s4_fine.csv"), index=False)
    print("saved")
