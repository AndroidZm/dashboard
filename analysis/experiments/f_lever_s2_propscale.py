"""s2: 纯比例放大 L —— 权重 ×L 且仓位上限 cap = 0.9×L（L>1.11 时融资）。

这是把基线收益流整体放大 L 倍的最干净口径（阻挡/成交次序基本与基线相同），
用来估计“经验 Kelly”（中位对数财富最大的 L）、回撤随 L 的增长、维持担保比例。
同时跑剔除 2024-02 与逐个剔除大簇 (LOCO)。80 种子。
输出 f_lever_out/s2_prop.csv
"""
import itertools
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import BIG, Data, cfg_win, month_mask, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "f_lever_out")
LS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
DROPS = ["", "2024-02", "2012-01", "2018-02", "2022-05", "2022-10"]
_D = None


def job(p):
    global _D
    if _D is None:
        _D = Data.load()
    L, br, win, drop = p
    kw = dict(k=L, cap=0.9 * L, borrow_rate=br)
    if drop:
        kw["mask"] = month_mask(_D, [drop])
    m = run_seeds(_D, cfg_win(win, **kw), range(80))
    return dict(L=L, borrow=br, win=win, drop=drop, **m)


if __name__ == "__main__":
    params = []
    for L, br, win, drop in itertools.product(LS, [0.05, 0.08], "ABF", DROPS):
        if win == "A" and drop not in ("", "2012-01"):
            continue
        if win == "B" and drop == "2012-01":
            continue
        params.append((L, br, win, drop))
    print(len(params))
    with Pool(3) as pool:
        rows = pool.map(job, params, chunksize=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "s2_prop.csv"), index=False)
    print("saved")
