"""s1: 全权重统一缩放 k × 仓位上限 cap × 融资利率 × sizing 口径，三个窗口，20 种子中位。

输出 f_lever_out/s1_grid.csv
"""
import itertools
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import Data, cfg_win, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "f_lever_out")
os.makedirs(OUT, exist_ok=True)
KS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
CAPS = [0.9, 1.0, 1.2, 1.5, 2.0]
_D = None


def job(p):
    global _D
    if _D is None:
        _D = Data.load()
    k, cap, br, sizing, win = p
    kw = dict(k=k, cap=cap, borrow_rate=br)
    if sizing == "mtm":
        kw.update(sizing="mtm", cap_basis="mtm")
    m = run_seeds(_D, cfg_win(win, **kw), range(20))
    return dict(k=k, cap=cap, borrow=br, sizing=sizing, win=win, **m)


if __name__ == "__main__":
    params = []
    for k, cap, sizing, win in itertools.product(KS, CAPS, ["cost", "mtm"], "ABF"):
        for br in ([0.06] if cap <= 1.0 else [0.05, 0.08]):
            params.append((k, cap, br, sizing, win))
    print(len(params), "configs")
    with Pool(3) as pool:
        rows = pool.map(job, params, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "s1_grid.csv"), index=False)
    print("saved")
