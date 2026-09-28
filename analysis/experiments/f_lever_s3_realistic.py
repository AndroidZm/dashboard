"""s3: 更现实的杠杆口径 + 固定金额 vs 按权益复利。

L 比例放大（权重×L，cap=0.9L），比较:
  cost      基线口径（按成本权益定额、成本口径上限），不模拟强平
  cost_mc   同上 + 维持担保比例 <130% 时按持有最久顺序强平到 150%
  mtm_mc    按盯市权益定额、盯市口径上限（券商按实时担保品授信的近似）+ 强平
  fixed_mc  固定金额：单笔 = 权重×L×初始资金（不复利），盯市口径上限 + 强平
80 种子。输出 f_lever_out/s3_real.csv
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
LS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
VAR = {
    "cost": dict(),
    "cost_mc": dict(margin_call=(1.3, 1.5)),
    "mtm_mc": dict(sizing="mtm", cap_basis="mtm", margin_call=(1.3, 1.5)),
    "fixed_mc": dict(sizing="fixed", cap_basis="mtm", margin_call=(1.3, 1.5)),
}
_D = None


def job(p):
    global _D
    if _D is None:
        _D = Data.load()
    L, br, var, win = p
    m = run_seeds(_D, cfg_win(win, k=L, cap=0.9 * L, borrow_rate=br, **VAR[var]), range(80))
    return dict(L=L, borrow=br, var=var, win=win, **m)


if __name__ == "__main__":
    params = list(itertools.product(LS, [0.05, 0.08], list(VAR), "ABF"))
    print(len(params))
    with Pool(3) as pool:
        rows = pool.map(job, params, chunksize=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "s3_real.csv"), index=False)
    print("saved")
