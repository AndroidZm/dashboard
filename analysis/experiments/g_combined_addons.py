"""核心（换仓 + 总仓位上限）之上叠加各个"脆弱"旋钮的全因子测试，80 种子。

因子:
  cap   0.9 / 1.0                     （核心，robust）
  W     权重 2/8/4 或 3/12/4           （调整档加码，A 窗口证据全来自 2008）
  E     移动止盈：+50% 后回落 10% 卖，止损放宽到 -70%（v_c_exit C1，fragile）
  I     闲置现金：沪深300 在 120 日均线上方（前一日判断）时，把超过总权益 30% 的现金买沪深300，每日再平衡（v_d_idle C3，fragile）
  V     剔除信号日前 60 日年化波动率 < 38% 的信号（v_e_select C1，fragile）
另有 swap off 的对照。窗口: A、B、F、B 去 2024-02、B/F 实盘口径（次日收盘 + 0.5% 滑点，指数费 0.2%）。
输出 g_combined_out/addons.csv
"""
import itertools
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import g_combined_engine as G
from e_select_rule import vol60

OUT = os.path.join(HERE, "g_combined_out")
os.makedirs(OUT, exist_ok=True)
D = G.Data.load()
V60 = vol60(D)
REG = G.reg_trend_h(D, "sh000300", ma=120, h=0.0, lag=1)
SW = dict(swap_policy="oldest", swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
EXIT = {"kind": "trail", "act": 0.5, "trail": 0.1, "sl": 0.7, "tp": None}


def build(cap, W, E, I, Vf, swap=True):
    kw = dict(SW) if swap else {}
    w = (0.03, 0.12, 0.04) if W else ((0.02, 0.08, 0.04) if swap else (0.02, 0.08, 0.06))
    cfg = G.GConfig(cap=cap, weights=w, **kw)
    if E:
        cfg = replace(cfg, exit_rule=EXIT)
    if I:
        cfg = replace(cfg, idx="sh000300", reserve=0.3, regime=REG, rebal="daily")
    if Vf:
        cfg = replace(cfg, mask=V60 >= 0.38)
    return cfg


EVALS = [("A", (), False), ("B", (), False), ("F", (), False), ("B", ("2024-02",), False),
         ("F", ("2024-02",), False), ("B", (), True), ("F", (), True), ("A", (), True)]


def job(a):
    swap, cap, W, E, I, Vf = a
    base = build(cap, W, E, I, Vf, swap)
    row = dict(swap=swap, cap=cap, W=W, E=E, I=I, V=Vf)
    for win, drop, real in EVALS:
        m = G.run_seeds(D, G.win_cfg(base, win, drop, realistic=real, D=D))
        t = win + ("n" if drop else "") + ("r" if real else "")
        for k in ("final", "cagr", "max_dd", "dd_p10", "dd_worst", "cagr_p10", "cagr_p90", "final_p10", "final_p90",
                  "avg_exposure", "avg_idx", "n_trades", "n_panic", "n_swap"):
            row[f"{t}_{k}"] = m[k]
    return row


if __name__ == "__main__":
    grid = [(True,) + g for g in itertools.product([0.9, 1.0], [False, True], [False, True], [False, True], [False, True])]
    grid += [(False, c, False, E, I, Vf) for c in (0.9, 1.0) for E in (False, True) for I in (False, True) for Vf in (False, True)]
    with Pool(3) as p:
        rows = p.map(job, grid, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "addons.csv"), index=False)
    print(len(df), "rows")
