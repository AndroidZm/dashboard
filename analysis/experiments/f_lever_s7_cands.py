"""s7: 候选点复核（80 种子）—— 剔簇、执行口径、盯市口径、逐年收益、融资路径指标。

候选：L=1.0(基线) / 1.1 / 1.4 / 1.5 / 2.2（cap=0.9L）+ 只提上限 cap=1.0。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import BIG, Data, cfg_win, month_mask, run, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "f_lever_out")
D = Data.load()
pd.set_option("display.width", 250)
CANDS = {"base L1.0": dict(k=1.0, cap=0.9), "L1.1": dict(k=1.1, cap=0.99), "cap1.0": dict(k=1.0, cap=1.0),
         "L1.4": dict(k=1.4, cap=1.26), "L1.5": dict(k=1.5, cap=1.35), "L2.2": dict(k=2.2, cap=1.98)}
VARS = {"主口径 融资5%": dict(borrow_rate=0.05), "融资8%": dict(borrow_rate=0.08),
        "次日收盘+0.5%滑点 融资8%": dict(borrow_rate=0.08, lag=1, slip=0.005),
        "盯市定额+盯市上限 融资8%": dict(borrow_rate=0.08, sizing="mtm", cap_basis="mtm"),
        "强平130%→150% 融资8%": dict(borrow_rate=0.08, margin_call=(1.3, 1.5))}
rows = []
for cn, c in CANDS.items():
    for vn, v in VARS.items():
        for w in "ABF":
            m = run_seeds(D, cfg_win(w, **c, **v), range(80))
            rows.append(dict(cand=cn, var=vn, win=w, drop="", **m))
    for w in "ABF":
        drops = ["2024-02"] + BIG if w != "A" else ["2012-01"]
        for dr in dict.fromkeys(drops):
            if w == "B" and dr == "2012-01":
                continue
            m = run_seeds(D, cfg_win(w, **c, borrow_rate=0.05, mask=month_mask(D, [dr])), range(80))
            rows.append(dict(cand=cn, var="剔簇 融资5%", win=w, drop=dr, **m))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "s7_cands.csv"), index=False)
df["txt"] = (df.final / 1e4).round(1).astype(str) + "/" + (df.cagr * 100).round(2).astype(str) + "/" + (df.max_dd * 100).round(1).astype(str)
for w in "BAF":
    print(f"=== 窗口 {w}：期末万/年化%/中位回撤%")
    x = df[(df.win == w) & (df["drop"] == "")]
    print(x.pivot_table(index="var", columns="cand", values="txt", aggfunc="first")[list(CANDS)].to_string())
    x = df[(df.win == w) & (df["drop"] != "")]
    print(x.pivot_table(index="drop", columns="cand", values="txt", aggfunc="first")[list(CANDS)].to_string())
print("=== 融资路径指标（主口径 融资5%）")
x = df[(df["var"] == "主口径 融资5%")]
print(x.pivot_table(index="cand", columns="win", values=["avg_exposure", "max_exposure", "max_debt_eq", "avg_debt_eq",
                                                          "days_borrow", "min_mr_worst", "dd_worst", "min_eq"]).round(2)
      .loc[list(CANDS)].to_string())

print("=== 逐年收益（全期 F，seed 13，融资 5%）")
yr = {}
for cn, c in CANDS.items():
    r = run(D, cfg_win("F", **c, borrow_rate=0.05, record=True), 13)
    eq = r["equity"]
    eq.index = pd.to_datetime(eq.index)
    y = eq.resample("YE").last()
    y = pd.concat([pd.Series([600000.0], index=[pd.Timestamp("2005-12-31")]), y]).pct_change().dropna() * 100
    yr[cn] = y.round(1)
Y = pd.DataFrame(yr)
Y.index = Y.index.year
print(Y.to_string())
print("亏损年数", (Y < 0).sum().to_dict())
print("最差年", Y.min().to_dict())
