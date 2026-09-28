"""h_critic_: 基准对照 + 瞬时冲击压力。

A 指数买入持有 / 纯沪深300 MA120 择时（前一日判断，当日收盘成交，关时现金 1.8%）在各窗口的年化与回撤，
  用来回答"利润最大化"最朴素的对照：不做信号、只做指数，能拿到多少。
B 瞬时冲击：逐日假设股票持仓瞬间 -37%、指数仓位瞬间 -25%，求最坏一天的回撤（20 种子中位）。
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import g_combined_engine as G
from d_idle_engine import idx_price, reg_trend_h
from g_combined_plans import make, D

G.WIN["B1"] = ("2016-01-01", "2020-12-31")
G.WIN["B2"] = ("2021-01-01", G.END_DATE)
G.WIN["A2"] = ("2009-01-01", "2016-01-28")


def stats(eq, t0, t1):
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr * 100, dd * 100


def timing(name, t0, t1, ma=120, fee=0.001, cash=0.018):
    P = idx_price(D, name)
    reg = reg_trend_h(D, name, ma=ma, h=0.0, lag=1)
    eq = np.empty(t1 - t0 + 1)
    v, on = 1.0, False
    for k, t in enumerate(range(t0, t1 + 1)):
        if k:
            dt = D.cal_days[t] - D.cal_days[t - 1]
            v = v * (P[t] / P[t - 1]) if on else v * (1 + cash) ** (dt / 365)
        want = bool(reg[t]) and t < t1
        if want != on:
            v *= 1 - fee
            on = want
        eq[k] = v
    return eq


if __name__ == "__main__":
    rows = []
    for w in ("A", "A2", "B", "B1", "B2", "F"):
        t0, t1 = D.day(G.WIN[w][0]), D.day_le(G.WIN[w][1])
        for nm in ("sh000300", "sh000905", "sh000852", "g2000p", "sh000001"):
            P = idx_price(D, nm)
            if nm != "g2000p" and np.isnan(D.index[nm].values[t0]):
                continue
            c, dd = stats(P[t0 : t1 + 1], t0, t1)
            rows.append(dict(win=w, what=f"买入持有 {nm}", cagr=c, dd=dd))
        for nm in ("sh000300", "g2000p"):
            c, dd = stats(timing(nm, t0, t1), t0, t1)
            rows.append(dict(win=w, what=f"纯择时 {nm} MA120", cagr=c, dd=dd))
    B = pd.DataFrame(rows)
    print("== 基准：年化% / 最大回撤%（价格指数，不含分红）")
    print(B.pivot(index="what", columns="win", values="cagr").round(2).to_string())
    print(B.pivot(index="what", columns="win", values="dd").round(1).to_string())

    print("\n== 瞬时冲击：股票持仓 -37% / 指数 -25%，最坏一天的冲击后回撤（20 种子中位；括号 = 最差种子）；另列 盯市最大仓位")
    PL = {"base": make(), "P1": make(S=True), "P2": make(S=True, C=True), "P3": make(S=True, C=True, I=True, res=0.0),
          "K1": make(C=True)}
    out = []
    for w in ("B", "F"):
        for nm, cfg in PL.items():
            vals, mx, dds = [], [], []
            for seed in range(20):
                r = G.run(D, replace(G.win_cfg(cfg, w), record=True), seed)
                eq = r["equity"].values
                pos = r["pos_value"].values
                iv = r["idx_value"].values
                pk = np.maximum.accumulate(eq)
                sh = (eq - 0.37 * pos - 0.25 * iv) / pk - 1
                vals.append(sh.min())
                mx.append(((pos + iv) / eq).max())
                dds.append(r["max_dd"])
            out.append(dict(win=w, plan=nm, shock_dd=np.median(vals) * 100, shock_worst=np.min(vals) * 100,
                            max_expo=np.median(mx) * 100, dd=np.median(dds) * 100))
    print(pd.DataFrame(out).round(1).to_string())
