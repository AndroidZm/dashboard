"""v_f_lever: 对抗性复核 f_lever（杠杆/整体放大）用的小工具。只调用 analysis/engine.py，不改它。

per_seed(): 逐种子跑 engine.run（record=True），并独立地从成交记录重算
  盯市持仓市值 pos、负债 debt=max(-(eq-pos),0)、维持担保比例 (pos+正现金)/debt、盯市总仓位 pos/eq。
summ(): 80 种子中位 + p10/p90 年化 + 回撤 p10/最差 + 最低维持担保比例等。
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Data, Config, run, END_DATE  # noqa: E402

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]


def mmask(D, months):
    ym = D.sig.signal_date.str[:7].values
    return ~np.isin(ym, list(months))


def Lcfg(win, L, **kw):
    s, e = WIN[win]
    base = dict(start=s, end=e, weights=(0.02 * L, 0.08 * L, 0.06 * L), cap=0.9 * L)
    base.update(kw)
    return Config(**base)


def one(D, cfg, seed, detail=False):
    if not detail:
        r = run(D, replace(cfg, record=False), seed)
        return dict(seed=seed, final=r["final"], cagr=r["cagr"], max_dd=r["max_dd"], n=r["n_trades"],
                    avg_exp=r["avg_exposure"], max_exp=r["max_exposure"], dd_date=r["dd_date"])
    r = run(D, replace(cfg, record=True), seed)
    eq = r["equity"].values
    t0 = D.day(cfg.start)
    T = len(eq)
    pos = np.zeros(T)
    for s, bt, st, amt in r["trades"][["s", "buy_t", "sell_t", "amount"]].itertuples(index=False):
        u = amt / (D.close[s, bt] * (1 + cfg.slip))
        pos[bt - t0: st - t0] += u * D.close[s, bt:st]
    cash = eq - pos
    debt = np.maximum(-cash, 0)
    posc = np.maximum(cash, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = np.where(debt > 1.0, (pos + posc) / np.maximum(debt, 1e-9), np.inf)
    k = int(np.argmin(mr))
    return dict(seed=seed, final=r["final"], cagr=r["cagr"], max_dd=r["max_dd"], n=r["n_trades"],
                avg_exp=r["avg_exposure"], max_exp=r["max_exposure"], dd_date=r["dd_date"],
                min_mr=float(mr.min()), mr_date=D.dates[t0 + k] if np.isfinite(mr.min()) else "",
                gross_at_min_mr=float(pos[k] / eq[k]) if np.isfinite(mr.min()) else np.nan,
                max_debt_eq=float(np.max(debt / eq)), days_borrow=float(np.mean(debt > 1.0)),
                min_eq=float(eq.min() / cfg.capital))


def per_seed(D, cfg, seeds=range(80), detail=False):
    return pd.DataFrame([one(D, cfg, s, detail) for s in seeds])


def summ(df):
    o = dict(final=df.final.median(), cagr=df.cagr.median(), max_dd=df.max_dd.median(),
             cagr_p10=df.cagr.quantile(0.1), cagr_p90=df.cagr.quantile(0.9),
             dd_p10=df.max_dd.quantile(0.1), dd_worst=df.max_dd.min(), n=df.n.median(),
             avg_exp=df.avg_exp.median(), max_exp=df.max_exp.median())
    o["calmar"] = o["cagr"] / abs(o["max_dd"])
    if "min_mr" in df:
        o["min_mr_med"] = df.min_mr.median()
        o["min_mr_worst"] = df.min_mr.min()
        o["frac_mr_lt130"] = float((df.min_mr < 1.3).mean())
        o["max_debt_eq_med"] = df.max_debt_eq.median()
        o["min_eq_med"] = df.min_eq.median()
    return o


def fmt(o):
    return (f"{o['final']/1e4:7.1f}万 {o['cagr']*100:5.2f}% dd {o['max_dd']*100:6.1f}% "
            f"(p10 {o['dd_p10']*100:6.1f} worst {o['dd_worst']*100:6.1f}) cagr p10/p90 "
            f"{o['cagr_p10']*100:5.2f}/{o['cagr_p90']*100:5.2f}")
