"""v_c_exit 验证用：独立实现的移动止盈出场（不引用 c_exit_lib），接到原 engine.run 上。

rule dict:
  {'kind':'fixed','tp':.4,'sl':.4}
  {'kind':'trail','act':a,'trail':w,'sl':sl,'tp':None}
  {'kind':'trail2','act':a,'trail':w,'act2':a2,'trail2':w2,'sl':sl,'tp':None}
逻辑（逐日循环写法，刻意与 c_exit_lib 的向量化写法不同）：
  买入日 e（与 engine.exits 同样的 lag 顺延），c0 = close[e]；从 e+1 起每日：
    r = close[t]/c0；peak = max(1, 历史 r 最大值, 含当日)；r >= 1+act 后永久激活；
    若当日有成交：r>=1+tp 或 r<=1-sl 或 (激活 且 r <= peak*(1-宽度)) -> 触发。
    宽度 = trail2 若 peak >= 1+act2 否则 trail。
  xlag>0：触发后第 xlag 个有成交日的收盘才卖（窗口末截断）。
engine.run 卖出时用 close[s, x]，所以只需要 x_arr。
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Data, Config, run, END_DATE  # noqa: E402

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
LOCO = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]
NaN = np.nan


@njit(cache=True)
def _exit_one(close, traded, e, end_t, tp, sl, act, trail, act2, trail2, xlag):
    c0 = close[e]
    peak = 1.0
    on = False
    x = end_t
    for t in range(e + 1, end_t + 1):
        r = close[t] / c0
        if r > peak:
            peak = r
        if (not np.isnan(act)) and r >= 1.0 + act:
            on = True
        if not traded[t]:
            continue
        hit = False
        if (not np.isnan(tp)) and r >= 1.0 + tp:
            hit = True
        if (not np.isnan(sl)) and r <= 1.0 - sl:
            hit = True
        if on:
            w = trail
            if (not np.isnan(act2)) and peak >= 1.0 + act2:
                w = trail2
            if r <= peak * (1.0 - w):
                hit = True
        if hit:
            x = t
            break
    if xlag > 0 and x < end_t:
        k = 0
        y = x
        while y < end_t and k < xlag:
            y += 1
            if traded[y]:
                k += 1
        x = y
    return x


def _f(v):
    return NaN if v is None else float(v)


_CACHE = {}


def my_exits(D, rule, end_t, lag=0, xlag=0):
    key = (tuple(sorted(rule.items())), end_t, lag, xlag)
    if key in _CACHE:
        return _CACHE[key]
    kind = rule.get("kind", "fixed")
    tp, sl = _f(rule.get("tp")), _f(rule.get("sl"))
    act = _f(rule.get("act")) if kind in ("trail", "trail2") else NaN
    trail = _f(rule.get("trail", 0.0)) if kind in ("trail", "trail2") else 0.0
    act2 = _f(rule.get("act2")) if kind == "trail2" else NaN
    trail2 = _f(rule.get("trail2", 0.0)) if kind == "trail2" else 0.0
    S = len(D.entry)
    e_arr = np.empty(S, np.int64)
    x_arr = np.empty(S, np.int64)
    ratio = np.empty(S)
    for s in range(S):
        e = int(D.entry[s]) + lag
        if e > end_t:
            e_arr[s], x_arr[s], ratio[s] = e, e, 1.0
            continue
        while lag and e < end_t and not D.traded[s, e]:
            e += 1
        x = _exit_one(D.close[s], D.traded[s], e, end_t, tp, sl, act, trail, act2, trail2, xlag)
        e_arr[s], x_arr[s] = e, x
        ratio[s] = D.close[s, x] / D.close[s, e]
    out = (e_arr, x_arr, ratio)
    _CACHE[key] = out
    return out


def patched(D, rule, xlag=0):
    Dx = copy.copy(D)

    def _exits(tp=None, sl=None, max_hold=None, end_t=None, lag=0):
        end_t = len(D.dates) - 1 if end_t is None else end_t
        return my_exits(D, rule, end_t, lag, xlag)
    Dx.exits = _exits
    return Dx


def month_mask(D, months):
    ym = D.sig.signal_date.str[:7].values
    return ~np.isin(ym, list(months))


def cfg_for(win, D=None, drop=(), **kw):
    s, e = WIN[win]
    c = Config(start=s, end=e, **kw)
    if drop:
        c = replace(c, mask=month_mask(D, drop))
    return c


def seeds_stats(D, rule, cfg, seeds=range(80), xlag=0):
    Dx = patched(D, rule, xlag)
    rows = [run(Dx, replace(cfg, record=False), s) for s in seeds]
    f = np.array([r["final"] for r in rows])
    c = np.array([r["cagr"] for r in rows])
    dd = np.array([r["max_dd"] for r in rows])
    ex = np.array([r["avg_exposure"] for r in rows])
    n = np.array([r["n_trades"] for r in rows])
    npan = np.array([r["n_by_tier"][2] for r in rows])
    return dict(final=np.median(f) / 1e4, cagr=np.median(c) * 100, dd=np.median(dd) * 100,
                c10=np.quantile(c, .1) * 100, c90=np.quantile(c, .9) * 100,
                dd_worst=dd.min() * 100, dd_p10=np.quantile(dd, .1) * 100, exp=np.median(ex) * 100,
                n=np.median(n), npan=np.median(npan), finals=f, cagrs=c)


# ------------------------------------------------------------- 并行
_D = None


def getD():
    global _D
    if _D is None:
        _D = Data.load()
    return _D


def _job(j):
    name, rule, win, drop, ns, kw, xlag = j
    D = getD()
    cfg = cfg_for(win, D, drop, **kw)
    m = seeds_stats(D, rule, cfg, range(ns), xlag)
    m.pop("finals"); m.pop("cagrs")
    return dict(name=name, win=win, drop="|".join(drop), xlag=xlag, **{k: kw.get(k) for k in ("lag", "slip")}, **m)


def evaluate(jobs, procs=3):
    """jobs: (name, rule, win, drop, nseeds, cfg_kw, xlag)"""
    D = getD()
    for name, rule, win, drop, ns, kw, xlag in jobs:   # 父进程预算出场，fork 共享
        t1 = D.day_le(WIN[win][1])
        my_exits(D, rule, t1, kw.get("lag", 0), xlag)
    if procs <= 1:
        return pd.DataFrame([_job(j) for j in jobs])
    from multiprocessing import Pool
    with Pool(procs) as p:
        return pd.DataFrame(p.map(_job, jobs, chunksize=max(1, len(jobs) // (procs * 6))))


def R(kind="trail", **kw):
    return {"kind": kind, **kw}


CANDS = {
    "base": {"kind": "fixed", "tp": 0.4, "sl": 0.4},
    "C3": {"kind": "trail2", "act": 0.5, "trail": 0.075, "act2": 1.0, "trail2": 0.3, "sl": 0.7, "tp": None},
    "C1": {"kind": "trail", "act": 0.5, "trail": 0.1, "sl": 0.7, "tp": None},
    "C2": {"kind": "trail", "act": 0.45, "trail": 0.05, "sl": 0.4, "tp": None},
    "C4": {"kind": "trail", "act": 1.0, "trail": 0.3, "sl": 0.7, "tp": None},
}
