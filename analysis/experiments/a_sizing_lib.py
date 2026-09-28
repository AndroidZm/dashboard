"""a_sizing 公用工具：窗口、剔簇 mask、并行网格评估。"""
from __future__ import annotations

import itertools
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from a_sizing_engine import SConfig, run, run_seeds, Data, END_DATE  # noqa: E402

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]

_D = None


def D():
    global _D
    if _D is None:
        _D = Data.load()
    return _D


def month_mask(months_out):
    ym = D().sig.signal_date.str[:7].values
    return ~np.isin(ym, list(months_out))


def cfg_for(params: dict, win: str, drop: tuple = ()):
    p = dict(params)
    start, end = WIN[win]
    p["start"], p["end"] = start, end
    if p.get("sizing") == "mtm":
        p.setdefault("cap_basis", "mtm")
    if drop:
        p["mask"] = month_mask(drop)
    return SConfig(**p)


def _eval(args):
    params, win, drop, nseeds = args
    m = run_seeds(D(), cfg_for(params, win, drop), seeds=range(nseeds))
    return {**{k: (str(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "win": win,
            "drop": "|".join(drop), "final": m["final"], "cagr": m["cagr"], "max_dd": m["max_dd"],
            "dd_p10": m["dd_p10"], "cagr_p10": m["cagr_p10"], "cagr_p90": m["cagr_p90"],
            "avg_exp": m["avg_exposure"], "max_exp": m["max_exposure"], "n": m["n_trades"],
            "n0": m["n0"], "n1": m["n1"], "n2": m["n2"], "vol": m["vol"], "min_cash": m["min_cash_frac"]}


def grid(param_list, wins=("B",), drops=((),), nseeds=20, procs=3):
    jobs = [(p, w, d, nseeds) for p in param_list for w in wins for d in drops]
    D()
    if procs <= 1:
        rows = [_eval(j) for j in jobs]
    else:
        with Pool(procs) as pool:
            rows = pool.map(_eval, jobs, chunksize=max(1, len(jobs) // (procs * 8)))
    return pd.DataFrame(rows)


def product(**kw):
    keys = list(kw)
    return [dict(zip(keys, vals)) for vals in itertools.product(*kw.values())]


def fmt(r):
    return f"{r['final']/1e4:7.1f}万 {r['cagr']*100:6.2f}% dd{r['max_dd']*100:6.1f}% exp{r['avg_exp']*100:4.0f}%"
