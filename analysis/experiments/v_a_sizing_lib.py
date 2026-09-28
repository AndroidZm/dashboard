"""v_a_sizing: 对抗性复核 a_sizing 结论用的公用工具（只用 analysis/engine.py 原引擎，不用对方的 a_sizing_engine 副本）。"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Data, Config, run, END_DATE  # noqa: E402

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]
_D = None


def D():
    global _D
    if _D is None:
        _D = Data.load()
    return _D


BASE = dict(weights=(0.02, 0.08, 0.06), cap=0.9, panic_max_pos=15, partial=False)
C1 = dict(weights=(0.02, 0.15, 0.06), cap=1.0, panic_max_pos=None, partial=True)
C1b = dict(weights=(0.02, 0.15, 0.06), cap=0.95, panic_max_pos=None, partial=True)
C2 = dict(weights=(0.0, 0.20, 0.06), cap=1.0, panic_max_pos=None, partial=True)
K1 = dict(weights=(0.02, 0.08, 0.06), cap=1.0, panic_max_pos=None, partial=True)
CANDS = {"BASE": BASE, "C1": C1, "C1b": C1b, "C2": C2, "K1": K1}


def mask_out(prefixes=(), years=()):
    sd = D().sig.signal_date.astype(str)
    m = np.ones(len(sd), bool)
    for p in list(prefixes) + [str(y) for y in years]:
        m &= ~sd.str.startswith(p).values
    return m


def cfg(params, win, mask=None, **kw):
    s, e = WIN[win]
    p = dict(params)
    p.update(kw)
    return Config(start=s, end=e, mask=mask, **p)


def seeds_raw(c: Config, seeds=range(80)):
    rows = []
    for sd in seeds:
        r = run(D(), replace(c, record=False), sd)
        rows.append((r["final"], r["cagr"], r["max_dd"], r["avg_exposure"], r["n_trades"], r["n_by_tier"][2]))
    return pd.DataFrame(rows, columns=["final", "cagr", "max_dd", "avg_exp", "n", "n2"])


def summ(df, base=None):
    out = dict(final=df.final.median() / 1e4, cagr=df.cagr.median() * 100, dd=df.max_dd.median() * 100,
               dd_p10=df.max_dd.quantile(0.1) * 100, dd_worst=df.max_dd.min() * 100,
               cagr_p10=df.cagr.quantile(0.1) * 100, cagr_p90=df.cagr.quantile(0.9) * 100,
               exp=df.avg_exp.median() * 100, n=df.n.median(), n2=df.n2.median())
    if base is not None:
        d = (df.cagr.values - base.cagr.values) * 100
        out.update(d_med=np.median(d), d_p10=np.percentile(d, 10), frac_pos=(d > 0).mean(),
                   d_medcagr=out["cagr"] - base.cagr.median() * 100, d_dd=out["dd"] - base.max_dd.median() * 100)
    return out


def fmt(o):
    s = f"{o['final']:7.1f}万 {o['cagr']:6.2f}% dd{o['dd']:6.1f}% (p10 {o['dd_p10']:6.1f}, worst {o['dd_worst']:6.1f}) cagr p10/p90 {o['cagr_p10']:5.2f}/{o['cagr_p90']:5.2f} exp{o['exp']:4.0f}% n{o['n']:5.1f} n2 {o['n2']:.0f}"
    if "d_med" in o:
        s += f" | dMedCAGR {o['d_medcagr']:+5.2f}pp paired {o['d_med']:+5.2f}pp (p10 {o['d_p10']:+5.2f}) pos {o['frac_pos']*100:3.0f}% dDD {o['d_dd']:+5.1f}"
    return s
