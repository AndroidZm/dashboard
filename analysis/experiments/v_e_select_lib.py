"""v_e_select 验证工具：窗口定义、去簇、80 种子统计（含 p10/p90 年化、最差种子回撤）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis'); sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from dataclasses import replace
import numpy as np, pandas as pd
from engine import Data, Config, run, run_seeds

WIN = {'A': ('2006-01-04', '2016-01-28'), 'B': ('2016-01-01', '2026-08-21'), 'F': ('2006-01-04', '2026-08-21')}
BIG = ['2012-01', '2018-02', '2022-05', '2022-10', '2024-02']
_D = None
def data():
    global _D
    if _D is None:
        _D = Data.load()
    return _D

VOL60 = np.load('/home/user/dashboard/analysis/experiments/v_e_select_out/vol60_theirs.npy')
VI = dict(zip((20, 40, 50, 60, 80, 120), np.load('/home/user/dashboard/analysis/experiments/v_e_select_out/vols_indep.npy')))
W = lambda k: (0.02 * k, 0.08 * k, 0.06 * k)

def cfg_of(thr=None, k=1.0, cap=0.9, vol=None, **kw):
    v = VOL60 if vol is None else vol
    mk = None if thr is None else (v >= thr)
    return Config(mask=mk, weights=W(k), cap=cap, **kw)

def months_mask(drop):
    D = data()
    mo = D.sig.signal_date.str[:7].values
    return ~np.isin(mo, list(drop))

def stats(cfg, win='B', seeds=range(80), drop=()):
    D = data()
    a, b = WIN[win]
    c = replace(cfg, start=a, end=b)
    if drop:
        mm = months_mask(drop)
        c = replace(c, mask=mm if c.mask is None else (c.mask & mm))
    rows = [run(D, c, s) for s in seeds]
    cagr = np.array([r['cagr'] for r in rows]); dd = np.array([r['max_dd'] for r in rows]); fin = np.array([r['final'] for r in rows])
    return dict(final=np.median(fin) / 1e4, cagr=np.median(cagr) * 100, dd=np.median(dd) * 100,
                p10=np.quantile(cagr, .1) * 100, p90=np.quantile(cagr, .9) * 100, worst_dd=dd.min() * 100,
                n=np.median([r['n_trades'] for r in rows]), exp=np.median([r['avg_exposure'] for r in rows]) * 100,
                npanic=np.median([r['n_by_tier'][2] for r in rows]), cagr_all=cagr)

def fmt(m):
    return f"{m['final']:7.1f}万 {m['cagr']:6.2f}% DD{m['dd']:6.1f}% p10/p90 {m['p10']:5.2f}/{m['p90']:5.2f} worstDD{m['worst_dd']:6.1f} n{m['n']:4.0f} exp{m['exp']:3.0f}% pan{m['npanic']:3.0f}"
