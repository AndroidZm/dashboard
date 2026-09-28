"""e_select 组合层面评估工具：同一规则在 A / B 窗口、去掉 2024-02、逐个去掉大簇 的结果（中位数）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis')
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from dataclasses import replace
import numpy as np
import pandas as pd
from engine import Data, Config, run_seeds, run

WIN = {'A': ('2006-01-04', '2016-01-28'), 'B': ('2016-01-01', '2026-08-21'), 'F': ('2006-01-04', '2026-08-21')}
BIG = ['2012-01', '2018-02', '2022-05', '2022-10', '2024-02']
_D = None


def data():
    global _D
    if _D is None:
        _D = Data.load()
    return _D


def month_mask(D, month):
    return D.sig.signal_date.str[:7].values != month


def ev(cfg: Config, win='B', seeds=range(20), drop_month=None):
    D = data()
    a, b = WIN[win]
    c = replace(cfg, start=a, end=b)
    if drop_month is not None:
        mm = month_mask(D, drop_month)
        c = replace(c, mask=mm if c.mask is None else (c.mask & mm))
    return run_seeds(D, c, seeds=seeds)


def line(m):
    return f"{m['final']/1e4:7.1f}万 {m['cagr']*100:5.2f}% DD{m['max_dd']*100:6.1f}% n{m['n_trades']:4.0f} exp{m['avg_exposure']*100:4.0f}%"


def full_report(cfg: Config, base: Config = None, seeds=range(80), label=''):
    """A、B、F、B去2024-02、LOCO(在 F 窗口上逐个去掉大簇) 相对基线的差值。"""
    base = base or Config()
    rows = []
    for win, drop in [('A', None), ('B', None), ('F', None), ('B', '2024-02')] + [('F', m) for m in BIG]:
        m1 = ev(cfg, win, seeds, drop)
        m0 = ev(base, win, seeds, drop)
        rows.append(dict(win=win, drop=drop or '-', final=m1['final'] / 1e4, cagr=m1['cagr'] * 100, dd=m1['max_dd'] * 100,
                         base_final=m0['final'] / 1e4, base_cagr=m0['cagr'] * 100, base_dd=m0['max_dd'] * 100,
                         d_cagr=(m1['cagr'] - m0['cagr']) * 100, d_dd=(m1['max_dd'] - m0['max_dd']) * 100))
    df = pd.DataFrame(rows)
    if label:
        print('#####', label)
    print(df.round(2).to_string(index=False), flush=True)
    return df
