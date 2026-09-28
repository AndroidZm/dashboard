"""独立复核用引擎（v_d_idle_ 验证任务）：逐日循环重写“闲置现金买指数 + 均线开关”。

不复用 d_idle_engine 的任何函数，只用 engine.Data 的数据和 D.exits（已验证）。
- 每个交易日：计息 -> 按收盘卖出到期信号 -> 按种子顺序买入当日信号（缺现金先卖指数）-> 再平衡指数
- rebal='daily' 每天检查容差；rebal='event' 只在有信号买/卖、regime 切换、起始日检查（与 d_idle 口径一致）
- regime 自己算：指数收盘 > ma 日简单均线（含当日）；reg_lag=1 用前一日判断、今日收盘成交
- idx=None 时应与 engine.run 完全一致
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine import Data, END_DATE  # noqa: E402

A = ("2006-01-04", "2016-01-28")
B = ("2016-01-01", END_DATE)
F = ("2006-01-04", END_DATE)


def my_regime(D, name, ma, reg_lag=0, kind="sma"):
    p = pd.Series(D.index[name].values.astype(float)).ffill()
    m = p.rolling(ma, min_periods=ma).mean()
    on = (p > m).values
    if reg_lag:
        on = np.r_[np.zeros(reg_lag, bool), on[:-reg_lag]]
    return on


def run2(D, start, end=END_DATE, idx=None, regime=None, reserve=0.0, band=0.01, idx_fee=0.001,
         seed=0, mask=None, lag=0, slip=0.0, fee=0.001, rebal="daily", capital=600_000.0,
         weights=(0.02, 0.08, 0.06), cap=0.9, panic_max=15, min_amount=5000.0, cash_rate=0.018,
         idx_slip=0.0, record=False):
    t0 = int(np.searchsorted(D.dates, start))
    t1 = int(np.searchsorted(D.dates, end, side="right") - 1)
    e_arr, x_arr, ratio = D.exits(0.4, 0.4, None, t1, lag)
    sel = (D.entry >= t0) & (e_arr <= t1)
    if mask is not None:
        sel &= mask
    ids = np.nonzero(sel)[0]
    rng = np.random.default_rng(seed)
    order = ids[np.lexsort((rng.random(len(ids)), e_arr[ids]))]
    buys_by_day = {}
    for s in order:
        buys_by_day.setdefault(int(e_arr[s]), []).append(int(s))
    P = None if idx is None else pd.Series(D.index[idx].values.astype(float)).ffill().values
    on = None if idx is None else (np.ones(len(D.dates), bool) if regime is None else regime)
    Icash = (1 + cash_rate) ** ((D.cal_days - D.cal_days[t0] + 1) / 365.0)
    cash = capital
    iu = 0.0
    held = {}
    cost_sum = 0.0
    exits_by_day = {}
    trades = []
    T = t1 - t0 + 1
    cash_c = np.zeros(T)
    idx_c = np.zeros(T)
    n_idx = 0
    fi_b = idx_fee + idx_slip
    for k in range(T):
        t = t0 + k
        if k > 0:
            cash *= Icash[t] / Icash[t - 1]
        event = (k == 0)
        # 1) 卖出
        for s in exits_by_day.pop(t, []):
            if s in held:
                bt, amount, units = held.pop(s)
                v = units * D.close[s, t] * (1 - fee - slip)
                cash += v
                cost_sum -= amount
                trades.append((s, bt, t, amount, "X"))
                event = True
        # 2) 买入
        if t in buys_by_day:
            event = True
            for s in buys_by_day[t]:
                iv = iu * P[t] if P is not None else 0.0
                equity = cash + iv + cost_sum
                tier = int(D.tier[s])
                w = weights[tier]
                if tier == 2 and panic_max is not None and len(held) >= panic_max:
                    continue
                amount = w * equity
                room = cap * equity - cost_sum
                if amount > room or amount < min_amount:
                    continue
                need = amount * (1 + fee)
                if cash < need and iu > 0:
                    v = min((need - cash) / (1 - fi_b), iu * P[t])
                    iu -= v / P[t]
                    cash += v * (1 - fi_b)
                    n_idx += 1
                cash -= need
                units = amount / (D.close[s, t] * (1 + slip))
                held[s] = (t, amount, units)
                cost_sum += amount
                exits_by_day.setdefault(int(x_arr[s]), []).append(s)
        # 3) 指数再平衡
        if P is not None and t < t1:
            if k > 0 and on[t] != on[t - 1]:
                event = True
            if rebal == "daily" or event:
                iv = iu * P[t]
                if not on[t]:
                    if iu > 0:
                        cash += iv * (1 - fi_b)
                        iu = 0.0
                        n_idx += 1
                else:
                    eq = cash + iv + cost_sum
                    target = max(0.0, cash + iv - reserve * eq)
                    diff = target - iv
                    if abs(diff) > band * eq:
                        if diff > 0:
                            iu += diff / (1 + fi_b) / P[t]
                            cash -= diff
                        else:
                            v = min(-diff, iv)
                            iu -= v / P[t]
                            cash += v * (1 - fi_b)
                        n_idx += 1
        if t == t1:
            for s in list(held):
                bt, amount, units = held.pop(s)
                cash += units * D.close[s, t] * (1 - fee - slip)
                cost_sum -= amount
                trades.append((s, bt, t, amount, "END"))
            if P is not None and iu > 0:
                cash += iu * P[t] * (1 - fi_b)
                iu = 0.0
        assert cash > -1e-6, (t, cash)
        cash_c[k] = cash
        idx_c[k] = iu * P[t] if P is not None else 0.0
    pos = np.zeros(T)
    for s, bt, st, amount, kind in trades:
        u = amount / (D.close[s, bt] * (1 + slip))
        pos[bt - t0: st - t0] += u * D.close[s, bt:st]
    eq = cash_c + idx_c + pos
    dd = eq / np.maximum.accumulate(eq) - 1
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    res = dict(final=cash, cagr=(cash / capital) ** (1 / years) - 1, max_dd=dd.min(), n_trades=len(trades),
               avg_exposure=float(np.mean(pos / eq)), avg_idx=float(np.mean(idx_c / eq)), n_idx=n_idx,
               n_panic=sum(1 for tr in trades if D.tier[tr[0]] == 2))
    if record:
        res["eq"] = pd.Series(eq, index=D.dates[t0:t1 + 1])
    return res


def seeds2(D, seeds=range(80), **kw):
    rows = [run2(D, seed=s, **kw) for s in seeds]
    df = pd.DataFrame(rows)
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    out["dd_worst"] = df.max_dd.min()
    out["final_p10"] = df.final.quantile(0.1)
    return out


def month_mask(D, months):
    return ~np.isin(D.sig.signal_date.str[:7].values, list(months))
