"""独立复核用引擎（v_b_panic_ 任务）：从 analysis/engine.py 的 run() 复制，独立重写“恐慌档被仓位上限挡住时换仓”。
不引用 b_panic_engine 的任何代码，用来交叉验证其结果。

VConfig 新增字段:
  swap        True 时启用：恐慌档新信号放不下（金额 > 仓位上限余量）时，按 sell_key 顺序卖出持仓，最多 max_n 只，
              事先模拟确认卖完能买才真正卖；卖完仍不够就一只不卖、也不买。
  max_n       一次最多卖几只
  keep_panic  True 时不卖已持有的恐慌档股票
  sell_key    'oldest'（买入日最早）| 'newest' | 'underwater'（浮亏最深）| 'random'（随机，安慰剂）
  tie         同一买入日的并列：'insert'（按买入执行顺序）| 'sidx'（按信号编号）| 'rev'（倒序）
  no_lev      True：买入金额+费用不得超过现金；超出但 >=95% 时缩小到现金可买，否则不买（与 b_panic 同语义）
              'strict'：超出就不买；False：同 engine（允许手续费造成的微小融资）
  swap_min_n30  只有新恐慌信号 n30 >= 该值才换仓
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine import Data, Config, END_DATE, _interest_index  # noqa: E402

WA = ("2006-01-04", "2016-01-28")
WB = ("2016-01-01", END_DATE)
WF = ("2006-01-04", END_DATE)
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]


@dataclass
class VConfig(Config):
    swap: bool = False
    max_n: int = 5
    keep_panic: bool = True
    sell_key: str = "oldest"
    tie: str = "insert"
    no_lev: object = True
    swap_min_n30: int = 0
    swap_tiers: tuple = (2,)
    sell_block: object = None      # (S,T) bool：True 的持仓当日不能被换仓卖出（如收盘封死跌停）
    swap_mask: object = None       # (S,) bool：只有该新信号为 True 时才允许换仓（按簇拆分用）


def run(D: Data, cfg: VConfig, seed: int = 0):
    t0, t1 = D.day(cfg.start), D.day_le(cfg.end)
    e_arr, x_arr, ratio = D.exits(cfg.tp, cfg.sl, cfg.max_hold, t1, cfg.lag)
    tiers = D.tier if cfg.tier_override is None else cfg.tier_override
    sel = (D.entry >= t0) & (e_arr <= t1)
    if cfg.mask is not None:
        sel &= cfg.mask
    idx = np.nonzero(sel)[0]
    rng = np.random.default_rng(seed)
    order = idx[np.lexsort((rng.random(len(idx)), e_arr[idx]))]
    rng2 = np.random.default_rng(10_000 + seed)   # 仅供 random 卖出对象

    I_cash = _interest_index(D, t0, cfg.cash_rate)
    I_borr = _interest_index(D, t0, cfg.borrow_rate)
    cash_pos = cfg.capital / I_cash[t0]
    cash_neg = 0.0
    held = {}
    seq = {}
    cost_sum = 0.0
    heap = []
    trades = []
    flows = []
    close = D.close
    nswap = 0
    nfail = 0

    def cash_at(t):
        return cash_pos * I_cash[t] - cash_neg * I_borr[t]

    def add_cash(t, v):
        nonlocal cash_pos, cash_neg
        if v >= 0:
            debt = cash_neg * I_borr[t]
            pay = min(v, debt)
            cash_neg -= pay / I_borr[t]
            cash_pos += (v - pay) / I_cash[t]
        else:
            avail = cash_pos * I_cash[t]
            use = min(-v, avail)
            cash_pos -= use / I_cash[t]
            cash_neg += (-v - use) / I_borr[t]
        flows.append((t, v))

    def sell(s, t, kind):
        nonlocal cost_sum
        bt, amount, units = held.pop(s)
        value = units * close[s, t] * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= amount
        trades.append((s, bt, t, amount, value - amount * (1 + cfg.fee), kind))

    def process_exits(upto):
        while heap and heap[0][0] <= upto:
            xt, s = heapq.heappop(heap)
            if s in held and held[s][0] <= xt:
                r = ratio[s]
                kind = "TP" if cfg.tp is not None and r >= 1 + cfg.tp else (
                    "SL" if cfg.sl is not None and r <= 1 - cfg.sl else ("END" if xt == t1 else "TIME"))
                sell(s, xt, kind)

    k_ins = 0
    for s in order:
        t = int(e_arr[s])
        process_exits(t)
        assert cfg.sizing == "cost" and cfg.cap_basis == "cost"
        cash = cash_at(t)
        equity = cash + cost_sum
        tier = int(tiers[s])
        w = cfg.weights[tier]
        if w <= 0:
            continue
        if tier == 2 and cfg.panic_max_pos is not None and len(held) >= cfg.panic_max_pos:
            continue
        amount = w * equity
        room = cfg.cap * equity - cost_sum
        if (amount > room and cfg.swap and tier in cfg.swap_tiers and D.n30[s] >= cfg.swap_min_n30
                and (cfg.swap_mask is None or cfg.swap_mask[s])):
            pool = [h for h in held if held[h][0] < t and not (cfg.keep_panic and tiers[h] == 2)
                    and not (cfg.sell_block is not None and cfg.sell_block[h, t])]
            if cfg.tie == "insert":
                pool.sort(key=lambda h: seq[h])
            elif cfg.tie == "sidx":
                pool.sort(key=lambda h: h)
            elif cfg.tie == "rev":
                pool.sort(key=lambda h: -seq[h])
            if cfg.sell_key == "oldest":
                pool.sort(key=lambda h: held[h][0])
            elif cfg.sell_key == "newest":
                pool.sort(key=lambda h: -held[h][0])
            elif cfg.sell_key == "underwater":
                pool.sort(key=lambda h: held[h][2] * close[h, t] / held[h][1])
            elif cfg.sell_key == "random":
                pool = list(rng2.permutation(pool)) if pool else pool
            c_cash, c_cost, plan, ok = cash, cost_sum, [], False
            for h in pool[: cfg.max_n]:
                _, a, u = held[h]
                c_cash += u * close[h, t] * (1 - cfg.fee - cfg.slip)
                c_cost -= a
                plan.append(h)
                e2 = c_cash + c_cost
                if w * e2 <= cfg.cap * e2 - c_cost:
                    ok = True
                    break
            if not ok:
                nfail += 1
                continue
            for h in plan:
                sell(h, t, "SWAP")
                nswap += 1
            cash = cash_at(t)
            equity = cash + cost_sum
            amount = w * equity
            room = cfg.cap * equity - cost_sum
        if amount > room:
            if not cfg.partial:
                continue
            amount = room
        if cfg.no_lev:
            lim = cash / (1 + cfg.fee)
            if amount > lim:
                if cfg.no_lev == "strict" or lim < 0.95 * amount:
                    continue
                amount = lim
        if amount < cfg.min_amount:
            continue
        units = amount / (close[s, t] * (1 + cfg.slip))
        add_cash(t, -amount * (1 + cfg.fee))
        held[s] = (t, amount, units)
        seq[s] = k_ins
        k_ins += 1
        cost_sum += amount
        heapq.heappush(heap, (int(x_arr[s]), s))
    process_exits(t1)
    for s in list(held):
        sell(s, t1, "END")

    final = cash_at(t1)
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    res = {"final": final, "cagr": (final / cfg.capital) ** (1 / years) - 1, "n_trades": len(trades)}
    T = t1 - t0 + 1
    pos = np.zeros(T)
    cost_curve = np.zeros(T)
    for s, bt, st, amount, pnl, kind in trades:
        u = amount / (close[s, bt] * (1 + cfg.slip))
        pos[bt - t0: st - t0] += u * close[s, bt:st]
        cost_curve[bt - t0: st - t0] += amount
    cp = np.zeros(T)
    cn = np.zeros(T)
    c_pos, c_neg = cfg.capital / I_cash[t0], 0.0
    flows.sort(key=lambda f: f[0])
    fi = 0
    for k in range(T):
        t = t0 + k
        while fi < len(flows) and flows[fi][0] == t:
            v = flows[fi][1]
            if v >= 0:
                debt = c_neg * I_borr[t]
                pay = min(v, debt)
                c_neg -= pay / I_borr[t]
                c_pos += (v - pay) / I_cash[t]
            else:
                avail = c_pos * I_cash[t]
                use = min(-v, avail)
                c_pos -= use / I_cash[t]
                c_neg += (-v - use) / I_borr[t]
            fi += 1
        cp[k], cn[k] = c_pos, c_neg
    cash_curve = cp * I_cash[t0: t1 + 1] - cn * I_borr[t0: t1 + 1]
    eq = cash_curve + pos
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    res["max_dd"] = dd.min()
    res["avg_exposure"] = float(np.mean(pos / eq))
    res["min_cash"] = float(cash_curve.min())
    res["n_swap"] = nswap
    res["n_fail"] = nfail
    res["n_panic"] = int(sum(1 for tr in trades if tiers[tr[0]] == 2))
    if cfg.record:
        res["equity"] = pd.Series(eq, index=D.dates[t0: t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t], tier=lambda d: tiers[d.s])
    return res


def limit_down(D, loose=False):
    """(S,T) 收盘封死跌停：日跌幅达到涨跌停限制（主板 10%，ST 5%，创业板 2020-08-24 起 20%，科创板 20%）且收盘=最低。
    loose=True：只要日跌幅 >= 限制的 95% 就算卖不掉（不要求收盘=最低）。"""
    c = D.close
    r = np.full(c.shape, np.nan)
    r[:, 1:] = c[:, 1:] / c[:, :-1] - 1
    code = D.sig.code.values
    name = D.sig.name.fillna("").values
    lim = np.full(c.shape, 0.10)
    cy = np.array([x.startswith(("300", "301")) for x in code])
    kc = np.array([x.startswith("688") for x in code])
    st = np.array(["ST" in x for x in name])
    t20 = int(np.searchsorted(D.dates, "2020-08-24"))
    lim[cy, t20:] = 0.20
    lim[kc, :] = 0.20
    lim[st, :] = 0.05
    traded = D.traded
    if loose:
        out = (r <= -lim * 0.95 + 1e-9) & traded
    else:
        low = D.low
        out = (r <= -lim + 0.004) & traded & (np.abs(c / low - 1) < 1e-4)
    return out


def run_seeds(D, cfg, seeds=range(80)):
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame(rows)
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    out["dd_worst"] = df.max_dd.min()
    out["cagr_min"] = df.cagr.min()
    out["_cagr_all"] = df.cagr.values
    return out


_M = {}


def masks(D):
    if not _M:
        ym = D.sig.signal_date.str[:7].values
        _M["ym"] = ym
        for c in BIG:
            _M["no" + c] = ym != c
    return _M
