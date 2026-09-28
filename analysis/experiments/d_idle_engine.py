"""闲置现金买指数 实验引擎（d_idle_ 任务专用）。

在 analysis/engine.py 的 run() 基础上复制并扩展；idx=None（或 regime 全 False）时与 engine.run 逐笔一致
（见 d_idle_check.py）。不修改 engine.py。

新增 IConfig 字段:
  idx        None | D.index 的列名 | 'g2000p'（国证2000，2009-12-31 前用中证1000 按比例接上）
  reserve    始终留作现金的比例（占总权益），高于它的闲置现金买指数
  idx_fee    指数每边费用（默认 0.1%）
  band       再平衡容差（占总权益）；现金偏离目标超过 band 才调整指数仓位
  regime     None 或 (T,) bool：当日收盘是否允许持有指数（False 当日收盘清掉指数）
  rebal      None | 'M'：是否每月第一个交易日额外做一次再平衡（无信号进出时也调整）
  idx_share  高于 reserve 的闲置现金中买指数的比例（1 = 全买）
口径:
  - 信号买入所需现金不足时，当日收盘卖指数补足（只卖差额）
  - 总权益（定仓用）= 现金 + 指数盯市 + 持仓成本；仓位上限 cap 只约束信号持仓（指数不占额度）
  - 每个有信号买卖的交易日收盘、档位/regime 切换日收盘做一次再平衡
  - 窗口末日指数按收盘卖出（扣费）
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine import Data, Config, END_DATE, _interest_index  # noqa: E402

WIN_A = ("2006-01-04", "2016-01-28")
WIN_B = ("2016-01-01", END_DATE)
WIN_F = ("2006-01-04", END_DATE)
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]


@dataclass
class IConfig(Config):
    idx: str | None = None
    reserve: float = 0.0
    idx_fee: float = 0.001
    band: float = 0.01
    regime: np.ndarray | None = field(default=None, repr=False)
    rebal: str | None = None
    idx_share: float = 1.0


_PX = {}


def idx_price(D: Data, name: str) -> np.ndarray:
    if name in _PX:
        return _PX[name]
    if name == "g2000p":
        g = D.index["sz399303"].values.astype(float)
        c = D.index["sh000852"].values.astype(float)
        k = int(np.argmax(~np.isnan(g)))
        p = g.copy()
        p[:k] = c[:k] * g[k] / c[k]
    elif "@tr" in name:
        # 全收益近似：价格指数 × (1+股息率)^(年)，例如 'sh000300@tr2.0' 表示年化 2.0% 股息再投资
        base, dy = name.split("@tr")
        p = idx_price(D, base) * (1 + float(dy) / 100) ** (D.cal_days / 365.25)
    else:
        p = D.index[name].values.astype(float)
    p = pd.Series(p).ffill().values
    _PX[name] = p
    return p


def run(D: Data, cfg: IConfig, seed: int = 0):
    t0, t1 = D.day(cfg.start), D.day_le(cfg.end)
    e_arr, x_arr, ratio = D.exits(cfg.tp, cfg.sl, cfg.max_hold, t1, cfg.lag)
    tiers = D.tier if cfg.tier_override is None else cfg.tier_override
    sel = (D.entry >= t0) & (e_arr <= t1)
    if cfg.mask is not None:
        sel &= cfg.mask
    idx = np.nonzero(sel)[0]
    rng = np.random.default_rng(seed)
    order = idx[np.lexsort((rng.random(len(idx)), e_arr[idx]))]

    I_cash = _interest_index(D, t0, cfg.cash_rate)
    I_borr = _interest_index(D, t0, cfg.borrow_rate)
    cash_pos = cfg.capital / I_cash[t0]
    cash_neg = 0.0
    held = {}
    cost_sum = 0.0
    heap = []
    trades = []
    flows = []
    iflows = []            # (t, delta_units)
    n_idx_trades = [0]
    idx_fee_paid = [0.0]
    P = idx_price(D, cfg.idx) if cfg.idx is not None else None
    reg = cfg.regime
    iu = 0.0
    fi = cfg.idx_fee

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

    def ival(t):
        return iu * P[t] if P is not None else 0.0

    def idx_trade(t, dv):
        """dv>0 买入市值 dv 的指数；dv<0 卖出市值 -dv。"""
        nonlocal iu
        du = dv / P[t]
        if dv < 0 and -du > iu:
            du = -iu
            dv = du * P[t]
        iu += du
        if iu < 1e-12:
            iu = 0.0
        add_cash(t, -dv * (1 + fi) if dv > 0 else -dv * (1 - fi))
        iflows.append((t, du))
        n_idx_trades[0] += 1
        idx_fee_paid[0] += abs(dv) * fi

    def rebalance(t):
        if P is None:
            return
        on = True if reg is None else bool(reg[t])
        cur = ival(t)
        if not on:
            if cur > 0:
                idx_trade(t, -cur)
            return
        cash = cash_at(t)
        eq = cash + cur + cost_sum
        # 目标: 现金 = reserve*eq + (1-idx_share)*(闲置 - reserve*eq)
        liquid = cash + cur
        spare = max(0.0, liquid - cfg.reserve * eq)
        target_idx = cfg.idx_share * spare
        diff = target_idx - cur
        if abs(diff) <= cfg.band * eq:
            return
        if diff > 0:
            idx_trade(t, diff / (1 + fi))
        elif cur > 0:
            idx_trade(t, max(diff, -cur))

    def sell(s, t, kind):
        nonlocal cost_sum
        bt, amount, units = held.pop(s)
        px = D.close[s, t]
        value = units * px * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= amount
        trades.append((s, bt, t, amount, value - amount * (1 + cfg.fee), kind))

    def mtm_positions(t):
        return sum(u * D.close[s, t] for s, (_, _, u) in held.items())

    def pop_exits_day(d):
        while heap and heap[0][0] == d:
            xt, s = heapq.heappop(heap)
            if s in held and held[s][0] <= xt:
                r = ratio[s]
                kind = "TP" if cfg.tp is not None and r >= 1 + cfg.tp else (
                    "SL" if cfg.sl is not None and r <= 1 - cfg.sl else ("END" if xt == t1 else "TIME"))
                sell(s, xt, kind)

    def flush_before(t):
        # 处理 < t 的所有卖出日，每个卖出日收盘再平衡
        while heap and heap[0][0] < t:
            d = heap[0][0]
            pop_exits_day(d)
            if d < t1:
                rebalance(d)

    # 事件日：信号买入日 ∪ regime 切换日 ∪ 月初（可选）∪ 起始日
    ev = set(int(e_arr[s]) for s in order)
    ev.add(t0)
    if P is not None:
        if reg is not None:
            rr = reg[t0 : t1 + 1].astype(np.int8)
            sw = np.nonzero(np.diff(rr) != 0)[0] + 1 + t0
            ev.update(int(x) for x in sw)
        if cfg.rebal == "M":
            mon = pd.Index(D.dates[t0 : t1 + 1]).str[:7]
            first = np.nonzero(np.r_[True, mon[1:] != mon[:-1]])[0] + t0
            ev.update(int(x) for x in first)
    ev = sorted(x for x in ev if x < t1) + [t1]
    ptr = 0
    n = len(order)
    state = {"held": held}
    for t in ev:
        flush_before(t)
        pop_exits_day(t)
        while ptr < n and int(e_arr[order[ptr]]) == t:
            s = order[ptr]
            ptr += 1
            cash = cash_at(t)
            iv = ival(t)
            if cfg.sizing == "cost":
                equity = cash + iv + cost_sum
            else:
                equity = cash + iv + mtm_positions(t)
            exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
            tier = int(tiers[s])
            if cfg.weight_fn is not None:
                w = cfg.weight_fn(s, t, state)
            else:
                w = cfg.weights[tier]
            if w <= 0:
                continue
            if tier == 2 and cfg.panic_max_pos is not None and len(held) >= cfg.panic_max_pos:
                if not cfg.panic_swap or not held:
                    continue
                oldest = min(held, key=lambda k: held[k][0])
                sell(oldest, t, "SWAP")
                cash = cash_at(t)
                equity = cash + iv + cost_sum if cfg.sizing == "cost" else cash + iv + mtm_positions(t)
                exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
            amount = w * equity
            room = cfg.cap * equity - exposure
            if amount > room:
                if not cfg.partial:
                    continue
                amount = room
            if amount < cfg.min_amount:
                continue
            need = amount * (1 + cfg.fee)
            if P is not None and cash < need and iu > 0:
                idx_trade(t, -((need - cash) / (1 - fi) * (1 + 1e-9)))
            units = amount / (D.close[s, t] * (1 + cfg.slip))
            add_cash(t, -need)
            held[s] = (t, amount, units)
            cost_sum += amount
            heapq.heappush(heap, (int(x_arr[s]), s))
        if t < t1:
            rebalance(t)
    # 末日
    pop_exits_day(t1)
    for s in list(held):
        sell(s, t1, "END")
    if P is not None and iu > 0:
        idx_trade(t1, -ival(t1))

    final = cash_at(t1)
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    res = {"final": final, "cagr": (final / cfg.capital) ** (1 / years) - 1, "n_trades": len(trades)}

    T = t1 - t0 + 1
    pos = np.zeros(T)
    cost_curve = np.zeros(T)
    for s, bt, st, amount, pnl, kind in trades:
        u = amount / (D.close[s, bt] * (1 + cfg.slip))
        pos[bt - t0 : st - t0] += u * D.close[s, bt:st]
        cost_curve[bt - t0 : st - t0] += amount
    cp = np.zeros(T)
    cn = np.zeros(T)
    c_pos, c_neg = cfg.capital / I_cash[t0], 0.0
    flows.sort(key=lambda f: f[0])
    k2 = 0
    for k in range(T):
        t = t0 + k
        while k2 < len(flows) and flows[k2][0] == t:
            v = flows[k2][1]
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
            k2 += 1
        cp[k], cn[k] = c_pos, c_neg
    cash_curve = cp * I_cash[t0 : t1 + 1] - cn * I_borr[t0 : t1 + 1]
    if P is not None:
        du = np.zeros(T)
        for t, d in iflows:
            du[t - t0] += d
        units = np.cumsum(du)
        units[np.abs(units) < 1e-9] = 0.0
        ivc = units * P[t0 : t1 + 1]
    else:
        ivc = np.zeros(T)
    eq = cash_curve + pos + ivc
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    res["max_dd"] = dd.min()
    res["dd_date"] = D.dates[t0 + int(dd.argmin())]
    res["avg_exposure"] = float(np.mean(pos / eq))
    res["avg_idx"] = float(np.mean(ivc / eq))
    res["avg_total"] = float(np.mean((pos + ivc) / eq))
    res["max_exposure"] = float(np.max(pos / eq))
    lr = np.diff(np.log(eq))
    res["vol"] = float(lr.std() * np.sqrt(244))
    res["n_idx_trades"] = n_idx_trades[0]
    res["idx_fees"] = idx_fee_paid[0]
    kinds = pd.Series([k for *_, k in trades], dtype=object)
    res["n_tp"] = int((kinds == "TP").sum())
    res["n_sl"] = int((kinds == "SL").sum())
    nb = tuple(int(sum(1 for tr in trades if tiers[tr[0]] == k)) for k in range(3))
    res["n_normal"], res["n_adjust"], res["n_panic"] = nb
    if cfg.record:
        res["equity"] = pd.Series(eq, index=D.dates[t0 : t1 + 1])
        res["idx_value"] = pd.Series(ivc, index=D.dates[t0 : t1 + 1])
        res["pos_value"] = pd.Series(pos, index=D.dates[t0 : t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).astype({"s": int, "buy_t": int, "sell_t": int}).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds(D: Data, cfg: IConfig, seeds=range(80)):
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, pd.Series, pd.DataFrame, str))} for r in rows])
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    out["dd_p90"] = df.max_dd.quantile(0.9)
    return out


def buy_hold(D: Data, name: str, start: str, end: str = END_DATE, capital=600_000.0, fee=0.001, frac=1.0, cash_rate=0.018):
    """frac 比例买入并持有指数，其余现金 1.8%。返回 final/cagr/max_dd。"""
    t0, t1 = D.day(start), D.day_le(end)
    P = idx_price(D, name)[t0 : t1 + 1]
    I = _interest_index(D, t0, cash_rate)[t0 : t1 + 1]
    units = capital * frac / (1 + fee) / P[0]
    cash = capital * (1 - frac) / I[0]
    eq = units * P + cash * I
    eq_final = units * P[-1] * (1 - fee) + cash * I[-1]
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"final": eq_final, "cagr": (eq_final / capital) ** (1 / years) - 1, "max_dd": dd}


# ---------- regime 工具 ----------

def signal_count_daily(D: Data, mask=None, window_days=30):
    """每个交易日收盘时，往前 window_days 个自然日（含当日）全表信号数（与 n_signals_30d 同口径）。"""
    sd = pd.to_datetime(D.sig.signal_date)
    if mask is not None:
        sd = sd[mask]
    cal = pd.to_datetime(pd.Series(D.dates))
    s_days = np.sort(((sd - pd.Timestamp("2005-01-01")).dt.days).values)
    c_days = ((cal - pd.Timestamp("2005-01-01")).dt.days).values
    hi = np.searchsorted(s_days, c_days, side="right")
    lo = np.searchsorted(s_days, c_days - window_days + 1, side="left")
    return hi - lo


def reg_quiet(D, mask=None, thr=5):
    """只在信号表安静（近 30 天信号数 <= thr，即平常档状态）时持有指数。"""
    return signal_count_daily(D, mask) <= thr


def reg_after(D, mask=None, thr=20, hold=120):
    """近 30 天信号数 > thr 的日子之后 hold 个交易日内持有指数（抄底后的反弹期）。"""
    n = signal_count_daily(D, mask)
    hot = np.nonzero(n > thr)[0]
    out = np.zeros(len(n), bool)
    for h in hot:
        out[h : h + hold + 1] = True
    return out


def reg_trend(D, name, ma=120):
    """指数收盘在 ma 日均线之上才持有。"""
    p = pd.Series(idx_price(D, name))
    m = p.rolling(ma, min_periods=ma).mean()
    return (p > m).values


def reg_trend_h(D, name, ma=120, h=0.0, lag=0):
    """带滞后带的均线规则：收盘 > MA*(1+h) 进入，< MA*(1-h) 退出；lag=1 表示用前一日收盘的判断在今日收盘成交。"""
    p = idx_price(D, name)
    m = pd.Series(p).rolling(ma, min_periods=ma).mean().values
    out = np.zeros(len(p), bool)
    on = False
    for t in range(len(p)):
        if np.isnan(m[t]):
            on = False
        elif not on and p[t] > m[t] * (1 + h):
            on = True
        elif on and p[t] < m[t] * (1 - h):
            on = False
        out[t] = on
    if lag:
        out = np.r_[np.zeros(lag, bool), out[:-lag]]
    return out


def reg_monthly(D, reg):
    """只在每月最后一个交易日收盘检查一次 regime，之后一个月保持不变（该日收盘成交）。"""
    mon = pd.Index(D.dates).str[:7]
    last = np.r_[mon[1:] != mon[:-1], True]
    out = np.zeros(len(reg), bool)
    cur = False
    for t in range(len(reg)):
        if last[t]:
            cur = bool(reg[t])
        out[t] = cur
    return out


def reg_dd(D, name, look=250, thr=0.3):
    """指数较 look 日最高回撤超过 thr 才持有（跌深才买）。"""
    p = pd.Series(idx_price(D, name))
    pk = p.rolling(look, min_periods=1).max()
    return (p / pk - 1 <= -thr).values


def month_mask(D, months):
    m = D.sig.signal_date.str[:7].values
    return ~np.isin(m, list(months))
