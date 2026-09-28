"""组合方案引擎（g_combined_ 任务专用）：把已验证的几条旋钮放进同一个回测循环。

基于 b_panic_engine.run（恐慌换仓，已被 v_b_panic 独立复核）复制扩展，另外合入:
  - d_idle_engine 的"闲置现金买指数 + 均线开关"（事件日或每日再平衡）
  - v_c_exit_lib.my_exits 的移动止盈出场（经 v_c_exit 独立复核，支持 lag）
不修改 engine.py。新增选项全部关闭时应与 engine.run / b_panic_engine.run / d_idle_engine.run
逐种子一致，见 g_combined_check.py。

GConfig 在 PConfig（见 b_panic_engine）之上新增:
  idx        None 或 D.index 的列名：闲置现金买的指数
  reserve    始终留作现金的比例（占总权益）
  idx_fee    指数每边费用
  band       再平衡容差（占总权益）
  regime     (T,) bool：当日收盘是否允许持有指数（False 当日收盘清掉）
  rebal      'event'（信号买卖日、regime 切换日再平衡，d_idle 口径）| 'daily'（每个交易日收盘再平衡，更保守）
  exit_rule  None（用 tp/sl 固定出场）或 v_c_exit_lib 的规则 dict（如移动止盈）
  xlag       出场触发后第 xlag 个成交日收盘才卖（执行延迟压力测试）
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from engine import Data, Config, END_DATE, _interest_index  # noqa: E402
from b_panic_engine import PConfig  # noqa: E402
from v_c_exit_lib import my_exits  # noqa: E402
from d_idle_engine import idx_price, reg_trend_h  # noqa: E402

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]


@dataclass
class GConfig(PConfig):
    idx: str | None = None
    reserve: float = 0.0
    idx_fee: float = 0.001
    band: float = 0.01
    regime: np.ndarray | None = field(default=None, repr=False)
    rebal: str = "event"
    exit_rule: dict | None = field(default=None, repr=False)
    xlag: int = 0


def run(D: Data, cfg: GConfig, seed: int = 0):
    t0, t1 = D.day(cfg.start), D.day_le(cfg.end)
    if cfg.exit_rule is None and not cfg.xlag:
        e_arr, x_arr, ratio = D.exits(cfg.tp, cfg.sl, cfg.max_hold, t1, cfg.lag)
    else:
        rule = cfg.exit_rule if cfg.exit_rule is not None else {"kind": "fixed", "tp": cfg.tp, "sl": cfg.sl}
        e_arr, x_arr, ratio = my_exits(D, rule, t1, cfg.lag, cfg.xlag)
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
    iflows = []
    close = D.close
    cap_np = cfg.cap if cfg.cap_np is None else cfg.cap_np
    P = idx_price(D, cfg.idx) if cfg.idx is not None else None
    reg = cfg.regime
    fi = cfg.idx_fee
    iu = 0.0
    n_idx = [0]
    fixed_exit = cfg.exit_rule is None

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
        n_idx[0] += 1

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
        spare = max(0.0, cash + cur - cfg.reserve * eq)
        diff = spare - cur
        if abs(diff) <= cfg.band * eq:
            return
        if diff > 0:
            idx_trade(t, diff / (1 + fi))
        elif cur > 0:
            idx_trade(t, max(diff, -cur))

    def sell(s, t, kind):
        nonlocal cost_sum
        bt, amount, units = held.pop(s)
        value = units * close[s, t] * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= amount
        trades.append((s, bt, t, amount, value - amount * (1 + cfg.fee), kind))

    def mtm_positions(t):
        return sum(u * close[s, t] for s, (_, _, u) in held.items())

    def pop_exits_day(d):
        while heap and heap[0][0] == d:
            xt, s = heapq.heappop(heap)
            if s in held and held[s][0] <= xt:
                r = ratio[s]
                if xt == t1:
                    kind = "END"
                elif fixed_exit:
                    kind = "TP" if cfg.tp is not None and r >= 1 + cfg.tp else (
                        "SL" if cfg.sl is not None and r <= 1 - cfg.sl else "TIME")
                else:
                    kind = "TP" if r >= 1 else "SL"
                sell(s, xt, kind)

    def flush_before(t):
        while heap and heap[0][0] < t:
            d = heap[0][0]
            pop_exits_day(d)
            if d < t1:
                rebalance(d)

    def swap_order(t):
        cand = []
        for h, (bt, amt, u) in held.items():
            if t - bt < cfg.swap_min_age or bt == t:
                continue
            if cfg.swap_keep_panic and tiers[h] == 2:
                continue
            val = u * close[h, t]
            cand.append((h, bt, amt, u, val, val / amt))
        p = cfg.swap_policy
        key = {"oldest": lambda c: c[1], "newest": lambda c: -c[1], "gain_pct": lambda c: -c[5],
               "gain_abs": lambda c: -(c[4] - c[2]), "flat": lambda c: abs(c[5] - 1),
               "underwater": lambda c: c[5]}[p]
        cand.sort(key=key)
        return cand

    def eqx(t, cash, iv):
        equity = cash + iv + (cost_sum if cfg.sizing == "cost" else mtm_positions(t))
        exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        return equity, exposure

    def buy_signal(s, t):
        nonlocal cost_sum
        cash = cash_at(t)
        iv = ival(t)
        equity, exposure = eqx(t, cash, iv)
        tier = int(tiers[s])
        w = cfg.weight_fn(s, t, state) if cfg.weight_fn is not None else cfg.weights[tier]
        if w <= 0:
            return
        pmp = cfg.panic_max_pos
        swap = cfg.swap_policy if D.n30[s] >= cfg.swap_min_n30 else None
        cap_t = cfg.cap if tier == 2 else cap_np
        if tier == 2 and not cfg.swap_fix and swap is not None:
            if pmp is not None and len(held) >= pmp:
                oldest = min(held, key=lambda k: held[k][0])
                sell(oldest, t, "SWAP")
                cash = cash_at(t)
                equity, exposure = eqx(t, cash, iv)
        else:
            blocked_pos = tier == 2 and pmp is not None and len(held) >= pmp
            amount = w * equity
            room = cap_t * equity - exposure
            blocked_cap = amount > room and not cfg.partial
            if tier in cfg.swap_tiers and swap is not None and (blocked_pos or (blocked_cap and cfg.swap_on == "both")):
                cand = swap_order(t)
                plan = []
                c_cash, c_cost = cash, cost_sum
                c_mtm = mtm_positions(t) if (cfg.sizing != "cost" or cfg.cap_basis != "cost") else 0.0
                ok = False
                need_pos = blocked_pos
                for c in cand:
                    if len(plan) >= cfg.swap_max_n:
                        break
                    h, bt, amt, u, val, r = c
                    plan.append(h)
                    c_cash += val * (1 - cfg.fee - cfg.slip)
                    c_cost -= amt
                    c_mtm -= val
                    need_pos = False
                    eq2 = c_cash + iv + (c_cost if cfg.sizing == "cost" else c_mtm)
                    ex2 = c_cost if cfg.cap_basis == "cost" else c_mtm
                    if w * eq2 <= cap_t * eq2 - ex2 or cfg.partial:
                        ok = True
                        break
                if not ok or need_pos:
                    return
                for h in plan:
                    sell(h, t, "SWAP")
                cash = cash_at(t)
                equity, exposure = eqx(t, cash, iv)
            elif blocked_pos:
                return
        amount = w * equity
        room = cap_t * equity - exposure
        if amount > room:
            if not cfg.partial:
                return
            amount = room
        if cfg.no_lev:
            lim = (cash + iv * (1 - fi)) / (1 + cfg.fee)
            if amount > lim:
                if cfg.partial or lim >= 0.95 * amount:
                    amount = lim
                else:
                    return
        if amount < cfg.min_amount:
            return
        need = amount * (1 + cfg.fee)
        if P is not None and cash < need and iu > 0:
            idx_trade(t, -((need - cash) / (1 - fi) * (1 + 1e-9)))
        units = amount / (close[s, t] * (1 + cfg.slip))
        add_cash(t, -need)
        held[s] = (t, amount, units)
        cost_sum += amount
        heapq.heappush(heap, (int(x_arr[s]), s))

    state = {"held": held}
    ev = set(int(e_arr[s]) for s in order)
    ev.add(t0)
    if P is not None:
        if cfg.rebal == "daily":
            ev.update(range(t0, t1))
        elif reg is not None:
            rr = reg[t0 : t1 + 1].astype(np.int8)
            ev.update(int(x) for x in (np.nonzero(np.diff(rr) != 0)[0] + 1 + t0))
    ev = sorted(x for x in ev if x < t1) + [t1]
    ptr, n = 0, len(order)
    for t in ev:
        flush_before(t)
        pop_exits_day(t)
        while ptr < n and int(e_arr[order[ptr]]) == t:
            buy_signal(order[ptr], t)
            ptr += 1
        if t < t1:
            rebalance(t)
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
        u = amount / (close[s, bt] * (1 + cfg.slip))
        pos[bt - t0 : st - t0] += u * close[s, bt:st]
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
        un = np.cumsum(du)
        un[np.abs(un) < 1e-9] = 0.0
        ivc = un * P[t0 : t1 + 1]
    else:
        ivc = np.zeros(T)
    eq = cash_curve + pos + ivc
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    res["max_dd"] = dd.min()
    res["dd_date"] = D.dates[t0 + int(dd.argmin())]
    res["avg_exposure"] = float(np.mean(pos / eq))
    res["avg_idx"] = float(np.mean(ivc / eq))
    res["max_exposure"] = float(np.max((pos + ivc) / eq))
    res["min_cash"] = float(cash_curve.min())
    kinds = [k for *_, k in trades]
    res["n_swap"] = int(sum(1 for k in kinds if k == "SWAP"))
    res["n_tp"] = int(sum(1 for k in kinds if k == "TP"))
    nb = tuple(int(sum(1 for tr in trades if tiers[tr[0]] == k)) for k in range(3))
    res["n_normal"], res["n_adjust"], res["n_panic"] = nb
    res["n_idx_trades"] = n_idx[0]
    if cfg.record:
        idxd = D.dates[t0 : t1 + 1]
        res["equity"] = pd.Series(eq, index=idxd)
        res["idx_value"] = pd.Series(ivc, index=idxd)
        res["pos_value"] = pd.Series(pos, index=idxd)
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds(D: Data, cfg: GConfig, seeds=range(80), raw=False):
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, str, pd.Series, pd.DataFrame))} for r in rows])
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["final_p10"] = df.final.quantile(0.1)
    out["final_p90"] = df.final.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    out["dd_worst"] = df.max_dd.min()
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else np.nan
    if raw:
        return out, df
    return out


_MM = {}


def month_mask(D: Data, months):
    key = tuple(sorted(months))
    if key not in _MM:
        ym = D.sig.signal_date.str[:7].values
        _MM[key] = ~np.isin(ym, list(months))
    return _MM[key]


def win_cfg(base: GConfig, win: str, drop=(), realistic=False, D: Data | None = None):
    """base 配置套上窗口 / 剔簇 / 实盘口径（信号次日收盘成交 + 每边 0.5% 滑点；指数费 0.2%）。"""
    st, en = WIN[win]
    kw = dict(start=st, end=en)
    if drop:
        m = month_mask(D, drop)
        kw["mask"] = m if base.mask is None else (base.mask & m)
    if realistic:
        kw.update(lag=1, slip=0.005)
        if base.idx is not None:
            # 指数开关本身已按前一日收盘判断（regime 用 lag=1 构造），这里只把指数费提到每边 0.2%
            kw["idx_fee"] = max(base.idx_fee, 0.002)
    return replace(base, **kw)
