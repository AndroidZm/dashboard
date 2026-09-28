"""c_exit 公用：自定义出场规则（移动止盈 / 分批止盈 / 时间出场变体）+ 支持“多腿”出场的组合引擎。

引擎 run_legs() 由 analysis/engine.py 的 run() 复制而来，唯一扩展是：一笔买入可以拆成若干“腿”，
每条腿占买入份额 frac_k，各自有卖出日 x_k 和价格比 ratio_k（分批止盈用）。只有一条腿、
且出场数组取自 D.exits() 时，与 engine.run 逐笔一致（见 c_exit_check.py）。

出场规则用 dict 描述（rule），由 exits_for(D, rule, end_t) 转为 legs:
  {'kind':'fixed', 'tp':0.4, 'sl':0.4, 'mh':None}
  {'kind':'trail', 'act':0.3, 'trail':0.15, 'sl':0.4, 'tp':None, 'mh':None}
        收盘首次 >= 1+act 后启动；之后收盘 <= 峰值收盘*(1-trail) 卖出；硬止损 sl、硬止盈 tp、最长持有 mh 照旧
  {'kind':'lock', 'act':0.3, 'floor':0.1, 'sl':0.4, 'tp':0.6}
        收盘首次 >= 1+act 后，止损抬到 1+floor（保本/锁利）
  {'kind':'tprof', 'tp':0.4, 'sl':0.4, 'mh':500, 'min_r':1.0}
        持有满 mh 后，只要收盘 >= min_r*成本 就卖出（亏损的继续拿到 tp/sl）
  {'kind':'decay', 'tp':0.6, 'tp2':0.2, 'mh':250, 'sl':0.4}
        持有满 mh 后，止盈线从 tp 降到 tp2
  {'kind':'legs', 'legs':[(frac, rule), ...]}    分批：每条腿一个单腿规则
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Data, Config, END_DATE, _interest_index  # noqa: E402

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]

_D = None


def getD():
    global _D
    if _D is None:
        _D = Data.load()
        _D._rule_cache = {}
    return _D


def month_mask(months_out):
    ym = getD().sig.signal_date.str[:7].values
    return ~np.isin(ym, list(months_out))


# ---------------------------------------------------------------- 出场规则
def _single_exit(D, rule, end_t):
    """单腿规则 -> (e_arr, x_arr, ratio)。"""
    k = rule.get("kind", "fixed")
    if k == "fixed":
        return _post(D, rule, *D.exits(rule.get("tp"), rule.get("sl"), rule.get("mh"), end_t, 0), end_t)
    S = len(D.entry)
    e_arr = D.entry.astype(np.int64).copy()
    x_arr = np.empty(S, np.int64)
    ratio = np.empty(S)
    tp, sl, mh = rule.get("tp"), rule.get("sl"), rule.get("mh")
    for s in range(S):
        e = int(e_arr[s])
        if e > end_t:
            x_arr[s], ratio[s] = e, 1.0
            continue
        c0 = D.close[s, e]
        r = D.close[s, e + 1: end_t + 1] / c0
        ok = D.traded[s, e + 1: end_t + 1]
        n = len(r)
        hit = np.zeros(n, bool)
        if tp is not None and k != "decay":
            hit |= r >= 1 + tp
        if sl is not None:
            hit |= r <= 1 - sl
        if k == "trail":
            pk = np.maximum.accumulate(np.maximum(r, 1.0)) if n else r
            act = np.maximum.accumulate(r >= 1 + rule["act"]) if n else r.astype(bool)
            hit |= act & (r <= pk * (1 - rule["trail"]))
        elif k == "trail2":
            # 两段式移动止盈：峰值 < 1+act2 时回撤 trail 卖出，峰值 >= 1+act2 后回撤放宽到 trail2
            pk = np.maximum.accumulate(np.maximum(r, 1.0)) if n else r
            act = np.maximum.accumulate(r >= 1 + rule["act"]) if n else r.astype(bool)
            width = np.where(pk >= 1 + rule["act2"], rule["trail2"], rule["trail"])
            hit |= act & (r <= pk * (1 - width))
        elif k == "lock":
            act = np.maximum.accumulate(r >= 1 + rule["act"]) if n else r.astype(bool)
            hit |= act & (r <= 1 + rule["floor"])
        elif k == "tprof":
            if mh is not None and mh - 1 < n:
                late = np.zeros(n, bool)
                late[mh - 1:] = True
                hit |= late & (r >= rule.get("min_r", 1.0))
        elif k == "decay":
            j = np.arange(n)
            thr = np.where(j >= (mh - 1), 1 + rule["tp2"], 1 + tp)
            hit |= r >= thr
        hit &= ok
        if k in ("fixed", "trail", "trail2", "lock") and mh is not None and mh < n:
            hit[mh - 1:] |= ok[mh - 1:]
        kk = np.argmax(hit) if hit.any() else -1
        x = e + 1 + kk if kk >= 0 else end_t
        x_arr[s] = x
        ratio[s] = D.close[s, x] / c0
    return _post(D, rule, e_arr, x_arr, ratio, end_t)


def _post(D, rule, e_arr, x_arr, ratio, end_t):
    """xlag: 触发后第 xlag 个有成交日的收盘才卖（执行延迟压力测试）。
    stress_q: 退市压力测试——持有期内收盘曾 <= 0.6 的信号中，按 stress_seed 抽出比例 q，
    视为“其实会退市”：有止损 sl 时按 1-sl 出场（止损先触发），无止损时按 0.05（-95%）出场，出场日不变。"""
    xlag = rule.get("xlag", 0)
    q = rule.get("stress_q", 0)
    if not xlag and not q:
        return e_arr, x_arr, ratio
    x_arr = x_arr.copy(); ratio = ratio.copy()
    S = len(e_arr)
    if q:
        u = np.random.default_rng(rule.get("stress_seed", 0)).random(S)
        sl = rule.get("sl")
        for s in range(S):
            e, x = int(e_arr[s]), int(x_arr[s])
            if x <= e or u[s] >= q:
                continue
            if (D.close[s, e + 1: x + 1] / D.close[s, e]).min() <= 0.6 + 1e-12:
                ratio[s] = 0.05 if sl is None or sl >= 0.95 else min(ratio[s], 1 - sl)
    if xlag:
        for s in range(S):
            e, x = int(e_arr[s]), int(x_arr[s])
            if x >= end_t or x <= e:
                continue
            k = 0
            y = x
            while y < end_t and k < xlag:
                y += 1
                if D.traded[s, y]:
                    k += 1
            ratio[s] = ratio[s] * D.close[s, y] / D.close[s, x]
            x_arr[s] = y
    return e_arr, x_arr, ratio


def _key(rule):
    if rule.get("kind") == "legs":
        return ("legs",) + tuple((f, _key(r)) for f, r in rule["legs"])
    return tuple(sorted(rule.items()))


def exits_for(D, rule, end_t):
    """-> (e_arr, [(frac, x_arr, ratio), ...])"""
    if not hasattr(D, "_rule_cache"):
        D._rule_cache = {}
    key = (_key(rule), end_t)
    if key in D._rule_cache:
        return D._rule_cache[key]
    if rule.get("kind") == "legs":
        legs = []
        e_arr = None
        for f, r in rule["legs"]:
            e_arr, x, ra = _single_exit(D, r, end_t)
            legs.append((f, x, ra))
    else:
        e_arr, x, ra = _single_exit(D, rule, end_t)
        legs = [(1.0, x, ra)]
    out = (e_arr, legs)
    D._rule_cache[key] = out
    return out


# ---------------------------------------------------------------- 引擎（多腿）
def run_legs(D: Data, cfg: Config, rule: dict, seed: int = 0):
    t0, t1 = D.day(cfg.start), D.day_le(cfg.end)
    e_arr, legs = exits_for(D, rule, t1)
    xmax = np.max(np.vstack([x for _, x, _ in legs]), axis=0)
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
    held = {}                 # s -> {'bt':t, 'legs':{k:(amount, units)}}
    cost_sum = 0.0
    heap = []                 # (exit_t, s, k)
    trades = []
    flows = []

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

    def sell_leg(s, k, t, kind, px=None):
        nonlocal cost_sum
        h = held[s]
        amount, units = h["legs"].pop(k)
        if px is None:
            px = D.close[s, t]
        value = units * px * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= amount
        trades.append((s, k, h["bt"], t, amount, units, value - amount * (1 + cfg.fee), kind))
        if not h["legs"]:
            del held[s]

    def mtm_positions(t):
        return sum(u * D.close[s, t] for s, h in held.items() for (_, u) in h["legs"].values())

    def process_exits(upto):
        while heap and heap[0][0] <= upto:
            xt, s, k = heapq.heappop(heap)
            if s in held and k in held[s]["legs"] and held[s]["bt"] <= xt:
                # 按出场数组的价格比成交（压力测试会改写 ratio；未改写时等于当日收盘）
                sell_leg(s, k, xt, "X", D.close[s, held[s]["bt"]] * legs[k][2][s])

    state = {"held": held}
    t = t0
    for s in order:
        t = int(e_arr[s])
        process_exits(t)
        cash = cash_at(t)
        equity = cash + cost_sum if cfg.sizing == "cost" else cash + mtm_positions(t)
        exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        tier = int(tiers[s])
        w = cfg.weight_fn(s, t, state) if cfg.weight_fn is not None else cfg.weights[tier]
        if w <= 0:
            continue
        if tier == 2 and cfg.panic_max_pos is not None and len(held) >= cfg.panic_max_pos:
            if not cfg.panic_swap or not held:
                continue
            oldest = min(held, key=lambda q: held[q]["bt"])
            for k in list(held[oldest]["legs"]):
                sell_leg(oldest, k, t, "SWAP")
            cash = cash_at(t)
            equity = cash + cost_sum if cfg.sizing == "cost" else cash + mtm_positions(t)
            exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        amount = w * equity
        room = cfg.cap * equity - exposure
        if amount > room:
            if not cfg.partial:
                continue
            amount = room
        if amount < cfg.min_amount:
            continue
        units = amount / (D.close[s, t] * (1 + cfg.slip))
        add_cash(t, -amount * (1 + cfg.fee))
        held[s] = {"bt": t, "legs": {}}
        for k, (f, x, _) in enumerate(legs):
            held[s]["legs"][k] = (amount * f, units * f)
            heapq.heappush(heap, (int(x[s]), s, k))
        cost_sum += amount
    process_exits(t1)
    for s in list(held):
        for k in list(held[s]["legs"]):
            sell_leg(s, k, t1, "END")

    final = cash_at(t1)
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    res = {"final": final, "cagr": (final / cfg.capital) ** (1 / years) - 1,
           "n_trades": len({tr[0] for tr in trades})}
    T = t1 - t0 + 1
    pos = np.zeros(T)
    for s, k, bt, st, amount, units, pnl, kind in trades:
        pos[bt - t0: st - t0] += units * D.close[s, bt:st]
    cp = np.zeros(T)
    cn = np.zeros(T)
    c_pos, c_neg = cfg.capital / I_cash[t0], 0.0
    flows.sort(key=lambda f: f[0])
    fi = 0
    for kk in range(T):
        tt = t0 + kk
        while fi < len(flows) and flows[fi][0] == tt:
            v = flows[fi][1]
            if v >= 0:
                debt = c_neg * I_borr[tt]
                pay = min(v, debt)
                c_neg -= pay / I_borr[tt]
                c_pos += (v - pay) / I_cash[tt]
            else:
                avail = c_pos * I_cash[tt]
                use = min(-v, avail)
                c_pos -= use / I_cash[tt]
                c_neg += (-v - use) / I_borr[tt]
            fi += 1
        cp[kk], cn[kk] = c_pos, c_neg
    cash_curve = cp * I_cash[t0: t1 + 1] - cn * I_borr[t0: t1 + 1]
    eq = cash_curve + pos
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    res["max_dd"] = dd.min()
    res["dd_date"] = D.dates[t0 + int(dd.argmin())]
    res["avg_exposure"] = float(np.mean(pos / eq))
    res["max_exposure"] = float(np.max(pos / eq))
    lr = np.diff(np.log(eq))
    res["vol"] = float(lr.std() * np.sqrt(244))
    bought = sorted({tr[0] for tr in trades})
    res["n0"] = int(sum(tiers[q] == 0 for q in bought))
    res["n1"] = int(sum(tiers[q] == 1 for q in bought))
    res["n2"] = int(sum(tiers[q] == 2 for q in bought))
    # 已成交持仓的逐笔（按信号合并各腿）收益与持有天数
    if trades:
        tr = pd.DataFrame(trades, columns=["s", "k", "bt", "st", "amount", "units", "pnl", "kind"])
        g = tr.groupby("s").agg(amount=("amount", "sum"), pnl=("pnl", "sum"), bt=("bt", "first"), st=("st", "max"))
        res["trade_mean_ret"] = float((g.pnl / g.amount).mean())
        res["trade_med_hold"] = float((g.st - g.bt).median())
        res["pnl_sum"] = float(g.pnl.sum())
    if cfg.record:
        res["equity"] = pd.Series(eq, index=D.dates[t0: t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "k", "buy_t", "sell_t", "amount", "units", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds_legs(D, cfg, rule, seeds=range(80)):
    rows = [run_legs(D, replace(cfg, record=False), rule, s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, str, pd.Series, pd.DataFrame))} for r in rows])
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    return out


# ---------------------------------------------------------------- 逐笔统计（不经组合）
def per_trade(D, rule, win, mask=None):
    """窗口内全部信号（不考虑资金约束）的逐笔统计。"""
    t0, t1 = D.day(WIN[win][0]), D.day_le(WIN[win][1])
    e_arr, legs = exits_for(D, rule, t1)
    sel = (D.entry >= t0) & (e_arr <= t1)
    if mask is not None:
        sel &= mask
    ret = sum(f * ra for f, _, ra in legs)[sel] - 1
    hold = np.max(np.vstack([x for _, x, _ in legs]), axis=0)[sel] - e_arr[sel]
    # 资金占用加权：每条腿的持有天数按份额加权
    hold_w = sum(f * (x - e_arr) for f, x, _ in legs)[sel]
    return {"pt_n": int(sel.sum()), "pt_mean": float(ret.mean()), "pt_med": float(np.median(ret)),
            "pt_win": float((ret > 0).mean()), "pt_hold_med": float(np.median(hold)),
            "pt_hold_mean": float(hold_w.mean()),
            "pt_ret_per_yr": float(ret.sum() / max(hold_w.sum(), 1) * 244)}


# ---------------------------------------------------------------- 并行评估
def cfg_for(win, drop=(), **kw):
    start, end = WIN[win]
    c = Config(start=start, end=end, **kw)
    if drop:
        c = replace(c, mask=month_mask(drop))
    return c


def _eval(job):
    name, rule, win, drop, nseeds, kw = job
    D = getD()
    m = run_seeds_legs(D, cfg_for(win, drop, **kw), rule, seeds=range(nseeds))
    out = {"name": name, "win": win, "drop": "|".join(drop), "final": m["final"] / 1e4, "cagr": m["cagr"] * 100,
           "max_dd": m["max_dd"] * 100, "dd_p10": m["dd_p10"] * 100, "cagr_p10": m["cagr_p10"] * 100,
           "avg_exp": m["avg_exposure"], "n": m["n_trades"], "n0": m["n0"], "n1": m["n1"], "n2": m["n2"],
           "tr_mean": m.get("trade_mean_ret", np.nan), "tr_hold": m.get("trade_med_hold", np.nan)}
    return out


def evaluate(jobs, procs=3):
    """jobs: list of (name, rule, win, drop_tuple, nseeds, cfg_kwargs)"""
    getD()
    # 预先在父进程算好出场缓存，fork 后子进程共享
    seen = set()
    for name, rule, win, drop, ns, kw in jobs:
        t1 = getD().day_le(WIN[win][1])
        k = (_key(rule), t1)
        if k not in seen:
            exits_for(getD(), rule, t1)
            seen.add(k)
    if procs <= 1:
        return pd.DataFrame([_eval(j) for j in jobs])
    with Pool(procs) as p:
        return pd.DataFrame(p.map(_eval, jobs, chunksize=max(1, len(jobs) // (procs * 8))))


# ---------------------------------------------------------------- 给其它引擎用：替换 D.exits
def patch_exits(D, rule):
    """返回 D 的浅拷贝，其 .exits(tp, sl, max_hold, end_t, lag) 忽略参数、改用单腿规则 rule 的出场。
    这样 engine.run / 其它基于 engine 的实验引擎无需改动即可使用移动止盈等规则（仅支持 lag=0）。"""
    import copy
    if rule.get("kind") == "legs":
        raise ValueError("分批规则请用 run_legs")
    Dx = copy.copy(D)

    def _exits(tp=None, sl=None, max_hold=None, end_t=None, lag=0):
        if lag:
            raise ValueError("patch_exits 只支持 lag=0")
        end_t = len(D.dates) - 1 if end_t is None else end_t
        e, legs = exits_for(D, rule, end_t)
        return e, legs[0][1], legs[0][2]
    Dx.exits = _exits
    return Dx
