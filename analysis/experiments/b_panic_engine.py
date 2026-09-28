"""恐慌档 / 资金回收 实验引擎（b_panic_ 任务专用）。

在 analysis/engine.py 的 run() 基础上复制并扩展，新增选项全部关闭时与 engine.run 逐笔一致
（见 b_panic_check.py 的复现检查）。不修改 engine.py。

新增 PConfig 字段:
  swap_policy   None | 'oldest' | 'newest' | 'gain_pct'(浮盈比例最大=最接近+40%止盈) |
                'gain_abs'(浮盈金额最大) | 'flat'(|浮盈|最小=最接近成本) | 'underwater'(浮亏最深) |
                'loss_min'(浮盈最小，同 underwater 的反向：先卖浮亏最浅/浮盈最小)
  swap_on       'maxpos' 只在持仓数挡住时换仓（engine 口径） | 'both' 仓位上限挡住时也换
  swap_fix      True: 事先判断卖出后能否买入，买不了就不卖（修正 engine 先卖后发现仓位不够的问题）
                False: 与 engine.panic_swap 完全一致（只支持 'oldest'）
  swap_max_n    仓位上限挡住时，为腾出额度最多连续卖几只
  swap_min_age  持有满多少个交易日的持仓才可被换掉
  swap_keep_panic  True: 不卖恐慌档持仓（避免同一簇内来回倒）
  cap_np        平常/调整档可用的总仓位上限（默认 = cap；< cap 即给恐慌档预留“干火药”）
  panic_mult    None 或 callable(n30)->倍数，恐慌档权重 = weights[2] * 倍数
  rule_mask     (S,) bool；为 False 的恐慌信号按基线口径（权重 base_w2、持仓 15 只、无换仓、cap_np=cap）
                用于按簇拆分“规则只在某一簇生效”带来的增量
  no_lev        True: 买入金额不超过现金（cap<=1 时防止手续费造成的微小融资）
  swap_tiers    哪些档位的新信号被挡时允许换仓（默认只有恐慌档 (2,)）
  swap_min_n30  只有信号的 n_signals_30d >= 该值时才换仓（簇强度门槛）
  swap_partial  True: 仓位上限挡住时只卖出“刚好够买”的那一部分（减仓），而不是整只卖掉（仅 cost 口径）
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine import Data, Config, END_DATE, _interest_index, run as engine_run  # noqa: E402


@dataclass
class PConfig(Config):
    swap_policy: str | None = None
    swap_on: str = "maxpos"
    swap_fix: bool = True
    swap_max_n: int = 3
    swap_min_age: int = 0
    swap_keep_panic: bool = False
    cap_np: float | None = None
    panic_mult: object = field(default=None, repr=False)
    rule_mask: np.ndarray | None = field(default=None, repr=False)
    base_w2: float = 0.06
    base_pmp: int | None = 15
    no_lev: bool = True
    swap_tiers: tuple = (2,)
    swap_min_n30: int = 0
    swap_partial: bool = False


WIN_A = ("2006-01-04", "2016-01-28")
WIN_B = ("2016-01-01", END_DATE)
WIN_F = ("2006-01-04", END_DATE)
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]


def run(D: Data, cfg: PConfig, seed: int = 0):
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
    close = D.close
    cap_np = cfg.cap if cfg.cap_np is None else cfg.cap_np

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
        px = close[s, t]
        value = units * px * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= amount
        trades.append((s, bt, t, amount, value - amount * (1 + cfg.fee), kind))

    def sell_part(s, t, f, kind):
        nonlocal cost_sum
        bt, amount, units = held[s]
        px = close[s, t]
        value = f * units * px * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= f * amount
        held[s] = (bt, amount * (1 - f), units * (1 - f))
        trades.append((s, bt, t, f * amount, value - f * amount * (1 + cfg.fee), kind))

    def mtm_positions(t):
        return sum(u * close[s, t] for s, (_, _, u) in held.items())

    def process_exits(upto):
        while heap and heap[0][0] <= upto:
            xt, s = heapq.heappop(heap)
            if s in held and held[s][0] <= xt:
                r = ratio[s]
                kind = "TP" if cfg.tp is not None and r >= 1 + cfg.tp else (
                    "SL" if cfg.sl is not None and r <= 1 - cfg.sl else ("END" if xt == t1 else "TIME"))
                sell(s, xt, kind)

    def swap_order(t, exclude_today=True):
        cand = []
        for h, (bt, amt, u) in held.items():
            if t - bt < cfg.swap_min_age:
                continue
            if exclude_today and bt == t:
                continue
            if cfg.swap_keep_panic and tiers[h] == 2:
                continue
            val = u * close[h, t]
            r = val / amt
            cand.append((h, bt, amt, u, val, r))
        p = cfg.swap_policy
        if p == "oldest":
            key = lambda c: c[1]
        elif p == "newest":
            key = lambda c: -c[1]
        elif p == "gain_pct":
            key = lambda c: -c[5]
        elif p == "gain_abs":
            key = lambda c: -(c[4] - c[2])
        elif p == "flat":
            key = lambda c: abs(c[5] - 1)
        elif p == "underwater":
            key = lambda c: c[5]
        else:
            raise ValueError(p)
        cand.sort(key=key)
        return cand

    state = {"held": held}
    skips = []
    for s in order:
        t = int(e_arr[s])
        process_exits(t)
        cash = cash_at(t)
        equity = cash + cost_sum if cfg.sizing == "cost" else cash + mtm_positions(t)
        exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        tier = int(tiers[s])
        active = cfg.rule_mask is None or bool(cfg.rule_mask[s])
        if cfg.weight_fn is not None:
            w = cfg.weight_fn(s, t, state)
        else:
            w = cfg.weights[tier]
        if tier == 2:
            if not active:
                w = cfg.base_w2
            elif cfg.panic_mult is not None:
                w = w * cfg.panic_mult(D.n30[s])
        if w <= 0:
            continue
        pmp = cfg.panic_max_pos if active else cfg.base_pmp
        swap = cfg.swap_policy if (active and D.n30[s] >= cfg.swap_min_n30) else None
        cap_t = cfg.cap if tier == 2 else cap_np

        if tier == 2 and not cfg.swap_fix and swap is not None:
            # 与 engine.panic_swap 完全一致的口径
            if pmp is not None and len(held) >= pmp:
                oldest = min(held, key=lambda k: held[k][0])
                sell(oldest, t, "SWAP")
                cash = cash_at(t)
                equity = cash + cost_sum if cfg.sizing == "cost" else cash + mtm_positions(t)
                exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        else:
            blocked_pos = tier == 2 and pmp is not None and len(held) >= pmp
            amount = w * equity
            room = cap_t * equity - exposure
            blocked_cap = amount > room and not cfg.partial
            if tier in cfg.swap_tiers and swap is not None and (blocked_pos or (blocked_cap and cfg.swap_on == "both")):
                # 模拟卖出直到能买入
                cand = swap_order(t)
                plan = []
                c_cash, c_cost, c_mtm = cash, cost_sum, (mtm_positions(t) if (cfg.sizing != "cost" or cfg.cap_basis != "cost") else 0.0)
                ok = False
                need_pos = blocked_pos
                part_f = None
                for c in cand:
                    if len(plan) >= cfg.swap_max_n:
                        break
                    h, bt, amt, u, val, r = c
                    if cfg.swap_partial and not blocked_pos and cfg.sizing == "cost" and cfg.cap_basis == "cost":
                        g = val * (1 - cfg.fee - cfg.slip) - amt
                        cw = cap_t - w
                        f = (c_cost - cw * (c_cash + c_cost)) / (cw * g + amt) * 1.001 + 1e-9
                        if 0 < f < 1:
                            plan.append(h)
                            part_f = f
                            ok = True
                            break
                    plan.append(h)
                    c_cash += val * (1 - cfg.fee - cfg.slip)
                    c_cost -= amt
                    c_mtm -= val
                    need_pos = False
                    eq2 = c_cash + c_cost if cfg.sizing == "cost" else c_cash + c_mtm
                    ex2 = c_cost if cfg.cap_basis == "cost" else c_mtm
                    a2 = w * eq2
                    if a2 <= cap_t * eq2 - ex2 or cfg.partial:
                        ok = True
                        break
                if not ok or need_pos:
                    skips.append((s, t, "swap_fail", len(cand), len(plan)))
                    continue
                for i, h in enumerate(plan):
                    if part_f is not None and i == len(plan) - 1:
                        sell_part(h, t, part_f, "SWAP")
                    else:
                        sell(h, t, "SWAP")
                cash = cash_at(t)
                equity = cash + cost_sum if cfg.sizing == "cost" else cash + mtm_positions(t)
                exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
            elif blocked_pos:
                continue
        amount = w * equity
        room = cap_t * equity - exposure
        if amount > room:
            if not cfg.partial:
                continue
            amount = room
        if cfg.no_lev:
            lim = cash / (1 + cfg.fee)
            if amount > lim:
                if cfg.partial or lim >= 0.95 * amount:
                    amount = lim
                else:
                    continue
        if amount < cfg.min_amount:
            continue
        units = amount / (close[s, t] * (1 + cfg.slip))
        add_cash(t, -amount * (1 + cfg.fee))
        held[s] = (t, amount, units)
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
        pos[bt - t0 : st - t0] += u * close[s, bt:st]
        cost_curve[bt - t0 : st - t0] += amount
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
    cash_curve = cp * I_cash[t0 : t1 + 1] - cn * I_borr[t0 : t1 + 1]
    eq = cash_curve + pos
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    res["max_dd"] = dd.min()
    res["dd_date"] = D.dates[t0 + int(dd.argmin())]
    res["avg_exposure"] = float(np.mean(pos / eq))
    res["max_exposure"] = float(np.max(pos / eq))
    res["min_cash"] = float(cash_curve.min())
    kinds = [k for *_, k in trades]
    res["n_swap"] = int(sum(1 for k in kinds if k == "SWAP"))
    res["n_tp"] = int(sum(1 for k in kinds if k == "TP"))
    res["n_by_tier"] = tuple(int(sum(1 for tr in trades if tiers[tr[0]] == k)) for k in range(3))
    res["n_panic"] = res["n_by_tier"][2]
    res["n_swap_fail"] = len(skips)
    if cfg.record:
        res["skips"] = skips
        res["equity"] = pd.Series(eq, index=D.dates[t0 : t1 + 1])
        res["cost_equity"] = pd.Series(cash_curve + cost_curve, index=D.dates[t0 : t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds(D: Data, cfg: PConfig, seeds=range(80)):
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, str, pd.Series, pd.DataFrame))} for r in rows])
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    out["dd_worst"] = df.max_dd.min()
    return out


_MASKS = {}


def masks(D: Data):
    if not _MASKS:
        ym = D.sig.signal_date.str[:7].values
        _MASKS["ym"] = ym
        _MASKS["no2402"] = ym != "2024-02"
        for c in BIG:
            _MASKS["no" + c] = ym != c
    return _MASKS


def fmt(m):
    return f"{m['final']/1e4:7.1f}万 {m['cagr']*100:6.2f}% dd{m['max_dd']*100:6.1f}% exp{m['avg_exposure']*100:5.1f}% np{m.get('n_panic', float('nan')):5.0f}"
