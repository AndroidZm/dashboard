"""f_lever 专用引擎：engine.run 的拷贝 + 杠杆相关的额外记录/机制。

在 engine.run 基础上增加:
  - 逐日负债曲线、维持担保比例 (持仓市值 + 正现金) / 融资负债，返回路径最小值 min_mr
  - 最大盯市总仓位 max_gross (持仓市值 / 净值)
  - sizing='fixed'：单笔 = 权重 × 初始资金（不随权益复利，固定金额），仓位上限仍按净值
  - margin_call=(警戒线, 恢复线)：逐日检查维持担保比例，低于平仓线（如 1.30）时按持有最久顺序强平到恢复线
    （简化：当日收盘价成交，扣费）。None = 不模拟（基线）。
  - kscale：把 cfg.weights 统一乘 k（等价于直接传缩放后的 weights，这里只是方便）

用 check_baseline() 先确认与 engine.run 完全一致。
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
import engine as E  # noqa: E402
from engine import Data, Config, END_DATE, _interest_index  # noqa: E402,F401

WIN = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", END_DATE), "F": ("2006-01-04", END_DATE)}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]


@dataclass
class LConfig(Config):
    margin_call: tuple | None = None      # (平仓线, 恢复线)，例如 (1.30, 1.50)
    k: float = 1.0                        # 权重统一缩放


def month_mask(D, months_out):
    ym = D.sig.signal_date.str[:7].values
    return ~np.isin(ym, list(months_out))


def run(D: Data, cfg: LConfig, seed: int = 0):
    t0, t1 = D.day(cfg.start), D.day_le(cfg.end)
    e_arr, x_arr, ratio = D.exits(cfg.tp, cfg.sl, cfg.max_hold, t1, cfg.lag)
    tiers = D.tier if cfg.tier_override is None else cfg.tier_override
    weights = tuple(w * cfg.k for w in cfg.weights)
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
    n_mc = 0            # 强平次数（天）
    mc_days = []

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
        px = D.close[s, t]
        value = units * px * (1 - cfg.fee - cfg.slip)
        add_cash(t, value)
        cost_sum -= amount
        trades.append((s, bt, t, amount, value - amount * (1 + cfg.fee), kind))

    def mtm_positions(t):
        return sum(u * D.close[s, t] for s, (_, _, u) in held.items())

    last_checked = [t0 - 1]

    def margin_check(upto):
        """逐日检查 (last_checked, upto] 的维持担保比例，必要时强平。"""
        nonlocal n_mc
        if cfg.margin_call is None:
            return
        lo, hi = cfg.margin_call
        for t in range(last_checked[0] + 1, upto + 1):
            # 先处理当日自然出场
            process_exits(t, check=False)
            debt = cash_neg * I_borr[t] - 0.0
            if debt <= 0 or not held:
                continue
            pos = mtm_positions(t) + cash_pos * I_cash[t]
            if pos / debt >= lo:
                continue
            n_mc += 1
            mc_days.append(t)
            # 按持有最久顺序卖，直到比例恢复到 hi
            for s in sorted(held, key=lambda k: held[k][0]):
                sell(s, t, "MC")
                debt = cash_neg * I_borr[t]
                if debt <= 0:
                    break
                pos = mtm_positions(t) + cash_pos * I_cash[t]
                if pos / debt >= hi:
                    break
        last_checked[0] = max(last_checked[0], upto)

    def process_exits(upto, check=True):
        if check and cfg.margin_call is not None:
            margin_check(upto)
        while heap and heap[0][0] <= upto:
            xt, s = heapq.heappop(heap)
            if s in held and held[s][0] <= xt:
                r = ratio[s]
                kind = "TP" if cfg.tp is not None and r >= 1 + cfg.tp else (
                    "SL" if cfg.sl is not None and r <= 1 - cfg.sl else ("END" if xt == t1 else "TIME"))
                sell(s, xt, kind)

    state = {"held": held}
    for s in order:
        t = int(e_arr[s])
        process_exits(t)
        cash = cash_at(t)
        if cfg.sizing == "cost":
            equity = cash + cost_sum
        else:
            equity = cash + mtm_positions(t)
        exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        tier = int(tiers[s])
        if cfg.weight_fn is not None:
            w = cfg.weight_fn(s, t, state)
        else:
            w = weights[tier]
        if w <= 0:
            continue
        if tier == 2 and cfg.panic_max_pos is not None and len(held) >= cfg.panic_max_pos:
            if not cfg.panic_swap or not held:
                continue
            oldest = min(held, key=lambda k: held[k][0])
            sell(oldest, t, "SWAP")
            cash = cash_at(t)
            equity = cash + cost_sum if cfg.sizing == "cost" else cash + mtm_positions(t)
            exposure = cost_sum if cfg.cap_basis == "cost" else mtm_positions(t)
        if cfg.sizing == "fixed":
            eq_now = cash + mtm_positions(t)
            amount = w * cfg.capital
            room = cfg.cap * eq_now - exposure
        else:
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
        held[s] = (t, amount, units)
        cost_sum += amount
        heapq.heappush(heap, (int(x_arr[s]), s))
    process_exits(t1)
    for s in list(held):
        sell(s, t1, "END")

    final = cash_at(t1)
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    res = {"final": final, "cagr": (max(final, 1e-9) / cfg.capital) ** (1 / years) - 1, "n_trades": len(trades)}

    T = t1 - t0 + 1
    pos = np.zeros(T)
    cost_curve = np.zeros(T)
    for s, bt, st, amount, pnl, kind in trades:
        u = amount / (D.close[s, bt] * (1 + cfg.slip))
        pos[bt - t0: st - t0] += u * D.close[s, bt:st]
        cost_curve[bt - t0: st - t0] += amount
    cp = np.zeros(T)
    cn = np.zeros(T)
    c_pos, c_neg = cfg.capital / I_cash[t0], 0.0
    flows.sort(key=lambda f: f[0])
    fi = 0
    for kk in range(T):
        t = t0 + kk
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
        cp[kk], cn[kk] = c_pos, c_neg
    posc = cp * I_cash[t0: t1 + 1]
    debt = cn * I_borr[t0: t1 + 1]
    cash_curve = posc - debt
    eq = cash_curve + pos
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    res["max_dd"] = dd.min()
    res["dd_date"] = D.dates[t0 + int(dd.argmin())]
    res["avg_exposure"] = float(np.mean(pos / eq))
    res["max_exposure"] = float(np.max(pos / eq))
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = np.where(debt > 1e-6, (pos + posc) / np.maximum(debt, 1e-6), np.inf)
    res["min_mr"] = float(mr.min())
    res["mr_date"] = D.dates[t0 + int(mr.argmin())] if np.isfinite(mr.min()) else ""
    res["max_debt_eq"] = float(np.max(debt / eq))
    res["avg_debt_eq"] = float(np.mean(debt / eq))
    res["days_borrow"] = float(np.mean(debt > 1e-6))
    res["min_eq"] = float(eq.min() / cfg.capital)
    lr = np.diff(np.log(np.maximum(eq, 1e-9)))
    res["vol"] = float(lr.std() * np.sqrt(244))
    res["n_mc"] = n_mc
    kinds = pd.Series([k for *_, k in trades])
    res["n_tp"] = int((kinds == "TP").sum())
    res["n_sl"] = int((kinds == "SL").sum())
    res["n_by_tier"] = tuple(int(sum(1 for tr in trades if tiers[tr[0]] == k)) for k in range(3))
    res["n_panic"] = res["n_by_tier"][2]
    if cfg.record:
        res["equity"] = pd.Series(eq, index=D.dates[t0: t1 + 1])
        res["pos"] = pd.Series(pos, index=D.dates[t0: t1 + 1])
        res["debt"] = pd.Series(debt, index=D.dates[t0: t1 + 1])
        res["mr"] = pd.Series(mr, index=D.dates[t0: t1 + 1])
        res["cost_equity"] = pd.Series(cash_curve + cost_curve, index=D.dates[t0: t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds(D: Data, cfg: LConfig, seeds=range(80)):
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, pd.Series, pd.DataFrame, str))}
                       for r in rows])
    num = df.select_dtypes("number")
    out = num.median().to_dict()
    out["mean_log"] = float(np.mean(np.log(np.maximum(df.final.values, 1e-9) / cfg.capital)))
    out["cagr_p10"] = num.cagr.quantile(0.1)
    out["dd_p10"] = num.max_dd.quantile(0.1)       # 10% 分位（更差的回撤）
    out["dd_worst"] = num.max_dd.min()
    out["min_mr_worst"] = num.min_mr.min()
    out["mc_any"] = float((num.n_mc > 0).mean())
    out["final_min"] = num.final.min()
    return out


def cfg_win(win, **kw):
    s, e = WIN[win]
    return LConfig(start=s, end=e, **kw)


def check_baseline(D=None, seeds=range(80)):
    D = D or Data.load()
    for w in "BFA":
        a = E.run_seeds(D, E.Config(start=WIN[w][0], end=WIN[w][1]), seeds)
        b = run_seeds(D, cfg_win(w), seeds)
        print(w, "engine", round(a["final"] / 1e4, 2), round(a["cagr"] * 100, 2), round(a["max_dd"] * 100, 2),
              "| f_lever", round(b["final"] / 1e4, 2), round(b["cagr"] * 100, 2), round(b["max_dd"] * 100, 2),
              "n", a["n_trades"], b["n_trades"])
    # 逐种子完全一致性检查（含杠杆）
    for c in [dict(cap=1.5, weights=(0.04, 0.16, 0.12)), dict(cap=0.9, sizing="mtm", cap_basis="mtm")]:
        for sd in range(5):
            a = E.run(D, E.Config(**c), sd)
            b = run(D, LConfig(**c), sd)
            assert abs(a["final"] - b["final"]) < 1e-6 and abs(a["max_dd"] - b["max_dd"]) < 1e-12, (c, sd)
    print("per-seed identical incl. cap=1.5 leverage: OK")


if __name__ == "__main__":
    check_baseline()
