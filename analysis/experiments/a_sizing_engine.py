"""a_sizing: engine.run 的副本，只加了三处（不改 analysis/engine.py）:

1. no_borrow（默认 True）：单笔金额不超过当时可用现金/(1+fee)，保证 cap<=1 时绝不融资。
   在 sizing='mtm' + cap_basis='cost' 这类组合下，原引擎可能让现金变负（隐性杠杆），这里截断。
   基线参数下该约束从不生效，结果与 engine.run 逐位一致（见 __main__ 自检）。
2. weight_fn 的 state 里多给 cash / equity / exposure / tier（原引擎只有 held）。
   结果里多给 n0/n1/n2（各档成交笔数，数值型，run_seeds 可取中位）和 min_cash_frac（最低现金/权益）。
3. slots=N：单笔 = 权益/N（三档同权），等价于 weights=(1/N,)*3，只是写法方便。

其余逻辑（执行顺序 RNG、出场、计息、盯市曲线）逐行照抄 engine.run。
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Data, Config, END_DATE, _interest_index  # noqa: E402
import engine as _base  # noqa: E402


@dataclass
class SConfig(Config):
    no_borrow: bool = True
    slots: int | None = None      # 单笔 = 权益 / slots（覆盖 weights）


def run(D: Data, cfg: SConfig, seed: int = 0):
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
    weights = cfg.weights if cfg.slots is None else (1.0 / cfg.slots,) * 3

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

    def process_exits(upto):
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
            state["cash"], state["equity"], state["exposure"], state["tier"] = cash, equity, exposure, tier
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
        amount = w * equity
        room = cfg.cap * equity - exposure
        if amount > room:
            if not cfg.partial:
                continue
            amount = room
        if cfg.no_borrow:
            # 可用现金不够（cap=1 的手续费零头，或 mtm/cost 混用的隐性杠杆）时截到现金为止
            amount = min(amount, max(cash, 0.0) / (1 + cfg.fee))
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
    res["min_cash_frac"] = float(np.min(cash_curve / eq))
    lr = np.diff(np.log(eq))
    res["vol"] = float(lr.std() * np.sqrt(244))
    kinds = pd.Series([k for *_, k in trades], dtype=object)
    res["n_tp"] = int((kinds == "TP").sum())
    res["n_sl"] = int((kinds == "SL").sum())
    nb = [int(sum(1 for tr in trades if tiers[tr[0]] == k)) for k in range(3)]
    res["n0"], res["n1"], res["n2"] = nb
    res["n_by_tier"] = tuple(nb)
    if cfg.record:
        res["equity"] = pd.Series(eq, index=D.dates[t0 : t1 + 1])
        res["cost_equity"] = pd.Series(cash_curve + cost_curve, index=D.dates[t0 : t1 + 1])
        res["exposure"] = pd.Series(pos / eq, index=D.dates[t0 : t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds(D: Data, cfg: SConfig, seeds=range(80)):
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, pd.Series, pd.DataFrame, str))} for r in rows])
    out = df.median().to_dict()
    out["cagr_p10"] = df.cagr.quantile(0.1)
    out["cagr_p90"] = df.cagr.quantile(0.9)
    out["dd_p10"] = df.max_dd.quantile(0.1)
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else np.nan
    return out


if __name__ == "__main__":
    D = Data.load()
    for start, end in [("2016-01-01", END_DATE), ("2006-01-04", END_DATE), ("2006-01-04", "2016-01-28")]:
        a = _base.run_seeds(D, Config(start=start, end=end))
        b = run_seeds(D, SConfig(start=start, end=end))
        print(start, end, "engine:", round(a["final"] / 1e4, 2), round(a["cagr"] * 100, 2), round(a["max_dd"] * 100, 2),
              "| copy:", round(b["final"] / 1e4, 2), round(b["cagr"] * 100, 2), round(b["max_dd"] * 100, 2),
              "n_tier", b["n0"], b["n1"], b["n2"])
        for sd in range(5):
            ra, rb = _base.run(D, Config(start=start, end=end), sd), run(D, SConfig(start=start, end=end), sd)
            assert abs(ra["final"] - rb["final"]) < 1e-6 and abs(ra["max_dd"] - rb["max_dd"]) < 1e-12
    # 对照：cap=1.0 下原引擎与 no_borrow 版差异（仅手续费级别）
    for cfg in [dict(cap=1.0, weights=(0.05, 0.1, 0.1), panic_max_pos=None, partial=True),
                dict(cap=1.0, weights=(0.05, 0.1, 0.1), panic_max_pos=None, sizing="mtm", cap_basis="cost")]:
        a = _base.run_seeds(D, Config(**cfg), seeds=range(20))
        b = run_seeds(D, SConfig(**cfg), seeds=range(20))
        print(cfg, "engine", round(a["final"] / 1e4, 1), "copy", round(b["final"] / 1e4, 1), "min_cash", round(b["min_cash_frac"], 4))
    print("reproduction OK")
