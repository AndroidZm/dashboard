"""抄底信号组合回测引擎（复现 DATA_README 的 2/8/6 规则，并把各个旋钮参数化）。

用法:
    from engine import Data, Config, run, run_seeds
    D = Data.load()                       # 首次会从 CSV 构建缓存 analysis/cache.npz
    r = run_seeds(D, Config())            # 80 个种子，返回中位数等统计

基线口径（与 results/seeds80_* 对齐）:
  - 初始 60 万，信号日收盘买入，单笔 = 档位权重 × 当时“成本口径”总权益（持仓按成本计）
  - 总仓位（成本口径）上限 90%，放不下整笔就不买；单笔 < 5000 元不买
  - 恐慌档持仓数 >= 15 时不买
  - 收盘价首次触及 +TP / -SL 时按当日收盘卖出；窗口末日按收盘清仓
  - 闲置现金按年化 1.8%（按自然日复利）计息；买卖各 0.1% 费用
  - 同一天多笔信号的执行顺序由种子决定
"""
from __future__ import annotations

import glob
import heapq
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.npz")
END_DATE = "2026-08-21"


class Data:
    """信号 + 按统一交易日历对齐的价格矩阵（行 = 信号，列 = 交易日）。"""

    def __init__(self, z):
        self.dates = z["dates"].astype(str)                 # (T,)
        self.cal_days = z["cal_days"]                       # (T,) 距 2005-01-01 的自然日
        self.close = z["close"]                             # (S,T) 等比复权收盘，停牌日前向填充
        self.open = z["open"]                               # (S,T) 开盘（停牌日为 NaN）
        self.high = z["high"]
        self.low = z["low"]
        self.traded = z["traded"]                           # (S,T) 当日是否有成交
        self.entry = z["entry"]                             # (S,) 信号日在日历中的下标
        self.tier = z["tier"]                               # (S,)
        self.n30 = z["n30"]
        self.sig = pd.read_csv(os.path.join(ROOT, "signals/signals_main_2006_2026.csv"), dtype={"code": str})
        self.index = pd.read_csv(os.path.join(ROOT, "index/index_daily_close.csv")).set_index("date").loc[self.dates]
        self._exit_cache = {}

    @classmethod
    def load(cls):
        if not os.path.exists(CACHE):
            build_cache()
        return cls(np.load(CACHE))

    def day(self, date: str) -> int:
        """第一个 >= date 的交易日下标。"""
        return int(np.searchsorted(self.dates, date))

    def day_le(self, date: str) -> int:
        """最后一个 <= date 的交易日下标。"""
        return int(np.searchsorted(self.dates, date, side="right") - 1)

    def exits(self, tp, sl, max_hold=None, end_t=None, lag=0):
        """每个信号的 (买入日下标, 卖出日下标, 卖出/买入价格比)。

        tp/sl 为比例（0.4 = ±40%），None 表示不设该边。max_hold 为最多持有的交易日数。
        lag=1 表示信号次日收盘买入。窗口末 end_t 前未触发的按 end_t 收盘计。
        """
        end_t = len(self.dates) - 1 if end_t is None else end_t
        key = (tp, sl, max_hold, end_t, lag)
        if key in self._exit_cache:
            return self._exit_cache[key]
        S = len(self.entry)
        e_arr = np.empty(S, np.int64)
        x_arr = np.empty(S, np.int64)
        ratio = np.empty(S)
        for s in range(S):
            e = int(self.entry[s]) + lag
            if e > end_t:
                e_arr[s], x_arr[s], ratio[s] = e, e, 1.0
                continue
            # 次日买入时如停牌，顺延到下一个有成交的日子
            while lag and e < end_t and not self.traded[s, e]:
                e += 1
            c0 = self.close[s, e]
            r = self.close[s, e + 1 : end_t + 1] / c0
            ok = self.traded[s, e + 1 : end_t + 1]
            hit = np.zeros(len(r), bool)
            if tp is not None:
                hit |= r >= 1 + tp
            if sl is not None:
                hit |= r <= 1 - sl
            hit &= ok
            if max_hold is not None:
                if max_hold < len(hit):
                    hit[max_hold - 1 :] |= ok[max_hold - 1 :]
            k = np.argmax(hit) if hit.any() else -1
            x = e + 1 + k if k >= 0 else end_t
            e_arr[s], x_arr[s] = e, x
            ratio[s] = self.close[s, x] / c0
        out = (e_arr, x_arr, ratio)
        self._exit_cache[key] = out
        return out


@dataclass
class Config:
    start: str = "2016-01-01"
    end: str = END_DATE
    capital: float = 600_000.0
    weights: tuple = (0.02, 0.08, 0.06)       # 平常 / 调整 / 恐慌 单笔占总权益
    cap: float = 0.90                          # 总仓位上限（成本口径）
    panic_max_pos: int | None = 15             # 恐慌档持仓数达到该值后不买（None 不限）
    min_amount: float = 5000.0
    tp: float | None = 0.40
    sl: float | None = 0.40
    max_hold: int | None = None                # 最长持有交易日
    fee: float = 0.001
    cash_rate: float = 0.018                   # 闲置现金年化
    borrow_rate: float = 0.06                  # 现金为负（融资）时的年化成本
    sizing: str = "cost"                       # 'cost' 成本口径权益 / 'mtm' 盯市权益
    cap_basis: str = "cost"                    # 仓位上限按 'cost' 或 'mtm' 计
    partial: bool = False                      # 超出仓位上限时 True=只买剩余额度，False=整笔不买（基线）
    lag: int = 0                               # 1 = 次日收盘买入
    slip: float = 0.0                          # 每边滑点
    panic_swap: bool = False                   # 恐慌期被挡时卖掉持有最久的一只再买
    mask: np.ndarray | None = field(default=None, repr=False)  # 只用 mask 为 True 的信号
    tier_override: np.ndarray | None = field(default=None, repr=False)  # 自定义档位 (S,)
    weight_fn: object = field(default=None, repr=False)  # weight_fn(s, t, state)->权重，覆盖 weights
    priority: np.ndarray | None = field(default=None, repr=False)  # (S,) 同日信号执行优先级，大者先买；None=随机
    record: bool = False                       # 返回逐日权益与成交


def _interest_index(D: Data, t0: int, rate: float):
    days = D.cal_days - D.cal_days[t0] + 1   # 起始日当天已计一天息（与 equity_curve 一致）
    return (1 + rate) ** (days / 365.0)


def run(D: Data, cfg: Config, seed: int = 0):
    t0, t1 = D.day(cfg.start), D.day_le(cfg.end)
    e_arr, x_arr, ratio = D.exits(cfg.tp, cfg.sl, cfg.max_hold, t1, cfg.lag)
    tiers = D.tier if cfg.tier_override is None else cfg.tier_override
    sel = (D.entry >= t0) & (e_arr <= t1)
    if cfg.mask is not None:
        sel &= cfg.mask
    idx = np.nonzero(sel)[0]
    rng = np.random.default_rng(seed)
    # 同日信号按种子随机排序
    keys = (rng.random(len(idx)), e_arr[idx]) if cfg.priority is None else \
        (rng.random(len(idx)), -np.nan_to_num(cfg.priority[idx], nan=-np.inf), e_arr[idx])
    order = idx[np.lexsort(keys)]

    I_cash = _interest_index(D, t0, cfg.cash_rate)
    I_borr = _interest_index(D, t0, cfg.borrow_rate)
    # 现金以“贴现单位”记账：正余额按 cash_rate、负余额按 borrow_rate 增长
    cash_pos = cfg.capital / I_cash[t0]
    cash_neg = 0.0
    held = {}                 # s -> (buy_t, amount, units)
    cost_sum = 0.0
    heap = []                 # (exit_t, s)
    trades = []
    flows = []                # (t, delta_cash) 用于逐日曲线

    def cash_at(t):
        return cash_pos * I_cash[t] - cash_neg * I_borr[t]

    def add_cash(t, v):
        nonlocal cash_pos, cash_neg
        if v >= 0:
            # 先还融资
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
        held[s] = (t, amount, units)
        cost_sum += amount
        heapq.heappush(heap, (int(x_arr[s]), s))
    process_exits(t1)
    for s in list(held):
        sell(s, t1, "END")

    final = cash_at(t1)
    years = (D.cal_days[t1] - D.cal_days[t0] + 1) / 365.25
    res = {"final": final, "cagr": (final / cfg.capital) ** (1 / years) - 1, "n_trades": len(trades)}

    # 逐日盯市曲线（交易日）
    T = t1 - t0 + 1
    pos = np.zeros(T)
    cost_curve = np.zeros(T)
    for s, bt, st, amount, pnl, kind in trades:
        u = amount / (D.close[s, bt] * (1 + cfg.slip))
        pos[bt - t0 : st - t0] += u * D.close[s, bt:st]
        cost_curve[bt - t0 : st - t0] += amount
    # 现金曲线：逐笔重放（用贴现单位保证与上面一致）
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
    lr = np.diff(np.log(eq))
    res["vol"] = float(lr.std() * np.sqrt(244))
    kinds = pd.Series([k for *_, k in trades])
    res["n_tp"] = int((kinds == "TP").sum())
    res["n_sl"] = int((kinds == "SL").sum())
    res["n_by_tier"] = tuple(int(sum(1 for tr in trades if tiers[tr[0]] == k)) for k in range(3))
    if cfg.record:
        res["equity"] = pd.Series(eq, index=D.dates[t0 : t1 + 1])
        res["cost_equity"] = pd.Series(cash_curve + cost_curve, index=D.dates[t0 : t1 + 1])
        res["trades"] = pd.DataFrame(trades, columns=["s", "buy_t", "sell_t", "amount", "pnl", "kind"]).assign(
            buy_date=lambda d: D.dates[d.buy_t], sell_date=lambda d: D.dates[d.sell_t],
            code=lambda d: D.sig.code.values[d.s], tier=lambda d: tiers[d.s])
    return res


def run_seeds(D: Data, cfg: Config, seeds=range(80)):
    """跑多个执行顺序种子，返回中位数统计（结论一律看中位数）。"""
    rows = [run(D, replace(cfg, record=False), s) for s in seeds]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (tuple, pd.Series, pd.DataFrame))} for r in rows])
    num = df.select_dtypes("number")
    out = num.median().to_dict()
    out["cagr_p10"] = num.cagr.quantile(0.1)
    out["cagr_p90"] = num.cagr.quantile(0.9)
    out["dd_p10"] = num.max_dd.quantile(0.1)
    # 同回撤年化：把收益按回撤线性缩放到基线回撤 -28.49% 的近似（仅作比较参考）
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else np.nan
    return out


def build_cache():
    sig = pd.read_csv(os.path.join(ROOT, "signals/signals_main_2006_2026.csv"), dtype={"code": str})
    idx = pd.read_csv(os.path.join(ROOT, "index/index_daily_close.csv"))
    dates = idx.date.values.astype(str)
    dates = dates[dates <= END_DATE]
    px = pd.concat(pd.read_csv(f, dtype={"code": str})
                   for f in sorted(glob.glob(os.path.join(ROOT, "prices/signal_stocks_adj_daily_part*.csv.gz"))))
    px = px[px.date <= END_DATE]
    codes = sig.code.unique()
    pos = {d: i for i, d in enumerate(dates)}
    S, T = len(sig), len(dates)
    per = {}
    for c, g in px[px.code.isin(codes)].groupby("code"):
        g = g[g.date.isin(pos)]
        ti = g.date.map(pos).values
        arr = {k: np.full(T, np.nan) for k in ("open", "high", "low", "close")}
        for k in arr:
            arr[k][ti] = g[k].values
        tr = np.zeros(T, bool)
        tr[ti] = True
        cl = pd.Series(arr["close"]).ffill().bfill().values
        per[c] = (cl, arr["open"], arr["high"], arr["low"], tr)
    close = np.empty((S, T))
    op, hi, lo = np.empty((S, T)), np.empty((S, T)), np.empty((S, T))
    traded = np.empty((S, T), bool)
    for i, c in enumerate(sig.code.values):
        close[i], op[i], hi[i], lo[i], traded[i] = per[c]
    entry = np.array([pos[d] for d in sig.signal_date.values])
    cal = (pd.to_datetime(pd.Series(dates)) - pd.Timestamp("2005-01-01")).dt.days.values
    np.savez_compressed(CACHE, dates=np.array([str(d) for d in dates], dtype="U10"), cal_days=cal, close=close, open=op, high=hi, low=lo, traded=traded,
                        entry=entry, tier=sig.tier.values, n30=sig.n_signals_30d.values)
