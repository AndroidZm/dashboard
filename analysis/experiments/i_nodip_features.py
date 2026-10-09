"""i_nodip_: 为每笔信号构造“信号日收盘时已知”的特征 + 持有期标签（是否从未跌破买入价等）。

输出 i_nodip_out/features.csv，一行一笔信号（与 signals_main 行序一致，列 s 为行号）。
所有特征只用信号日及之前的数据；同日信号数 n_same_day 在收盘前严格说不可知（见 REPORT 第 1 条），单独标出。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Data  # noqa: E402

OUT = os.path.join(HERE, "i_nodip_out")
os.makedirs(OUT, exist_ok=True)


def ann_vol(c, t, n):
    if t - n < 1:
        return np.nan
    r = np.diff(np.log(c[t - n : t + 1]))
    return float(r.std() * np.sqrt(244))


def main():
    D = Data.load()
    sig = D.sig.copy()
    S = len(sig)
    e = D.entry
    close, op, hi, lo, traded = D.close, D.open, D.high, D.low, D.traded
    dates = pd.to_datetime(pd.Series(D.dates))
    T = len(D.dates)

    # 小盘指数：国证2000，2009-12-31 之前用中证1000 代理（README：日收益相关 0.992）
    idx = D.index.copy()
    sm = idx["sz399303"].astype(float).to_numpy().copy()
    proxy = pd.Series(idx["sh000852"].astype(float).to_numpy()).ffill().bfill().to_numpy()
    first = int(np.argmax(~np.isnan(sm)))
    scale = sm[first] / proxy[first]
    sm[:first] = proxy[:first] * scale
    sm = pd.Series(sm).ffill().bfill().to_numpy()
    hs = pd.Series(idx["sh000300"].astype(float).to_numpy()).ffill().bfill().to_numpy()
    sh = pd.Series(idx["sh000001"].astype(float).to_numpy()).ffill().bfill().to_numpy()

    def idx_feats(a, t, p):
        f = {}
        for n in (1, 3, 5, 10, 20, 60, 120, 250):
            f[f"{p}_ret{n}"] = a[t] / a[t - n] - 1 if t - n >= 0 else np.nan
        for n in (20, 60, 250):
            w = a[max(0, t - n + 1) : t + 1]
            f[f"{p}_dd{n}"] = a[t] / w.max() - 1
            f[f"{p}_bounce{n}"] = a[t] / w.min() - 1
            f[f"{p}_days_since_low{n}"] = len(w) - 1 - int(np.argmin(w))
            f[f"{p}_days_since_high{n}"] = len(w) - 1 - int(np.argmax(w))
        for n in (5, 20, 60, 120, 250):
            w = a[max(0, t - n + 1) : t + 1]
            f[f"{p}_ma{n}"] = a[t] / w.mean() - 1
        f[f"{p}_vol20"] = ann_vol(a, t, 20)
        f[f"{p}_up_days5"] = int((np.diff(a[t - 5 : t + 1]) > 0).sum()) if t >= 5 else np.nan
        return f

    # 全表信号的日历，用来算簇位置
    sig_t = e.copy()
    sig_day = dates.iloc[sig_t].values
    sd = pd.to_datetime(sig.signal_date)
    cnt_by_t = np.bincount(sig_t, minlength=T)

    rows = []
    for s in range(S):
        t = int(e[s])
        c = close[s]
        f = {"s": s}
        # ---- 个股 ----
        f["ret1"] = c[t] / c[t - 1] - 1
        f["gap"] = op[s, t] / c[t - 1] - 1
        f["oc"] = c[t] / op[s, t] - 1
        f["rng"] = (hi[s, t] - lo[s, t]) / c[t - 1]
        f["clv"] = (c[t] - lo[s, t]) / (hi[s, t] - lo[s, t]) if hi[s, t] > lo[s, t] else np.nan
        for n in (3, 5, 10, 20, 60, 120, 250):
            f[f"ret{n}"] = c[t] / c[t - n] - 1 if t - n >= 0 else np.nan
        for n in (20, 60, 250, 500):
            w = c[max(0, t - n + 1) : t + 1]
            f[f"dd{n}"] = c[t] / w.max() - 1
            f[f"bounce{n}"] = c[t] / w.min() - 1
            f[f"days_since_low{n}"] = len(w) - 1 - int(np.argmin(w))
            f[f"days_since_high{n}"] = len(w) - 1 - int(np.argmax(w))
        w = c[: t + 1]
        f["dd_all"] = c[t] / w.max() - 1
        for n in (5, 10, 20, 60, 120, 250):
            w = c[max(0, t - n + 1) : t + 1]
            f[f"ma{n}"] = c[t] / w.mean() - 1
        f["ma20_slope5"] = c[t - 4 : t + 1].mean() / c[t - 9 : t - 4].mean() - 1 if t >= 9 else np.nan
        f["ma60_slope20"] = c[t - 19 : t + 1].mean() / c[t - 79 : t - 59].mean() - 1 if t >= 79 else np.nan
        for n in (10, 20, 60, 250):
            f[f"vol{n}"] = ann_vol(c, t, n)
        r20 = np.diff(np.log(c[t - 20 : t + 1]))
        f["maxdrop20"] = float(np.exp(r20.min()) - 1)
        f["maxup20"] = float(np.exp(r20.max()) - 1)
        f["up_days5"] = int((np.diff(c[t - 5 : t + 1]) > 0).sum())
        f["up_days20"] = int((np.diff(c[t - 20 : t + 1]) > 0).sum())
        k = 0
        while t - k - 1 >= 0 and c[t - k] > c[t - k - 1]:
            k += 1
        f["streak_up"] = k
        hl = (hi[s, t - 19 : t + 1] - lo[s, t - 19 : t + 1]) / c[t - 20 : t]
        f["amp20"] = float(np.nanmean(hl)) if np.isfinite(hl).any() else np.nan
        f["susp20"] = int((~traded[s, t - 19 : t + 1]).sum())
        f["bars_before_signal"] = int(sig.bars_before_signal.iloc[s])
        f["price_raw"] = float(sig.buy_price_raw.iloc[s])
        f["log_price"] = float(np.log(sig.buy_price_raw.iloc[s]))
        name = str(sig.name.iloc[s])
        f["is_st"] = int("ST" in name.upper())
        code = str(sig.code.iloc[s])
        f["board"] = {"60": "沪主板", "00": "深主板", "30": "创业板", "68": "科创板"}.get(code[:2], "其它")
        f["code3"] = code[:3]
        f["prior_signals_same_stock"] = int(((sig.code.values == code) & (e < t)).sum())
        # ---- 指数 ----
        f.update(idx_feats(sm, t, "sm"))
        f.update(idx_feats(hs, t, "hs"))
        f["sh_ret1"] = sh[t] / sh[t - 1] - 1
        f["sh_ret20"] = sh[t] / sh[t - 20] - 1
        f["sh_dd250"] = sh[t] / sh[max(0, t - 249) : t + 1].max() - 1
        f["rel1"] = f["ret1"] - f["sm_ret1"]
        f["rel5"] = f["ret5"] - f["sm_ret5"]
        f["rel20"] = f["ret20"] - f["sm_ret20"]
        f["rel60"] = f["ret60"] - f["sm_ret60"]
        f["rel250"] = f["ret250"] - f["sm_ret250"] if not np.isnan(f["ret250"]) else np.nan
        # ---- 簇位置 ----
        day = dates.iloc[t]
        f["n_signals_30d"] = int(sig.n_signals_30d.iloc[s])
        f["n_same_day"] = int(cnt_by_t[t])
        f["n_prev1"] = int(cnt_by_t[t - 1])
        f["n_prev5"] = int(cnt_by_t[t - 5 : t].sum())
        f["n_prev10"] = int(cnt_by_t[t - 10 : t].sum())
        f["n_prev30_excl"] = f["n_signals_30d"] - f["n_same_day"]
        f["n_prev60_bars"] = int(cnt_by_t[t - 60 : t].sum())
        prev = sd[(sd < day)]
        f["days_since_prev_signal"] = int((day - prev.max()).days) if len(prev) else 9999
        win = sd[(sd < day) & (sd >= day - pd.Timedelta(days=29))]
        if len(win):
            start = win.min()
            f["days_since_cluster_start"] = int((day - start).days)
            t_start = int(np.searchsorted(D.dates, start.strftime("%Y-%m-%d")))
            f["sm_ret_since_cluster_start"] = sm[t] / sm[t_start] - 1
            f["stock_ret_since_cluster_start"] = c[t] / c[t_start] - 1
        else:
            f["days_since_cluster_start"] = 0
            f["sm_ret_since_cluster_start"] = 0.0
            f["stock_ret_since_cluster_start"] = 0.0
        # ---- 标签 ----
        x = int(np.searchsorted(D.dates, sig.exit_date.iloc[s]))
        seg = c[t + 1 : x + 1]
        f["mae_close"] = float(seg.min() / c[t] - 1) if len(seg) else 0.0
        lows = lo[s, t + 1 : x + 1]
        f["mae_low"] = float(np.nanmin(lows) / c[t] - 1) if len(lows) and np.isfinite(lows).any() else np.nan
        # 买入后 5/10/20 根内的最低收盘（早期是否跌破）
        for n in (5, 10, 20, 60):
            seg_n = c[t + 1 : min(x, t + n) + 1]
            f[f"min_close_{n}"] = float(seg_n.min() / c[t] - 1) if len(seg_n) else 0.0
        f["exit_kind"] = sig.exit_kind.iloc[s]
        f["ret_pct"] = float(sig.ret_pct.iloc[s])
        f["held_bars"] = int(sig.held_bars.iloc[s])
        f["mfe_pct"] = float(sig.mfe_pct.iloc[s])
        f["bars_to_plus20"] = sig.bars_to_plus20.iloc[s]
        rows.append(f)

    F = pd.DataFrame(rows)
    F.insert(1, "code", sig.code.values)
    F.insert(2, "name", sig.name.values)
    F.insert(3, "signal_date", sig.signal_date.values)
    F.insert(4, "tier", sig.tier.values)
    F.insert(5, "ym", sig.signal_date.str[:7].values)
    F["nodip0"] = (F.mae_close >= -1e-12).astype(int)
    F["nodip3"] = (F.mae_close >= -0.03).astype(int)
    F["nodip5"] = (F.mae_close >= -0.05).astype(int)
    F["nodip_low"] = (F.mae_low >= -1e-12).astype(int)
    F["tp"] = (F.exit_kind == "TP").astype(int)
    F["fast_tp60"] = ((F.exit_kind == "TP") & (F.held_bars <= 60)).astype(int)
    # 未平仓且持有不足 120 根的，“是否跌破”还没定，标签不可用
    F["label_ok"] = ~((F.exit_kind == "OPEN") & (F.held_bars < 120) & (F.mae_close >= -1e-12))
    F["window"] = np.where(F.signal_date < "2016-01-01", "A", "B")
    F.to_csv(os.path.join(OUT, "features.csv"), index=False, encoding="utf-8-sig")
    print(F.shape, "nodip0:", F.nodip0.sum(), "label_ok:", F.label_ok.sum())


if __name__ == "__main__":
    main()
