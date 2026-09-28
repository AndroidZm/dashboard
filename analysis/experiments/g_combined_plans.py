"""三套操作方案（稳健 / 进取 / 激进）的最终对比表。可独立运行：

    python analysis/experiments/g_combined_plans.py        # 约 3~6 分钟（3 进程）

引擎: g_combined_engine.py（已由 g_combined_check.py 证明在各选项单独打开时与 engine.run /
b_panic_engine / d_idle_engine / v_c_exit_lib 逐种子完全一致）。结论一律看 80 个执行种子的中位数。

旋钮（复核结论）:
  S  恐慌换仓（robust）: 恐慌档新信号被总仓位上限挡住时，按买入日期从早到晚卖出非恐慌档持仓，卖到刚好够买为止
     （卖完仍买不下就一只不卖），恐慌档单笔 4%，取消"恐慌持仓满 15 只不买"。
  C  总仓位上限 90% -> 100%（robust，本质是多用闲置现金，回撤加深约 4~5pp）
  W  平常/调整档 2%/8% -> 3%/12%（fragile：A 窗口的增益全部来自 2008-09 的几笔，B 窗口约 +0.2）
  E  移动止盈：收盘首次 >= +50% 后，从最高收盘回落 10% 卖；止损放宽到 -70%（fragile：少数股票、幸存者偏差）
  I  闲置现金：沪深300 收盘在 120 日均线上方（前一日判断）时，超过总权益 res 的现金买沪深300，每日再平衡（fragile）
输出: 终端表格 + g_combined_out/plans_*.csv
"""
from __future__ import annotations

import copy
import itertools
import math
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import g_combined_engine as G
import v_c_exit_lib as VX

OUT = os.path.join(HERE, "g_combined_out")
os.makedirs(OUT, exist_ok=True)
G.WIN["B1"] = ("2016-01-01", "2020-12-31")
G.WIN["B2"] = ("2021-01-01", G.END_DATE)
D = G.Data.load()
REG = G.reg_trend_h(D, "sh000300", ma=120, h=0.0, lag=1)
SWAP = dict(swap_policy="oldest", swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
EXIT = {"kind": "trail", "act": 0.5, "trail": 0.1, "sl": 0.7, "tp": None}


def make(S=False, C=False, W=False, E=False, I=False, res=0.3):
    kw = dict(SWAP) if S else {}
    wn, wa = (0.03, 0.12) if W else (0.02, 0.08)
    cfg = G.GConfig(cap=1.0 if C else 0.9, weights=(wn, wa, 0.04 if S else 0.06), **kw)
    if E:
        cfg = replace(cfg, exit_rule=EXIT)
    if I:
        cfg = replace(cfg, idx="sh000300", reserve=res, regime=REG, rebal="daily")
    return cfg


PLANS = {
    "BASE 基线 2/8/6 cap90%": make(),
    "P1 稳健 = S(换仓)": make(S=True),
    "  P1 + E(移动止盈)": make(S=True, E=True),
    "  P1 + I(沪深300择时,留30%)": make(S=True, I=True),
    "P2 进取 = S + C(满仓)": make(S=True, C=True),
    "  P2 + W(3/12)": make(S=True, C=True, W=True),
    "  P2 + E": make(S=True, C=True, E=True),
    "  P2 + I(留30%)": make(S=True, C=True, I=True),
    "P3 激进 = S + C + I(不留现金)": make(S=True, C=True, I=True, res=0.0),
    "  P3 + W": make(S=True, C=True, W=True, I=True, res=0.0),
    "  P3 + W + E(全部叠加)": make(S=True, C=True, W=True, E=True, I=True, res=0.0),
}
KEYP = ["P1 稳健 = S(换仓)", "P2 进取 = S + C(满仓)", "P3 激进 = S + C + I(不留现金)"]
EVALS = {
    "B": ("B", (), False), "A": ("A", (), False), "F": ("F", (), False),
    "Bn": ("B", ("2024-02",), False), "Fn": ("F", ("2024-02",), False),
    "Br": ("B", (), True), "Ar": ("A", (), True), "Fr": ("F", (), True),
    "B1": ("B1", (), False), "B2": ("B2", (), False), "B2r": ("B2", (), True),
}
LOCO = {
    "A-2012-01": ("A", ("2012-01",)), "B-2018-02": ("B", ("2018-02",)), "B-2022-05": ("B", ("2022-05",)),
    "B-2022-10": ("B", ("2022-10",)), "B-2024-02": ("B", ("2024-02",)), "B-4大簇": ("B", tuple(G.BIG[1:])),
    "F-2012-01": ("F", ("2012-01",)), "F-2018-02": ("F", ("2018-02",)), "F-2022-05": ("F", ("2022-05",)),
    "F-2022-10": ("F", ("2022-10",)), "F-2024-02": ("F", ("2024-02",)), "F-5大簇": ("F", tuple(G.BIG)),
}
COLS = ["final", "cagr", "max_dd", "avg_exposure", "avg_idx", "n_trades", "n_panic", "n_swap"]


def _job(a):
    name, ev = a
    if ev in EVALS:
        win, drop, real = EVALS[ev]
    else:
        (win, drop), real = LOCO[ev], False
    cfg = PLANS[name] if name in PLANS else SHAP[name]
    _, df = G.run_seeds(D, G.win_cfg(cfg, win, drop, realistic=real, D=D), raw=True)
    return name, ev, df[COLS].reset_index(drop=True)


def summ(df):
    return dict(final=df.final.median(), cagr=df.cagr.median(), dd=df.max_dd.median(), dd_p10=df.max_dd.quantile(0.1),
                dd_worst=df.max_dd.min(), c10=df.cagr.quantile(0.1), c90=df.cagr.quantile(0.9),
                f10=df.final.quantile(0.1), f90=df.final.quantile(0.9), exp=df.avg_exposure.median(),
                idx=df.avg_idx.median(), trades=df.n_trades.median(), npanic=df.n_panic.median(), nswap=df.n_swap.median())


def paired(df, base):
    d = (df.cagr.values - base.cagr.values) * 100
    return np.median(d), (d > 0).mean() * 100


# ---------- Shapley 贡献（S、C、E、I 四个旋钮的 16 种组合，W 另算）----------
FACT = ["S", "C", "E", "I"]
SHAP = {}
for bits in itertools.product([0, 1], repeat=4):
    SHAP["shap_" + "".join(map(str, bits))] = make(**{f: bool(b) for f, b in zip(FACT, bits)})
SHAP_EV = ["B", "A", "F", "Bn", "Br", "Fr"]


def shapley(vals):
    n = len(FACT)
    out = {}
    for i, f in enumerate(FACT):
        tot = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            if bits[i]:
                continue
            k = sum(bits)
            wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            b1 = list(bits)
            b1[i] = 1
            tot += wgt * (vals["".join(map(str, b1))] - vals["".join(map(str, bits))])
        out[f] = tot
    return out


# ---------- 幸存者偏差压力：持有期曾跌破 -40% 的信号抽 q 比例，在跌破后下一交易日按 -95% "退市" ----------
def delist_D(q, draw):
    rng = np.random.default_rng(1000 + draw)
    Dx = copy.copy(D)
    Dx.close = D.close.copy()
    Dx._exit_cache = {}
    for s in range(len(D.entry)):
        e = int(D.entry[s])
        c0 = D.close[s, e]
        hit = np.nonzero((D.close[s, e + 1:] <= 0.6 * c0) & D.traded[s, e + 1:])[0]
        if len(hit) and rng.random() < q:
            tb = e + 1 + hit[0]
            Dx.close[s, tb + 1:] = 0.05 * c0
    return Dx


def stress(names, q=0.3, draws=5):
    rows = []
    for dr in range(draws):
        Dx = delist_D(q, dr)
        VX._CACHE.clear()
        for win in ("B", "F"):
            base = G.run_seeds(Dx, G.win_cfg(PLANS["BASE 基线 2/8/6 cap90%"], win))
            for nm in names:
                m = G.run_seeds(Dx, G.win_cfg(PLANS[nm], win))
                rows.append(dict(plan=nm, win=win, draw=dr, d=(m["cagr"] - base["cagr"]) * 100, cagr=m["cagr"] * 100,
                                 base=base["cagr"] * 100))
    VX._CACHE.clear()
    return pd.DataFrame(rows)


def f1(x):
    return f"{x / 1e4:6.1f}"


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    jobs = [(n, e) for n in PLANS for e in EVALS]
    jobs += [(n, e) for n in ["BASE 基线 2/8/6 cap90%"] + KEYP + ["  P3 + W + E(全部叠加)"] for e in LOCO]
    jobs += [(n, e) for n in SHAP for e in SHAP_EV]
    with Pool(3) as p:
        res = p.map(_job, jobs, chunksize=2)
    R = {(n, e): df for n, e, df in res}
    BASE = "BASE 基线 2/8/6 cap90%"

    # ---- 表 1：主表 ----
    rows = []
    for n in PLANS:
        r = {"方案": n}
        for e in EVALS:
            s = summ(R[(n, e)])
            r[e + "_final"], r[e + "_cagr"], r[e + "_dd"] = s["final"], s["cagr"], s["dd"]
            r[e + "_ddp10"], r[e + "_ddworst"], r[e + "_c10"], r[e + "_c90"] = s["dd_p10"], s["dd_worst"], s["c10"], s["c90"]
            r[e + "_f10"], r[e + "_f90"], r[e + "_exp"], r[e + "_idx"] = s["f10"], s["f90"], s["exp"], s["idx"]
            r[e + "_trades"], r[e + "_npanic"], r[e + "_nswap"] = s["trades"], s["npanic"], s["nswap"]
            dpp, win = paired(R[(n, e)], R[(BASE, e)])
            r[e + "_dpp"], r[e + "_win"] = dpp, win
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "plans_main.csv"), index=False)

    def line(e, title):
        print(f"\n== {title}  (80 种子中位：期末万 / 年化% / 盯市最大回撤%  [回撤 p10 / 最差种子]  年化 p10~p90  | 同种子配对 vs 基线：年化差中位 pp, 胜出种子%)")
        for _, r in T.iterrows():
            print(f"{r['方案']:30s} {f1(r[e+'_final'])} {r[e+'_cagr']*100:6.2f} {r[e+'_dd']*100:6.1f} "
                  f"[{r[e+'_ddp10']*100:6.1f} {r[e+'_ddworst']*100:6.1f}]  {r[e+'_c10']*100:5.2f}~{r[e+'_c90']*100:5.2f}"
                  f"  | {r[e+'_dpp']:+5.2f} {r[e+'_win']:5.0f}%   仓位{r[e+'_exp']*100:3.0f}% 指数{r[e+'_idx']*100:3.0f}% 笔{r[e+'_trades']:4.0f} 恐慌{r[e+'_npanic']:3.0f} 换仓{r[e+'_nswap']:3.0f}")

    # ---- 表 0：三套方案一览（年化% / 盯市最大回撤%，80 种子中位）----
    print("\n== 三套方案一览（期末万 / 年化% / 最大回撤%；实盘 = 信号次日收盘成交 + 每边 0.5% 滑点；种子 = B 窗口年化 p10~p90 与回撤 p10/最差）")
    hdr = f"{'方案':30s} {'B 2016~':>22s} {'A 2006~2016':>22s} {'全期 2006~':>23s} {'B去2024-02':>11s} {'B实盘':>14s} {'全期实盘':>14s} {'2021~':>7s} {'B种子':>24s}"
    print(hdr)
    for n in [BASE, "P1 稳健 = S(换仓)", "  P1 + I(沪深300择时,留30%)", "P2 进取 = S + C(满仓)", "  P2 + I(留30%)",
              "P3 激进 = S + C + I(不留现金)", "  P3 + W + E(全部叠加)"]:
        r = T[T["方案"] == n].iloc[0]
        cell = lambda e: f"{r[e+'_final']/1e4:7.1f}/{r[e+'_cagr']*100:5.2f}/{r[e+'_dd']*100:5.1f}"
        print(f"{n:30s} {cell('B'):>22s} {cell('A'):>22s} {cell('F'):>23s} {r['Bn_cagr']*100:11.2f} "
              f"{r['Br_cagr']*100:6.2f}/{r['Br_dd']*100:6.1f} {r['Fr_cagr']*100:6.2f}/{r['Fr_dd']*100:6.1f} {r['B2_cagr']*100:7.2f} "
              f"{r['B_c10']*100:5.2f}~{r['B_c90']*100:5.2f} {r['B_ddp10']*100:5.1f}/{r['B_ddworst']*100:5.1f}")

    line("B", "窗口 B 2016-01-01~2026-08-21")
    line("A", "窗口 A 2006-01-04~2016-01-28")
    line("F", "全期 2006-01-04~2026-08-21")
    line("Bn", "B 去掉 2024-02 全部信号（基线同样去掉）")
    line("Fn", "全期 去掉 2024-02")
    line("Br", "B 实盘口径：信号次日收盘成交 + 每边 0.5% 滑点（指数费 0.2%）")
    line("Ar", "A 实盘口径")
    line("Fr", "全期 实盘口径")
    line("B1", "子区间 2016-01~2020-12")
    line("B2", "子区间 2021-01~2026-08")
    line("B2r", "子区间 2021-01~2026-08 实盘口径")

    # ---- 表 2：剔簇 ----
    print("\n== 剔簇：同种子配对年化差中位 pp（方案 vs 同样剔簇后的基线）；括号内为方案自身年化% / 回撤%")
    lr = []
    for n in KEYP + ["  P3 + W + E(全部叠加)"]:
        r = {"方案": n}
        for e in LOCO:
            dpp, _ = paired(R[(n, e)], R[(BASE, e)])
            s = summ(R[(n, e)])
            r[e] = f"{dpp:+.2f} ({s['cagr']*100:.2f}/{s['dd']*100:.1f})"
        lr.append(r)
    L = pd.DataFrame(lr).set_index("方案").T
    print(L.to_string())
    L.to_csv(os.path.join(OUT, "plans_loco.csv"))

    # ---- 表 3：Shapley ----
    print("\n== 旋钮贡献（年化 pp）：S 换仓 / C 满仓 / E 移动止盈 / I 沪深300择时 的 16 种组合，Shapley 值；"
          "'单独' = 只在基线上加这一个；'最后加' = 其余三个都在时再加它")
    sr = []
    for e in SHAP_EV:
        vals = {k[5:]: R[(k, e)].cagr.median() * 100 for k in SHAP}
        sh = shapley(vals)
        for i, f in enumerate(FACT):
            solo = "".join("1" if j == i else "0" for j in range(4))
            allb = "1111"
            rest = "".join("0" if j == i else "1" for j in range(4))
            sr.append(dict(win=e, lever=f, shapley=sh[f], alone=vals[solo] - vals["0000"], last=vals[allb] - vals[rest]))
        sr.append(dict(win=e, lever="合计(1111-0000)", shapley=vals["1111"] - vals["0000"], alone=np.nan, last=np.nan))
    SH = pd.DataFrame(sr)
    SH.to_csv(os.path.join(OUT, "plans_shapley.csv"), index=False)
    print(SH.pivot(index="lever", columns="win", values="shapley").round(2).to_string())
    print("-- 单独加:")
    print(SH.pivot(index="lever", columns="win", values="alone").round(2).dropna().to_string())
    print("-- 最后加:")
    print(SH.pivot(index="lever", columns="win", values="last").round(2).dropna().to_string())
    # 两两交互（B 与 Br）
    for e in ("B", "Br", "F"):
        v = {k[5:]: R[(k, e)].cagr.median() * 100 for k in SHAP}
        print(f"-- 交互 {e}: S+E 单独之和 {v['1000']+v['0010']-2*v['0000']:+.2f} vs 同时 {v['1010']-v['0000']:+.2f}; "
              f"S+C 之和 {v['1000']+v['0100']-2*v['0000']:+.2f} vs 同时 {v['1100']-v['0000']:+.2f}; "
              f"E+I 之和 {v['0010']+v['0001']-2*v['0000']:+.2f} vs 同时 {v['0011']-v['0000']:+.2f}")

    # ---- 表 4：W 在 A 窗口的增益来源 ----
    yr = D.sig.signal_date.str[:4].values
    print("\n== W(调整档 12%) 的增益来源：A 窗口剔除 2008 年信号后（80 种子中位年化%）")
    for n0, n1 in [("P2 进取 = S + C(满仓)", "  P2 + W(3/12)")]:
        for lab, mk in [("A 全部", None), ("A 去2008", yr != "2008"), ("F 去2008", yr != "2008")]:
            win = "F" if lab.startswith("F") else "A"
            a = G.run_seeds(D, replace(G.win_cfg(PLANS[n0], win), mask=mk))["cagr"] * 100
            b = G.run_seeds(D, replace(G.win_cfg(PLANS[n1], win), mask=mk))["cagr"] * 100
            print(f"  {lab}: 不加 W {a:.2f}  加 W {b:.2f}  差 {b-a:+.2f}")

    # ---- 表 5：幸存者偏差压力（只影响取消 -40% 止损的 E）----
    print("\n== 幸存者偏差压力：持有期曾跌破 -40% 的信号随机 30% 在跌破后次日按 -95% 退市（5 次抽样 × 80 种子，当日口径）")
    ST = stress(KEYP + ["  P1 + E(移动止盈)", "  P2 + E", "  P3 + W + E(全部叠加)"], q=0.3, draws=5)
    ST.to_csv(os.path.join(OUT, "plans_stress.csv"), index=False)
    g = ST.groupby(["plan", "win"]).agg(d_med=("d", "median"), d_min=("d", "min"), d_max=("d", "max"),
                                         cagr=("cagr", "median"), base=("base", "median"))
    print(g.round(2).to_string())

    # ---- 表 6：跨窗口选参（来自 g_combined_grid.py 的 20 种子网格）----
    gp = os.path.join(OUT, "grid.csv")
    if os.path.exists(gp):
        gd = pd.read_csv(gp)
        sw = gd[gd.swap]
        print("\n== 跨窗口选参（换仓族 cap×平常×调整×恐慌 共 336 组，20 种子）：在一个窗口选年化最高，到另一个窗口看")
        for sel, oth in (("A", "B"), ("B", "A")):
            r = sw.loc[sw[sel + "_cagr"].idxmax()]
            print(f"  在 {sel} 选: cap {r.cap} 权重 {r.wn:.2f}/{r.wa:.2f}/{r.wp:.2f} -> {sel} {r[sel+'_cagr']*100:.2f}%  "
                  f"{oth} {r[oth+'_cagr']*100:.2f}% / 回撤 {r[oth+'_dd']*100:.1f}%  ({oth} 族内最高 {sw[oth+'_cagr'].max()*100:.2f}%)")
        for cap in (0.9, 0.95, 1.0):
            c = sw[sw.cap == cap]
            print(f"  cap {cap}: B 年化 {c.B_cagr.min()*100:.2f}~{c.B_cagr.max()*100:.2f}  A {c.A_cagr.min()*100:.2f}~{c.A_cagr.max()*100:.2f}"
                  f"  B 回撤 {c.B_dd.min()*100:.1f}~{c.B_dd.max()*100:.1f}")
        ns = gd[~gd.swap]
        m = sw.merge(ns, on=["cap", "wn", "wa", "wp"], suffixes=("", "_ns"))
        for e in ("A", "B", "B_no2402"):
            d = (m[e + "_cagr"] - m[e + "_cagr_ns"]) * 100
            print(f"  换仓 vs 同 cap/权重不换仓 ({e}): 中位 {d.median():+.2f}pp, 为正 {(d>0).mean()*100:.0f}% (n={len(d)})")
    print("\n完成。CSV 在", OUT)
