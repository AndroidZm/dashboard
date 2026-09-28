"""v_f_lever s4: 可行性与压力。
(a) 监管可行杠杆：按盯市口径把总仓位上限压到 1+折算率/融资保证金比例（0.65/1.0 → 1.65；0.5/1.0 → 1.5），
    盯市定额，融资 8%，80 种子。对照候选的成本口径 cap=0.9L。
(b) 冲击压力：中位路径（seed 13 及全部 80 种子）逐日施加一次 -37% 持仓瞬时冲击（2015 年国证2000 最差 20 日 -41% × beta 0.9），
    看冲击后相对前高的回撤、是否跌破 130% 维持担保比例；以及实际路径上距离 130% 还差多少跌幅。
(c) L=3 用 main engine（不强平）独立数一下有多少种子跌破 130%；并用 f_lever_engine 的强平版对照。
(d) 抽查 rejected 里的数字：只放大权重（cap 0.9）、k0.75 cap1.0。
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v_f_lever_lib import WIN, Config, Data, Lcfg, fmt, mmask, per_seed, run, summ  # noqa: E402

D = Data.load()
print("=== (a) 监管可行杠杆（盯市定额 + 盯市上限，融资 8%）")
for gmax in [1.65, 1.5]:
    for L in [1.5, 1.8, 2.2, 2.5]:
        for w in "BAF":
            cfg = Lcfg(w, L, borrow_rate=0.08, sizing="mtm", cap_basis="mtm", cap=min(0.9 * L, gmax))
            o = summ(per_seed(D, cfg, detail=True))
            print(f"Gmax {gmax} L{L} {w}: {fmt(o)} | minMR worst {o['min_mr_worst']:.2f} maxGross {o['max_exp']:.2f}")
    for w in "BAF":
        cfg = Lcfg(w, 1.0, borrow_rate=0.08, sizing="mtm", cap_basis="mtm", cap=gmax, weights=(0.02 * gmax / 0.9, 0.08 * gmax / 0.9, 0.06 * gmax / 0.9))
        o = summ(per_seed(D, cfg))
        print(f"Gmax {gmax} 比例放满(L={gmax/0.9:.2f}) {w}: {fmt(o)}")


def shock_stats(D, cfg, seed, shock=0.37):
    r = run(D, replace(cfg, record=True), seed)
    eq = r["equity"].values
    t0 = D.day(cfg.start)
    pos = np.zeros(len(eq))
    for s, bt, st, amt in r["trades"][["s", "buy_t", "sell_t", "amount"]].itertuples(index=False):
        pos[bt - t0: st - t0] += amt / (D.close[s, bt] * (1 + cfg.slip)) * D.close[s, bt:st]
    debt = np.maximum(pos - eq, 0)
    peak = np.maximum.accumulate(eq)
    eq_after = eq - pos * shock
    sdd = eq_after / peak - 1
    mr_after = np.where(debt > 1, pos * (1 - shock) / np.maximum(debt, 1e-9), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.where(debt > 1, 1 - 1.3 * debt / np.maximum(pos, 1e-9), np.inf)   # 还要跌多少才到 130%
    return dict(stressed_dd=sdd.min(), frac_days_mc=float(np.mean(mr_after < 1.3)),
                min_need=float(need.min()), need_date=D.dates[t0 + int(np.argmin(need))] if np.isfinite(need.min()) else "",
                wipe=float(np.min(eq_after / eq)))


print("\n=== (b) 冲击压力：逐日 -37% 瞬时持仓冲击（80 种子中位 / 最差）")
for L in [1.0, 1.1, 1.5, 2.0, 2.2, 2.5]:
    for w in "BF":
        rows = pd.DataFrame([shock_stats(D, Lcfg(w, L, borrow_rate=0.08), s) for s in range(80)])
        print(f"L{L} {w}: 冲击后最大回撤 中位 {rows.stressed_dd.median()*100:.1f}% 最差 {rows.stressed_dd.min()*100:.1f}% | "
              f"冲击会触发 130% 的交易日占比 中位 {rows.frac_days_mc.median()*100:.1f}% | 实际路径距 130% 最小还差跌幅 中位 "
              f"{rows.min_need.median()*100:.1f}% 最差 {rows.min_need.min()*100:.1f}% (日期众数 {rows.need_date.mode().iloc[0]})")

print("\n=== (c) L=3 / 2.75（cap 0.9L，融资 5%）main engine 不强平：跌破 130% 的种子比例")
for L in [2.5, 2.75, 3.0]:
    for w in "BAF":
        o = summ(per_seed(D, Lcfg(w, L, borrow_rate=0.05), detail=True))
        print(f"L{L} {w}: {fmt(o)} | 跌破130%种子比例 {o['frac_mr_lt130']:.2f} minMR 中位 {o['min_mr_med']:.2f}")
from f_lever_engine import LConfig, run_seeds as frs  # noqa: E402
for L in [2.5, 3.0]:
    for w in "BA":
        s, e = WIN[w]
        m = frs(D, LConfig(start=s, end=e, k=L, cap=0.9 * L, borrow_rate=0.05, margin_call=(1.3, 1.5)))
        print(f"f_lever_engine 强平版 L{L} {w}: {m['final']/1e4:.1f}万 dd {m['max_dd']*100:.1f}% 被强平种子比例 {m['mc_any']:.2f}")

print("\n=== (d) 抽查 rejected 数字（80 种子）")
for k in [1.25, 1.5, 1.75, 2.0]:
    for w in "BA":
        s, e = WIN[w]
        o = summ(per_seed(D, Config(start=s, end=e, weights=(0.02 * k, 0.08 * k, 0.06 * k), cap=0.9)))
        print(f"k_only {k} cap0.9 {w}: {fmt(o)}")
for w in "BA":
    s, e = WIN[w]
    o = summ(per_seed(D, Config(start=s, end=e, weights=(0.015, 0.06, 0.045), cap=1.0)))
    print(f"k0.75 cap1.0 {w}: {fmt(o)}")
