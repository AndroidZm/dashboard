"""h_critic_: 对 g_combined 三套方案的补充攻击性检查（不改 engine.py / g_combined_engine.py）。

1  换仓卖出是否发生在停牌日（引擎 swap_order 没检查 D.traded）
2  档位的"当日可知性"：n30 含当日全表信号。严格因果版 = 前 29 天信号数 + 1（只知道自己）
3  档位阈值敏感性（平常/调整 与 调整/恐慌 分界）
4  终点敏感性：B 窗口终点放在 2024-01-31（2024-02 簇之前）等
5  现金利率敏感性
6  费用敏感性（换仓提高换手）
输出 h_critic_out/*.csv
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import g_combined_engine as G
from g_combined_plans import make, D

OUT = os.path.join(HERE, "h_critic_out")
os.makedirs(OUT, exist_ok=True)

PL = {"base": make(), "P1": make(S=True), "P2": make(S=True, C=True), "P3": make(S=True, C=True, I=True, res=0.0),
      "K1": make(C=True)}

# ---- 档位定义 ----
d = pd.to_datetime(D.sig.signal_date)
dv = d.values
n30 = D.n30.astype(int)
same = np.array([(dv == x).sum() for x in dv])
prev = n30 - same  # 前 29 个自然日（不含当日）


def tier_of(n, a=5, b=20):
    return np.where(n <= a, 0, np.where(n <= b, 1, 2)).astype(D.tier.dtype)


TIERS = {
    "orig(5,20)": None,
    "causal(prev+1)": tier_of(prev + 1),
    "thr(5,15)": tier_of(n30, 5, 15),
    "thr(5,25)": tier_of(n30, 5, 25),
    "thr(5,30)": tier_of(n30, 5, 30),
    "thr(3,20)": tier_of(n30, 3, 20),
    "thr(8,20)": tier_of(n30, 8, 20),
}
assert (tier_of(n30) == D.tier).all()


def job(a):
    kind, plan, key, win, drop, extra = a
    cfg = PL[plan]
    if kind == "tier" and TIERS[key] is not None:
        cfg = replace(cfg, tier_override=TIERS[key])
    c = G.win_cfg(cfg, win if win in G.WIN else "B", drop, realistic=bool(extra.get("real")), D=D)
    if win not in G.WIN:
        c = replace(c, end=win)
    for k, v in extra.items():
        if k != "real":
            c = replace(c, **{k: v})
    _, df = G.run_seeds(D, c, raw=True)
    return dict(kind=kind, plan=plan, key=key, win=win, drop="|".join(drop), real=bool(extra.get("real")),
                cagr=df.cagr.values, final=df.final.values, dd=df.max_dd.values, npanic=df.n_panic.values)


def swap_suspended():
    rows = []
    for plan in ("P1", "P2"):
        for win in ("A", "B"):
            for real in (False, True):
                tot = sus = 0
                for seed in range(20):
                    r = G.run(D, replace(G.win_cfg(PL[plan], win, realistic=real), record=True), seed)
                    t = r["trades"]
                    sw = t[t.kind == "SWAP"]
                    tot += len(sw)
                    sus += int((~D.traded[sw.s.values, sw.sell_t.values]).sum())
                rows.append(dict(plan=plan, win=win, real=real, swaps=tot, suspended=sus))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print("== 1 换仓卖出落在停牌日（20 种子合计）")
    print(swap_suspended().to_string())

    print("\n== 档位：严格因果版与原版不同的恐慌信号（按月）")
    tc = TIERS["causal(prev+1)"]
    diff = (D.tier == 2) & (tc < 2)
    print(pd.Series(D.sig.signal_date.str[:7].values[diff]).value_counts().sort_index().to_string())

    jobs = []
    for key in TIERS:
        for plan in ("base", "P1", "P2", "P3"):
            for win, drop in (("A", ()), ("B", ()), ("F", ()), ("B", ("2024-02",)), ("A", ("2012-01",))):
                jobs.append(("tier", plan, key, win, drop, {}))
    ENDS = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-01-31",
            "2024-12-31", "2025-06-30", "2025-12-31"]
    for e in ENDS:
        for plan in ("base", "P1", "P2", "P3", "K1"):
            for real in (False, True):
                jobs.append(("end", plan, e, e, (), {"real": real}))
    for cr in (0.01, 0.025, 0.03):
        for plan in ("base", "P1", "P2", "K1"):
            for win in ("B", "F"):
                jobs.append(("cash", plan, str(cr), win, (), {"cash_rate": cr}))
    for fee in (0.0015, 0.002):
        for plan in ("base", "P1", "P2", "K1"):
            for win in ("B", "F"):
                jobs.append(("fee", plan, str(fee), win, (), {"fee": fee}))
                jobs.append(("fee", plan, str(fee) + "r", win, (), {"fee": fee, "real": True}))
    print(f"\n{len(jobs)} jobs")
    with Pool(3) as p:
        res = p.map(job, jobs, chunksize=2)
    R = {(r["kind"], r["plan"], r["key"], r["win"], r["drop"], r["real"]): r for r in res}
    rows = []
    for r in res:
        b = R[(r["kind"], "base", r["key"], r["win"], r["drop"], r["real"])]
        dd = (r["cagr"] - b["cagr"]) * 100
        rows.append(dict(kind=r["kind"], plan=r["plan"], key=r["key"], win=r["win"], drop=r["drop"], real=r["real"],
                         cagr=np.median(r["cagr"]) * 100, final=np.median(r["final"]) / 1e4, dd=np.median(r["dd"]) * 100,
                         dd_worst=r["dd"].min() * 100, npanic=np.median(r["npanic"]),
                         d_pp=np.median(dd), win_pct=(dd > 0).mean() * 100))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "checks.csv"), index=False)
    for kind in ("tier", "end", "cash", "fee"):
        s = T[T.kind == kind].copy()
        s["cell"] = s.apply(lambda x: f"{x.cagr:6.2f}/{x.dd:6.1f} {x.d_pp:+5.2f}({x.win_pct:3.0f}%) np{x.npanic:3.0f}", axis=1)
        s["col"] = s.win + np.where(s["drop"] != "", "-" + s["drop"], "") + np.where(s.real, "r", "")
        print(f"\n== {kind}: 年化%/回撤%  配对差pp(胜出%)  恐慌笔数")
        print(s.pivot_table(index=["key", "plan"], columns="col", values="cell", aggfunc="first").to_string())
