"""h_critic_: 第 1 档候选对比。L11 = 杠杆复核判为 robust 的"权重×1.1、上限 99%、不融资"，组合方案没有纳入。
比较 基线 / P1(换仓) / L11 / P1+L11 / P2 / C(cap1.0)，当日与实盘口径，含剔簇与阈值(5,15)。"""
import os, sys
from dataclasses import replace
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import g_combined_engine as G
from g_combined_plans import make, D, SWAP
from h_critic_checks import TIERS
PL = {"base": make(), "P1": make(S=True),
      "L11": G.GConfig(weights=(0.022, 0.088, 0.066), cap=0.99),
      "P1+L11": G.GConfig(weights=(0.022, 0.088, 0.044), cap=0.99, **SWAP),
      "P2": make(S=True, C=True), "C": make(C=True)}
EV = {"B": ("B", ()), "Bn": ("B", ("2024-02",)), "B-4": ("B", tuple(G.BIG[1:])),
      "A": ("A", ()), "A-1201": ("A", ("2012-01",)), "F": ("F", ()), "F-5": ("F", tuple(G.BIG)), "B2": ("B2", ())}
G.WIN["B2"] = ("2021-01-01", G.END_DATE)
def job(a):
    p, e, tk, real = a
    w, drop = EV[e]
    cfg = PL[p] if TIERS[tk] is None else replace(PL[p], tier_override=TIERS[tk])
    _, df = G.run_seeds(D, G.win_cfg(cfg, w, drop, realistic=real, D=D), raw=True)
    return a, df
if __name__ == "__main__":
    jobs = [(p, e, "orig(5,20)", r) for p in PL for e in EV for r in (False, True)]
    jobs += [(p, e, "thr(5,15)", r) for p in PL for e in ("B", "Bn", "A", "F") for r in (False, True)]
    with Pool(3) as pool:
        R = dict(pool.map(job, jobs, chunksize=2))
    rows = []
    for (p, e, tk, r), df in R.items():
        b = R[("base", e, tk, r)]
        d = (df.cagr.values - b.cagr.values) * 100
        rows.append(dict(plan=p, ev=e, tier=tk, real=r, cagr=df.cagr.median() * 100, dd=df.max_dd.median() * 100,
                         ddw=df.max_dd.min() * 100, final=df.final.median() / 1e4, d=np.median(d), win=(d > 0).mean() * 100))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, "h_critic_out", "lvl1.csv"), index=False)
    T["cell"] = T.apply(lambda x: f"{x.cagr:5.2f}/{x.dd:5.1f}[{x.ddw:5.1f}] {x.d:+5.2f}({x.win:3.0f}%)", axis=1)
    pd.set_option("display.width", 300)
    for (tk, r), s in T.groupby(["tier", "real"]):
        print(f"\n== 档位 {tk} {'实盘' if r else '当日'}   年化/中位回撤[最差种子] 配对差(胜出%)")
        print(s.pivot(index="plan", columns="ev", values="cell").to_string())
