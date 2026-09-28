"""h_critic_: 实盘口径（次日收盘 + 0.5% 滑点）下的剔簇 / 阈值 / 因果档位检查。"""
import os, sys
from dataclasses import replace
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import g_combined_engine as G
from g_combined_plans import make, D
from h_critic_checks import TIERS
PL = {"base": make(), "P1": make(S=True), "P2": make(S=True, C=True), "P3": make(S=True, C=True, I=True, res=0.0), "C": make(C=True)}
EV = {"B": ("B", ()), "Bn": ("B", ("2024-02",)), "B-4": ("B", tuple(G.BIG[1:])), "B-18/24": ("B", ("2018-02", "2024-02")),
      "A": ("A", ()), "A-1201": ("A", ("2012-01",)), "F": ("F", ()), "F-5": ("F", tuple(G.BIG))}
def job(a):
    plan, ev, tk, real = a
    w, drop = EV[ev]
    cfg = PL[plan]
    if TIERS[tk] is not None:
        cfg = replace(cfg, tier_override=TIERS[tk])
    _, df = G.run_seeds(D, G.win_cfg(cfg, w, drop, realistic=real, D=D), raw=True)
    return (plan, ev, tk, real), df
if __name__ == "__main__":
    jobs = [(p, e, "orig(5,20)", True) for p in PL for e in EV]
    jobs += [(p, e, tk, r) for p in PL for e in ("B", "Bn", "A", "F") for tk in ("thr(5,15)", "thr(5,30)", "causal(prev+1)") for r in (True,)]
    jobs += [(p, e, "orig(5,20)", False) for p in PL for e in ("B-18/24",)]
    with Pool(3) as pool:
        R = dict(pool.map(job, jobs, chunksize=2))
    rows = []
    for (p, e, tk, r), df in R.items():
        b = R[("base", e, tk, r)]
        d = (df.cagr.values - b.cagr.values) * 100
        rows.append(dict(plan=p, ev=e, tier=tk, real=r, cagr=df.cagr.median() * 100, dd=df.max_dd.median() * 100,
                         d=np.median(d), win=(d > 0).mean() * 100))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, "h_critic_out", "real.csv"), index=False)
    T["cell"] = T.apply(lambda x: f"{x.cagr:5.2f}/{x.dd:5.1f} {x.d:+5.2f}({x.win:3.0f}%)", axis=1)
    pd.set_option("display.width", 250)
    for (tk, r), s in T.groupby(["tier", "real"]):
        print(f"\n== 档位 {tk} {'实盘' if r else '当日'}")
        print(s.pivot(index="plan", columns="ev", values="cell").to_string())
