"""h_critic_: (1) 回撤日期核对；(2) P1 × 整体放大 L 的邻域（L=1.0~1.2，cap=0.9L，恐慌 4%×L，不融资）。"""
import os, sys
from collections import Counter
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import g_combined_engine as G
from g_combined_plans import make, D, SWAP
def ddmonths(cfg, w):
    return Counter(G.run(D, G.win_cfg(cfg, w), s)["dd_date"][:7] for s in range(80)).most_common(3)
def mk(L):
    return G.GConfig(weights=(0.02 * L, 0.08 * L, 0.04 * L), cap=min(0.9 * L, 1.0), **SWAP)
EV = {"B": ("B", ()), "Bn": ("B", ("2024-02",)), "B-4": ("B", tuple(G.BIG[1:])), "A": ("A", ()), "F": ("F", ())}
def job(a):
    L, e, real = a
    w, drop = EV[e]
    cfg = make() if L == 0 else mk(L)
    _, df = G.run_seeds(D, G.win_cfg(cfg, w, drop, realistic=real, D=D), raw=True)
    return a, df
if __name__ == "__main__":
    print("P1 B 回撤月份", ddmonths(make(S=True), "B"))
    print("P2 B 回撤月份", ddmonths(make(S=True, C=True), "B"))
    print("P3 F 回撤月份", ddmonths(make(S=True, C=True, I=True, res=0.0), "F"))
    print("P1+L11 B 回撤月份", ddmonths(mk(1.1), "B"))
    Ls = [0, 1.0, 1.05, 1.1, 1.15, 1.2]
    jobs = [(L, e, r) for L in Ls for e in EV for r in (False, True)]
    with Pool(3) as pool:
        R = dict(pool.map(job, jobs))
    rows = []
    for (L, e, r), df in R.items():
        b = R[(0, e, r)]
        d = (df.cagr.values - b.cagr.values) * 100
        rows.append(dict(L=L, ev=e + ("r" if r else ""), cell=f"{df.cagr.median()*100:5.2f}/{df.max_dd.median()*100:5.1f}[{df.max_dd.min()*100:5.1f}] {np.median(d):+5.2f}({(d>0).mean()*100:3.0f}%)"))
    pd.set_option("display.width", 300)
    print(pd.DataFrame(rows).pivot(index="L", columns="ev", values="cell").to_string())
