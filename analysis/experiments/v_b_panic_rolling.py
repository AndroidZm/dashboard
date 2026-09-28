"""复核 7：(a) 滚动 5 年窗口（逐年起点）与 B 窗口起点平移；(b) 同种子配对胜率；(c) 出场 ±30%/±50% 下换仓是否仍有增益。80 种子。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, pandas as pd
from v_b_panic_engine import Data, VConfig, run_seeds, WA, WB, WF, END_DATE

D = Data.load()
SW = dict(panic_max_pos=None, swap=True, max_n=99, keep_panic=True, sell_key="oldest")
C = {"base": VConfig(), "L1": VConfig(weights=(0.02, 0.08, 0.04), **SW),
     "L1b": VConfig(weights=(0.03, 0.12, 0.04), **SW), "L1b_ns": VConfig(weights=(0.03, 0.12, 0.06)),
     "L2": VConfig(cap=1.0, weights=(0.03, 0.12, 0.04), **SW), "L2_ns": VConfig(cap=1.0, weights=(0.03, 0.12, 0.06)),
     "L3": VConfig(cap=1.0, weights=(0.03, 0.16, 0.04), **SW), "L3_ns": VConfig(cap=1.0, weights=(0.03, 0.16, 0.06))}
WINS = [(f"R{y}-{y+5}", f"{y}-01-01", f"{y+5}-01-01") for y in range(2006, 2022)]
WINS += [(f"B@{s}", s, END_DATE) for s in ("2016-07-01", "2017-01-01", "2017-07-01", "2018-06-01", "2019-01-01", "2020-01-01", "2021-01-01")]
WINS += [("A@2007", "2007-01-04", WA[1]), ("A@2009", "2009-01-05", WA[1]), ("A->2013", WA[0], "2013-12-31"), ("A->2014", WA[0], "2014-12-31"),
         ("B->2021", WB[0], "2021-12-31"), ("B->2023", WB[0], "2023-12-31")]


def ev(w):
    tag, st, en = w
    row = dict(win=tag)
    for n, c in C.items():
        m = run_seeds(D, replace(c, start=st, end=en))
        row[n] = m["cagr"] * 100
        row[n + "_dd"] = m["max_dd"] * 100
    return row


if __name__ == "__main__":
    with Pool(3) as p:
        rows = p.map(ev, WINS)
    df = pd.DataFrame(rows).set_index("win")
    out = pd.DataFrame({"base": df.base, "base_dd": df.base_dd, "L1-base": df.L1 - df.base, "L1_dd": df.L1_dd,
                        "L1b-base": df.L1b - df.base, "L1b-ns": df.L1b - df.L1b_ns,
                        "L2-base": df.L2 - df.base, "L2-ns": df.L2 - df.L2_ns, "L2_dd": df.L2_dd,
                        "L3-base": df.L3 - df.base, "L3-ns": df.L3 - df.L3_ns, "L3_dd": df.L3_dd})
    pd.set_option("display.width", 250)
    print(out.round(2).to_string())
    r = out.loc[[i for i in out.index if i.startswith("R")]]
    print("rolling 5y: frac>0", (r[["L1-base", "L1b-base", "L1b-ns", "L2-base", "L2-ns", "L3-base", "L3-ns"]] > 0).mean().round(2).to_dict())

    # (b) 配对
    from v_b_panic_engine import run
    for n in ("L1", "L1b", "L2", "L3"):
        for tag, (st, en) in (("A", WA), ("B", WB), ("F", WF)):
            a = np.array([run(D, replace(C[n], start=st, end=en), s)["cagr"] for s in range(80)])
            b = np.array([run(D, replace(C["base"], start=st, end=en), s)["cagr"] for s in range(80)])
            print(f"paired {n} {tag}: frac seeds cand>base {np.mean(a > b):.2f}; min diff {np.min(a-b)*100:+.2f}pp; min cand cagr {a.min()*100:.2f} vs base max {b.max()*100:.2f}")
    # (c) 出场 ±30/±50
    for tp in (0.3, 0.5):
        for tag, (st, en) in (("A", WA), ("B", WB), ("F", WF)):
            b = run_seeds(D, VConfig(start=st, end=en, tp=tp, sl=tp))
            l1 = run_seeds(D, replace(C["L1"], start=st, end=en, tp=tp, sl=tp))
            l2 = run_seeds(D, replace(C["L2"], start=st, end=en, tp=tp, sl=tp))
            l2n = run_seeds(D, replace(C["L2_ns"], start=st, end=en, tp=tp, sl=tp))
            print(f"tp/sl={tp} {tag}: base {b['cagr']*100:.2f} L1 {(l1['cagr']-b['cagr'])*100:+.2f} L2-vs-ns {(l2['cagr']-l2n['cagr'])*100:+.2f}")
