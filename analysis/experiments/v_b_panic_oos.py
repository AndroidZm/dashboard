"""复核 6：样本外选择（80 种子）。
(1) 换仓规则族（cap .9, 2/8 不变）：卖出对象 × 恐慌权重 × 是否保留恐慌持仓；A 选→B 看、B 选→A 看，含次日+滑点。
(2) cap × 平常 × 调整 × 恐慌权重 × 有/无换仓；按绝对回撤约束 -30% / -40% / 不限，在一个窗口选、另一窗口看。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, pandas as pd
from v_b_panic_engine import Data, VConfig, run_seeds, WA, WB, WF

D = Data.load()
OUT = "/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/"


def ev(job):
    key, cfg = job
    row = dict(key)
    for tag, (st, en), lag, slip in (("A", WA, 0, 0), ("B", WB, 0, 0), ("F", WF, 0, 0), ("Ax", WA, 1, 0.005), ("Bx", WB, 1, 0.005)):
        m = run_seeds(D, replace(cfg, start=st, end=en, lag=lag, slip=slip))
        row[tag] = m["cagr"] * 100; row[tag + "_dd"] = m["max_dd"] * 100; row[tag + "_fin"] = m["final"] / 1e4
    return row


if __name__ == "__main__":
    jobs = []
    for sk, w2, kp in itertools.product(["oldest", "newest", "underwater", "random"], [0.02, 0.03, 0.04, 0.05, 0.06, 0.08], [True, False]):
        jobs.append((dict(g="fam", sk=sk, w2=w2, kp=kp), VConfig(weights=(0.02, 0.08, w2), panic_max_pos=None, swap=True, max_n=99, keep_panic=kp, sell_key=sk)))
    for cap, wn, wa in itertools.product([0.9, 0.95, 1.0], [0.02, 0.03, 0.04], [0.06, 0.08, 0.10, 0.12, 0.16]):
        jobs.append((dict(g="grid", cap=cap, wn=wn, wa=wa, sw="-", w2=0.06), VConfig(cap=cap, weights=(wn, wa, 0.06))))
        for w2 in (0.03, 0.04, 0.06):
            jobs.append((dict(g="grid", cap=cap, wn=wn, wa=wa, sw="oldest", w2=w2),
                         VConfig(cap=cap, weights=(wn, wa, w2), panic_max_pos=None, swap=True, max_n=99, keep_panic=True)))
    jobs.append((dict(g="base"), VConfig()))
    with Pool(3) as p:
        rows = p.map(ev, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "v_oos.csv", index=False)
    print("done", len(df))
