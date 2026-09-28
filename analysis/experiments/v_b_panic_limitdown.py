"""复核 4：换仓卖出发生在恐慌日，很多微盘股收盘封死跌停、实际卖不掉。
(a) 原规则下换仓卖出有多少笔落在封死跌停日；(b) 禁止卖出封死跌停(严格)/跌幅>=95%限制(宽松)的持仓后重跑；(c) 叠加次日成交+0.5%滑点。80 种子。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
import numpy as np, pandas as pd
from v_b_panic_engine import Data, VConfig, run, run_seeds, WA, WB, WF, limit_down

D = Data.load(); LD = limit_down(D); LL = limit_down(D, True)
SW = dict(panic_max_pos=None, swap=True, max_n=5, keep_panic=True, sell_key="oldest")
C = {"L1": VConfig(weights=(0.02, 0.08, 0.04), **SW), "L1b": VConfig(weights=(0.03, 0.12, 0.04), **SW),
     "L2": VConfig(cap=1.0, weights=(0.03, 0.12, 0.04), **SW), "L3": VConfig(cap=1.0, weights=(0.03, 0.16, 0.04), **SW)}
NS = {"L2": VConfig(cap=1.0, weights=(0.03, 0.12, 0.06)), "L3": VConfig(cap=1.0, weights=(0.03, 0.16, 0.06))}
# (a)
for n, c in C.items():
    for tag, (st, en) in (("B", WB), ("A", WA)):
        tot = ld = ll = 0
        for sd in range(20):
            tr = run(D, replace(c, start=st, end=en, record=True), sd)["trades"]
            sw = tr[tr.kind == "SWAP"]
            tot += len(sw); ld += int(LD[sw.s.values, sw.sell_t.values].sum()); ll += int(LL[sw.s.values, sw.sell_t.values].sum())
        print(f"{n} {tag}: swap sells/seed {tot/20:.1f}, on locked limit-down {ld/tot*100:.0f}%, on >=95%-limit drop {ll/tot*100:.0f}%")
base = {}
for lag, slip in ((0, 0.0), (1, 0.005)):
    for tag, (st, en) in (("B", WB), ("A", WA), ("F", WF)):
        base[(lag, tag)] = run_seeds(D, VConfig(start=st, end=en, lag=lag, slip=slip))
for n, c in C.items():
    for blk_name, blk in (("none", None), ("strict", LD), ("loose", LL)):
        for lag, slip in ((0, 0.0), (1, 0.005)):
            line = f"{n:4s} block={blk_name:6s} lag{lag} slip{slip}:"
            for tag, (st, en) in (("B", WB), ("A", WA), ("F", WF)):
                m = run_seeds(D, replace(c, start=st, end=en, lag=lag, slip=slip, sell_block=blk))
                b = base[(lag, tag)]
                line += f" {tag} {m['final']/1e4:6.1f}万 {m['cagr']*100:5.2f}% ({(m['cagr']-b['cagr'])*100:+.2f}) dd{m['max_dd']*100:5.1f} sw{m['n_swap']:3.0f} fail{m['n_fail']:3.0f} |"
                if n in NS:
                    mn = run_seeds(D, replace(NS[n], start=st, end=en, lag=lag, slip=slip))
                    line += f" vsNoSwap {(m['cagr']-mn['cagr'])*100:+.2f} |"
            print(line, flush=True)
