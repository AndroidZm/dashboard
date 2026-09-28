"""检查 1：原引擎（engine.run，非对方副本）80 种子复现 C1/C1b/C2/K1/基线，A/B/F 三窗口；配对差、种子离散、最差种子回撤。"""
from v_a_sizing_lib import *
for w in "BAF":
    b = seeds_raw(cfg(BASE, w))
    print(f"=== window {w}")
    print(f"  BASE {fmt(summ(b))}")
    for n in ["C1", "C1b", "C2", "K1"]:
        x = seeds_raw(cfg(CANDS[n], w))
        print(f"  {n:4s} {fmt(summ(x, b))}")
