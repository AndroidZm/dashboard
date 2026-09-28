"""复核 2：80 种子复现 基线 + L1/L1b/L2/L3 在 A/B/F，当日成交、次日成交、次日+0.5%滑点；种子分位与最差回撤。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from v_b_panic_engine import Data, VConfig, run_seeds, WA, WB, WF

D = Data.load()
SW = dict(panic_max_pos=None, swap=True, max_n=5, keep_panic=True, sell_key="oldest")
C = {
    "base": VConfig(),
    "cap1_noswap": VConfig(cap=1.0),
    "wa12_noswap": VConfig(weights=(0.03, 0.12, 0.06)),
    "cap1_wa12_noswap": VConfig(cap=1.0, weights=(0.03, 0.12, 0.06)),
    "cap1_wa16_noswap": VConfig(cap=1.0, weights=(0.03, 0.16, 0.06)),
    "L1": VConfig(weights=(0.02, 0.08, 0.04), **SW),
    "L1b": VConfig(weights=(0.03, 0.12, 0.04), **SW),
    "L2": VConfig(cap=1.0, weights=(0.03, 0.12, 0.04), **SW),
    "L3": VConfig(cap=1.0, weights=(0.03, 0.16, 0.04), **SW),
}
for lag, slip in ((0, 0.0), (1, 0.0), (1, 0.005)):
    print(f"==== lag={lag} slip={slip}")
    base = {}
    for n, c in C.items():
        line = n.ljust(17)
        for tag, (st, en) in (("B", WB), ("A", WA), ("F", WF)):
            m = run_seeds(D, replace(c, start=st, end=en, lag=lag, slip=slip))
            if n == "base":
                base[tag] = m
            d = (m["cagr"] - base[tag]["cagr"]) * 100
            line += (f" {tag}:{m['final']/1e4:6.1f}万 {m['cagr']*100:5.2f}%({d:+.2f}) dd{m['max_dd']*100:5.1f} "
                     f"p10/90 {m['cagr_p10']*100:5.2f}/{m['cagr_p90']*100:5.2f} worstDD{m['dd_worst']*100:5.1f} np{m['n_panic']:3.0f} sw{m['n_swap']:3.0f} |")
        print(line, flush=True)
