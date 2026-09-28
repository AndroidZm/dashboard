"""阶段 14：成本压力测试——次日收盘成交 + 每边 0.5% 滑点；以及 80 种子分位数。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F

D = Data.load()
SW5 = dict(swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
C = {
    "base": PConfig(),
    "L1_swap": PConfig(swap_policy="oldest", weights=(0.02, 0.08, 0.04), **SW5),
    "cap1_noswap": PConfig(cap=1.0),
    "L2_cap1_wa12": PConfig(cap=1.0, swap_policy="oldest", weights=(0.03, 0.12, 0.04), **SW5),
}
for lag, slip in ((0, 0.0), (1, 0.005), (0, 0.005)):
    print(f"--- lag={lag} slip={slip}")
    for n, c in C.items():
        line = n.ljust(14)
        for tag, (st, en) in (("A", WIN_A), ("B", WIN_B), ("F", WIN_F)):
            m = run_seeds(D, replace(c, start=st, end=en, lag=lag, slip=slip))
            line += f" {tag}: {m['final']/1e4:6.1f}万 {m['cagr']*100:5.2f}% [p10 {m['cagr_p10']*100:5.2f} p90 {m['cagr_p90']*100:5.2f}] dd{m['max_dd']*100:6.1f} (p10 {m['dd_p10']*100:6.1f}) |"
        print(line)
