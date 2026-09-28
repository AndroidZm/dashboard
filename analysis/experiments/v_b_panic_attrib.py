"""复核 8：A / B 窗口内“只在某个月启用换仓”的逐簇归因（相对同配置不换仓），80 种子；含次日+0.5%滑点。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
import numpy as np
from v_b_panic_engine import Data, VConfig, run_seeds, WA, WB

D = Data.load(); ym = D.sig.signal_date.str[:7].values; t2 = D.tier == 2
months = [m for m in sorted(set(ym[t2])) if ((ym == m) & t2).sum() >= 2]
SW = dict(panic_max_pos=None, swap=True, max_n=99, keep_panic=True)
for n, w, cap in (("L1", (0.02, 0.08, 0.04), 0.9), ("L2", (0.03, 0.12, 0.04), 1.0)):
    for lag, slip in ((0, 0), (1, 0.005)):
        for tag, (st, en) in (("A", WA), ("B", WB)):
            ns = run_seeds(D, VConfig(start=st, end=en, weights=w, cap=cap, panic_max_pos=None, lag=lag, slip=slip))
            line = f"{n} {tag} lag{lag}:"
            for m in months:
                if not (st[:7] <= m <= en[:7]):
                    continue
                r = run_seeds(D, VConfig(start=st, end=en, weights=w, cap=cap, lag=lag, slip=slip, swap_mask=(ym == m), **SW))
                line += f" {m}:{(r['cagr']-ns['cagr'])*100:+.2f}"
            print(line, flush=True)
