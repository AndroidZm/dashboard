"""v_f_lever s1: 独立复现候选数字（80 种子，engine.run 直接跑），含种子离散、最差回撤、维持担保比例。"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v_f_lever_lib import Data, Config, Lcfg, WIN, fmt, per_seed, summ  # noqa: E402

D = Data.load()
rows = []
CANDS = [("base", 1.0, {}), ("L1.1", 1.1, {}), ("cap1.0", None, dict(cap=1.0)),
         ("L1.4", 1.4, {}), ("L1.5", 1.5, {}), ("L2.0", 2.0, {}), ("L2.2", 2.2, {}), ("L2.5", 2.5, {})]
for name, L, extra in CANDS:
    for w in "BAF":
        for br in ([0.05, 0.08] if (L or 1) > 1.11 else [0.06]):
            if L is None:
                s, e = WIN[w]
                cfg = Config(start=s, end=e, borrow_rate=br, **extra)
            else:
                cfg = Lcfg(w, L, borrow_rate=br, **extra)
            o = summ(per_seed(D, cfg, detail=True))
            rows.append(dict(cand=name, win=w, borrow=br, **o))
            print(f"{name:7s} {w} br{br:.2f} {fmt(o)} | minMR med {o['min_mr_med']:.2f} worst {o['min_mr_worst']:.2f}"
                  f" <1.3 {o['frac_mr_lt130']:.2f} | maxDebt/eq {o['max_debt_eq_med']:.2f} maxGross {o['max_exp']:.2f}"
                  f" calmar {o['calmar']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "v_f_lever_out_s1.csv"), index=False)
