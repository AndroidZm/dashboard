"""第一轮筛选：指数 × 现金保留比例 × regime，三个窗口，20 个种子。"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_idle_out")
SEEDS = range(20)
WINS = {"A": I.WIN_A, "B": I.WIN_B, "F": I.WIN_F}
IDX = ["sh000300", "sh000905", "sh000852", "g2000p", "sz399006"]
RES = [0.0, 0.1, 0.2, 0.3, 0.5]


def regimes(nm):
    R = {"always": None,
         "quiet5": I.reg_quiet(D, thr=5),
         "quiet0": I.reg_quiet(D, thr=0),
         }
    for thr, lab in [(20, "panic"), (5, "adj")]:
        for h in (60, 120, 250):
            R[f"after_{lab}{h}"] = I.reg_after(D, thr=thr, hold=h)
    for ma in (60, 120, 200, 250):
        R[f"ma{ma}"] = I.reg_trend(D, nm, ma)
    for thr in (0.2, 0.3):
        R[f"dd{int(thr*100)}"] = I.reg_dd(D, nm, 250, thr)
    return R


rows = []
t = time.time()
# 基线与买入持有
for w, (st, en) in WINS.items():
    m = I.run_seeds(D, I.IConfig(start=st, end=en), SEEDS)
    rows.append(dict(win=w, idx="none", reserve=1.0, regime="base", **m))
    for nm in IDX:
        if nm == "sz399006" and w != "B":
            continue
        bh = I.buy_hold(D, nm, st, en)
        rows.append(dict(win=w, idx=nm, reserve=-1, regime="buyhold", **bh))
for nm in IDX:
    R = regimes(nm)
    for (rn, rg), res, (w, (st, en)) in itertools.product(R.items(), RES, WINS.items()):
        if nm == "sz399006" and w != "B":
            continue
        m = I.run_seeds(D, I.IConfig(start=st, end=en, idx=nm, reserve=res, regime=rg), SEEDS)
        rows.append(dict(win=w, idx=nm, reserve=res, regime=rn, **m))
    print(nm, "done", round(time.time() - t), flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "grid1.csv"), index=False)
