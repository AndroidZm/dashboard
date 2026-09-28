"""第二轮：均线择时 regime 的细网格（指数 × MA × 滞后带 × 次日成交 × 现金保留），三个窗口，20 个种子。
另算纯指数择时（不做信号，mask 全 False）作对照。"""
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
NOSIG = np.zeros(len(D.sig), bool)
rows = []
t = time.time()
for nm in IDX:
    for ma, h, lag in itertools.product([40, 60, 90, 120, 150, 200, 250], [0.0, 0.03], [0, 1]):
        rg = I.reg_trend_h(D, nm, ma, h, lag)
        for w, (st, en) in WINS.items():
            if nm == "sz399006" and w != "B":
                continue
            for res in [0.0, 0.1, 0.2, 0.3]:
                m = I.run_seeds(D, I.IConfig(start=st, end=en, idx=nm, reserve=res, regime=rg), SEEDS)
                rows.append(dict(win=w, idx=nm, ma=ma, h=h, lag=lag, reserve=res, sig=1, **m))
            m = I.run(D, I.IConfig(start=st, end=en, idx=nm, reserve=0.0, regime=rg, mask=NOSIG), 0)
            rows.append(dict(win=w, idx=nm, ma=ma, h=h, lag=lag, reserve=0.0, sig=0, **{k: v for k, v in m.items() if not isinstance(v, (tuple, str))}))
    print(nm, "done", round(time.time() - t), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "grid2.csv"), index=False)
print("ALLDONE", flush=True)
