"""v_d_idle 复核 2：邻域/平台 + 样本外选择 + 未测过的指数（上证综指、深证成指、中小板指）+ 子区间。20 种子。
使用 d_idle_engine（已在 s1 中用独立引擎逐种子验证一致）。"""
import sys, os, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
S = range(20)
WINS = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", E.END_DATE), "F": ("2006-01-04", E.END_DATE),
        "A1": ("2006-01-04", "2010-12-31"), "A2": ("2011-01-01", "2016-01-28"),
        "B1": ("2016-01-01", "2020-12-31"), "B2": ("2021-01-01", E.END_DATE)}
rows = []
t = time.time()
for w, (st, en) in WINS.items():
    m = I.run_seeds(D, I.IConfig(start=st, end=en), S)
    rows.append(dict(idx="none", ma=0, reserve=1.0, lag=0, win=w, **m))
# 沪深300 邻域
for ma, res, lag in itertools.product([40, 60, 90, 120, 150, 200, 250], [0.0, 0.1, 0.2, 0.3, 0.4], [0, 1]):
    rg = I.reg_trend_h(D, "sh000300", ma, 0.0, lag)
    for w, (st, en) in WINS.items():
        m = I.run_seeds(D, I.IConfig(start=st, end=en, idx="sh000300", reserve=res, regime=rg), S)
        rows.append(dict(idx="sh000300", ma=ma, reserve=res, lag=lag, win=w, **m))
print("300 done", round(time.time() - t), flush=True)
# 未测过的指数
for nm in ["sh000001", "sz399001", "sz399005"]:
    for ma, res in itertools.product([60, 90, 120, 150, 200, 250], [0.0, 0.2]):
        rg = I.reg_trend_h(D, nm, ma, 0.0, 0)
        for w, (st, en) in WINS.items():
            m = I.run_seeds(D, I.IConfig(start=st, end=en, idx=nm, reserve=res, regime=rg), S)
            rows.append(dict(idx=nm, ma=ma, reserve=res, lag=0, win=w, **m))
    print(nm, "done", round(time.time() - t), flush=True)
os.makedirs(os.path.join(HERE, "v_d_idle_out"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "v_d_idle_out", "s2_grid.csv"), index=False)
print("ALLDONE")
