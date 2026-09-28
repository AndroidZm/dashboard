"""v_d_idle 复核 5（80 种子）：B1/B2 子区间、MA 邻域、未测过的上证综指/深证成指、A 窗口内选出的 MA40。"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
W = {"A": ("2006-01-04", "2016-01-28"), "B": ("2016-01-01", E.END_DATE), "F": ("2006-01-04", E.END_DATE),
     "B1": ("2016-01-01", "2020-12-31"), "B2": ("2021-01-01", E.END_DATE)}
CF = {"base": None}
for ma in (90, 120, 150, 250):
    for res in (0.0, 0.2):
        CF[f"300_ma{ma}_r{res}"] = ("sh000300", ma, res, 0)
CF["300_ma120_r0.3_lag1"] = ("sh000300", 120, 0.3, 1)
CF["300_ma90_r0.3_lag1"] = ("sh000300", 90, 0.3, 1)
CF["300_ma150_r0.3_lag1"] = ("sh000300", 150, 0.3, 1)
CF["300_ma40_r0.0"] = ("sh000300", 40, 0.0, 0)
for nm in ("sh000001", "sz399001"):
    for res in (0.0, 0.2):
        CF[f"{nm}_ma120_r{res}"] = (nm, 120, res, 0)
        CF[f"{nm}_ma250_r{res}"] = (nm, 250, res, 0)
rows = []
base = {}
for name, p in CF.items():
    line = []
    for w, (st, en) in W.items():
        if p is None:
            m = I.run_seeds(D, I.IConfig(start=st, end=en))
            base[w] = m
        else:
            nm, ma, res, lag = p
            m = I.run_seeds(D, I.IConfig(start=st, end=en, idx=nm, reserve=res, regime=I.reg_trend_h(D, nm, ma, 0.0, lag)))
        rows.append(dict(cfg=name, win=w, final=m["final"], cagr=m["cagr"], max_dd=m["max_dd"], p10=m["cagr_p10"], p90=m["cagr_p90"], dd_p10=m["dd_p10"]))
        line.append(f"{w} {m['final']/1e4:6.1f}万 {m['cagr']*100:5.2f}% ({100*(m['cagr']-base[w]['cagr']):+5.2f}) dd {m['max_dd']*100:5.1f}")
    print(f"{name:24s}", " | ".join(line), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "v_d_idle_out", "s5_confirm.csv"), index=False)
print("ALLDONE")
