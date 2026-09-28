"""v_d_idle 复核 1：独立引擎对基线逐种子一致性；C1~C4 在 A/B/全期 80 种子复现（两套引擎对照）。"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I
import v_d_idle_engine2 as V

D = E.Data.load()
W = {"A": V.A, "B": V.B, "F": V.F}

# 1) idx=None 与 engine.run 逐种子
for w, (st, en) in W.items():
    mx = 0
    for s in range(20):
        a = E.run(D, E.Config(start=st, end=en), s)
        b = V.run2(D, st, en, seed=s)
        mx = max(mx, abs(a["final"] - b["final"]), abs(a["max_dd"] - b["max_dd"]) * 1e6)
    print("baseline check", w, "max diff", mx, flush=True)

# 2) regime 一致性
for ma in (120, 250):
    for lag in (0, 1):
        a = I.reg_trend_h(D, "sh000300", ma, 0.0, lag)
        b = V.my_regime(D, "sh000300", ma, lag)
        print("regime diff ma", ma, "lag", lag, int((a != b).sum()))

C = {
    "C1": dict(idx="sh000300", ma=120, reserve=0.2, lag=0),
    "C2": dict(idx="sh000300", ma=120, reserve=0.0, lag=0),
    "C3": dict(idx="sh000300", ma=120, reserve=0.3, lag=1),
    "C4": dict(idx="sh000852", ma=60, reserve=0.0, lag=0),
}
rows = []
fmt = lambda m: f"{m['final']/1e4:7.1f}万 {m['cagr']*100:6.2f}% dd {m['max_dd']*100:6.1f} p10cagr {m['cagr_p10']*100:5.2f} p90 {m['cagr_p90']*100:5.2f} ddp10 {m['dd_p10']*100:6.1f}"
for w, (st, en) in W.items():
    b = V.seeds2(D, start=st, end=en)
    print(w, "base   ", fmt(b), "worst", round(b["dd_worst"] * 100, 1), flush=True)
    rows.append(dict(cand="base", win=w, eng="v_daily", **b))
    for c, p in C.items():
        rg_their = I.reg_trend_h(D, p["idx"], p["ma"], 0.0, p["lag"])
        rg_mine = V.my_regime(D, p["idx"], p["ma"], p["lag"])
        m1 = I.run_seeds(D, I.IConfig(start=st, end=en, idx=p["idx"], reserve=p["reserve"], regime=rg_their))
        m2 = V.seeds2(D, start=st, end=en, idx=p["idx"], reserve=p["reserve"], regime=rg_mine, rebal="event")
        m3 = V.seeds2(D, start=st, end=en, idx=p["idx"], reserve=p["reserve"], regime=rg_mine, rebal="daily")
        print(w, c, "their ", fmt(m1))
        print(w, c, "v_evt ", fmt(m2), "worst", round(m2["dd_worst"] * 100, 1), "idx", round(m2["avg_idx"] * 100, 1))
        print(w, c, "v_day ", fmt(m3), "worst", round(m3["dd_worst"] * 100, 1), flush=True)
        rows.append(dict(cand=c, win=w, eng="their", **m1))
        rows.append(dict(cand=c, win=w, eng="v_event", **m2))
        rows.append(dict(cand=c, win=w, eng="v_daily", **m3))
os.makedirs(os.path.join(HERE, "v_d_idle_out"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "v_d_idle_out", "s1_repro.csv"), index=False)
print("ALLDONE")
