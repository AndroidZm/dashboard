"""v_d_idle 复核 4：全期逐年收益差（80 种子中位）、起点滚动到期末、B 窗口去掉最好年份后的累计增益。"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I
import v_d_idle_engine2 as V

D = E.Data.load()
C = {"C1": dict(reserve=0.2, rl=0), "C2": dict(reserve=0.0, rl=0), "C3": dict(reserve=0.3, rl=1)}


def yearly(eq):
    y = eq.groupby(eq.index.str[:4]).last()
    first = eq.iloc[0]
    prev = np.r_[first, y.values[:-1]]
    return pd.Series(y.values / prev - 1, index=y.index)


for win, (st, en) in {"F": V.F, "B": V.B}.items():
    yb = pd.DataFrame({s: yearly(V.run2(D, st, en, seed=s, record=True)["eq"]) for s in range(80)})
    hs = pd.Series(D.index["sh000300"].values, index=D.dates).loc[D.dates[D.day(st)]:en]
    hsy = yearly(hs)
    out = pd.DataFrame({"base": yb.median(axis=1) * 100, "hs300": hsy * 100})
    for c, p in C.items():
        rg = V.my_regime(D, "sh000300", 120, p["rl"])
        yc = pd.DataFrame({s: yearly(V.run2(D, st, en, seed=s, idx="sh000300", reserve=p["reserve"], regime=rg, rebal="event", record=True)["eq"]) for s in range(80)})
        out[c] = yc.median(axis=1) * 100
        out[c + "_diff"] = ((1 + yc) / (1 + yb) - 1).median(axis=1) * 100
    print("====", win)
    print(out.round(1).to_string())
    if win == "B":
        for c in C:
            d = 1 + out[c + "_diff"] / 100
            print(c, "cum wealth ratio", round(d.prod(), 3), "without best year", round(d.prod() / d.max(), 3),
                  "without best 2 years", round(d.prod() / d.nlargest(2).prod(), 3), "best years", d.nlargest(2).index.tolist(),
                  "2016-2020", round(d.loc["2016":"2020"].prod(), 3), "2021-2026", round(d.loc["2021":"2026"].prod(), 3))

# 起点滚动到期末（80 种子）
print("==== rolling start to 2026-08-21 (80 seeds): cagr diff pp / dd diff pp")
rg0 = I.reg_trend_h(D, "sh000300", 120)
rg1 = I.reg_trend_h(D, "sh000300", 120, 0, 1)
for y in range(2006, 2025):
    st = f"{y}-01-01"
    b = I.run_seeds(D, I.IConfig(start=st))
    c1 = I.run_seeds(D, I.IConfig(start=st, idx="sh000300", reserve=0.2, regime=rg0))
    c2 = I.run_seeds(D, I.IConfig(start=st, idx="sh000300", reserve=0.0, regime=rg0))
    c3 = I.run_seeds(D, I.IConfig(start=st, idx="sh000300", reserve=0.3, regime=rg1))
    print(y, f"base {b['cagr']*100:5.2f}% dd {b['max_dd']*100:5.1f} | C1 {100*(c1['cagr']-b['cagr']):+5.2f} dd {100*(c1['max_dd']-b['max_dd']):+5.1f} | C2 {100*(c2['cagr']-b['cagr']):+5.2f} dd {100*(c2['max_dd']-b['max_dd']):+5.1f} | C3 {100*(c3['cagr']-b['cagr']):+5.2f} dd {100*(c3['max_dd']-b['max_dd']):+5.1f}", flush=True)
