"""滚动起点检验：起点 2006..2023 各年 1 月初、终点 2026-08-21，以及 5 年滚动窗口；比较年化与回撤相对基线的差。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_idle_out")
SEEDS = range(20)
CFG = {
    "base": dict(),
    "300_always_r0": dict(idx="sh000300"),
    "300_always_r20": dict(idx="sh000300", reserve=0.2),
    "300_ma120": dict(idx="sh000300", regime=I.reg_trend_h(D, "sh000300", 120)),
    "300_ma200": dict(idx="sh000300", regime=I.reg_trend_h(D, "sh000300", 200)),
    "300_ma250": dict(idx="sh000300", regime=I.reg_trend_h(D, "sh000300", 250)),
    "300_ma120_lag1": dict(idx="sh000300", regime=I.reg_trend_h(D, "sh000300", 120, 0, 1)),
    "905_ma120": dict(idx="sh000905", regime=I.reg_trend_h(D, "sh000905", 120)),
    "852_ma120": dict(idx="sh000852", regime=I.reg_trend_h(D, "sh000852", 120)),
    "g2000_ma120": dict(idx="g2000p", regime=I.reg_trend_h(D, "g2000p", 120)),
    "g2000_ma60": dict(idx="g2000p", regime=I.reg_trend_h(D, "g2000p", 60)),
}
rows = []
for kind in ["toend", "5y"]:
    for y in range(2006, 2024 if kind == "toend" else 2022):
        st = f"{y}-01-01"
        en = I.END_DATE if kind == "toend" else f"{y+4}-12-31"
        for name, kw in CFG.items():
            m = I.run_seeds(D, I.IConfig(start=st, end=en, **kw), SEEDS)
            rows.append(dict(kind=kind, start=y, cfg=name, final=m["final"], cagr=m["cagr"], max_dd=m["max_dd"], avg_idx=m["avg_idx"]))
    print(kind, "done", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "rolling.csv"), index=False)
print("ALLDONE", flush=True)
