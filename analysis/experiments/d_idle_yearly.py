"""逐年拆解：全期 2006-2026，基线 vs 沪深300 均线 regime（留 0/20% 现金）vs 纯指数择时 vs 买入持有沪深300。
年收益取 80 个种子逐年收益的中位数；另给最大回撤日期与切换次数。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
st, en = I.WIN_F
rg120 = I.reg_trend_h(D, "sh000300", 120)
rg250 = I.reg_trend_h(D, "sh000300", 250)
t0, t1 = D.day(st), D.day_le(en)
sw = np.abs(np.diff(rg120[t0:t1 + 1].astype(int))).sum()
print("MA120 regime switches in full window:", sw, "per year", round(sw / 20.6, 1), "| on-fraction", rg120[t0:t1 + 1].mean().round(2))
sw = np.abs(np.diff(rg250[t0:t1 + 1].astype(int))).sum()
print("MA250 regime switches:", sw, "per year", round(sw / 20.6, 1), "| on-fraction", rg250[t0:t1 + 1].mean().round(2))
CF = {
    "base": dict(),
    "300ma120_r0": dict(idx="sh000300", regime=rg120),
    "300ma120_r20": dict(idx="sh000300", regime=rg120, reserve=0.2),
    "300ma250_r0": dict(idx="sh000300", regime=rg250),
    "timing_only": dict(idx="sh000300", regime=rg120, mask=np.zeros(len(D.sig), bool)),
    "bh300": dict(idx="sh000300", mask=np.zeros(len(D.sig), bool)),
}
yr = {}
dd = {}
for k, kw in CF.items():
    ys = []
    ds = []
    for s in range(80 if k not in ("timing_only", "bh300") else 1):
        r = I.run(D, I.IConfig(start=st, end=en, record=True, **kw), s)
        e = r["equity"]
        e.index = pd.to_datetime(e.index)
        y = e.resample("YE").last()
        y = pd.concat([pd.Series([600000.0], index=[pd.Timestamp("2005-12-31")]), y])
        ys.append(y.pct_change().dropna())
        ds.append((r["max_dd"], r["dd_date"]))
    yr[k] = pd.concat(ys, axis=1).median(axis=1)
    dds = sorted(ds)
    dd[k] = dds[len(dds) // 2]
out = pd.DataFrame(yr) * 100
out.index = out.index.year
print(out.round(1).to_string())
print("geo mean (%):", ((1 + out / 100).prod() ** (1 / len(out)) * 100 - 100).round(2).to_dict())
print("median maxDD & date:", {k: (round(v[0] * 100, 1), v[1]) for k, v in dd.items()})
