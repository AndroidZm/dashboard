"""B 窗口逐年（80 种子逐年收益中位）：基线 vs 沪深300 MA120（留 0/20% 现金），以及去掉任一年后的累计超额；
另打印 A/全期 最大回撤区间。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
rg = I.reg_trend_h(D, "sh000300", 120)
CF = {"base": dict(), "ma120_r0": dict(idx="sh000300", regime=rg), "ma120_r20": dict(idx="sh000300", regime=rg, reserve=0.2)}


def yearly(st, en, kw):
    ys = []
    for s in range(80):
        e = I.run(D, I.IConfig(start=st, end=en, record=True, **kw), s)["equity"]
        e.index = pd.to_datetime(e.index)
        y = e.resample("YE").last()
        y = pd.concat([pd.Series([600000.0], index=[pd.Timestamp(st) - pd.Timedelta(days=1)]), y])
        ys.append(y.pct_change().dropna())
    return pd.concat(ys, axis=1).median(axis=1)


st, en = I.WIN_B
Y = pd.DataFrame({k: yearly(st, en, kw) for k, kw in CF.items()})
Y.index = Y.index.year
print((Y * 100).round(1).to_string())
for k in ["ma120_r0", "ma120_r20"]:
    rel = (1 + Y[k]) / (1 + Y["base"])
    tot = rel.prod()
    print(k, "cum relative wealth", round(tot, 3), "| leave-one-year-out min", round(min(tot / rel), 3), "(drop", rel.idxmax(), ")",
          "| years better", int((rel > 1).sum()), "of", len(rel))
# 回撤区间（种子 13）
for w, (st, en) in {"A": I.WIN_A, "F": I.WIN_F}.items():
    for k, kw in CF.items():
        r = I.run(D, I.IConfig(start=st, end=en, record=True, **kw), 13)
        e = r["equity"]
        pk = e.cummax()
        dd = e / pk - 1
        tmin = dd.idxmin()
        tpk = e.loc[:tmin].idxmax()
        print(w, k, f"maxDD {dd.min()*100:.1f}% peak {tpk} trough {tmin}", "idx share at peak %.0f%%" % (r["idx_value"].loc[tpk] / e.loc[tpk] * 100),
              "pos share at peak %.0f%%" % (r["pos_value"].loc[tpk] / e.loc[tpk] * 100))
