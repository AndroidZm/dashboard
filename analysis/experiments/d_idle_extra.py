"""补充检验：创业板指均线在 2010-06~2016-01 的样本外；沪深300 加 2% 股息的全收益近似（信号仍按价格指数均线）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
S = range(20)
for st, en in [("2010-06-01", "2016-01-28"), I.WIN_B]:
    b = I.run_seeds(D, I.IConfig(start=st, end=en), S)
    print(f"[{st}~{en}] base {b['final']/1e4:.0f}万 {b['cagr']*100:.2f}% dd {b['max_dd']*100:.1f}")
    for nm, ma, h in [("sz399006", 250, 0.03), ("sz399006", 200, 0.03), ("sz399006", 250, 0), ("sz399006", 120, 0), ("sh000300", 120, 0), ("sh000300", 250, 0.03)]:
        m = I.run_seeds(D, I.IConfig(start=st, end=en, idx=nm, regime=I.reg_trend_h(D, nm, ma, h)), S)
        print(f"   {nm} ma{ma} h{h}: {m['final']/1e4:.0f}万 {m['cagr']*100:.2f}% dd {m['max_dd']*100:.1f}")
print("TR proxy (CSI300 + 2%/yr dividends), regime from price index MA120:")
rg = I.reg_trend_h(D, "sh000300", 120)
for w, (st, en) in {"A": I.WIN_A, "B": I.WIN_B, "F": I.WIN_F}.items():
    for res in [0.0, 0.2]:
        m = I.run_seeds(D, I.IConfig(start=st, end=en, idx="sh000300@tr2.0", regime=rg, reserve=res), S)
        m0 = I.run_seeds(D, I.IConfig(start=st, end=en, idx="sh000300", regime=rg, reserve=res), S)
        print(f"  {w} r{res}: TR {m['final']/1e4:.0f}万 {m['cagr']*100:.2f}% dd {m['max_dd']*100:.1f} | price {m0['final']/1e4:.0f}万 {m0['cagr']*100:.2f}%")
