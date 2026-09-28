"""h_critic_: 2026 年 1~8 月 P1/P2 为何大幅跑输基线（逐年收益、年初持仓、2026 年新买入）。"""
import os, sys
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import g_combined_engine as G
from g_combined_plans import make, D
PL = {"base": make(), "P1": make(S=True), "P2": make(S=True, C=True)}
rows = []
yrs = [str(y) for y in range(2016, 2027)]
for nm, cfg in PL.items():
    for seed in range(20):
        r = G.run(D, replace(G.win_cfg(cfg, "B"), record=True), seed)
        eq = r["equity"]; tr = r["trades"]
        ye = eq.groupby(eq.index.str[:4]).last()
        yr = ye / ye.shift(1).fillna(6e5) - 1
        t25 = D.day_le("2025-12-31")
        held25 = tr[(tr.buy_t <= t25) & (tr.sell_t > t25)]
        b26 = tr[tr.buy_t > t25]
        pos = r["pos_value"]
        d = dict(plan=nm, seed=seed, **{f"r{y}": yr.get(y, np.nan) * 100 for y in yrs},
                 held_2025end=len(held25), held_panic_2025end=int((held25.tier == 2).sum()),
                 held_cost_share=held25.amount.sum() / eq.loc[D.dates[t25]] * 100,
                 held_2024buy=int((held25.buy_date.str[:4] <= "2024").sum()),
                 buys_2026=len(b26), exp_2026=float((pos / eq)[eq.index >= "2026-01-01"].mean() * 100))
        rows.append(d)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("== 逐年收益% (20 种子中位)")
print(df.groupby("plan")[[f"r{y}" for y in yrs]].median().round(1).to_string())
print(df.groupby("plan")[["held_2025end", "held_panic_2025end", "held_2024buy", "held_cost_share", "buys_2026", "exp_2026"]].median().round(1).to_string())
# 2026 年信号
s26 = D.sig[D.sig.signal_date >= "2026-01-01"]
print("2026 信号数", len(s26), s26.groupby([s26.signal_date.str[:7], "tier"]).size().to_string())
