"""v_f_lever s2: L 线细网格（0.8~2.6，步长 0.05，80 种子）→ 邻域形状 + 双向样本外选择（中位回撤 / p10 回撤两种约束）。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v_f_lever_lib import Data, Lcfg, per_seed, summ  # noqa: E402

D = Data.load()
OUT = os.path.join(HERE, "v_f_lever_out_s2.csv")
if not os.path.exists(OUT):
    rows = []
    for L in np.round(np.arange(0.8, 2.601, 0.05), 2):
        for w in "ABF":
            for br in [0.05, 0.08]:
                o = summ(per_seed(D, Lcfg(w, L, borrow_rate=br)))
                rows.append(dict(L=L, win=w, borrow=br, **o))
    pd.DataFrame(rows).to_csv(OUT, index=False)
df = pd.read_csv(OUT)

pd.set_option("display.width", 250)
for br in [0.05, 0.08]:
    x = df[df.borrow == br]
    print(f"=== 融资 {br:.0%}: L 线（期末万 / 年化 / 中位回撤 / p10回撤 / calmar）")
    t = x.assign(txt=lambda d: (d.final / 1e4).round(1).astype(str) + "/" + (d.cagr * 100).round(2).astype(str) + "/"
                 + (d.max_dd * 100).round(1).astype(str) + "/" + (d.dd_p10 * 100).round(1).astype(str) + "/" + d.calmar.round(3).astype(str))
    print(t.pivot_table(index="L", columns="win", values="txt", aggfunc="first").to_string())
    # 单调性 / 平滑度: 期末相对 L 的局部斜率
    for w in "AB":
        y = x[x.win == w].sort_values("L")
        d = np.diff(np.log(y.final.values))
        print(w, "log期末逐步增量 min/max", d.min().round(4), d.max().round(4), "回撤逐步变化 min/max",
              np.diff(y.max_dd.values).min().round(4), np.diff(y.max_dd.values).max().round(4))

print("\n=== 双向样本外选择（在选窗取满足回撤约束的最大期末 L，到评窗评估）")
for br in [0.05, 0.08]:
    x = df[df.borrow == br]
    for crit in ["max_dd", "dd_p10"]:
        for lim_name, lim in [("-30%", -0.305), ("-40%", -0.405)]:
            for sel, ev in [("A", "B"), ("B", "A")]:
                a = x[(x.win == sel) & (x[crit] >= lim)]
                L = a.loc[a.final.idxmax(), "L"]
                ms = x[(x.win == sel) & (x.L == L)].iloc[0]
                me = x[(x.win == ev) & (x.L == L)].iloc[0]
                b = x[(x.win == ev) & (x[crit] >= lim)]
                Lo = b.loc[b.final.idxmax(), "L"]
                base = x[(x.win == ev) & (x.L == 1.0)].iloc[0]
                print(f"融资{br:.0%} 约束 {crit}>={lim_name} 选{sel}→评{ev}: L={L:.2f} 选窗 {ms.final/1e4:.1f}万/{ms.max_dd*100:.1f}%/p10 {ms.dd_p10*100:.1f}"
                      f" | 评窗 {me.final/1e4:.1f}万/{me.max_dd*100:.1f}%/p10 {me.dd_p10*100:.1f} (基线 {base.final/1e4:.1f}万)"
                      f" | 评窗事后 L={Lo:.2f} | 约束{'满足' if me[crit] >= lim else '违反'}")
