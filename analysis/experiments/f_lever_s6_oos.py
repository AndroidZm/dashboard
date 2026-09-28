"""s6: 样本外协议 —— 在一个窗口按风险档位选参数，在另一个窗口评估（两个方向）。

族 1  L 比例放大（权重×L，cap=0.9L），80 种子（s4 + s2 结果）
族 2  k × cap 全网格（s1，20 种子筛选），被选中的点再用 80 种子复核
风险档位：中位回撤 >= -30% / >= -40% / 不限（不限时取中位期末最大）
融资利率 5%（主口径），另报 8%。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import Data, cfg_win, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "f_lever_out")
D = Data.load()
LIM = {"≤-30%": -0.305, "≤-40%": -0.405, "不限": -9}


def fmt(m):
    return f"{m['final']/1e4:7.1f}万 {m['cagr']*100:5.2f}% dd {m['max_dd']*100:6.1f}% (p10 {m['dd_p10']*100:6.1f}%)"


base = {w: run_seeds(D, cfg_win(w), range(80)) for w in "AB"}
print("基线", {w: fmt(base[w]) for w in "AB"})

# ---- 族 1：L
s4 = pd.read_csv(os.path.join(OUT, "s4_fine.csv"))
s2 = pd.read_csv(os.path.join(OUT, "s2_prop.csv"))
s4 = s4[(s4.fam == "L") & (s4["drop"].isna())]
s2 = s2[s2["drop"].isna()].rename(columns={"L": "k"})
Lf = pd.concat([s4, s2[~s2.k.isin(s4.k)]], ignore_index=True)
Lf = Lf[Lf.k <= 2.5]  # 2.5 以上 cap>2.25 超出监管 2 倍上限，另行讨论
print("\n=== 族 1：L 比例放大（80 种子）")
for br in [0.05, 0.08]:
    X = Lf[Lf.borrow == br]
    for sel, ev in [("A", "B"), ("B", "A")]:
        for lvl, lim in LIM.items():
            a = X[(X.win == sel) & (X.max_dd >= lim)]
            L = a.loc[a.final.idxmax(), "k"]
            ms = X[(X.win == sel) & (X.k == L)].iloc[0]
            me = X[(X.win == ev) & (X.k == L)].iloc[0]
            # 评估窗口在同档位下的“事后最优”
            b = X[(X.win == ev) & (X.max_dd >= lim)]
            Lopt = b.loc[b.final.idxmax(), "k"]
            mo = X[(X.win == ev) & (X.k == Lopt)].iloc[0]
            print(f"融资{br:.0%} 选{sel}→评{ev} {lvl:5s}: 选 L={L:4.2f} | 选窗 {fmt(ms)} | 评窗 {fmt(me)}"
                  f" | 评窗事后最优 L={Lopt:4.2f} {fmt(mo)} | 评窗约束{'满足' if me.max_dd >= lim else '违反'}")

# ---- 族 2：k × cap 网格
s1 = pd.read_csv(os.path.join(OUT, "s1_grid.csv"))
s1 = s1[(s1.sizing == "cost") & (s1.borrow != 0.08)]
print("\n=== 族 2：k × cap 全网格（20 种子选，80 种子复核）")
rows = []
for sel, ev in [("A", "B"), ("B", "A")]:
    for lvl, lim in LIM.items():
        a = s1[(s1.win == sel) & (s1.max_dd >= lim)]
        r = a.loc[a.final.idxmax()]
        k, cap, br = r.k, r.cap, r.borrow
        ms = run_seeds(D, cfg_win(sel, k=k, cap=cap, borrow_rate=br), range(80))
        me = run_seeds(D, cfg_win(ev, k=k, cap=cap, borrow_rate=br), range(80))
        print(f"选{sel}→评{ev} {lvl:5s}: 选 k={k} cap={cap} | 选窗(80) {fmt(ms)} | 评窗(80) {fmt(me)} | 基线评窗 {fmt(base[ev])}"
              f" | 评窗约束{'满足' if me['max_dd'] >= lim else '违反'}")
        rows.append(dict(sel=sel, ev=ev, lvl=lvl, k=k, cap=cap, sel_final=ms["final"], sel_dd=ms["max_dd"],
                         ev_final=me["final"], ev_cagr=me["cagr"], ev_dd=me["max_dd"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "s6_oos_grid.csv"), index=False)

# ---- 前沿检查：k×cap 网格里有没有点明显高于 L 线（同窗口同回撤下期末更高）
print("\n=== 前沿检查：网格点相对 L 线（按回撤插值）的期末超额（20 种子网格 vs 80 种子 L 线）")
X = Lf[Lf.borrow == 0.05]
for w in "AB":
    line = X[X.win == w].sort_values("max_dd", ascending=False)
    g = s1[s1.win == w].copy()
    g["line_final"] = np.interp(-g.max_dd, -line.max_dd.values, line.final.values)
    g["excess_pct"] = (g.final / g.line_final - 1) * 100
    g = g[g.max_dd >= line.max_dd.min()]
    top = g.sort_values("excess_pct", ascending=False).head(6)
    print(w, "网格点数", len(g), "高于 L 线 >5% 的点数", int((g.excess_pct > 5).sum()), "低于 L 线 >5% 的点数", int((g.excess_pct < -5).sum()))
    print(top[["k", "cap", "borrow", "final", "max_dd", "line_final", "excess_pct"]].round(3).to_string(index=False))
