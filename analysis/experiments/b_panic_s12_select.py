"""阶段 12：分风险档的样本外选择。网格 = cap × 平常权重 × 调整权重 × 恐慌权重，有/无“恐慌换仓”。
在一个窗口上按回撤约束选年化最高者，拿到另一个窗口看（A->B 与 B->A）。20 种子筛选。
回撤约束按同窗口基线回撤的倍数：档1 <= 1.12x（~-30%/-28%）、档2 <= 1.5x（~-40%）、档3 不限。"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D)
SE = range(20)
WINS = (("A", WIN_A, None), ("B", WIN_B, None), ("Bx", WIN_B, M["no2402"]), ("F", WIN_F, None))
OUT = "/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/b_s12.csv"
if not os.path.exists(OUT):
    rows = []
    for cap, wn, wa in itertools.product([0.9, 0.95, 1.0], [0.02, 0.03, 0.04], [0.06, 0.08, 0.10, 0.12, 0.16]):
        for sw, w2 in [(None, 0.06), ("oldest", 0.03), ("oldest", 0.04), ("oldest", 0.06)]:
            row = dict(cap=cap, wn=wn, wa=wa, swap=sw or "-", w2=w2)
            for tag, (st, en), mk in WINS:
                kw = dict(swap_policy=sw, swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5) if sw else {}
                m = run_seeds(D, PConfig(start=st, end=en, mask=mk, cap=cap, weights=(wn, wa, w2), **kw), SE)
                row[tag] = m["cagr"] * 100
                row[tag + "_dd"] = m["max_dd"] * 100
                row[tag + "_fin"] = m["final"] / 1e4
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT, index=False)
df = pd.read_csv(OUT)
base = df[(df.cap == 0.9) & (df.wn == 0.02) & (df.wa == 0.08) & (df.swap == "-")].iloc[0]
print("baseline(20 seeds):", {k: round(base[k], 2) for k in ("A", "A_dd", "B", "B_dd", "Bx", "F", "F_dd")})
pd.set_option("display.width", 250)
cols = ["cap", "wn", "wa", "swap", "w2", "A", "A_dd", "B", "B_dd", "Bx", "Bx_dd", "F", "F_dd", "F_fin"]
for lvl, k in (("档1", 1.12), ("档2", 1.5), ("档3", 99)):
    for sel, oth in (("A", "B"), ("B", "A")):
        for fam in ("-", "oldest"):
            sub = df[(df.swap == fam) if fam == "-" else (df.swap != "-")]
            ok = sub[sub[sel + "_dd"] >= base[sel + "_dd"] * k]
            best = ok.sort_values(sel, ascending=False).head(3)
            print(f"{lvl} 在{sel}上选 ({'无换仓' if fam == '-' else '有换仓'}), 前3:")
            print(best[cols].round(2).to_string(index=False))
    print()
# 平滑性：有换仓 w2=0.04, cap=0.9 时对 (wn, wa) 的表
for cap in (0.9, 1.0):
    s = df[(df.swap == "oldest") & (df.w2 == 0.04) & (df.cap == cap)]
    for col in ("A", "B", "Bx", "F_dd"):
        print(f"cap={cap} swap w2=.04 {col}"); print(s.pivot(index="wn", columns="wa", values=col).round(2).to_string())
    s = df[(df.swap == "-") & (df.cap == cap)]
    for col in ("A", "B"):
        print(f"cap={cap} NO swap {col}"); print(s.pivot(index="wn", columns="wa", values=col).round(2).to_string())
