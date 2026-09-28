"""样本外选择：在一个窗口按三个风险档（回撤 >=-30.5% / >=-40.5% / 不限）选期末中位最高的配置，看另一个窗口。
配置族 = grid1（常驻/安静期/恐慌后/回撤买入/均线）+ grid2（均线细网格、滞后带、次日成交），均为 20 种子中位。"""
import os
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_idle_out")
g1 = pd.read_csv(os.path.join(OUT, "grid1.csv"))
g2 = pd.read_csv(os.path.join(OUT, "grid2.csv"))
g2 = g2[g2.sig == 1].copy()
g1 = g1[~g1.regime.str.startswith("ma") & ~g1.regime.isin(["buyhold"])].copy()
g1["key"] = g1.idx + "|" + g1.regime + "|r" + g1.reserve.astype(str)
g2["key"] = g2.idx + "|ma" + g2.ma.astype(str) + "_h" + g2.h.astype(str) + "_lag" + g2.lag.astype(str) + "|r" + g2.reserve.astype(str)
df = pd.concat([g1[["key", "win", "final", "cagr", "max_dd"]], g2[["key", "win", "final", "cagr", "max_dd"]]])
p = df.pivot_table(index="key", columns="win", values=["final", "cagr", "max_dd"])
p.columns = [f"{a}_{b}" for a, b in p.columns]
base = p.loc[[k for k in p.index if k.startswith("none|base")]].iloc[0]
print("base:", {k: round(v, 3) for k, v in base.items()})
for fam_name, fam in [("all", p), ("excl_399006", p[~p.index.str.startswith("sz399006")]), ("csi300_only", p[p.index.str.startswith("sh000300")]),
                      ("smallcap_only", p[p.index.str.startswith(("sh000852", "g2000p", "sh000905"))])]:
    print("\n=== family:", fam_name, len(fam))
    for tr, te in [("A", "B"), ("B", "A"), ("B", "F")]:
        f = fam.dropna(subset=[f"final_{tr}"])
        for lvl, lim in [("dd30", -0.305), ("dd40", -0.405), ("free", -1)]:
            c = f[f[f"max_dd_{tr}"] >= lim]
            if c.empty:
                print(tr, lvl, "none"); continue
            k = c[f"final_{tr}"].idxmax()
            r = c.loc[k]
            te_final = r.get(f"final_{te}", np.nan)
            print(f"train {tr} {lvl}: {k:42s} train {r[f'final_{tr}']/1e4:7.0f}万 dd {r[f'max_dd_{tr}']*100:5.1f} (base {base[f'final_{tr}']/1e4:.0f}) -> test {te} {te_final/1e4:7.0f}万 dd {r.get(f'max_dd_{te}', np.nan)*100:5.1f} (base {base[f'final_{te}']/1e4:.0f}/{base[f'max_dd_{te}']*100:.1f})")
        # top-10 on train and their test rank (dd30)
    if fam_name == "all":
        f = fam.dropna(subset=["final_A", "final_B"])
        print("\nSpearman rank corr of final A vs B across configs:", round(f.final_A.rank().corr(f.final_B.rank()), 3), "n", len(f))
        good = f[(f.final_A > base.final_A) & (f.final_B > base.final_B)]
        print("configs beating base in BOTH windows:", len(good), "of", len(f))
        print(good.assign(FA=good.final_A / 1e4, FB=good.final_B / 1e4, dA=good.max_dd_A * 100, dB=good.max_dd_B * 100)[["FA", "dA", "FB", "dB"]].round(0).sort_values("FB", ascending=False).head(40).to_string())
