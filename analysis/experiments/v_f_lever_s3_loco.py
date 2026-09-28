"""v_f_lever s3: 剔簇复核（逐个剔 5 大簇 + 一次剔掉全部 5 大簇），每个候选对同样剔簇的基线比，80 种子。
另跑执行口径：次日收盘 + 0.5% 滑点（融资 8%）、盯市定额/上限，以及融资利率 10% 压力。"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v_f_lever_lib import BIG, Data, Lcfg, mmask, per_seed, summ  # noqa: E402

D = Data.load()
CANDS = {"base": 1.0, "L1.1": 1.1, "L1.5": 1.5, "L2.2": 2.2, "L2.5": 2.5}
rows = []
drops = [("", [])] + [(m, [m]) for m in BIG] + [("ALL5", BIG)]
for w in "BAF":
    for dn, dl in drops:
        if w == "A" and dn not in ("", "2012-01", "ALL5"):
            continue
        if w == "B" and dn == "2012-01":
            continue
        for cn, L in CANDS.items():
            kw = dict(borrow_rate=0.08)
            if dl:
                kw["mask"] = mmask(D, dl)
            o = summ(per_seed(D, Lcfg(w, L, **kw)))
            rows.append(dict(win=w, drop=dn or "-", cand=cn, **o))
df = pd.DataFrame(rows)
base = df[df.cand == "base"].set_index(["win", "drop"])
df["gain_pct"] = [100 * (r.final / base.loc[(r.win, r["drop"]), "final"] - 1) for _, r in df.iterrows()]
df["d_dd"] = [100 * (r.max_dd - base.loc[(r.win, r["drop"]), "max_dd"]) for _, r in df.iterrows()]
df["txt"] = ((df.final / 1e4).round(1).astype(str) + "/" + (df.max_dd * 100).round(1).astype(str)
             + " (+" + df.gain_pct.round(0).astype(int).astype(str) + "%)")
pd.set_option("display.width", 250)
print("剔簇（融资 8%）：期末万/中位回撤 (相对同剔簇基线的期末增幅)")
for w in "BAF":
    print("窗口", w)
    print(df[df.win == w].pivot_table(index="drop", columns="cand", values="txt", aggfunc="first")[list(CANDS)].to_string())
df.to_csv(os.path.join(HERE, "v_f_lever_out_s3.csv"), index=False)

print("\n=== 执行口径（80 种子）")
VARS = {"次日+0.5%滑点 融资8%": dict(borrow_rate=0.08, lag=1, slip=0.005),
        "次日+0.5%滑点 融资10%": dict(borrow_rate=0.10, lag=1, slip=0.005),
        "盯市定额+盯市上限 融资8%": dict(borrow_rate=0.08, sizing="mtm", cap_basis="mtm"),
        "次日+0.5%滑点 剔2024-02 融资8%": dict(borrow_rate=0.08, lag=1, slip=0.005, mask=mmask(D, ["2024-02"]))}
for vn, v in VARS.items():
    for w in "BAF":
        res = {cn: summ(per_seed(D, Lcfg(w, L, **v))) for cn, L in CANDS.items()}
        b = res["base"]
        print(f"{vn} {w}: " + " | ".join(
            f"{cn} {o['final']/1e4:.1f}万/{o['cagr']*100:.2f}%/{o['max_dd']*100:.1f}% p10cagr {o['cagr_p10']*100:.2f} ddw {o['dd_worst']*100:.1f}"
            for cn, o in res.items()))
