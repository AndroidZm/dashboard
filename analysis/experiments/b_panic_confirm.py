"""80 种子确认：候选规则 vs 基线，在 A / B / 全期、去掉 2024-02、逐个去掉大簇 (LOCO)、次日成交 lag=1 下的表现。
用法: python b_panic_confirm.py <候选名...>   （候选定义在 CANDS）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from dataclasses import replace
from b_panic_engine import Data, PConfig, run_seeds, WIN_A, WIN_B, WIN_F, BIG, masks

D = Data.load(); M = masks(D)
SE = range(int(os.environ.get("NSEED", 80)))
SW = dict(swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=3)
CANDS = {
    "base": PConfig(),
    "oldest_w4": PConfig(swap_policy="oldest", weights=(0.02, 0.08, 0.04), **SW),
    "oldest_w3": PConfig(swap_policy="oldest", weights=(0.02, 0.08, 0.03), **SW),
    "oldest_w5": PConfig(swap_policy="oldest", weights=(0.02, 0.08, 0.05), **SW),
    "flat_w4": PConfig(swap_policy="flat", weights=(0.02, 0.08, 0.04), **SW),
    "under_w2": PConfig(swap_policy="underwater", weights=(0.02, 0.08, 0.02), **SW),
    "under_w4": PConfig(swap_policy="underwater", weights=(0.02, 0.08, 0.04), **SW),
    "gainpct_w4": PConfig(swap_policy="gain_pct", weights=(0.02, 0.08, 0.04), **SW),
    "newest_w4": PConfig(swap_policy="newest", weights=(0.02, 0.08, 0.04), **SW),
}
extra = {}
if os.environ.get("EXTRA"):
    exec(open(os.environ["EXTRA"]).read(), globals(), extra)
    CANDS.update(extra.get("CANDS", {}))


def evaluate(c):
    out = {}
    for tag, (st, en) in (("A", WIN_A), ("B", WIN_B), ("F", WIN_F)):
        out[tag] = run_seeds(D, replace(c, start=st, end=en), SE)
        out[tag + "_lag1"] = run_seeds(D, replace(c, start=st, end=en, lag=1), SE)
    out["Bx2402"] = run_seeds(D, replace(c, start=WIN_B[0], end=WIN_B[1], mask=M["no2402"]), SE)
    out["Fx2402"] = run_seeds(D, replace(c, start=WIN_F[0], end=WIN_F[1], mask=M["no2402"]), SE)
    for cl in BIG:
        out["Fx" + cl] = run_seeds(D, replace(c, start=WIN_F[0], end=WIN_F[1], mask=M["no" + cl]), SE)
        w = WIN_A if cl < "2016" else WIN_B
        out[("A" if cl < "2016" else "B") + "x" + cl] = run_seeds(D, replace(c, start=w[0], end=w[1], mask=M["no" + cl]), SE)
    return out


names = sys.argv[1:] or list(CANDS)
if "base" not in names:
    names = ["base"] + names
res = {n: evaluate(CANDS[n]) for n in names}
b = res["base"]
keys = list(b)
print("baseline:", " ".join(f"{k}:{b[k]['final']/1e4:.1f}万/{b[k]['cagr']*100:.2f}%/{b[k]['max_dd']*100:.1f}" for k in ("A", "B", "F", "B_lag1", "F_lag1")))
rows = []
for n in names:
    r = res[n]
    row = {"cand": n}
    for k in keys:
        row[k] = f"{(r[k]['cagr'] - b[k]['cagr'])*100:+.2f}/{r[k]['max_dd']*100:.1f}"
    rows.append(row)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
df = pd.DataFrame(rows).set_index("cand").T
print("每格 = 年化增量pp(相对同口径基线) / 该口径盯市最大回撤%")
print(df.to_string())
print("\n绝对值 (期末万 / 年化% / 回撤% / 恐慌成交笔数 / 换仓笔数):")
for n in names:
    r = res[n]
    print(n, " | ".join(f"{k}:{r[k]['final']/1e4:.1f}/{r[k]['cagr']*100:.2f}/{r[k]['max_dd']*100:.1f}/np{r[k]['n_panic']:.0f}/sw{r[k]['n_swap']:.0f}" for k in ("A", "B", "F", "Bx2402")))
