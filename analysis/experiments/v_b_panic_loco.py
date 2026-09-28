"""复核 5：去簇 / 逐个去大簇 (LOCO) / 同时去掉全部 5 个大簇；以及“只在某个月启用换仓”的逐簇归因（相对同权重不换仓）。80 种子。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
import numpy as np, pandas as pd
from v_b_panic_engine import Data, VConfig, run_seeds, WA, WB, WF, BIG

D = Data.load()
ym = D.sig.signal_date.str[:7].values
SW = dict(panic_max_pos=None, swap=True, max_n=5, keep_panic=True, sell_key="oldest")
C = {"L1": ((0.02, 0.08, 0.04), 0.9), "L1b": ((0.03, 0.12, 0.04), 0.9), "L2": ((0.03, 0.12, 0.04), 1.0), "L3": ((0.03, 0.16, 0.04), 1.0)}
allbig = ~np.isin(ym, BIG)
SCEN = [("A", WA, None), ("A-2012-01", WA, ym != "2012-01"),
        ("B", WB, None)] + [(f"B-{c}", WB, ym != c) for c in BIG[1:]] + [("B-2024-02-2018-02", WB, ~np.isin(ym, ["2024-02", "2018-02"])),
        ("B-all4big", WB, ~np.isin(ym, BIG[1:])), ("F", WF, None)] + [(f"F-{c}", WF, ym != c) for c in BIG] + [("F-all5big", WF, allbig)]
rows = []
for tag, (st, en), mk in SCEN:
    b = run_seeds(D, VConfig(start=st, end=en, mask=mk))
    row = dict(scen=tag, base=b["cagr"] * 100, base_dd=b["max_dd"] * 100)
    for n, (w, cap) in C.items():
        m = run_seeds(D, VConfig(start=st, end=en, mask=mk, weights=w, cap=cap, **SW))
        ns = run_seeds(D, VConfig(start=st, end=en, mask=mk, weights=(w[0], w[1], 0.06), cap=cap))
        row[n + "_vsBase"] = (m["cagr"] - b["cagr"]) * 100
        row[n + "_vsNoSwap"] = (m["cagr"] - ns["cagr"]) * 100
        row[n + "_dd"] = m["max_dd"] * 100
    rows.append(row)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
df = pd.DataFrame(rows).set_index("scen")
pd.set_option("display.width", 250)
print(df.round(2).to_string())

# 逐簇归因：只在月份 m 的恐慌信号上启用换仓（相对同权重、不限恐慌持仓、不换仓）
tier2 = D.tier == 2
months = sorted(set(ym[tier2]))
cnt = {m: int(((ym == m) & tier2).sum()) for m in months}
print("panic months:", cnt)
small = [m for m in months if cnt[m] < 10]
print("small panic months (<10 panic signals):", small)
for n, (w, cap) in C.items():
    line = n + " F-window, swap only in month (vs same cfg no swap):"
    nos = run_seeds(D, VConfig(start=WF[0], end=WF[1], weights=w, cap=cap, panic_max_pos=None))
    full = run_seeds(D, VConfig(start=WF[0], end=WF[1], weights=w, cap=cap, **SW))
    line += f" ALL {(full['cagr']-nos['cagr'])*100:+.2f} |"
    for lab, sel in [(m, ym == m) for m in months if cnt[m] >= 2] + [("SMALL<10", np.isin(ym, small)), ("BIG5", np.isin(ym, BIG))]:
        m = run_seeds(D, VConfig(start=WF[0], end=WF[1], weights=w, cap=cap, **SW, swap_mask=sel))
        line += f" {lab}:{(m['cagr']-nos['cagr'])*100:+.2f}"
    print(line, flush=True)
