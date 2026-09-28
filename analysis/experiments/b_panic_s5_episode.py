"""阶段 5：候选换仓规则的按簇拆分 + 机制检查。
(1) 规则只在某一个恐慌簇生效（其余照基线），看 A / B / 全期 的 CAGR 增量 -> 增量来自哪一簇；
(2) 换出去的持仓若继续持有到自然出场的收益 vs 换进来的恐慌股收益（逐笔、按金额）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from dataclasses import replace
from b_panic_engine import Data, PConfig, run, run_seeds, WIN_A, WIN_B, WIN_F, masks

D = Data.load(); M = masks(D); ym = M["ym"]
SE = range(int(os.environ.get("NSEED", 80)))
CANDS = {
    "oldest_w4": PConfig(swap_policy="oldest", swap_on="both", panic_max_pos=None, weights=(0.02, 0.08, 0.04), swap_keep_panic=True),
    "flat_w4": PConfig(swap_policy="flat", swap_on="both", panic_max_pos=None, weights=(0.02, 0.08, 0.04), swap_keep_panic=True),
    "gainpct_w6": PConfig(swap_policy="gain_pct", swap_on="both", panic_max_pos=None, weights=(0.02, 0.08, 0.06), swap_keep_panic=True),
}
if len(sys.argv) > 1:
    CANDS = {k: v for k, v in CANDS.items() if k in sys.argv[1:]}
tier2_months = sorted(set(ym[D.tier == 2]))
print("tier2 months:", {m: int(((ym == m) & (D.tier == 2)).sum()) for m in tier2_months})
e_all, x_all, r_all = D.exits(0.4, 0.4, None, len(D.dates) - 1, 0)
for name, c in CANDS.items():
    print("=" * 30, name)
    for tag, (st, en) in (("A", WIN_A), ("B", WIN_B), ("F", WIN_F)):
        b = run_seeds(D, PConfig(start=st, end=en), SE)
        full = run_seeds(D, replace(c, start=st, end=en), SE)
        line = f"{tag}: base {b['cagr']*100:.2f}% dd{b['max_dd']*100:.1f} | full rule {full['cagr']*100-b['cagr']*100:+.2f}pp dd{full['max_dd']*100:.1f} np{full['n_panic']:.0f} |"
        for mth in tier2_months:
            if not (st <= mth + "-31" and mth + "-01" <= en):
                continue
            if ((ym == mth) & (D.tier == 2)).sum() < 2:
                continue
            m = run_seeds(D, replace(c, start=st, end=en, rule_mask=(ym == mth)), SE)
            line += f" {mth}:{(m['cagr']-b['cagr'])*100:+.2f}"
        print(line)
    # 机制：单条路径逐笔
    for tag, (st, en) in (("A", WIN_A), ("B", WIN_B)):
        t1 = D.day_le(en)
        e, x, r = D.exits(0.4, 0.4, None, t1, 0)
        rows = []
        for sd in range(20):
            res = run(D, replace(c, start=st, end=en, record=True), sd)
            tr = res["trades"]
            sw = tr[tr.kind == "SWAP"]
            for _, q in sw.iterrows():
                s = int(q.s); t = int(q.sell_t)
                # 继续持有到自然出场：从卖出日价格到自然出场价格的比
                cont = D.close[s, x[s]] / D.close[s, t]
                rows.append(dict(seed=sd, ym=D.dates[t][:7], kind="sold", amt=q.amount + q.pnl, fwd=cont - 1, days=x[s] - t, cur=(q.amount + q.pnl) / q.amount - 1))
            pb = tr[(tr.tier == 2)]
            for _, q in pb.iterrows():
                s = int(q.s)
                rows.append(dict(seed=sd, ym=D.dates[int(q.buy_t)][:7], kind="panic_buy", amt=q.amount, fwd=r[s] - 1, days=x[s] - int(q.buy_t), cur=0))
        df = pd.DataFrame(rows)
        if len(df):
            g = df.groupby(["ym", "kind"]).apply(lambda d: pd.Series(dict(n=len(d) / 20, amt_w_fwd=np.average(d.fwd, weights=d.amt) * 100, med_days=d.days.median(), cur=d.cur.mean() * 100)), include_groups=False)
            print(tag, "mechanism (per seed avg n; amount-weighted forward return % to natural exit; cur = unrealized at swap)")
            print(g.round(1).to_string())
