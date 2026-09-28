"""v_d_idle 复核 3（80 种子）：剔除 2024-02、逐个剔除大簇、一次剔除全部 5 个大簇；
实盘口径（信号次日成交 + 每边 0.5% 滑点；regime 次日；指数额外滑点）；每日再平衡口径。"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I
import v_d_idle_engine2 as V

D = E.Data.load()
W = {"A": V.A, "B": V.B, "F": V.F}
C = {
    "C1": dict(idx="sh000300", ma=120, reserve=0.2, rl=0),
    "C2": dict(idx="sh000300", ma=120, reserve=0.0, rl=0),
    "C3": dict(idx="sh000300", ma=120, reserve=0.3, rl=1),
    "C4": dict(idx="sh000852", ma=60, reserve=0.0, rl=0),
}
BIG = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]
DROPS = {"none": None, **{d: [d] for d in BIG}, "all5": BIG}
rows = []
t = time.time()


def rec(cand, win, drop, exe, m):
    rows.append(dict(cand=cand, win=win, drop=drop, exe=exe, **{k: m[k] for k in ["final", "cagr", "max_dd", "cagr_p10", "cagr_p90", "dd_p10", "avg_idx"] if k in m},
                     dd_worst=m.get("dd_worst", np.nan)))


def seeds_their(cfg):
    out = I.run_seeds(D, cfg)
    return out


# 簇剔除
for dname, months in DROPS.items():
    mk = None if months is None else I.month_mask(D, months)
    for w, (st, en) in W.items():
        if w == "A" and dname not in ("none", "2012-01", "all5"):
            continue
        if w == "B" and dname == "2012-01":
            continue
        rec("base", w, dname, "same", I.run_seeds(D, I.IConfig(start=st, end=en, mask=mk)))
        for c, p in C.items():
            rg = I.reg_trend_h(D, p["idx"], p["ma"], 0.0, p["rl"])
            rec(c, w, dname, "same", I.run_seeds(D, I.IConfig(start=st, end=en, mask=mk, idx=p["idx"], reserve=p["reserve"], regime=rg)))
    print(dname, round(time.time() - t), flush=True)

# 执行口径：信号 lag=1 + slip 0.005；regime lag=1；指数费用 0.1%/0.2%/0.6%
for w, (st, en) in W.items():
    rec("base", w, "none", "lag1slip", I.run_seeds(D, I.IConfig(start=st, end=en, lag=1, slip=0.005)))
    for c, p in C.items():
        rg1 = I.reg_trend_h(D, p["idx"], p["ma"], 0.0, 1)
        for fee in (0.001, 0.002, 0.006):
            rec(c, w, "none", f"lag1slip_idxfee{fee}", I.run_seeds(D, I.IConfig(start=st, end=en, lag=1, slip=0.005, idx=p["idx"], reserve=p["reserve"], regime=rg1, idx_fee=fee)))
        # 每日再平衡（独立引擎）
        rg = V.my_regime(D, p["idx"], p["ma"], p["rl"])
        rec(c, w, "none", "daily_rebal", V.seeds2(D, start=st, end=en, idx=p["idx"], reserve=p["reserve"], regime=rg, rebal="daily"))
        rec(c, w, "none", "daily_rebal_lag1slip_fee2", V.seeds2(D, start=st, end=en, idx=p["idx"], reserve=p["reserve"], regime=V.my_regime(D, p["idx"], p["ma"], 1),
                                                                rebal="daily", lag=1, slip=0.005, idx_fee=0.002))
    print("exe", w, round(time.time() - t), flush=True)
os.makedirs(os.path.join(HERE, "v_d_idle_out"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "v_d_idle_out", "s3_robust.csv"), index=False)
print("ALLDONE")
