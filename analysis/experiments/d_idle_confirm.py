"""终选确认（80 种子）：A / B / 全期；剔除 2024-02；逐个剔除大簇（LOCO）；执行口径敏感性。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import engine as E
import d_idle_engine as I

D = E.Data.load()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_idle_out")
r300 = lambda ma, h=0.0, lag=0: I.reg_trend_h(D, "sh000300", ma, h, lag)
CF = {
    "base": dict(),
    "300_always_r0": dict(idx="sh000300"),
    "300_always_r20": dict(idx="sh000300", reserve=0.2),
    "300_ma120_r0": dict(idx="sh000300", regime=r300(120)),
    "300_ma120_r10": dict(idx="sh000300", regime=r300(120), reserve=0.1),
    "300_ma120_r20": dict(idx="sh000300", regime=r300(120), reserve=0.2),
    "300_ma150_r20": dict(idx="sh000300", regime=r300(150), reserve=0.2),
    "300_ma200h3_r0": dict(idx="sh000300", regime=r300(200, 0.03)),
    "300_ma200h3_r20": dict(idx="sh000300", regime=r300(200, 0.03), reserve=0.2),
    "300_ma250h3_r0": dict(idx="sh000300", regime=r300(250, 0.03)),
    "300_ma120_r0_lag1": dict(idx="sh000300", regime=r300(120, 0, 1)),
    "300_ma120_r20_lag1": dict(idx="sh000300", regime=r300(120, 0, 1), reserve=0.2),
    "300_ma120_r0_monthly": dict(idx="sh000300", regime=I.reg_monthly(D, r300(120))),
    "300_ma120_r20_monthly": dict(idx="sh000300", regime=I.reg_monthly(D, r300(120)), reserve=0.2),
    "300_ma120_r0_fee3": dict(idx="sh000300", regime=r300(120), idx_fee=0.003),
    "300_ma120_r0_band3": dict(idx="sh000300", regime=r300(120), band=0.03),
    "300_ma120_r0_band0": dict(idx="sh000300", regime=r300(120), band=0.0),
    "852_ma60_r0": dict(idx="sh000852", regime=I.reg_trend_h(D, "sh000852", 60)),
    "g2000_ma40_r0": dict(idx="g2000p", regime=I.reg_trend_h(D, "g2000p", 40)),
}
MAIN = ["base", "300_always_r0", "300_ma120_r0", "300_ma120_r20", "300_ma200h3_r20", "300_ma250h3_r0", "852_ma60_r0"]
rows = []
t = time.time()
for name, kw in CF.items():
    for w, (st, en) in {"A": I.WIN_A, "B": I.WIN_B, "F": I.WIN_F}.items():
        m = I.run_seeds(D, I.IConfig(start=st, end=en, **kw))
        rows.append(dict(cfg=name, win=w, drop="none", **{k: m[k] for k in ["final", "cagr", "max_dd", "cagr_p10", "cagr_p90", "dd_p10", "avg_exposure", "avg_idx", "n_trades", "n_idx_trades"]}))
    if name in MAIN:
        for drop in I.BIG:
            mk = I.month_mask(D, [drop])
            for w, (st, en) in {"A": I.WIN_A, "B": I.WIN_B, "F": I.WIN_F}.items():
                if w == "A" and drop != "2012-01":
                    continue
                if w == "B" and drop == "2012-01":
                    continue
                m = I.run_seeds(D, I.IConfig(start=st, end=en, mask=mk, **kw))
                rows.append(dict(cfg=name, win=w, drop=drop, **{k: m[k] for k in ["final", "cagr", "max_dd", "cagr_p10", "cagr_p90", "dd_p10", "avg_exposure", "avg_idx", "n_trades", "n_idx_trades"]}))
    print(name, round(time.time() - t), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "confirm.csv"), index=False)
print("ALLDONE", flush=True)
