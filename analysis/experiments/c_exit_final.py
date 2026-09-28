"""最终候选全套复核（80 种子）：A/B/F，12 个大月份逐一剔除 + 三大簇同时剔除，执行延迟+滑点，
幸存者偏差压力 q=0.2/0.3（3 个抽样种子 x 40 执行种子），同种子配对胜率，逐笔统计，回撤日期。"""
import time, sys
from c_exit_lib import *
from c_exit_confirm import T, FX
from c_exit_confirm2 import MONTHS

def T2(a, t, a2, t2, sl):
    return {"kind": "trail2", "act": a, "trail": t, "act2": a2, "trail2": t2, "sl": sl, "tp": None}

FINAL = {
    "base": FX(0.4, 0.4),
    "tr_a45_t05_sl40": T(0.45, 0.05, 0.4),
    "t2_a50_t075_x100_t30_sl40": T2(0.5, 0.075, 1.0, 0.3, 0.4),
    "t2_a50_t075_x100_t35_sl40": T2(0.5, 0.075, 1.0, 0.35, 0.4),
    "tr_a50_t10_sl70": T(0.5, 0.1, 0.7),
    "t2_a50_t075_x100_t30_sl70": T2(0.5, 0.075, 1.0, 0.3, 0.7),
    "t2_a50_t10_x150_t35_noSL": T2(0.5, 0.1, 1.5, 0.35, None),
    "tr_a70_t20_noSL": T(0.7, 0.2, None),
    "tr_a60_t30_sl70": T(0.6, 0.3, 0.7),
    "tr_a100_t30_noSL": T(1.0, 0.3, None),
}
if __name__ == "__main__":
    t = time.time()
    D = getD()
    jobs = []
    for n, r in FINAL.items():
        for w in "ABF":
            jobs.append((n, r, w, (), 80, {}))
            jobs.append((n + "|xlag1_slip", {**r, "xlag": 1}, w, (), 80, {"slip": 0.005}))
            for q in [0.2, 0.3]:
                for ss in range(3):
                    jobs.append((f"{n}|q{q}|{ss}", {**r, "stress_q": q, "stress_seed": ss}, w, (), 40, {}))
        for m in MONTHS:
            wm = "A" if m < "2016" else "B"
            jobs.append((n, r, wm, (m,), 80, {}))
            jobs.append((n, r, "F", (m,), 80, {}))
        jobs.append((n, r, "A", ('2012-01', '2012-08', '2012-12'), 80, {}))
        jobs.append((n, r, "B", ('2022-05', '2022-10', '2024-02'), 80, {}))
        jobs.append((n, r, "F", ('2012-01', '2022-05', '2024-02'), 80, {}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/final.csv", index=False)
    # 配对胜率、回撤日期、逐笔统计
    rows = []
    for w in "ABF":
        base_f = np.array([run_legs(D, cfg_for(w), FINAL["base"], s)["final"] for s in range(80)])
        for n, r in FINAL.items():
            rs = [run_legs(D, cfg_for(w), r, s) for s in range(80)]
            f = np.array([x["final"] for x in rs])
            dd = pd.Series([x["dd_date"][:7] for x in rs]).value_counts()
            rows.append({"name": n, "win": w, "paired_win": float(np.mean(f > base_f)), "dd_month_mode": dd.index[0],
                         "dd_month_share": float(dd.iloc[0] / 80), "tr_mean": float(np.median([x["trade_mean_ret"] for x in rs])),
                         "tr_hold": float(np.median([x["trade_med_hold"] for x in rs])), **per_trade(D, r, w)})
    pd.DataFrame(rows).to_csv("c_exit_out/final_extra.csv", index=False)
    print("elapsed", time.time() - t)
