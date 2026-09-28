"""决选第二轮（80 种子）：保留 -40% 止损的移动止盈 vs 放宽/取消止损，扩展留一簇（所有 >=8 笔的月份）+ 执行延迟压力。"""
import time
from c_exit_lib import *
from c_exit_confirm import T, FX
CANDS2 = {
    "base": FX(0.4, 0.4),
    "tr_a40_t05_sl40": T(0.4, 0.05, 0.4),
    "tr_a45_t05_sl40": T(0.45, 0.05, 0.4),
    "tr_a45_t075_sl40": T(0.45, 0.075, 0.4),
    "tr_a50_t075_sl40": T(0.5, 0.075, 0.4),
    "tr_a50_t10_sl40": T(0.5, 0.1, 0.4),
    "tr_a50_t10_sl70": T(0.5, 0.1, 0.7),
    "tr_a50_t10_noSL": T(0.5, 0.1, None),
    "tr_a70_t20_sl40": T(0.7, 0.2, 0.4),
    "tr_a70_t20_noSL": T(0.7, 0.2, None),
}
MONTHS = ['2008-09', '2012-01', '2012-08', '2012-12', '2018-02', '2018-07', '2018-10', '2021-02', '2022-05', '2022-10', '2024-02', '2026-07']
if __name__ == "__main__":
    t = time.time()
    jobs = []
    for n, r in CANDS2.items():
        for w in "ABF":
            jobs.append((n, r, w, (), 80, {}))
            jobs.append((n + "|xlag1_slip", {**r, "xlag": 1}, w, (), 80, {"slip": 0.005}))
        for m in MONTHS:
            w = "A" if m < "2016" else "B"
            jobs.append((n, r, w, (m,), 80, {}))
            jobs.append((n, r, "F", (m,), 80, {}))
        jobs.append((n, r, "A", ('2012-01', '2012-08', '2012-12'), 80, {}))
        jobs.append((n, r, "B", ('2022-05', '2022-10', '2024-02'), 80, {}))
        jobs.append((n, r, "F", ('2012-01', '2022-05', '2024-02'), 80, {}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/confirm2.csv", index=False)
    print("elapsed", time.time() - t)
