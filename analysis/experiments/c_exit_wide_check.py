"""宽移动止盈（第 2/3 风险档候选）加 -70% 灾难止损的复核：A/B/F、留一簇、2024-02、执行延迟、幸存者压力。"""
from c_exit_lib import *
from c_exit_confirm import T, FX
from c_exit_confirm2 import MONTHS
C = {"base": FX(0.4, 0.4), "tr_a100_t30_sl70": T(1.0, 0.3, 0.7), "tr_a80_t30_sl70": T(0.8, 0.3, 0.7), "tr_a100_t25_sl70": T(1.0, 0.25, 0.7)}
if __name__ == "__main__":
    jobs = []
    for n, r in C.items():
        for w in "ABF":
            jobs.append((n, r, w, (), 80, {}))
            jobs.append((n + "|xlag1_slip", {**r, "xlag": 1}, w, (), 80, {"slip": 0.005}))
            for ss in range(3):
                jobs.append((f"{n}|q0.2|{ss}", {**r, "stress_q": 0.2, "stress_seed": ss}, w, (), 40, {}))
        for m in MONTHS:
            jobs.append((n, r, "A" if m < "2016" else "B", (m,), 80, {}))
            jobs.append((n, r, "F", (m,), 80, {}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/wide_check.csv", index=False)
