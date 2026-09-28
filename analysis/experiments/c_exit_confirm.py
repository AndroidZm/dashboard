"""决选复核（80 种子）：窗口 A / B / F，剔除 2024-02，逐一剔除大簇（留一簇），执行延迟（触发次日收盘卖）与滑点。"""
import time
from c_exit_lib import *

def T(act, tr, sl=None, tp=None):
    return {"kind": "trail", "act": act, "trail": tr, "sl": sl, "tp": tp}
def FX(tp, sl, mh=None):
    return {"kind": "fixed", "tp": tp, "sl": sl, "mh": mh}

CANDS = {
    "base_tp40_sl40": FX(0.4, 0.4),
    "fx_tp40_noSL": FX(0.4, None),
    "fx_tp45_noSL": FX(0.45, None),
    "fx_tp50_noSL": FX(0.5, None),
    "fx_tp50_sl40": FX(0.5, 0.4),
    "tr_a45_t10_noSL": T(0.45, 0.1),
    "tr_a50_t10_noSL": T(0.5, 0.1),
    "tr_a50_t10_sl40": T(0.5, 0.1, 0.4),
    "tr_a50_t10_sl60": T(0.5, 0.1, 0.6),
    "tr_a60_t15_noSL": T(0.6, 0.15),
    "tr_a70_t20_noSL": T(0.7, 0.2),
}

if __name__ == "__main__":
    t = time.time()
    jobs = []
    for n, r in CANDS.items():
        for w in "ABF":
            jobs.append((n, r, w, (), 80, {}))
        for w in "BF":
            jobs.append((n, r, w, ("2024-02",), 80, {}))
        for m in BIG:
            w = "A" if m < "2016" else "B"
            jobs.append((n, r, w, (m,), 80, {}))
            jobs.append((n, r, "F", (m,), 80, {}))
        for w in "ABF":   # 执行压力：触发后次日收盘卖 + 每边 0.5% 滑点
            jobs.append((n + "|xlag1_slip", {**r, "xlag": 1}, w, (), 80, {"slip": 0.005}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/confirm.csv", index=False)
    print("elapsed", time.time() - t)
