"""两段式移动止盈：+act 启动、回撤 trail 卖出；峰值超过 +act2 后回撤放宽到 trail2。20 种子筛选 A/B/F。"""
import itertools, time
from c_exit_lib import *
from c_exit_confirm import T, FX
if __name__ == "__main__":
    t = time.time()
    R = {"base": FX(0.4, 0.4), "tr_a50_t10_sl40": T(0.5, 0.1, 0.4), "tr_a50_t10_noSL": T(0.5, 0.1), "tr_a70_t20_noSL": T(0.7, 0.2)}
    for a, tr, a2, tr2, sl in itertools.product([0.45, 0.5], [0.075, 0.1], [0.8, 1.0, 1.5], [0.15, 0.2, 0.25, 0.3], [0.4, None]):
        R[f"t2_a{a}_t{tr}_a2{a2}_t2{tr2}_sl{sl}"] = {"kind": "trail2", "act": a, "trail": tr, "act2": a2, "trail2": tr2, "sl": sl, "tp": None}
    jobs = [(n, r, w, (), 20, {}) for n, r in R.items() for w in "ABF"]
    df = evaluate(jobs)
    df.to_csv("c_exit_out/trail2.csv", index=False)
    print("elapsed", time.time() - t)
    p = df.pivot_table(index="name", columns="win", values="cagr")
    b = p.loc["base"]
    d = (p - b).round(2)
    d["min"] = d.min(axis=1)
    d = d.join(df.pivot_table(index="name", columns="win", values="final").round(0).add_prefix("fin_")).join(df.pivot_table(index="name", columns="win", values="max_dd").round(1).add_prefix("dd_"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(d.sort_values("min", ascending=False).to_string())
