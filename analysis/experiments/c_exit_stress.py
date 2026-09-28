"""幸存者偏差压力测试：持有期内收盘曾跌破 -40% 的信号中抽出比例 q 视为“其实退市”，
无止损规则按 -95% 出场，有止损 sl 的规则按 -sl 出场（止损先触发）。5 个抽样种子 x 40 个执行种子取中位。"""
import time
from c_exit_lib import *
from c_exit_confirm import CANDS

NAMES = ["base_tp40_sl40", "fx_tp40_noSL", "fx_tp50_noSL", "fx_tp50_sl40", "tr_a50_t10_noSL", "tr_a50_t10_sl40",
         "tr_a50_t10_sl60", "tr_a60_t15_noSL", "tr_a70_t20_noSL"]
if __name__ == "__main__":
    t = time.time()
    jobs = []
    for n in NAMES:
        for q in [0.1, 0.2, 0.3, 0.5]:
            for ss in range(5):
                for w in "ABF":
                    jobs.append((f"{n}|q{q}|ss{ss}", {**CANDS[n], "stress_q": q, "stress_seed": ss}, w, (), 40, {}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/stress.csv", index=False)
    print("elapsed", time.time() - t)
    df["cand"] = df.name.str.split("|").str[0]
    df["q"] = df.name.str.split("|").str[1]
    g = df.groupby(["cand", "q", "win"])[["final", "cagr", "max_dd"]].median().reset_index()
    base = g[g.cand == "base_tp40_sl40"].set_index(["q", "win"])
    g["dC"] = g.cagr - [base.loc[(q, w), "cagr"] for q, w in zip(g.q, g.win)]
    pd.set_option("display.width", 250)
    print(g.pivot_table(index=["q", "win"], columns="cand", values="final").reindex(columns=NAMES).round(0).to_string())
    print(g.pivot_table(index=["q", "win"], columns="cand", values="dC").reindex(columns=NAMES).round(2).to_string())
    print(g.pivot_table(index=["q", "win"], columns="cand", values="max_dd").reindex(columns=NAMES).round(1).to_string())
