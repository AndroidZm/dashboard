"""出场规则与仓位的交互（资金占用问题）：权重整体放大 k 倍（上限仍 90%、不融资），以及取消恐慌档 15 只上限。80 种子。"""
import time
from c_exit_lib import *
from c_exit_confirm import CANDS, T, FX
C = {k: CANDS[k] for k in ["base_tp40_sl40", "fx_tp40_noSL", "tr_a50_t10_sl40", "tr_a50_t10_noSL", "tr_a50_t10_sl60", "tr_a70_t20_noSL"]}
C["tr_a45_t05_sl40"] = T(0.45, 0.05, 0.4)
C["fx_tp30_sl40"] = FX(0.3, 0.4)
if __name__ == "__main__":
    t = time.time()
    jobs = []
    for n, r in C.items():
        for k in [1.0, 1.5, 2.0, 3.0]:
            for w in "ABF":
                jobs.append((f"{n}|k{k}", r, w, (), 80, {"weights": (0.02 * k, 0.08 * k, 0.06 * k)}))
        for w in "ABF":
            jobs.append((f"{n}|k1.0_nomax", r, w, (), 80, {"panic_max_pos": None}))
            jobs.append((f"{n}|k2.0_nomax", r, w, (), 80, {"panic_max_pos": None, "weights": (0.04, 0.16, 0.12)}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/sizing.csv", index=False)
    print("elapsed", time.time() - t)
    df["cand"] = df.name.str.split("|").str[0]; df["k"] = df.name.str.split("|").str[1]
    pd.set_option("display.width", 250)
    for v in ["final", "max_dd", "avg_exp", "n2"]:
        print(v)
        print(df.pivot_table(index=["k", "win"], columns="cand", values=v).reindex(columns=list(C)).round(1 if v != "avg_exp" else 2).to_string())
