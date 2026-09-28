"""止损宽度 x 移动止盈平台（80 种子）+ 幸存者偏差压力（q=0.2/0.3，3 个抽样种子 x 40 执行种子）。"""
import itertools, time
from c_exit_lib import *
from c_exit_confirm import T, FX
if __name__ == "__main__":
    t = time.time()
    R = {"base": FX(0.4, 0.4)}
    for a, tr, sl in itertools.product([0.45, 0.5, 0.55], [0.075, 0.1, 0.125], [0.4, 0.5, 0.6, 0.7, None]):
        R[f"tr_a{a}_t{tr}_sl{sl}"] = T(a, tr, sl)
    for tp, sl in itertools.product([0.4, 0.45, 0.5], [0.4, 0.5, 0.6, 0.7, None]):
        R[f"fx_tp{tp}_sl{sl}"] = FX(tp, sl)
    jobs = []
    for n, r in R.items():
        for w in "ABF":
            jobs.append((n + "|q0", r, w, (), 80, {}))
            for q in [0.2, 0.3]:
                for ss in range(3):
                    jobs.append((f"{n}|q{q}|{ss}", {**r, "stress_q": q, "stress_seed": ss}, w, (), 40, {}))
    df = evaluate(jobs)
    df.to_csv("c_exit_out/slgrid.csv", index=False)
    print("elapsed", time.time() - t)
