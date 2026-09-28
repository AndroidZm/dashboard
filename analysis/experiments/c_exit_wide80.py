"""80 种子：宽移动止盈（偏向窗口 A 的“让利润奔跑”）与两段式移动止盈，窗口 A / B / F。"""
import itertools, time
from c_exit_lib import *
from c_exit_confirm import T, FX
if __name__ == "__main__":
    t = time.time()
    R = {"base": (FX(0.4, 0.4), dict(fam="base"))}
    for a, tr, sl in itertools.product([0.5, 0.6, 0.7, 0.8, 1.0], [0.2, 0.25, 0.3, 0.35, 0.4], [0.4, None]):
        R[f"tr_a{a}_t{tr}_sl{sl}"] = (T(a, tr, sl), dict(fam="trail", act=a, tr=tr, sl=sl))
    for tr, a2, tr2, sl in itertools.product([0.075, 0.1], [0.8, 1.0, 1.5], [0.2, 0.25, 0.3, 0.35], [0.4, 0.7, None]):
        R[f"t2_a0.5_t{tr}_a2{a2}_t2{tr2}_sl{sl}"] = ({"kind": "trail2", "act": 0.5, "trail": tr, "act2": a2, "trail2": tr2, "sl": sl, "tp": None},
                                                     dict(fam="trail2", tr=tr, act2=a2, tr2=tr2, sl=sl))
    jobs = [(n, r, w, (), 80, {}) for n, (r, _) in R.items() for w in "ABF"]
    df = evaluate(jobs)
    meta = pd.DataFrame([{"name": n, **m} for n, (_, m) in R.items()])
    df = df.merge(meta, on="name")
    df.to_csv("c_exit_out/wide80.csv", index=False)
    print("elapsed", time.time() - t)
