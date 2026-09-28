"""80 种子复核：移动止盈网格（sl 0.4 / 不设），窗口 A / B / F（全期），并计算 3x3 邻域平均（平台检查）。"""
import itertools, time
from c_exit_lib import *
if __name__ == "__main__":
    t = time.time()
    D = getD()
    R = [("fixed_tp0.4_sl0.4", {"kind": "fixed", "tp": 0.4, "sl": 0.4, "mh": None}, dict(fam="base"))]
    for act, tr, sl in itertools.product([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7], [0.05, 0.075, 0.1, 0.125, 0.15, 0.2], [0.4, None]):
        R.append((f"trail_a{act}_t{tr}_sl{sl}", {"kind": "trail", "act": act, "trail": tr, "sl": sl, "tp": None}, dict(fam="trail", act=act, tr=tr, sl=sl)))
    for tp, sl in itertools.product([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7], [0.4, None]):
        R.append((f"fixed_tp{tp}_sl{sl}", {"kind": "fixed", "tp": tp, "sl": sl, "mh": None}, dict(fam="fixed", tp=tp, sl=sl)))
    jobs = [(n, r, w, (), 80, {}) for n, r, _ in R for w in "ABF"]
    df = evaluate(jobs)
    meta = pd.DataFrame([{"name": n, "win": w, **m, **per_trade(D, r, w)} for n, r, m in R for w in "ABF"])
    df = df.merge(meta, on=["name", "win"])
    df.to_csv("c_exit_out/grid_fine80.csv", index=False)
    print("elapsed", time.time() - t)
