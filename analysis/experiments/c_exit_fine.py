"""细网格：移动止盈（启动线 act x 回撤幅度 trail x 硬止损 sl，不设硬止盈）与固定止盈 tp x sl，20 种子，窗口 A / B。"""
import itertools, time
from c_exit_lib import *
if __name__ == "__main__":
    t = time.time()
    D = getD()
    R = []
    for act, tr, sl in itertools.product([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7], [0.05, 0.075, 0.1, 0.125, 0.15, 0.2], [0.4, 0.5, 0.6, 0.7, None]):
        R.append((f"trail_a{act}_t{tr}_sl{sl}", {"kind": "trail", "act": act, "trail": tr, "sl": sl, "tp": None}, dict(fam="trail", act=act, tr=tr, sl=sl)))
    for tp, sl in itertools.product([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7], [0.4, 0.5, 0.6, 0.7, None]):
        R.append((f"fixed_tp{tp}_sl{sl}", {"kind": "fixed", "tp": tp, "sl": sl, "mh": None}, dict(fam="fixed", tp=tp, sl=sl)))
    jobs = [(n, r, w, (), 20, {}) for n, r, _ in R for w in "AB"]
    df = evaluate(jobs)
    meta = pd.DataFrame([{"name": n, "win": w, **m, **per_trade(D, r, w)} for n, r, m in R for w in "AB"])
    df = df.merge(meta, on=["name", "win"])
    df.to_csv("c_exit_out/grid_fine.csv", index=False)
    print("elapsed", time.time() - t)
