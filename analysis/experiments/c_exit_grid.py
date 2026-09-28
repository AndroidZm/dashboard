"""固定 TP x SL x 最长持有 网格（组合层面，20 种子筛选），窗口 A / B 各跑一遍，并附逐笔统计。"""
import itertools, time
from c_exit_lib import *
TP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, None]
SL = [0.2, 0.3, 0.4, 0.5, 0.6, None]
MH = [None, 250, 500, 750]
if __name__ == "__main__":
    t = time.time()
    D = getD()
    jobs = []
    for tp, sl, mh in itertools.product(TP, SL, MH):
        rule = {"kind": "fixed", "tp": tp, "sl": sl, "mh": mh}
        name = f"tp{tp}_sl{sl}_mh{mh}"
        for w in "AB":
            jobs.append((name, rule, w, (), 20, {}))
    df = evaluate(jobs)
    pts = []
    for tp, sl, mh in itertools.product(TP, SL, MH):
        rule = {"kind": "fixed", "tp": tp, "sl": sl, "mh": mh}
        for w in "AB":
            pts.append({"name": f"tp{tp}_sl{sl}_mh{mh}", "win": w, "tp": tp, "sl": sl, "mh": mh, **per_trade(D, rule, w)})
    df = df.merge(pd.DataFrame(pts), on=["name", "win"])
    df.to_csv("c_exit_out/grid_fixed.csv", index=False)
    print("elapsed", time.time() - t)
