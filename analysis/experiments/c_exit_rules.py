"""自定义出场机制网格（组合层面，20 种子筛选，窗口 A / B）：
移动止盈 trail、锁利 lock、分批止盈 legs、到期只卖盈利 tprof、到期降低止盈线 decay。"""
import itertools, time
from c_exit_lib import *

def rules():
    out = []
    out.append(("base", {"kind": "fixed", "tp": 0.4, "sl": 0.4, "mh": None}))
    for act, tr, sl, tp in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.15, 0.2, 0.25, 0.3], [0.4, None], [None, 1.0]):
        out.append((f"trail_a{act}_t{tr}_sl{sl}_tp{tp}", {"kind": "trail", "act": act, "trail": tr, "sl": sl, "tp": tp}))
    for act, fl, tp in itertools.product([0.3, 0.4, 0.5, 0.6], [0.0, 0.1, 0.2, 0.3], [0.6, 0.8, 1.0, None]):
        if fl < act and (tp is None or tp > act):
            out.append((f"lock_a{act}_f{fl}_tp{tp}", {"kind": "lock", "act": act, "floor": fl, "sl": 0.4, "tp": tp}))
    sec = {"tp0.8": {"kind": "fixed", "tp": 0.8, "sl": 0.4, "mh": None},
           "tp1.0": {"kind": "fixed", "tp": 1.0, "sl": 0.4, "mh": None},
           "tpNone": {"kind": "fixed", "tp": None, "sl": 0.4, "mh": None},
           "trail.6_.2": {"kind": "trail", "act": 0.6, "trail": 0.2, "sl": 0.4, "tp": None},
           "lock.6_.3": {"kind": "lock", "act": 0.6, "floor": 0.3, "sl": 0.4, "tp": None}}
    for f, tp1, (sn, sr) in itertools.product([0.33, 0.5, 0.67], [0.3, 0.4, 0.5], sec.items()):
        out.append((f"legs_f{f}_tp{tp1}_{sn}", {"kind": "legs", "legs": [(f, {"kind": "fixed", "tp": tp1, "sl": 0.4, "mh": None}), (1 - f, sr)]}))
    for tp, mh, mr in itertools.product([0.4, 0.5, 0.6, 0.8, None], [250, 500, 750], [1.0, 1.1, 1.2]):
        out.append((f"tprof_tp{tp}_mh{mh}_r{mr}", {"kind": "tprof", "tp": tp, "sl": 0.4, "mh": mh, "min_r": mr}))
    for tp, tp2, mh in itertools.product([0.6, 0.8, 1.0], [0.1, 0.2, 0.3], [250, 500]):
        out.append((f"decay_tp{tp}_tp2{tp2}_mh{mh}", {"kind": "decay", "tp": tp, "tp2": tp2, "sl": 0.4, "mh": mh}))
    return out

if __name__ == "__main__":
    t = time.time()
    D = getD()
    R = rules()
    print(len(R), "rules")
    jobs = [(n, r, w, (), 20, {}) for n, r in R for w in "AB"]
    df = evaluate(jobs)
    pts = pd.DataFrame([{"name": n, "win": w, **per_trade(D, r, w)} for n, r in R for w in "AB"])
    df = df.merge(pts, on=["name", "win"])
    df.to_csv("c_exit_out/grid_rules.csv", index=False)
    print("elapsed", time.time() - t)
