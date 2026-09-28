"""复现检查：d_idle_engine 在 idx=None / regime 全 False 时与 engine.run 逐种子一致。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import engine as E
import d_idle_engine as I

D = E.Data.load()
for st, en in [I.WIN_B, I.WIN_F, I.WIN_A]:
    diffs = []
    for s in range(20):
        a = E.run(D, E.Config(start=st, end=en), s)
        b = I.run(D, I.IConfig(start=st, end=en), s)
        c = I.run(D, I.IConfig(start=st, end=en, idx="sh000300", regime=np.zeros(len(D.dates), bool)), s)
        diffs.append(max(abs(a["final"] - b["final"]), abs(a["max_dd"] - b["max_dd"]) * 1e6, abs(a["final"] - c["final"])))
    t = time.time()
    m = I.run_seeds(D, I.IConfig(start=st, end=en))
    print(st, en, "max abs diff over 20 seeds:", max(diffs), "| 80-seed median final %.1f万 cagr %.2f%% dd %.1f%% ntr %d (%.1fs)" % (
        m["final"] / 1e4, m["cagr"] * 100, m["max_dd"] * 100, m["n_trades"], time.time() - t))
