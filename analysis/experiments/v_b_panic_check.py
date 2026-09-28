"""复核 1：(a) 独立引擎关掉换仓时与 engine.run 逐种子一致；(b) 候选配置在独立引擎与 b_panic_engine 下逐种子对比。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
import numpy as np
from v_b_panic_engine import Data, VConfig, run, WA, WB, WF
from engine import Config, run as erun
import b_panic_engine as B

D = Data.load()
t = time.time()
for st, en in (WB, WA, WF):
    d = max(abs(erun(D, Config(start=st, end=en), sd)["final"] - run(D, VConfig(start=st, end=en, no_lev=False), sd)["final"]) for sd in range(80))
    d2 = max(abs(erun(D, Config(start=st, end=en), sd)["max_dd"] - run(D, VConfig(start=st, end=en, no_lev=True), sd)["max_dd"]) for sd in range(80))
    print("baseline eq check", st, en, "max|dfinal|", d, "max|ddd|", d2)
print("t", time.time() - t)
SW = dict(swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5, swap_policy="oldest")
C = {"L1": ((0.02, 0.08, 0.04), 0.9), "L1b": ((0.03, 0.12, 0.04), 0.9), "L2": ((0.03, 0.12, 0.04), 1.0), "L3": ((0.03, 0.16, 0.04), 1.0)}
for n, (w, cap) in C.items():
    for st, en in (WB, WA, WF):
        for lag, slip in ((0, 0.0), (1, 0.005)):
            dif = []
            for sd in range(80):
                a = run(D, VConfig(start=st, end=en, weights=w, cap=cap, panic_max_pos=None, swap=True, max_n=5, lag=lag, slip=slip), sd)
                b = B.run(D, B.PConfig(start=st, end=en, weights=w, cap=cap, lag=lag, slip=slip, **SW), sd)
                dif.append(abs(a["final"] / b["final"] - 1))
            print(n, st, lag, slip, "max rel diff final (mine vs theirs):", f"{max(dif):.2e}", "n_diff>1e-9:", sum(x > 1e-9 for x in dif))
print("t", time.time() - t)
