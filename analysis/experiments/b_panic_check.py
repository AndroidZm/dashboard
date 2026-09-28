"""复现检查：b_panic_engine 在新选项关闭时必须与 engine.run 逐种子一致。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from b_panic_engine import Data, Config, PConfig, run, run_seeds, engine_run, WIN_A, WIN_B, WIN_F, fmt
from engine import run_seeds as e_run_seeds

D = Data.load()
for st, en in (WIN_B, WIN_F, WIN_A):
    for kw in ({}, {"panic_swap": True}, {"panic_max_pos": None}, {"cap": 1.0}):
        diffs = []
        for sd in range(80):
            a = engine_run(D, Config(start=st, end=en, **kw), sd)
            ckw = dict(kw)
            if kw.get("panic_swap"):
                ckw = {"swap_policy": "oldest", "swap_fix": False}
            b = run(D, PConfig(start=st, end=en, no_lev=False, **ckw), sd)
            diffs.append(abs(a["final"] - b["final"]) + abs(a["max_dd"] - b["max_dd"]))
        print(st, en, kw, "max abs diff over 80 seeds:", max(diffs))
    print(" baseline:", fmt(run_seeds(D, PConfig(start=st, end=en))))
