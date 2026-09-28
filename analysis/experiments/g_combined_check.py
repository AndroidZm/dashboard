"""g_combined_engine 复现检查：各选项单独打开时，与原来已复核过的引擎逐种子比较。

1) 全部关闭 → engine.run（基线、K1 cap1+partial、L=1.1）
2) 恐慌换仓 → b_panic_engine.run（L1、L2，含次日+滑点）
3) 闲置现金买沪深300 → d_idle_engine.run（C1 lag0 留20%、C3 lag1 留30%）
4) 移动止盈出场 → engine.run + v_c_exit_lib.patched（C1 a0.5/t0.1/sl0.7）
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from engine import Data, Config, run as e_run
import b_panic_engine as BP
import d_idle_engine as DI
import v_c_exit_lib as VX
import g_combined_engine as G

D = Data.load()
SEEDS = range(80)
WINS = G.WIN
SW = dict(swap_policy="oldest", swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)


def cmp(name, fa, fb):
    t = time.time()
    worst = 0.0
    for w, (st, en) in WINS.items():
        for sd in SEEDS:
            a, b = fa(st, en, sd), fb(st, en, sd)
            worst = max(worst, abs(a["final"] - b["final"]) / a["final"], abs(a["max_dd"] - b["max_dd"]))
    print(f"{name:42s} max rel diff (final) / abs diff (dd) over 3 windows x 80 seeds: {worst:.2e}  ({time.time()-t:.1f}s)")
    return worst


ok = True
# 1) 全关
for nm, kw in [("baseline", {}), ("K1 cap1 partial", dict(cap=1.0, partial=True, panic_max_pos=None)),
               ("L1.1", dict(weights=(0.022, 0.088, 0.066), cap=0.99))]:
    ok &= cmp("engine vs G: " + nm,
              lambda st, en, sd, kw=kw: e_run(D, Config(start=st, end=en, **kw), sd),
              lambda st, en, sd, kw=kw: G.run(D, G.GConfig(start=st, end=en, no_lev=False, **kw), sd)) < 1e-9
# 2) 换仓
for nm, kw in [("L1 swap 2/8/4 cap.9", dict(weights=(0.02, 0.08, 0.04), cap=0.9)),
               ("L2 swap 3/12/4 cap1", dict(weights=(0.03, 0.12, 0.04), cap=1.0)),
               ("L2 lag1 slip.5%", dict(weights=(0.03, 0.12, 0.04), cap=1.0, lag=1, slip=0.005))]:
    ok &= cmp("b_panic vs G: " + nm,
              lambda st, en, sd, kw=kw: BP.run(D, BP.PConfig(start=st, end=en, **SW, **kw), sd),
              lambda st, en, sd, kw=kw: G.run(D, G.GConfig(start=st, end=en, **SW, **kw), sd)) < 1e-9
# 3) 指数
for nm, lag, res in [("idle C1 hs300 MA120 lag0 res.2", 0, 0.2), ("idle C3 hs300 MA120 lag1 res.3", 1, 0.3)]:
    rg = DI.reg_trend_h(D, "sh000300", ma=120, h=0.0, lag=lag)
    ok &= cmp("d_idle vs G: " + nm,
              lambda st, en, sd, rg=rg, res=res: DI.run(D, DI.IConfig(start=st, end=en, idx="sh000300", reserve=res, regime=rg), sd),
              lambda st, en, sd, rg=rg, res=res: G.run(D, G.GConfig(start=st, end=en, idx="sh000300", reserve=res, regime=rg, no_lev=False), sd)) < 1e-9
# 4) 出场
RULE = {"kind": "trail", "act": 0.5, "trail": 0.1, "sl": 0.7, "tp": None}
DX = VX.patched(D, RULE)
ok &= cmp("engine+patched exits vs G: trail a.5 t.1 sl.7",
          lambda st, en, sd: e_run(DX, Config(start=st, end=en), sd),
          lambda st, en, sd: G.run(D, G.GConfig(start=st, end=en, exit_rule=RULE, no_lev=False), sd)) < 1e-9
print("ALL EQUAL" if ok else "MISMATCH")

# 基线数字（80 种子中位）
for w in "BAF":
    st, en = WINS[w]
    m = G.run_seeds(D, G.GConfig(start=st, end=en))
    print(f"baseline {w}: {m['final']/1e4:.1f}万 {m['cagr']*100:.2f}% dd {m['max_dd']*100:.1f}% trades {m['n_trades']:.0f}")
# 每日再平衡口径（v_d_idle 的复核数字：C3 A 501.1 万、全期 1725 万）
rg = DI.reg_trend_h(D, "sh000300", ma=120, h=0.0, lag=1)
for w in "BAF":
    st, en = WINS[w]
    m = G.run_seeds(D, G.GConfig(start=st, end=en, idx="sh000300", reserve=0.3, regime=rg, rebal="daily"))
    print(f"idle C3 daily-rebal {w}: {m['final']/1e4:.1f}万 {m['cagr']*100:.2f}% dd {m['max_dd']*100:.1f}%")
