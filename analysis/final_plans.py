"""最终推荐方案对比表（REPORT.md 里的数字由本脚本生成）。

运行: python analysis/final_plans.py   （约 2~3 分钟）
输出: analysis/final_plans.csv 以及终端表格。
口径: 80 个执行顺序种子的中位数；"实盘" = 信号次日收盘成交 + 每边 0.5% 滑点。
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments"))
import g_combined_engine as G  # noqa: E402  换仓逻辑；换仓关闭时与 engine.run 逐种子一致

SWAP = dict(swap_policy="oldest", swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
PLANS = {
    "0 基线 2/8/6 上限90%": G.GConfig(),
    "1 稳健: 单笔×1.1 上限99%": G.GConfig(weights=(0.022, 0.088, 0.066), cap=0.99),
    "2 进取: ×1.1 + 恐慌换仓": G.GConfig(weights=(0.022, 0.088, 0.044), cap=0.99, **SWAP),
    "3 利润最大: ×1.5 融资(8%)": G.GConfig(weights=(0.03, 0.12, 0.09), cap=1.35, borrow_rate=0.08, no_lev=False),
}
WINDOWS = {
    "B 2016-2026": dict(start="2016-01-01"),
    "A 2006-2016": dict(start="2006-01-04", end="2016-01-28"),
    "全期 2006-2026": dict(start="2006-01-04"),
}


def job(args):
    plan, win, real, no2402 = args
    D = G.Data.load()
    cfg = replace(PLANS[plan], **WINDOWS[win])
    if real:
        cfg = replace(cfg, lag=1, slip=0.005)
    if no2402:
        cfg = replace(cfg, mask=~D.sig.signal_date.str.startswith("2024-02").values)
    m = G.run_seeds(D, cfg)
    return dict(plan=plan, window=win + (" 去2024-02" if no2402 else ""), exec="实盘" if real else "当日",
                final_wan=m["final"] / 1e4, cagr=m["cagr"] * 100, max_dd=m["max_dd"] * 100, dd_p10=m["dd_p10"] * 100)


if __name__ == "__main__":
    jobs = [(p, w, r, False) for p in PLANS for w in WINDOWS for r in (False, True)]
    jobs += [(p, "B 2016-2026", r, True) for p in PLANS for r in (False, True)]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(job, jobs))
    df = pd.DataFrame(rows).round(2)
    df.to_csv(os.path.join(HERE, "final_plans.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    for e in ("实盘", "当日"):
        print(f"\n== {e}（期末万元 / 年化% / 中位最大回撤% / 回撤p10%）")
        d = df[df.exec == e]
        d = d.assign(v=d.final_wan.map("{:.1f}".format) + " / " + d.cagr.map("{:.2f}".format) + " / "
                     + d.max_dd.map("{:.1f}".format) + " / " + d.dd_p10.map("{:.1f}".format))
        print(d.pivot(index="plan", columns="window", values="v").to_string())
