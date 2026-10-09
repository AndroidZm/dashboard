"""i_nodip_: 优先级路径是确定性的（没有同分，80 个种子结果相同），所以要和“随机顺序的分布”比：
算每条优先级路径在 80 个随机顺序里的百分位（年化、回撤），以及它在簇里实际换进/换出了哪些股票。
输出 i_nodip_out/priority_pct.txt
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Config, Data, run  # noqa: E402

OUT = os.path.join(HERE, "i_nodip_out")


def main():
    D = Data.load()
    F = pd.read_csv(os.path.join(OUT, "screen_score.csv"), dtype={"code": str, "code3": str})
    f = lambda c: F[c].values.astype(float)  # noqa: E731
    pri = {"ma120 低者先": -f("ma120"), "vol20 高者先": f("vol20"), "样本外模型分高者先": f("nodip0_个股_logit"),
           "ma120+vol20 (两者标准化相加)": (-(f("ma120") - np.nanmean(f("ma120"))) / np.nanstd(f("ma120")) + (f("vol20") - np.nanmean(f("vol20"))) / np.nanstd(f("vol20"))),
           "ma120 高者先(反向)": f("ma120")}
    ym = F.ym.values
    cases = {"A": dict(start="2006-01-04", end="2016-01-28"), "B": dict(start="2016-01-01"), "F": dict(start="2006-01-04"),
             "B1 2016-2020": dict(start="2016-01-01", end="2020-12-31"), "B2 2021-2026": dict(start="2021-01-01"),
             "A 实盘": dict(start="2006-01-04", end="2016-01-28", lag=1, slip=0.005), "B 实盘": dict(start="2016-01-01", lag=1, slip=0.005),
             "F 实盘": dict(start="2006-01-04", lag=1, slip=0.005),
             "B+稳健方案": dict(start="2016-01-01", weights=(0.022, 0.088, 0.066), cap=0.99),
             "F+稳健方案": dict(start="2006-01-04", weights=(0.022, 0.088, 0.066), cap=0.99)}
    for m in ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]:
        cases[f"F去{m}"] = dict(start="2006-01-04", mask=ym != m)
    rows = []
    fh = open(os.path.join(OUT, "priority_pct.txt"), "w")

    def emit(s):
        print(s)
        fh.write(s + "\n")

    for cn, cd in cases.items():
        base = [run(D, Config(**cd), s) for s in range(80)]
        bc = np.array([b["cagr"] for b in base]) * 100
        bd = np.array([b["max_dd"] for b in base]) * 100
        for pn, p in pri.items():
            r = run(D, Config(**cd, priority=p), 0)
            rows.append(dict(case=cn, 优先级=pn, cagr=r["cagr"] * 100, cagr_pct=(bc < r["cagr"] * 100).mean() * 100, base_med=np.median(bc),
                             dd=r["max_dd"] * 100, dd_pct=(bd < r["max_dd"] * 100).mean() * 100, base_dd_med=np.median(bd)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "priority_pct.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    emit("===== 优先级路径年化在 80 个随机顺序中的百分位（越高越好；50 = 和随机中位一样）")
    emit(R.pivot(index="优先级", columns="case", values="cagr_pct").reindex(list(pri))[list(cases)].round(0).to_string())
    emit("\n===== 年化：优先级路径 vs 随机中位")
    emit((R.assign(v=lambda d: d.cagr.map("{:.2f}".format) + " vs " + d.base_med.map("{:.2f}".format))).pivot(index="优先级", columns="case", values="v").reindex(list(pri))[list(cases)].to_string())
    emit("\n===== 最大回撤在随机顺序中的百分位（越高 = 回撤越浅于更多随机路径）")
    emit(R.pivot(index="优先级", columns="case", values="dd_pct").reindex(list(pri))[list(cases)].round(0).to_string())

    # ---- 簇里实际买了什么 ----
    sig = D.sig
    for cn, cd in {"B": dict(start="2016-01-01")}.items():
        rp = run(D, Config(**cd, priority=pri["ma120 低者先"], record=True), 0)
        tp = rp["trades"].assign(path="ma120优先")
        tb = pd.concat([run(D, Config(**cd, record=True), s)["trades"].assign(seed=s) for s in range(80)])
        for m in ["2018-02", "2022-05", "2022-10", "2024-02"]:
            a = tp[tp.buy_date.str.startswith(m)]
            b = tb[tb.buy_date.str.startswith(m)]
            emit(f"\n===== {m}：ma120 优先路径买入的股票（n={len(a)}）vs 随机顺序 80 条路径（平均每条 {len(b) / 80:.1f} 笔）")
            a = a.merge(F[["s", "ma120", "vol20", "nodip0", "held_bars", "ret_pct"]], on="s", how="left")
            emit(a[["buy_date", "code", "tier", "amount", "sell_date", "kind", "pnl", "ma120", "vol20", "nodip0", "held_bars", "ret_pct"]].assign(
                name=sig.name.values[a.s]).round(3).to_string(index=False))
            freq = b.groupby("code").agg(次数=("seed", "nunique"), 平均pnl=("pnl", "mean"), 持有=("sell_t", lambda x: np.nan))
            bb = b.merge(F[["code", "signal_date", "ma120", "nodip0", "held_bars", "ret_pct"]].assign(buy_date=lambda d: d.signal_date), on=["code", "buy_date"], how="left")
            emit(f"随机路径里被买到的股票：{bb.code.nunique()} 只；它们的 ma120 中位 {bb.ma120.median():.3f}，不跌破率 {bb.nodip0.mean():.2f}，持有中位 {bb.held_bars.median():.0f}，平均收益 {bb.ret_pct.mean():.1f}%，平均 pnl {bb.pnl.mean():.0f}")
            emit(f"ma120 优先路径：ma120 中位 {a.ma120.median():.3f}，不跌破率 {a.nodip0.mean():.2f}，持有中位 {a.held_bars.median():.0f}，平均收益 {a.ret_pct.mean():.1f}%，平均 pnl {a.pnl.mean():.0f}")
    fh.close()


if __name__ == "__main__":
    main()
