"""i_nodip_: 汇总成一个能手算的筛选规则，并检验它对“之后的信号”有没有用。

1) 导出“从没收盘跌破买入价”的信号名单 never_below_entry.csv
2) 不跌破组 vs 其余的特征对比表（全部 / 只看 2024-02 簇内 / 只看 2022-05 簇内）
3) 手算评分：深跌(ma120<=-30%) + 高波动(amp20>=6%) + 创业板，各 1 分；市场条件 = 小盘指数距 20 日低点反弹>=8% 且近 5 个交易日全表信号>=10
   按分数、按簇、按窗口看不跌破率 / 止盈率 / 收益 / 持有期
4) 用评分当“同日信号的买入优先级”（不减少交易，只改先买谁）跑组合回测；对照随机顺序、反向优先级、样本外模型得分
5) 给 2026 年的新信号打分
输出 i_nodip_out/screen.txt、screen_score.csv、screen_priority.csv、never_below_entry.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Config, Data, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "i_nodip_out")


def main():
    D = Data.load()
    F = pd.read_csv(os.path.join(OUT, "features.csv"), dtype={"code": str, "code3": str})
    F["创业板"] = (F.board == "创业板").astype(int)
    F["深跌"] = (F.ma120 <= -0.30).astype(int)
    F["高波动"] = (F.amp20 >= 0.06).astype(int)
    F["个股分"] = F["深跌"] + F["高波动"] + F["创业板"]
    F["市场条件"] = ((F.sm_bounce20 >= 0.08) & (F.n_prev5 >= 10)).astype(int)
    sc = pd.read_csv(os.path.join(OUT, "model_oos_scores.csv"), dtype={"code": str})
    F = F.merge(sc[["s", "nodip0_个股_logit", "nodip0_个股_gbm"]], on="s", how="left")
    fh = open(os.path.join(OUT, "screen.txt"), "w")
    pd.set_option("display.width", 300)

    def emit(s):
        print(s)
        fh.write(s + "\n")

    # ---- 1) 名单 ----
    cols = ["s", "code", "name", "signal_date", "tier", "exit_kind", "ret_pct", "held_bars", "mae_close", "mae_low", "board", "price_raw",
            "bars_before_signal", "ret1", "ret20", "ret60", "ma120", "dd250", "vol20", "amp20", "sm_bounce20", "sm_days_since_low20", "n_same_day", "n_prev5",
            "深跌", "高波动", "创业板", "个股分", "市场条件", "label_ok"]
    L = F[F.nodip0 == 1][cols].copy()
    L["mae_low"] = L.mae_low.round(4)
    L.rename(columns={"mae_close": "持有期最低收盘相对买价", "mae_low": "持有期最低价相对买价", "price_raw": "买入价(不复权)",
                      "ret1": "信号日涨幅", "ret20": "20日涨幅", "ret60": "60日涨幅", "ma120": "相对120日均线", "dd250": "距250日高点",
                      "vol20": "20日年化波动", "amp20": "20日均振幅", "sm_bounce20": "小盘指数距20日低点反弹", "sm_days_since_low20": "指数低点后天数",
                      "n_same_day": "同日信号数", "n_prev5": "前5日信号数", "label_ok": "结局已定"}).to_csv(
        os.path.join(OUT, "never_below_entry.csv"), index=False, encoding="utf-8-sig")
    emit(f"从没收盘跌破买入价：{int(F.nodip0.sum())} 笔（其中 {int((F.nodip0 == 1).sum() - F[F.nodip0 == 1].label_ok.sum())} 笔是 2026-07/08 的未平仓新信号，结局未定）；"
         f"盘中最低价也没跌破的只有 {int(F.nodip_low.sum())} 笔。")

    # ---- 2) 特征对比 ----
    G = F[F.label_ok].copy()
    feats = {"20日涨幅": "ret20", "60日涨幅": "ret60", "相对60日均线": "ma60", "相对120日均线": "ma120", "距250日高点": "dd250",
             "近20日最大单日跌幅": "maxdrop20", "10日年化波动": "vol10", "20日年化波动": "vol20", "20日均振幅": "amp20",
             "信号日涨幅": "ret1", "近3日涨幅": "ret3", "买入价(不复权)": "price_raw", "上市天数(交易日)": "bars_before_signal",
             "小盘指数距20日低点反弹": "sm_bounce20", "指数低点后天数": "sm_days_since_low20", "同日信号数": "n_same_day", "前5日信号数": "n_prev5"}
    shares = {"创业板占比": "创业板", "ST占比": "is_st", "深跌占比(ma120<=-30%)": "深跌", "高波动占比(amp20>=6%)": "高波动"}
    for tag, m in {"全部已定信号": np.ones(len(G), bool), "只看 2024-02 簇内": G.ym.values == "2024-02", "只看 2022-05 簇内": G.ym.values == "2022-05",
                   "两大簇以外": ~G.ym.isin(["2024-02", "2022-05"]).values}.items():
        g = G[m]
        t = pd.DataFrame({f"不跌破(n={int(g.nodip0.sum())})": [g.loc[g.nodip0 == 1, c].median() for c in feats.values()] + [g.loc[g.nodip0 == 1, c].mean() for c in shares.values()],
                          f"其余(n={int((g.nodip0 == 0).sum())})": [g.loc[g.nodip0 == 0, c].median() for c in feats.values()] + [g.loc[g.nodip0 == 0, c].mean() for c in shares.values()]},
                         index=list(feats) + list(shares))
        emit(f"\n===== 特征对比（中位数；占比为均值）：{tag}")
        emit(t.round(3).to_string())

    # ---- 3) 评分 ----
    def table(g, by):
        return g.groupby(by).agg(n=("s", "size"), 不跌破率=("nodip0", "mean"), 不跌破3pct率=("nodip3", "mean"), 止盈率=("tp", "mean"),
                                 平均收益=("ret_pct", "mean"), 持有中位=("held_bars", "median"), 快止盈60=("fast_tp60", "mean")).round(3)

    emit("\n===== 个股分（深跌+高波动+创业板）× 市场条件：全部已定信号")
    emit(table(G, ["市场条件", "个股分"]).to_string())
    emit("\n===== 个股分：分窗口")
    emit(table(G, ["window", "个股分"]).to_string())
    emit("\n===== 个股分：分大簇（≥20 笔的月份）")
    big = G[G.groupby("ym").s.transform("size") >= 20]
    emit(table(big, ["ym", "个股分"]).to_string())
    emit("\n===== 个股分=3 vs 0，逐簇不跌破率（≥10 笔的月份）")
    t = G[G.groupby("ym").s.transform("size") >= 10].groupby("ym").apply(
        lambda g: pd.Series({"n": len(g), "n_3分": int((g.个股分 == 3).sum()), "不跌破_3分": g.loc[g.个股分 == 3, "nodip0"].mean(),
                             "n_2分": int((g.个股分 == 2).sum()), "不跌破_2分": g.loc[g.个股分 == 2, "nodip0"].mean(),
                             "n_0-1分": int((g.个股分 <= 1).sum()), "不跌破_0-1分": g.loc[g.个股分 <= 1, "nodip0"].mean(),
                             "止盈_≥2分": g.loc[g.个股分 >= 2, "tp"].mean(), "止盈_0-1分": g.loc[g.个股分 <= 1, "tp"].mean(),
                             "持有_≥2分": g.loc[g.个股分 >= 2, "held_bars"].median(), "持有_0-1分": g.loc[g.个股分 <= 1, "held_bars"].median()}))
    emit(t.round(3).to_string())
    # 样本外模型得分的分位（按月分组交叉验证，每笔的得分来自没见过它那个月的模型）
    G["模型分位"] = pd.qcut(G["nodip0_个股_logit"].rank(method="first"), 5, labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"])
    emit("\n===== 样本外模型得分（个股特征 logit，按月分组交叉验证）五分位")
    emit(table(G, ["模型分位"]).to_string())
    emit(table(G[G.ym != "2024-02"], ["模型分位"]).rename(columns=lambda c: c).to_string() + "   <- 去掉 2024-02")
    F.to_csv(os.path.join(OUT, "screen_score.csv"), index=False, encoding="utf-8-sig")

    # ---- 4) 优先级回测 ----
    S = len(F)
    pri = {
        "随机顺序(基线)": None,
        "个股分高者先": F["个股分"].values.astype(float),
        "个股分低者先(反向)": -F["个股分"].values.astype(float),
        "跌得越深越先(ma120 低者先)": -F.ma120.values.astype(float),
        "样本外模型分高者先": F["nodip0_个股_logit"].values.astype(float),
        "样本外模型分低者先(反向)": -F["nodip0_个股_logit"].values.astype(float),
    }
    windows = {"B 2016-2026": dict(start="2016-01-01"), "A 2006-2016": dict(start="2006-01-04", end="2016-01-28"), "全期": dict(start="2006-01-04")}
    rows = []
    no2402 = ~F.signal_date.str.startswith("2024-02").values
    for pn, p in pri.items():
        for wn, wd in windows.items():
            for ex_name, ex in {"当日": {}, "实盘": dict(lag=1, slip=0.005)}.items():
                r = run_seeds(D, Config(**wd, priority=p, **ex))
                rows.append(dict(优先级=pn, window=wn, exec=ex_name, final_wan=r["final"] / 1e4, cagr=r["cagr"] * 100, max_dd=r["max_dd"] * 100, n_trades=r["n_trades"],
                                 cagr_p10=r["cagr_p10"] * 100, cagr_p90=r["cagr_p90"] * 100))
            if wn == "B 2016-2026":
                r = run_seeds(D, Config(**wd, priority=p, mask=no2402))
                rows.append(dict(优先级=pn, window="B 去2024-02", exec="当日", final_wan=r["final"] / 1e4, cagr=r["cagr"] * 100, max_dd=r["max_dd"] * 100, n_trades=r["n_trades"],
                                 cagr_p10=r["cagr_p10"] * 100, cagr_p90=r["cagr_p90"] * 100))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "screen_priority.csv"), index=False, encoding="utf-8-sig")
    emit("\n===== 评分只用来决定同日信号谁先买（规则其余不变，80 种子中位：期末万/年化%/回撤%/笔数）")
    for e in ("当日", "实盘"):
        d = P[P.exec == e].assign(v=lambda d: d.final_wan.map("{:.0f}".format) + "/" + d.cagr.map("{:.2f}".format) + "/" + d.max_dd.map("{:.1f}".format) + "/n" + d.n_trades.astype(int).astype(str))
        emit(f"\n-- {e}")
        emit(d.pivot(index="优先级", columns="window", values="v").reindex(list(pri)).to_string())

    # ---- 5) 2026 年新信号 ----
    N = F[F.signal_date >= "2026-01-01"][["code", "name", "signal_date", "tier", "exit_kind", "ret_pct", "held_bars", "mae_close", "ma120", "amp20", "board", "深跌", "高波动", "创业板", "个股分", "sm_bounce20", "n_prev5", "市场条件", "nodip0_个股_logit"]]
    emit("\n===== 2026 年的信号按同一规则打分（结局：到 2026-08-21 为止）")
    emit(N.round(3).to_string(index=False))
    fh.close()


if __name__ == "__main__":
    main()
