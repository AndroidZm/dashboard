"""i_nodip_: 同一天的信号里，哪些个股特征能区分“不跌破”和“跌破”。
按信号日分层（只用当天 >= 10 笔的日子），算分层 AUC 和置换 p 值；再列出大日子里两组的中位数对比。
输出 i_nodip_out/withinday.txt
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "i_nodip_out")

STOCK = ["ret1", "gap", "oc", "rng", "clv", "ret3", "ret5", "ret10", "ret20", "ret60", "ret120", "ret250",
         "dd20", "dd60", "dd250", "dd500", "dd_all", "bounce20", "bounce60", "bounce250", "bounce500",
         "days_since_low20", "days_since_low60", "days_since_low250", "days_since_high250", "days_since_high500",
         "ma5", "ma10", "ma20", "ma60", "ma120", "ma250", "ma20_slope5", "ma60_slope20",
         "vol10", "vol20", "vol60", "vol250", "maxdrop20", "maxup20", "up_days5", "up_days20", "streak_up", "amp20",
         "susp20", "bars_before_signal", "log_price", "is_st", "board_cyb", "board_sh", "board_kcb",
         "prior_signals_same_stock", "rel1", "rel5", "rel20", "rel60", "rel250", "stock_ret_since_cluster_start"]


def auc(x, y):
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan, 0
    return mannwhitneyu(x[y == 1], x[y == 0]).statistic / (n1 * n0), n1 * n0


def strat(df, col, label, key):
    num = den = 0.0
    for _, g in df.groupby(key):
        a, w = auc(g[col].values.astype(float), g[label].values)
        if np.isnan(a):
            continue
        num += a * w
        den += w
    return num / den if den else np.nan


def main():
    F = pd.read_csv(os.path.join(OUT, "features.csv"), dtype={"code": str, "code3": str})
    F = F[F.label_ok].copy()
    F["board_cyb"] = (F.board == "创业板").astype(int)
    F["board_sh"] = (F.board == "沪主板").astype(int)
    F["board_kcb"] = (F.board == "科创板").astype(int)
    big_days = F.groupby("signal_date").s.transform("size") >= 10
    B = F[big_days].copy()
    print("大日子：", B.groupby("signal_date").agg(n=("s", "size"), nodip=("nodip0", "sum")).T.to_string())
    rng = np.random.default_rng(0)
    rows = []
    for label in ("nodip0", "nodip3"):
        for c in STOCK:
            obs = strat(B, c, label, "signal_date")
            if np.isnan(obs):
                continue
            cnt = 0
            d = B[["signal_date", c, label]].copy()
            for _ in range(500):
                d[label] = d.groupby("signal_date")[label].transform(lambda v: rng.permutation(v.values))
                a = strat(d, c, label, "signal_date")
                if abs(a - 0.5) >= abs(obs - 0.5) - 1e-12:
                    cnt += 1
            # 只用 2024-02 的大日子 / 只用 2024-02 以外的大日子
            o24 = strat(B[B.ym == "2024-02"], c, label, "signal_date")
            oo = strat(B[B.ym != "2024-02"], c, label, "signal_date")
            rows.append(dict(label=label, feature=c, auc_day_strat=obs, p_perm=(cnt + 1) / 501,
                             auc_2402_days=o24, auc_other_days=oo,
                             med_pos=np.nanmedian(B.loc[B[label] == 1, c]), med_neg=np.nanmedian(B.loc[B[label] == 0, c])))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "withinday.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    with open(os.path.join(OUT, "withinday.txt"), "w") as fh:
        for label in ("nodip0", "nodip3"):
            sub = R[R.label == label].copy()
            sub["dev"] = (sub.auc_day_strat - 0.5).abs()
            s = f"\n===== {label}：按信号日分层的 AUC（大日子 n>=10，共 {B.signal_date.nunique()} 天 {len(B)} 笔）\n" + \
                sub.sort_values("dev", ascending=False).head(25).drop(columns="dev").round(3).to_string(index=False)
            print(s)
            fh.write(s + "\n")
        # 2024-02-20 单日对比
        for day in ("2024-02-20", "2024-02-19", "2022-05-06", "2022-05-12", "2012-01-12"):
            g = F[F.signal_date == day]
            cols = ["ret1", "oc", "clv", "ret5", "ret20", "bounce60", "dd250", "dd_all", "ma60", "vol60", "amp20",
                    "maxdrop20", "bars_before_signal", "log_price", "rel5", "rel20", "board_cyb", "is_st"]
            t = g.groupby("nodip0")[cols].median().T
            t.columns = [f"跌破(n={(g.nodip0 == 0).sum()})", f"不跌破(n={(g.nodip0 == 1).sum()})"]
            s = f"\n--- {day} 两组中位数\n" + t.round(3).to_string()
            print(s)
            fh.write(s + "\n")


if __name__ == "__main__":
    main()
