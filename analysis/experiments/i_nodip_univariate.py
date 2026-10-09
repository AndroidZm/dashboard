"""i_nodip_: 单变量检验。每个特征算三种 AUC：
  全样本 AUC（含择时效应）、按月分层 AUC（同一个月内比较，剔掉择时/簇效应）、A/B 窗口各自的 AUC，
  以及按月分层的置换检验 p 值。输出 i_nodip_out/univariate.csv。
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "i_nodip_out")

EXCLUDE = {"s", "code", "name", "signal_date", "tier", "ym", "board", "code3", "exit_kind", "ret_pct", "held_bars",
           "mfe_pct", "bars_to_plus20", "mae_close", "mae_low", "min_close_5", "min_close_10", "min_close_20",
           "min_close_60", "nodip0", "nodip3", "nodip5", "nodip_low", "tp", "fast_tp60", "label_ok", "window"}


def auc(x, y):
    """AUC = P(x_pos > x_neg)，NaN 自动剔除；类别不全返回 nan。"""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, n1, n0
    u = mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided").statistic
    return u / (n1 * n0), n1, n0


def strat_auc(df, col, label, min_n=8):
    """按月分层：每个月内 AUC，用 n1*n0 加权（等价于分层 Mann-Whitney）。"""
    num = den = 0.0
    months = []
    for ym, g in df.groupby("ym"):
        if len(g) < min_n:
            continue
        a, n1, n0 = auc(g[col].values.astype(float), g[label].values)
        if np.isnan(a):
            continue
        w = n1 * n0
        num += a * w
        den += w
        months.append(ym)
    return (num / den if den else np.nan), len(months)


def perm_p(df, col, label, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs, k = strat_auc(df, col, label)
    if np.isnan(obs):
        return np.nan
    d = df[["ym", col, label]].copy()
    cnt = 0
    for _ in range(n_perm):
        d[label] = d.groupby("ym")[label].transform(lambda v: rng.permutation(v.values))
        a, _ = strat_auc(d, col, label)
        if abs(a - 0.5) >= abs(obs - 0.5) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main():
    F = pd.read_csv(os.path.join(OUT, "features.csv"), dtype={"code": str, "code3": str})
    F = F[F.label_ok].copy()
    F["board_cyb"] = (F.board == "创业板").astype(int)
    F["board_sh"] = (F.board == "沪主板").astype(int)
    F["board_kcb"] = (F.board == "科创板").astype(int)
    feats = [c for c in F.columns if c not in EXCLUDE and F[c].dtype != object]
    rows = []
    for label in ("nodip0", "nodip3"):
        for c in feats:
            x = F[c].values.astype(float)
            a_all, _, _ = auc(x, F[label].values)
            a_A, n1A, _ = auc(x[F.window.values == "A"], F[label].values[F.window.values == "A"])
            a_B, n1B, _ = auc(x[F.window.values == "B"], F[label].values[F.window.values == "B"])
            a_s, k = strat_auc(F, c, label)
            a_s_no2402, _ = strat_auc(F[F.ym != "2024-02"], c, label)
            a_2402, _, _ = auc(x[F.ym.values == "2024-02"], F[label].values[F.ym.values == "2024-02"])
            a_2205, _, _ = auc(x[F.ym.values == "2022-05"], F[label].values[F.ym.values == "2022-05"])
            rows.append(dict(label=label, feature=c, auc_all=a_all, auc_A=a_A, auc_B=a_B, auc_strat=a_s, n_months=k,
                             auc_strat_no2402=a_s_no2402, auc_2402=a_2402, auc_2205=a_2205,
                             med_pos=np.nanmedian(x[F[label].values == 1]), med_neg=np.nanmedian(x[F[label].values == 0])))
    R = pd.DataFrame(rows)
    # 置换 p 值只算分层 AUC 偏离 0.5 最大的前 25 个特征（每个标签）
    R["p_strat"] = np.nan
    for label in ("nodip0", "nodip3"):
        sub = R[R.label == label].copy()
        sub["dev"] = (sub.auc_strat - 0.5).abs()
        top = sub.sort_values("dev", ascending=False).head(25)
        for i, r in top.iterrows():
            R.loc[i, "p_strat"] = perm_p(F, r.feature, label)
    R.to_csv(os.path.join(OUT, "univariate.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for label in ("nodip0", "nodip3"):
        sub = R[R.label == label].copy()
        sub["dev"] = (sub.auc_strat - 0.5).abs()
        print(f"\n===== {label}: 按月分层 AUC 偏离 0.5 最大的特征（AUC>0.5 表示该特征越大越不跌破）")
        print(sub.sort_values("dev", ascending=False).head(30).drop(columns="dev").round(3).to_string(index=False))
        sub["dev_all"] = (sub.auc_all - 0.5).abs()
        print(f"\n===== {label}: 全样本 AUC 最大的特征（含择时）")
        print(sub.sort_values("dev_all", ascending=False).head(15).drop(columns=["dev", "dev_all"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
