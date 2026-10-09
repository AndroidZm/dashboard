"""i_nodip_: 多变量模型能否在样本外识别“不跌破买入价”的信号。

三组特征：择时（指数状态 + 信号簇位置）、个股（信号日前的走势/波动/价位/板块）、两者合并。
验证方式：按月分组的 GroupKFold、留一大簇、A→B / B→A 时间切分。
评价：样本外 AUC；样本外按月分层 AUC（剔掉择时后还剩多少选股能力）；预测分前 20% 的实际不跌破率 / 止盈率 / 收益。
输出 i_nodip_out/model.txt、model_oos_scores.csv。
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "i_nodip_out")

TIMING = ["sm_ret1", "sm_ret3", "sm_ret5", "sm_ret10", "sm_ret20", "sm_ret60", "sm_ret120", "sm_ret250",
          "sm_dd20", "sm_dd60", "sm_dd250", "sm_bounce20", "sm_bounce60", "sm_bounce250",
          "sm_days_since_low20", "sm_days_since_low60", "sm_days_since_low250", "sm_days_since_high250",
          "sm_ma5", "sm_ma20", "sm_ma60", "sm_ma120", "sm_ma250", "sm_vol20", "sm_up_days5",
          "hs_ret1", "hs_ret5", "hs_ret20", "hs_ret60", "hs_ret250", "hs_dd250", "hs_bounce20", "hs_ma20", "hs_ma60",
          "sh_ret1", "sh_ret20", "sh_dd250",
          "n_signals_30d", "n_prev1", "n_prev5", "n_prev10", "n_prev30_excl", "n_prev60_bars",
          "days_since_prev_signal", "days_since_cluster_start", "sm_ret_since_cluster_start", "tier"]
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
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    return mannwhitneyu(x[y == 1], x[y == 0]).statistic / (y.sum() * (1 - y).sum())


def strat_auc(score, y, groups, min_n=8):
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_n:
            continue
        a = auc(score[m], y[m])
        if np.isnan(a):
            continue
        w = y[m].sum() * (1 - y[m]).sum()
        num += a * w
        den += w
    return num / den if den else np.nan


def models():
    return {
        "logit": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced")),
        "gbm": HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=0),
    }


def oos_scores(F, feats, label, scheme):
    """返回每笔信号的样本外得分（按 scheme 划分）。"""
    X = F[feats].values.astype(float)
    y = F[label].values
    out = {}
    for mname in ("logit", "gbm"):
        sc = np.full(len(F), np.nan)
        if scheme == "gkf":
            splits = GroupKFold(n_splits=5).split(X, y, groups=F.ym.values)
        elif scheme == "loco":  # 留一大簇（其余月份合为一折）
            big = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]
            rest = ~F.ym.isin(big).values
            splits = [(np.nonzero(F.ym.values != b)[0], np.nonzero(F.ym.values == b)[0]) for b in big]
            splits.append((np.nonzero(~rest)[0], np.nonzero(rest)[0]))
        elif scheme == "A2B":
            splits = [(np.nonzero(F.window.values == "A")[0], np.nonzero(F.window.values == "B")[0])]
        elif scheme == "B2A":
            splits = [(np.nonzero(F.window.values == "B")[0], np.nonzero(F.window.values == "A")[0])]
        for tr, te in splits:
            if y[tr].sum() < 3:
                continue
            m = models()[mname]
            m.fit(X[tr], y[tr])
            sc[te] = m.predict_proba(X[te])[:, 1]
        out[mname] = sc
    return out


def evaluate(F, sc, label, tag):
    y = F[label].values
    m = np.isfinite(sc)
    r = dict(tag=tag, n=int(m.sum()), auc=auc(sc[m], y[m]), auc_strat_month=strat_auc(sc[m], y[m], F.ym.values[m]),
             auc_no2402=auc(sc[m & (F.ym.values != "2024-02")], y[m & (F.ym.values != "2024-02")]),
             auc_A=auc(sc[m & (F.window.values == "A")], y[m & (F.window.values == "A")]),
             auc_B=auc(sc[m & (F.window.values == "B")], y[m & (F.window.values == "B")]))
    # 前 20% 的实际表现 vs 其余
    q = np.nanquantile(sc[m], 0.8)
    top = m & (sc >= q)
    rest = m & (sc < q)
    r.update(base_rate=y[m].mean(), top20_rate=y[top].mean(), rest_rate=y[rest].mean(),
             top20_tp=F.tp.values[top].mean(), rest_tp=F.tp.values[rest].mean(),
             top20_ret=F.ret_pct.values[top].mean(), rest_ret=F.ret_pct.values[rest].mean(),
             top20_held=np.median(F.held_bars.values[top]), rest_held=np.median(F.held_bars.values[rest]),
             top20_share_2402=(F.ym.values[top] == "2024-02").mean())
    # 分层：前 20% 按月内分位
    F2 = F.loc[m].copy()
    F2["sc"] = sc[m]
    F2["rk"] = F2.groupby("ym").sc.rank(pct=True)
    big = F2[F2.groupby("ym").s.transform("size") >= 10]
    r["top20_within_month_rate"] = big.loc[big.rk >= 0.8, label].mean()
    r["rest_within_month_rate"] = big.loc[big.rk < 0.8, label].mean()
    return r


def main():
    F = pd.read_csv(os.path.join(OUT, "features.csv"), dtype={"code": str, "code3": str})
    F = F[F.label_ok].reset_index(drop=True)
    F["board_cyb"] = (F.board == "创业板").astype(int)
    F["board_sh"] = (F.board == "沪主板").astype(int)
    F["board_kcb"] = (F.board == "科创板").astype(int)
    sets = {"择时": TIMING, "个股": STOCK, "合并": TIMING + STOCK}
    rows = []
    scores = F[["s", "code", "name", "signal_date", "ym", "window", "nodip0", "nodip3", "tp", "ret_pct", "held_bars"]].copy()
    for label in ("nodip0", "nodip3"):
        for sname, feats in sets.items():
            for scheme in ("gkf", "loco", "A2B", "B2A"):
                res = oos_scores(F, feats, label, scheme)
                for mname, sc in res.items():
                    rows.append(evaluate(F, sc, label, f"{label}|{sname}|{scheme}|{mname}"))
                    if scheme == "gkf":
                        scores[f"{label}_{sname}_{mname}"] = sc
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "model.csv"), index=False, encoding="utf-8-sig")
    scores.to_csv(os.path.join(OUT, "model_oos_scores.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    with open(os.path.join(OUT, "model.txt"), "w") as fh:
        for label in ("nodip0", "nodip3"):
            sub = R[R.tag.str.startswith(label)]
            s = f"\n===== 标签 {label}：样本外表现\n" + sub.round(3).to_string(index=False)
            print(s)
            fh.write(s + "\n")
    # 合并模型（logit, gkf）的系数：全样本拟合，看哪些特征在起作用
    for label in ("nodip0",):
        X = F[TIMING + STOCK].values.astype(float)
        m = models()["logit"].fit(X, F[label].values)
        coef = pd.Series(m[-1].coef_[0], index=TIMING + STOCK).sort_values()
        s = f"\n===== 全样本 logit 系数（标准化后），{label}\n负向前 15：\n{coef.head(15).round(3).to_string()}\n正向前 15：\n{coef.tail(15).round(3).to_string()}"
        print(s)
        with open(os.path.join(OUT, "model.txt"), "a") as fh:
            fh.write(s + "\n")


if __name__ == "__main__":
    main()
