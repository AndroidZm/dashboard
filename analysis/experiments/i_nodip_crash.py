"""i_nodip_: 规则搜索里反复出现的“跌得深 / 波动大”这一族特征，用固定阈值做复核：
按月分层 lift（同一个月内选中 vs 未选中的不跌破率之差）+ 置换 p 值；分窗口、分簇；对止盈率/收益/持有期的影响；
以及当过滤器用在组合里的结果（含同月随机剔除的安慰剂、放大单笔的变体）。
输出 i_nodip_out/crash.txt、crash_rules.csv、crash_portfolio.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Config, Data, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "i_nodip_out")

RULES = {
    "ma120<=-0.25": lambda F: F.ma120 <= -0.25,
    "ma120<=-0.30": lambda F: F.ma120 <= -0.30,
    "ma120<=-0.35": lambda F: F.ma120 <= -0.35,
    "dd250<=-0.45": lambda F: F.dd250 <= -0.45,
    "dd250<=-0.50": lambda F: F.dd250 <= -0.50,
    "dd250<=-0.55": lambda F: F.dd250 <= -0.55,
    "dd250<=-0.60": lambda F: F.dd250 <= -0.60,
    "ret60<=-0.30": lambda F: F.ret60 <= -0.30,
    "ret60<=-0.35": lambda F: F.ret60 <= -0.35,
    "dd_all<=-0.80": lambda F: F.dd_all <= -0.80,
    "vol250>=0.50": lambda F: F.vol250 >= 0.50,
    "vol250>=0.55": lambda F: F.vol250 >= 0.55,
    "vol60>=0.50": lambda F: F.vol60 >= 0.50,
    "vol60>=0.60": lambda F: F.vol60 >= 0.60,
    "创业板": lambda F: F.board == "创业板",
    "ret1>=0.03": lambda F: F.ret1 >= 0.03,
    "ret1<=0.005": lambda F: F.ret1 <= 0.005,
    "ma120<=-0.30 且 vol250>=0.50": lambda F: (F.ma120 <= -0.30) & (F.vol250 >= 0.50),
    "dd250<=-0.50 且 创业板": lambda F: (F.dd250 <= -0.50) & (F.board == "创业板"),
}


def strat_lift(F, sel, label, min_n=8):
    num = den = 0.0
    for ym, g in F.groupby("ym"):
        if len(g) < min_n:
            continue
        s = sel[g.index]
        n1, n0 = s.sum(), (~s).sum()
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0 / len(g)
        num += (g.loc[s, label].mean() - g.loc[~s, label].mean()) * w
        den += w
    return num / den if den else np.nan


def perm_p(F, sel, label, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs = strat_lift(F, sel, label)
    if np.isnan(obs):
        return np.nan
    cnt = 0
    lab = F[label].values.copy()
    G = F.copy()
    for _ in range(n):
        for ym, idx in F.groupby("ym").indices.items():
            lab[idx] = rng.permutation(F[label].values[idx])
        G[label] = lab
        if abs(strat_lift(G, sel, label)) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1)


def main():
    D = Data.load()
    F0 = pd.read_csv(os.path.join(OUT, "features.csv"), dtype={"code": str, "code3": str})
    F = F0[F0.label_ok].reset_index(drop=True)
    fh = open(os.path.join(OUT, "crash.txt"), "w")

    def emit(s):
        print(s)
        fh.write(s + "\n")

    rows = []
    for name, fn in RULES.items():
        sel = fn(F).fillna(False).values.astype(bool)
        r = dict(rule=name, n_sel=int(sel.sum()), cov=sel.mean())
        for tag, m in {"全部": np.ones(len(F), bool), "A": F.window.values == "A", "B": F.window.values == "B",
                       "去2024-02": F.ym.values != "2024-02", "2024-02": F.ym.values == "2024-02", "2022-05": F.ym.values == "2022-05",
                       "B去两大簇": (F.window.values == "B") & ~F.ym.isin(["2024-02", "2022-05"]).values}.items():
            a, b = F.nodip0.values[m & sel], F.nodip0.values[m & ~sel]
            r[f"{tag}:选中率"] = a.mean() if len(a) else np.nan
            r[f"{tag}:其余率"] = b.mean() if len(b) else np.nan
            r[f"{tag}:n选中"] = int(len(a))
        r["分层lift"] = strat_lift(F, sel, "nodip0")
        r["分层lift_去2402"] = strat_lift(F[F.ym != "2024-02"].reset_index(drop=True), sel[F.ym.values != "2024-02"], "nodip0")
        r["p_perm"] = perm_p(F, sel, "nodip0")
        r["分层lift_nodip3"] = strat_lift(F, sel, "nodip3")
        # 对交易结果的影响（全部）
        r["止盈率:选中"] = F.tp.values[sel].mean()
        r["止盈率:其余"] = F.tp.values[~sel].mean()
        r["收益:选中"] = F.ret_pct.values[sel].mean()
        r["收益:其余"] = F.ret_pct.values[~sel].mean()
        r["持有中位:选中"] = np.median(F.held_bars.values[sel])
        r["持有中位:其余"] = np.median(F.held_bars.values[~sel])
        r["分层lift_止盈"] = strat_lift(F, sel, "tp")
        r["分层lift_快止盈60"] = strat_lift(F, sel, "fast_tp60")
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "crash_rules.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    cols1 = ["rule", "n_sel", "cov", "全部:选中率", "全部:其余率", "A:选中率", "A:其余率", "A:n选中", "B:选中率", "B:其余率", "去2024-02:选中率", "去2024-02:其余率",
             "2024-02:选中率", "2024-02:其余率", "2022-05:选中率", "2022-05:其余率", "B去两大簇:选中率", "B去两大簇:其余率", "B去两大簇:n选中"]
    emit("===== 固定阈值规则：不跌破率（选中 vs 其余）")
    emit(R[cols1].round(3).to_string(index=False))
    cols2 = ["rule", "分层lift", "p_perm", "分层lift_去2402", "分层lift_nodip3", "止盈率:选中", "止盈率:其余", "收益:选中", "收益:其余", "持有中位:选中", "持有中位:其余", "分层lift_止盈", "分层lift_快止盈60"]
    emit("\n===== 按月分层 lift（同月内选中-未选中）、置换 p、对止盈/收益/持有期的影响")
    emit(R[cols2].round(3).to_string(index=False))

    # ---- 组合过滤 ----
    windows = {"B": dict(start="2016-01-01"), "A": dict(start="2006-01-04", end="2016-01-28"), "F": dict(start="2006-01-04")}
    tests = {"基线": None, "dd250<=-0.50": RULES["dd250<=-0.50"], "ma120<=-0.30": RULES["ma120<=-0.30"], "vol250>=0.50": RULES["vol250>=0.50"]}
    rng = np.random.default_rng(2)
    prow = []
    for name, fn in tests.items():
        m = np.ones(len(F0), bool) if fn is None else fn(F0).fillna(False).values.astype(bool)
        for wn, wd in windows.items():
            for variant, extra in {"过滤": {}, "过滤+单笔×1.5": dict(weights=(0.03, 0.12, 0.09)), "过滤+单笔×2": dict(weights=(0.04, 0.16, 0.12))}.items():
                if fn is None and variant != "过滤":
                    continue
                r = run_seeds(D, Config(**wd, mask=m, **extra))
                rr = run_seeds(D, Config(**wd, mask=m, lag=1, slip=0.005, **extra))
                row = dict(rule=name, variant=variant, window=wn, cagr=r["cagr"] * 100, max_dd=r["max_dd"] * 100, n_trades=r["n_trades"], final_wan=r["final"] / 1e4,
                           cagr_real=rr["cagr"] * 100, dd_real=rr["max_dd"] * 100)
                if fn is not None:
                    vals = []
                    for _ in range(8):
                        pm = np.zeros(len(F0), bool)
                        for ym, idx in F0.groupby("ym").indices.items():
                            kk = int(m[idx].sum())
                            if kk:
                                pm[rng.choice(idx, kk, replace=False)] = True
                        vals.append(run_seeds(D, Config(**wd, mask=pm, **extra), seeds=range(20))["cagr"] * 100)
                    row["placebo_cagr"] = float(np.mean(vals))
                    row["placebo_sd"] = float(np.std(vals))
                prow.append(row)
    P = pd.DataFrame(prow)
    P.to_csv(os.path.join(OUT, "crash_portfolio.csv"), index=False, encoding="utf-8-sig")
    emit("\n===== 组合：过滤器 / 过滤+放大单笔（80 种子中位，安慰剂=同月随机剔除同样数量 8 次×20 种子）")
    emit(P.round(2).to_string(index=False))
    fh.close()


if __name__ == "__main__":
    main()
