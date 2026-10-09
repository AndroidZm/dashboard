"""i_nodip_: 单条阈值规则的样本外检验 + 组合层面的收益影响。

1) 规则搜索：对每个特征、每个分位阈值、两个方向，算“选中的信号里不跌破率 / 止盈率 / 平均收益”，
   在一个切分上选、在另一个切分上验：B→A、A→B、2024-02→其它、其它→2024-02。
2) 组合回测：把规则当成过滤器（只买通过的信号），用 engine 跑 80 种子，对比基线和“同月随机剔除同样数量”的安慰剂。
输出 i_nodip_out/rules.txt、rules_search.csv、rules_portfolio.csv
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Config, Data, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "i_nodip_out")
from i_nodip_model import STOCK, TIMING  # noqa: E402


def search(F, label, train_mask, test_mask, min_sel=15):
    rows = []
    base_tr = F.loc[train_mask, label].mean()
    base_te = F.loc[test_mask, label].mean()
    for c in STOCK + TIMING:
        x = F[c].values.astype(float)
        qs = np.nanquantile(x[train_mask], np.linspace(0.1, 0.9, 9))
        for q in np.unique(qs):
            for direction in (">=", "<="):
                sel = (x >= q) if direction == ">=" else (x <= q)
                sel &= np.isfinite(x)
                tr, te = sel & train_mask, sel & test_mask
                if tr.sum() < min_sel or te.sum() < min_sel:
                    continue
                rows.append(dict(feature=c, dir=direction, thr=q,
                                 n_tr=int(tr.sum()), cov_tr=tr.sum() / train_mask.sum(), rate_tr=F.loc[tr, label].mean(), lift_tr=F.loc[tr, label].mean() - base_tr,
                                 n_te=int(te.sum()), cov_te=te.sum() / test_mask.sum(), rate_te=F.loc[te, label].mean(), lift_te=F.loc[te, label].mean() - base_te,
                                 tp_te=F.loc[te, "tp"].mean(), tp_te_rest=F.loc[test_mask & ~sel, "tp"].mean(),
                                 ret_te=F.loc[te, "ret_pct"].mean(), ret_te_rest=F.loc[test_mask & ~sel, "ret_pct"].mean()))
    R = pd.DataFrame(rows)
    return R


def portfolio(D, masks, windows):
    rows = []
    for mname, m in masks.items():
        for wname, wd in windows.items():
            for real in (False, True):
                ex = dict(lag=1, slip=0.005) if real else {}
                r = run_seeds(D, Config(**wd, mask=m, **ex))
                rows.append(dict(rule=mname, window=wname, exec="实盘" if real else "当日", n_sig=int(m[(D.dates[D.entry] >= wd["start"]) & (D.dates[D.entry] <= wd.get("end", "2026-12-31"))].sum()),
                                 final_wan=r["final"] / 1e4, cagr=r["cagr"] * 100, max_dd=r["max_dd"] * 100, n_trades=r["n_trades"], avg_exp=r["avg_exposure"] * 100))
    return pd.DataFrame(rows)


def main():
    D = Data.load()
    F = pd.read_csv(os.path.join(OUT, "features.csv"), dtype={"code": str, "code3": str})
    F["board_cyb"] = (F.board == "创业板").astype(int)
    F["board_sh"] = (F.board == "沪主板").astype(int)
    F["board_kcb"] = (F.board == "科创板").astype(int)
    ok = F.label_ok.values
    A = (F.window.values == "A") & ok
    B = (F.window.values == "B") & ok
    c2402 = (F.ym.values == "2024-02") & ok
    other = (F.ym.values != "2024-02") & ok
    pd.set_option("display.width", 300)
    fh = open(os.path.join(OUT, "rules.txt"), "w")

    def emit(s):
        print(s)
        fh.write(s + "\n")

    allR = []
    for label in ("nodip0", "nodip3"):
        for name, (tr, te) in {"B→A": (B, A), "A→B": (A, B), "2024-02→其它": (c2402, other), "其它→2024-02": (other, c2402)}.items():
            R = search(F, label, tr, te)
            R["label"], R["split"] = label, name
            allR.append(R)
            top = R[R.lift_tr > 0.05].sort_values("lift_tr", ascending=False).drop_duplicates("feature").head(12)
            emit(f"\n===== {label} | 在 {name.split('→')[0]} 上选 lift 最大的规则（每特征一条），到 {name.split('→')[1]} 上的表现（基准率 训练 {F.loc[tr, label].mean():.3f} / 测试 {F.loc[te, label].mean():.3f}）")
            emit(top[["feature", "dir", "thr", "n_tr", "cov_tr", "rate_tr", "lift_tr", "n_te", "cov_te", "rate_te", "lift_te", "tp_te", "tp_te_rest", "ret_te", "ret_te_rest"]].round(3).to_string(index=False))
    pd.concat(allR).to_csv(os.path.join(OUT, "rules_search.csv"), index=False, encoding="utf-8-sig")

    # 规则在四个切分上都为正的（样本外 lift > 0），按最小样本外 lift 排序
    R = pd.concat(allR)
    R = R[R.label == "nodip0"]
    key = ["feature", "dir", "thr"]
    g = R.groupby(key).agg(n_splits=("split", "nunique"), min_lift_te=("lift_te", "min"), mean_lift_te=("lift_te", "mean"), min_lift_tr=("lift_tr", "min"))
    g = g[(g.n_splits == 4) & (g.min_lift_tr > 0)].sort_values("min_lift_te", ascending=False)
    emit("\n===== nodip0：在四个切分上训练 lift 都 > 0 的规则，按样本外最小 lift 排序（前 25）")
    emit(g.head(25).round(3).to_string())

    # ---- 组合层面：几条有代表性的规则当过滤器 ----
    masks = {"基线(不过滤)": np.ones(len(F), bool)}
    cands = {
        "择时: 小盘指数5日涨幅>=6%": F.sm_ret5.values >= 0.06,
        "择时: 小盘指数距20日低点反弹>=10%": F.sm_bounce20.values >= 0.10,
        "择时: 指数低点后第4-8天": (F.sm_days_since_low20.values >= 4) & (F.sm_days_since_low20.values <= 8),
        "个股: 信号日涨幅>=3%": F.ret1.values >= 0.03,
        "个股: 创业板": F.board_cyb.values == 1,
        "个股: 60日波动率>=50%": F.vol60.values >= 0.5,
        "个股: 距250日高点跌幅>=50%": F.dd250.values <= -0.5,
    }
    masks.update({k: v & np.isfinite(v.astype(float)) for k, v in cands.items()})
    windows = {"B 2016-2026": dict(start="2016-01-01"), "A 2006-2016": dict(start="2006-01-04", end="2016-01-28"), "全期": dict(start="2006-01-04")}
    P = portfolio(D, masks, windows)
    # 安慰剂：同月随机剔除同样数量（10 次平均），只对 B 窗口当日口径
    rng = np.random.default_rng(1)
    plc = []
    for k, m in cands.items():
        vals = []
        for _ in range(10):
            pm = np.zeros(len(F), bool)
            for ym, idx in F.groupby("ym").indices.items():
                kk = int(m[idx].sum())
                if kk:
                    pm[rng.choice(idx, kk, replace=False)] = True
            r = run_seeds(D, Config(start="2016-01-01", mask=pm), seeds=range(20))
            vals.append(r["cagr"] * 100)
        plc.append(dict(rule=k, placebo_cagr_B=np.mean(vals)))
    P = P.merge(pd.DataFrame(plc), on="rule", how="left")
    P.to_csv(os.path.join(OUT, "rules_portfolio.csv"), index=False, encoding="utf-8-sig")
    emit("\n===== 组合回测：规则当过滤器（80 种子中位；安慰剂 = 同月随机剔除同样数量，20 种子×10 次，B 当日）")
    for e in ("当日", "实盘"):
        d = P[P.exec == e].assign(v=lambda d: d.final_wan.map("{:.0f}".format) + "万/" + d.cagr.map("{:.2f}".format) + "%/" + d.max_dd.map("{:.1f}".format) + "%/n" + d.n_trades.astype(int).astype(str))
        emit(f"\n-- {e}（期末/年化/回撤/成交笔数）")
        emit(d.pivot(index="rule", columns="window", values="v").join(P[P.exec == "当日"].drop_duplicates("rule").set_index("rule").placebo_cagr_B.round(2)).to_string())
    fh.close()


if __name__ == "__main__":
    main()
