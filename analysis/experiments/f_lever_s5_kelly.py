"""s5: Kelly 估计 —— 逐笔 Kelly vs 按簇 Kelly vs 连续时间 Kelly，以及簇内相关性、年块自助法。

(a) 逐笔 Kelly：把 663 笔（或窗口内）当成相互独立、依次下注的赌局，max E[log(1+f r)]
(b) 按月簇 Kelly：同一信号月的所有信号等权当成“一注”
(c) 连续时间 Kelly：基线路径（80 种子中位那条附近，用 seed 13）的日超额收益 μ/σ²，给出 L*（相对基线的放大倍数）
    以及持仓账户（只看持仓部分）的 μ_b/σ_b² —— 持仓暴露的 Kelly 倍数
(d) 簇内相关：同时持有的信号股日收益平均两两相关系数、等效独立注数
(e) 年块自助：按自然年重抽基线日收益，L* 的分布
(f) 压力：小盘指数历史最大跌幅与“触发 130% 维持担保比例所需持仓跌幅”
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import WIN, Data, LConfig, cfg_win, run  # noqa: E402

D = Data.load()
FEE = 0.001
rng = np.random.default_rng(0)


def kelly_discrete(r, fmax=30):
    """max_f mean log(1+f r)（f 可 >1），网格 + 细化。"""
    r = np.asarray(r)
    lo = -1 / r.max() + 1e-9 if r.max() > 0 else 0
    hi = min(fmax, 1 / (-r.min()) - 1e-9) if r.min() < 0 else fmax
    fs = np.linspace(0, hi, 20001)
    g = np.array([np.mean(np.log1p(f * r)) for f in fs[::20]])
    i = int(np.argmax(g)) * 20
    fs2 = fs[max(0, i - 40): i + 41]
    g2 = np.array([np.mean(np.log1p(f * r)) for f in fs2])
    j = int(np.argmax(g2))
    return fs2[j], g2[j], hi


print("=" * 20, "(a) 逐笔 Kelly")
t_end = len(D.dates) - 1
e, x, ratio = D.exits(0.4, 0.4, None, t_end, 0)
r_all = ratio * (1 - FEE) / (1 + FEE) - 1
held_years = (D.cal_days[x] - D.cal_days[e]) / 365.25
sd = D.sig.signal_date.values
for name, m in [("全部 663", np.ones(len(sd), bool)),
                ("窗口 A 2006-2016", (sd >= "2006-01-04") & (sd <= "2016-01-28")),
                ("窗口 B 2016-2026", sd >= "2016-01-01"),
                ("剔除 2024-02", ~pd.Series(sd).str.startswith("2024-02").values)]:
    r = r_all[m]
    f, g, hi = kelly_discrete(r)
    p = (r > 0.3).mean()
    q = (r < -0.3).mean()
    fb = (p - q) / 0.4 if p + q > 0 else np.nan
    print(f"{name}: n={m.sum()} 均值 {r.mean()*100:.1f}% 胜率(+40) {p*100:.1f}% 败率(-40) {q*100:.1f}% "
          f"持有中位 {np.median(held_years[m]):.2f} 年 | 逐笔 Kelly f*={f:.2f} (每笔下注 {f*100:.0f}% 权益), "
          f"二元近似 (p-q)/0.4={fb:.2f}, 每笔期望对数增长 {g:.3f}")

print("=" * 20, "(b) 按月簇 Kelly（同月信号等权 = 一注）")
ym = pd.Series(sd).str[:7].values
dfc = pd.DataFrame({"ym": ym, "r": r_all, "hy": held_years})
cl = dfc.groupby("ym").agg(r=("r", "mean"), n=("r", "size"), hy=("hy", "median")).reset_index()
for name, c in [("全部月", cl), ("B 窗口月", cl[cl.ym >= "2016-01"]), ("A 窗口月", cl[cl.ym < "2016-02"])]:
    f, g, hi = kelly_discrete(c.r.values)
    print(f"{name}: 簇数 {len(c)} 簇均收益均值 {c.r.mean()*100:.1f}% 最差簇 {c.r.min()*100:.1f}% "
          f"| 按簇 Kelly f*={f:.2f}（每个信号月押 {f*100:.0f}% 权益）")
# 按信号数加权的有效簇数
w = cl.n / cl.n.sum()
print("有效簇数 1/HHI =", round(1 / (w ** 2).sum(), 2))

# 各大簇等权篮子的逐日盯市路径最差点（持有到各自出场，出场后按现金）
print("--- 大簇等权篮子：盯市最低点 / 最终")
for mo in ["2012-01", "2018-02", "2018-10", "2022-05", "2022-10", "2024-02", "2008-09", "2012-12", "2018-07"]:
    idx = np.nonzero(ym == mo)[0]
    t0 = e[idx].min()
    t1 = x[idx].max()
    vals = []
    for s in idx:
        path = np.ones(t1 - t0 + 1)
        a, b = e[s] - t0, x[s] - t0
        path[a: b + 1] = D.close[s, e[s]: x[s] + 1] / D.close[s, e[s]]
        path[b + 1:] = path[b]
        vals.append(path)
    v = np.mean(vals, axis=0)
    print(f"{mo}: n={len(idx)} 篮子（各自入场日起算，未入场部分按 1）最低 {(v.min()-1)*100:.1f}% (第 {v.argmin()} 个交易日) 最终 {(v[-1]-1)*100:.1f}%")

print("=" * 20, "(c) 连续时间 Kelly（基线路径 seed 13）")
out_c = {}
for win in "ABF":
    r = run(D, cfg_win(win, record=True), 13)
    eq = r["equity"].values
    ret = np.diff(eq) / eq[:-1]
    days = D.cal_days[D.day(WIN[win][0]): D.day_le(WIN[win][1]) + 1]
    dt = np.diff(days) / 365.0
    rc = (1.018 ** dt) - 1
    ex = ret - rc
    mu = ex.mean() * 244
    var = ex.var() * 244
    Lstar = mu / var
    # 持仓账户日收益
    tr = r["trades"]
    T0 = D.day(WIN[win][0])
    Tn = len(eq)
    pnl = np.zeros(Tn)
    base = np.zeros(Tn)
    for s, bt, st, amt in tr[["s", "buy_t", "sell_t", "amount"]].itertuples(index=False):
        u = amt / D.close[s, bt]
        c = D.close[s, bt: st + 1]
        pnl[bt - T0 + 1: st - T0 + 1] += u * np.diff(c)
        base[bt - T0 + 1: st - T0 + 1] += u * c[:-1]
    ok = base > 0
    rb = pnl[ok] / base[ok]
    mub = rb.mean() * 244
    varb = rb.var() * 244
    out_c[win] = (mu, var, Lstar)
    # 月/季收益口径（包含回撤的持续性/自相关）
    ser = pd.Series(eq, index=pd.to_datetime(D.dates[T0: T0 + Tn]))
    for fr, nper in [("ME", 12), ("QE", 4)]:
        z = ser.resample(fr).last().pct_change().dropna().values - (1.018 ** (1 / nper) - 1)
        print(f"{win} 窗口 {fr} 收益口径: μ={z.mean()*nper*100:.2f}% σ={z.std()*np.sqrt(nper)*100:.1f}% → L*={z.mean()*nper/(z.var()*nper):.2f}")
    print(f"{win}: 策略日超额年化 μ={mu*100:.2f}% σ={np.sqrt(var)*100:.1f}% → L*=μ/σ²={Lstar:.2f}（相对基线的放大倍数）"
          f"| 持仓账户 μ_b={mub*100:.1f}% σ_b={np.sqrt(varb)*100:.1f}% → 持仓暴露 Kelly={(mub-0.05)/varb:.2f}~{(mub-0.08)/varb:.2f} 倍权益"
          f"（融资 5%~8%），持仓天数占比 {ok.mean()*100:.0f}%")
    # (e) 年块自助
    yrs = pd.Series(D.dates[T0 + 1: T0 + Tn]).str[:4].values
    uy = np.unique(yrs)
    groups = [ex[yrs == y] for y in uy]
    Ls = []
    for _ in range(4000):
        pick = rng.integers(0, len(uy), len(uy))
        z = np.concatenate([groups[i] for i in pick])
        Ls.append(z.mean() / z.var())
    Ls = np.array(Ls)
    loo = [np.concatenate([groups[j] for j in range(len(uy)) if j != i]) for i in range(len(uy))]
    loo_L = [z.mean() / z.var() for z in loo]
    print(f"   年块自助 L* 分位: p5 {np.percentile(Ls,5):.2f} p10 {np.percentile(Ls,10):.2f} p25 {np.percentile(Ls,25):.2f} "
          f"p50 {np.median(Ls):.2f} p75 {np.percentile(Ls,75):.2f}; P(L*<1)={np.mean(Ls<1)*100:.0f}%  P(L*<0)={np.mean(Ls<0)*100:.0f}% "
          f"| 逐年剔除 L* 最小 {min(loo_L):.2f}（剔 {uy[int(np.argmin(loo_L))]}）最大 {max(loo_L):.2f}（剔 {uy[int(np.argmax(loo_L))]}）")

print("=" * 20, "(d) 同时持仓信号股的日收益相关性")
ret_mat = np.diff(np.log(D.close), axis=1)  # (S,T-1)，停牌前向填充为 0 收益
for mo in ["2024-02", "2022-05", "2018-02", "2012-01", "2022-10"]:
    idx = np.nonzero(ym == mo)[0]
    t0 = e[idx].max()
    R = ret_mat[idx, t0: t0 + 120]
    C = np.corrcoef(R)
    n = len(idx)
    rho = (C.sum() - n) / (n * (n - 1))
    n_eff = n / (1 + (n - 1) * rho)
    print(f"{mo}: n={n} 入场后 120 个交易日日收益平均两两相关 ρ={rho:.2f} → 等效独立注数 n/(1+(n-1)ρ)={n_eff:.1f}；"
          f"15 只同时持有的等效独立注数={15/(1+14*rho):.1f}")
# 与国证2000 的 beta
ix = np.log(D.index["sz399303"].ffill().values)
ixr = np.diff(ix)
print("=" * 20, "(f) 压力测试：小盘指数历史最大跌幅")
for col, nm in [("sz399303", "国证2000"), ("sh000852", "中证1000"), ("sh000905", "中证500")]:
    s = D.index[col].dropna()
    for a, b in [("2007-10-01", "2008-12-31"), ("2015-06-01", "2016-02-29"), ("2018-01-01", "2019-01-31"),
                 ("2024-01-01", "2024-02-29"), ("2005-01-01", "2026-08-21")]:
        z = s[(s.index >= a) & (s.index <= b)]
        if len(z) < 5:
            continue
        dd = (z / z.cummax() - 1)
        # 最快 20/60 交易日跌幅
        r20 = (z / z.shift(20) - 1).min()
        r60 = (z / z.shift(60) - 1).min()
        print(f"{nm} {a}~{b}: 最大回撤 {dd.min()*100:.1f}%（谷 {dd.idxmin()}） 最差 20 日 {r20*100:.1f}% 最差 60 日 {r60*100:.1f}%")
print("触发维持担保比例 130% 所需的持仓跌幅（满仓在 cap 时，负债 = cap-1 倍权益）：")
for cap in [1.1, 1.2, 1.35, 1.5, 1.8, 2.0, 2.25]:
    debt = cap - 1
    x_need = 1 - 1.3 * debt / cap
    print(f"  总仓位 {cap:.2f} 倍权益: 持仓跌 {x_need*100:.0f}% 触发 130% 平仓线；此时权益损失 {x_need*cap*100:.0f}%")
