"""检查 4：整年剔除、不同起点/子区间、2008 年贡献、回撤发生时段、C1 相对 K1 的逐种子回撤比较。"""
from v_a_sizing_lib import *
from engine import Config as _C
import engine as E
NAMES = ["BASE", "K1", "C1", "C1b", "C2"]
def block(title, cfgs):
    print("====", title)
    for lab, w, kw in cfgs:
        m = kw.pop("mask", None); s, e = kw.pop("start", None), kw.pop("end", None)
        res = {}
        for n in NAMES:
            c = cfg(CANDS[n], w, mask=m, **kw)
            if s: c = replace(c, start=s)
            if e: c = replace(c, end=e)
            res[n] = seeds_raw(c)
        line = f"{lab:28s} base {res['BASE'].final.median()/1e4:6.1f}/{res['BASE'].cagr.median()*100:5.2f}/{res['BASE'].max_dd.median()*100:5.1f} |"
        for n in NAMES[1:]:
            d = np.median((res[n].cagr.values - res['BASE'].cagr.values) * 100)
            dk = np.median((res[n].cagr.values - res['K1'].cagr.values) * 100)
            line += f" {n} {d:+5.2f}pp(vsK1 {dk:+5.2f}) dd{res[n].max_dd.median()*100:5.1f} |"
        print(line)
block("整年剔除", [
    ("A -Y2008", "A", dict(mask=mask_out(("2008",)))),
    ("A -Y2008 -Y2012", "A", dict(mask=mask_out(("2008", "2012")))),
    ("F -Y2008 -Y2012", "F", dict(mask=mask_out(("2008", "2012")))),
    ("B -Y2018", "B", dict(mask=mask_out(("2018",)))),
    ("B -Y2022", "B", dict(mask=mask_out(("2022",)))),
    ("B -Y2024", "B", dict(mask=mask_out(("2024",)))),
    ("F -all5big+2008", "F", dict(mask=mask_out(tuple(BIG) + ("2008",)))),
])
block("不同起点/子区间", [
    ("2009-01-05~2016-01-28", "A", dict(start="2009-01-05")),
    ("2006-01-04~2011-06-30", "A", dict(end="2011-06-30")),
    ("2011-07-01~2016-01-28", "A", dict(start="2011-07-01")),
    ("2016-01-01~2020-12-31", "B", dict(end="2020-12-31")),
    ("2021-01-04~2026-08-21", "B", dict(start="2021-01-04")),
    ("2017-01-03~2026-08-21", "B", dict(start="2017-01-03")),
    ("2018-01-02~2026-08-21", "B", dict(start="2018-01-02")),
    ("2019-01-02~2026-08-21", "B", dict(start="2019-01-02")),
    ("2020-01-02~2026-08-21", "B", dict(start="2020-01-02")),
    ("2009-01-05~2026-08-21", "F", dict(start="2009-01-05")),
])
# 2008 贡献与回撤时段
Dd = D()
print("==== 2008 signals:", Dd.sig[Dd.sig.signal_date.str.startswith("2008")][["signal_date", "code", "tier", "n_signals_30d", "exit_kind", "ret_pct", "held_days"]].to_string())
for n in ["BASE", "K1", "C1", "C2"]:
    pnl08, amt08, ddd = [], [], []
    for sd in range(80):
        r = run(Dd, cfg(CANDS[n], "A", record=True), sd)
        tr = r["trades"]; x = tr[tr.buy_date.str.startswith("2008")]
        pnl08.append(x.pnl.sum()); amt08.append(x.amount.sum()); ddd.append(r["dd_date"][:7])
    rB = [run(Dd, cfg(CANDS[n], "B"), sd)["dd_date"][:7] for sd in range(80)]
    print(f"{n}: 2008 buys mean amount {np.mean(amt08)/1e4:.1f}万 pnl {np.mean(pnl08)/1e4:.1f}万 | A dd month {pd.Series(ddd).value_counts().head(3).to_dict()} | B dd month {pd.Series(rB).value_counts().head(3).to_dict()}")
# C1 vs K1 逐种子回撤
for w in "BF":
    c1 = seeds_raw(cfg(C1, w)); k1 = seeds_raw(cfg(K1, w))
    print(w, "C1 dd better than K1 in", ((c1.max_dd - k1.max_dd) > 0).mean(), "median diff", np.median((c1.max_dd - k1.max_dd) * 100))
