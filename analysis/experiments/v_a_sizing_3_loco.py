"""检查 3/4/5：剔 2024-02、逐个剔大簇、剔整年 2008、次日+0.5% 滑点、换一组种子（80..159、160..239）。
每项都给：相对同口径基线的中位年化差、配对差、胜率、回撤；以及 C1/C1b/C2 相对 K1（只提 cap）的增量。"""
from v_a_sizing_lib import *
from multiprocessing import Pool

SCEN = []
for w in "BAF":
    SCEN.append((w, "all", None, {}))
SCEN += [("B", "-2024-02", ("2024-02",), {}), ("F", "-2024-02", ("2024-02",), {})]
for m in BIG:
    w = "A" if m < "2016" else "B"
    SCEN += [(w, "-" + m, (m,), {}), ("F", "-" + m, (m,), {})]
SCEN += [("A", "-Y2008", ("2008",), {}), ("F", "-Y2008", ("2008",), {}),
         ("A", "-Y2008-2012-01", ("2008", "2012-01"), {}), ("F", "-Y2008-2012-01", ("2008", "2012-01"), {}),
         ("F", "-2008,2012-01,2022-05,2024-02", ("2008", "2012-01", "2022-05", "2024-02"), {}),
         ("B", "-2022-05,2024-02", ("2022-05", "2024-02"), {}),
         ("B", "-2018-02,2024-02", ("2018-02", "2024-02"), {})]
for w in "BAF":
    SCEN.append((w, "lag1+slip0.5%", None, dict(lag=1, slip=0.005)))
for w in "BAF":
    SCEN.append((w, "seeds80-159", None, {"_seeds": range(80, 160)}))
    SCEN.append((w, "seeds160-239", None, {"_seeds": range(160, 240)}))

NAMES = ["BASE", "K1", "C1", "C1b", "C2"]

def ev(sc):
    w, lab, drop, kw = sc
    kw = dict(kw)
    seeds = kw.pop("_seeds", range(80))
    m = mask_out(drop) if drop else None
    res = {n: seeds_raw(cfg(CANDS[n], w, mask=m, **kw), seeds) for n in NAMES}
    rows = []
    for n in NAMES:
        o = summ(res[n], res["BASE"])
        dk = (res[n].cagr.values - res["K1"].cagr.values) * 100
        rows.append(dict(win=w, scen=lab, name=n, final=o["final"], cagr=o["cagr"], dd=o["dd"], dd_p10=o["dd_p10"], dd_worst=o["dd_worst"],
                         cagr_p10=o["cagr_p10"], cagr_p90=o["cagr_p90"], d_base_med=o["d_medcagr"], d_base_paired=o["d_med"], pos=o["frac_pos"],
                         d_dd=o["d_dd"], d_K1_paired=np.median(dk), pos_vs_K1=(dk > 0).mean(), n=o["n"]))
    return rows

if __name__ == "__main__":
    D()
    with Pool(3) as p:
        out = p.map(ev, SCEN, chunksize=1)
    df = pd.DataFrame([r for rr in out for r in rr])
    df.to_csv("v_a_sizing_out/loco.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    df["k"] = df.win + ":" + df.scen
    order = list(dict.fromkeys(df.k))
    for col in ["d_base_paired", "d_base_med", "d_dd", "d_K1_paired", "dd"]:
        print("====", col)
        print(df.pivot_table(index="k", columns="name", values=col, sort=False).reindex(order).round(2).to_string())
    print("==== final/cagr/dd for lag1 & seeds")
    x = df[df.scen.str.contains("lag|seeds|all")]
    print(x[["k", "name", "final", "cagr", "dd", "dd_p10", "dd_worst", "cagr_p10", "cagr_p90", "pos"]].round(2).to_string())
