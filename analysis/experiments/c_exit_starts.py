"""起点敏感性：不同起始日到 2026-08-21（80 种子），看候选相对基线的年化差是否依赖起点。"""
import time
from c_exit_lib import *
from c_exit_final import FINAL
NAMES = ["base", "tr_a45_t05_sl40", "tr_a50_t10_sl70", "t2_a50_t075_x100_t30_sl70", "t2_a50_t075_x100_t35_sl40", "tr_a100_t30_noSL"]
STARTS = ["2008-01-01", "2010-01-01", "2012-01-01", "2014-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]

def _ev(job):
    n, st = job
    D = getD()
    m = run_seeds_legs(D, Config(start=st), FINAL[n], seeds=range(80))
    return dict(name=n, start=st, final=m["final"] / 1e4, cagr=m["cagr"] * 100, max_dd=m["max_dd"] * 100)

if __name__ == "__main__":
    t = time.time()
    D = getD()
    for n in NAMES:
        exits_for(D, FINAL[n], len(D.dates) - 1)
    from multiprocessing import Pool
    with Pool(3) as p:
        rows = p.map(_ev, [(n, s) for n in NAMES for s in STARTS])
    df = pd.DataFrame(rows)
    df.to_csv("c_exit_out/starts.csv", index=False)
    b = df[df.name == "base"].set_index("start")
    df["dC"] = df.cagr - df.start.map(b.cagr)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index="start", columns="name", values="dC").reindex(columns=NAMES).round(2).to_string())
    print(df.pivot_table(index="start", columns="name", values="max_dd").reindex(columns=NAMES).round(1).to_string())
    print(b[["final", "cagr", "max_dd"]].round(2))
    print("elapsed", time.time() - t)
