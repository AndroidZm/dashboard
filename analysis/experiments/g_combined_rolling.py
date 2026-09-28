"""滚动起点 / 滚动 5 年窗口：检查各组合的增益是否只来自某一段行情（当日口径与实盘口径）。
输出 g_combined_out/rolling.csv"""
import os, sys
from dataclasses import replace
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd, numpy as np
import g_combined_engine as G
from g_combined_plans import make, D

CF = {"base": make(), "S": make(S=True), "E": make(E=True), "S+E": make(S=True, E=True), "S+C": make(S=True, C=True),
      "S+C+E": make(S=True, C=True, E=True), "S+C+W": make(S=True, C=True, W=True),
      "S+I": make(S=True, I=True), "S+C+I.3": make(S=True, C=True, I=True), "S+C+I0": make(S=True, C=True, I=True, res=0.0),
      "S+C+W+E+I0": make(S=True, C=True, W=True, E=True, I=True, res=0.0), "C": make(C=True), "I": make(I=True)}
STARTS = [f"{y}-01-01" for y in (2008, 2010, 2012, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)]
WIN5 = [(f"{y}-01-01", f"{y+5}-12-31") for y in range(2006, 2021)]


def job(a):
    nm, st, en, real = a
    c = replace(CF[nm], start=st, end=en)
    if real:
        c = replace(c, lag=1, slip=0.005, idx_fee=max(c.idx_fee, 0.002) if c.idx else c.idx_fee)
    m = G.run_seeds(D, c)
    return dict(cfg=nm, start=st, end=en, real=real, cagr=m["cagr"], dd=m["max_dd"])


if __name__ == "__main__":
    jobs = [(n, s, G.END_DATE, r) for n in CF for s in STARTS for r in (False, True)]
    jobs += [(n, s, e, r) for n in CF for s, e in WIN5 for r in (False, True)]
    with Pool(3) as p:
        rows = p.map(job, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "g_combined_out", "rolling.csv"), index=False)
    pd.set_option("display.width", 250)
    for kind, sub in (("滚动起点(到2026-08)", df[df.end == G.END_DATE]), ("滚动5年窗口", df[df.end != G.END_DATE])):
        for real in (False, True):
            s = sub[sub.real == real]
            pv = s.pivot(index="start", columns="cfg", values="cagr") * 100
            d = pv.sub(pv["base"], axis=0).drop(columns="base")
            print(f"\n== {kind} {'实盘口径' if real else '当日口径'}: 相对基线的年化差 pp（base 列为基线年化%）")
            d.insert(0, "base%", pv["base"])
            print(d.round(2).to_string())
            print("   为正比例:", ((d.drop(columns='base%') > 0).mean() * 100).round(0).to_dict())
