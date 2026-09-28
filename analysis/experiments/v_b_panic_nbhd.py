"""复核 3：每个候选参数 ±1 格邻域 + 卖出对象安慰剂(random) + 并列顺序；80 种子，A/B/F/B去2024-02。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from multiprocessing import Pool
import pandas as pd
from v_b_panic_engine import Data, VConfig, run_seeds, WA, WB, WF, masks

D = Data.load(); M = masks(D)
WINS = (("A", WA, None), ("B", WB, None), ("Bx", WB, M["no2024-02"]), ("F", WF, None))
SW = dict(panic_max_pos=None, swap=True, max_n=5, keep_panic=True, sell_key="oldest")
CENTER = {"L1": dict(weights=(0.02, 0.08, 0.04), cap=0.9), "L1b": dict(weights=(0.03, 0.12, 0.04), cap=0.9),
          "L2": dict(weights=(0.03, 0.12, 0.04), cap=1.0), "L3": dict(weights=(0.03, 0.16, 0.04), cap=1.0)}
jobs = [("base", "-", VConfig())]
for n in ("cap1_noswap",):
    jobs.append((n, "-", VConfig(cap=1.0)))
for L, c in CENTER.items():
    wn, wa, w2 = c["weights"]
    jobs.append((L, "center", VConfig(**c, **SW)))
    jobs.append((L, "noswap(w2=.06,pmp15)", VConfig(weights=(wn, wa, 0.06), cap=c["cap"])))
    for x in (0.02, 0.03, 0.05, 0.06, 0.08):
        jobs.append((L, f"w2={x}", VConfig(weights=(wn, wa, x), cap=c["cap"], **SW)))
    for x in (1, 2, 3, 4, 6, 8, 99):
        jobs.append((L, f"max_n={x}", VConfig(**c, **{**SW, "max_n": x})))
    jobs.append((L, "keep_panic=F", VConfig(**c, **{**SW, "keep_panic": False})))
    for k in ("newest", "underwater", "random"):
        jobs.append((L, f"sell={k}", VConfig(**c, **{**SW, "sell_key": k})))
    for k in ("sidx", "rev"):
        jobs.append((L, f"tie={k}", VConfig(**c, **{**SW, "tie": k})))
    for d in (-0.01, 0.01) if L.startswith("L1") or True else ():
        if wn + d > 0:
            jobs.append((L, f"wn={wn+d:.2f}", VConfig(weights=(wn + d, wa, w2), cap=c["cap"], **SW)))
    for d in (-0.02, 0.02, 0.04):
        jobs.append((L, f"wa={wa+d:.2f}", VConfig(weights=(wn, wa + d, w2), cap=c["cap"], **SW)))
    for cc in (0.85, 0.95, 1.0) if c["cap"] == 0.9 else (0.9, 0.95):
        jobs.append((L, f"cap={cc}", VConfig(weights=c["weights"], cap=cc, **SW)))
    if c["cap"] == 1.0:
        jobs.append((L, "no_lev=strict", VConfig(**c, **{**SW}, no_lev="strict")))
        jobs.append((L, "no_lev=False(engine)", VConfig(**c, **{**SW}, no_lev=False)))


def ev(job):
    L, var, cfg = job
    row = dict(cand=L, var=var)
    for tag, (st, en), mk in WINS:
        m = run_seeds(D, replace(cfg, start=st, end=en, mask=mk))
        row[tag] = m["cagr"] * 100; row[tag + "_dd"] = m["max_dd"] * 100
        if tag == "B":
            row["B_p10"] = m["cagr_p10"] * 100; row["B_wdd"] = m["dd_worst"] * 100; row["B_sw"] = m["n_swap"]
    return row


if __name__ == "__main__":
    with Pool(3) as p:
        rows = p.map(ev, jobs)
    df = pd.DataFrame(rows)
    b = df.iloc[0]
    for t in ("A", "B", "Bx", "F"):
        df["d" + t] = df[t] - b[t]
    df.to_csv("/tmp/claude-0/-home-user-dashboard/165eccdd-32ad-5f72-8bde-c1779d405297/scratchpad/v_nbhd.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(df[["cand", "var", "A", "dA", "A_dd", "B", "dB", "B_dd", "B_wdd", "B_sw", "Bx", "dBx", "F", "dF", "F_dd"]].round(2).to_string(index=False))
