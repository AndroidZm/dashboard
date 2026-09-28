"""检查 2：独立网格（原引擎，80 种子），wn×wa×cap（partial=True, 不限恐慌持仓, wp=0.06）+ 若干扰动，A/B/F 三窗口。
用于：跨窗口选择（两个方向、三档回撤约束）、C1 邻域是否平台。"""
from v_a_sizing_lib import *
from multiprocessing import Pool
import time

WN = [0.0, 0.01, 0.02, 0.03, 0.04]
WA = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
CAP = [0.9, 0.95, 1.0]
P = [dict(weights=(a, b, 0.06), cap=c, panic_max_pos=None, partial=True) for a in WN for b in WA for c in CAP]
# 不开 partial 的对照 + 基线
P += [dict(weights=(a, b, 0.06), cap=c, panic_max_pos=None, partial=False) for a in [0.02] for b in WA for c in CAP]
P += [BASE]

def key(p):
    return f"{p['weights'][0]:.2f}/{p['weights'][1]:.2f}/{p['weights'][2]:.2f} cap{p['cap']} pmp{p['panic_max_pos']} part{int(p['partial'])}"

def ev(j):
    p, w = j
    x = seeds_raw(cfg(p, w))
    o = summ(x)
    return dict(key=key(p), wn=p['weights'][0], wa=p['weights'][1], cap=p['cap'], partial=p['partial'], pmp=p['panic_max_pos'], win=w, **o)

if __name__ == "__main__":
    D(); t = time.time()
    jobs = [(p, w) for p in P for w in "ABF"]
    with Pool(3) as pool:
        rows = pool.map(ev, jobs, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv("v_a_sizing_out/grid.csv", index=False)
    print(len(jobs), "runs", time.time() - t)
