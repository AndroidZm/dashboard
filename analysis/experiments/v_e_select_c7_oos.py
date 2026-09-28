"""v_e_select 检查：在 grid80.csv 上做双向选参（原始 argmax 与 3x3 邻域平滑 argmax），三个回撤档，报告对侧窗口结果。"""
import numpy as np, pandas as pd
df = pd.read_csv('/home/user/dashboard/analysis/experiments/v_e_select_out/grid80.csv')
thrs = sorted(df.thr.unique()); ks = sorted(df.k.unique())
def smooth(sub, col):
    p = sub.pivot(index='k', columns='thr', values=col)
    # 只在有筛选的阈值之间平滑（thr=0 表示不筛选，单独处理）
    q = p.copy()
    for i, k in enumerate(p.index):
        for j, t in enumerate(p.columns):
            ii = [x for x in (i - 1, i, i + 1) if 0 <= x < len(p.index)]
            jj = [x for x in (j - 1, j, j + 1) if 0 <= x < len(p.columns)] if t > 0 else [j]
            jj = [x for x in jj if (p.columns[x] > 0) == (t > 0)]
            q.iloc[i, j] = p.iloc[ii, jj].values.mean()
    return q.stack().rename(col + '_s').reset_index()
fams = {'C1族 (k=1,cap0.9)': df[(df.k == 1.0) & (df.cap == 0.9)], 'C2族 (cap0.9, thr x k)': df[df.cap == 0.9], 'C3族 (cap<=1.0, thr x k x cap)': df}
for fname, sub in fams.items():
    print('=====', fname)
    parts = []
    for cap, g in sub.groupby('cap'):
        g = g.copy()
        for col in ('A', 'B'):
            s = smooth(g, col); g = g.merge(s, on=['k', 'thr'])
        parts.append(g)
    g = pd.concat(parts).reset_index(drop=True)
    for src, dst in (('A', 'B'), ('B', 'A')):
        for lim in (-30, -40, -100):
            ok = g[g[src + 'dd'] >= lim]
            for how in ('raw', 's'):
                col = src if how == 'raw' else src + '_s'
                r = ok.loc[ok[col].idxmax()]
                print(f"选于{src} DD>={lim:4d} {how:3s}: thr {r.thr:.2f} k {r.k:.2f} cap {r.cap:.2f} | {src} {r[src]:.2f} ({r[src+'dd']:.1f}) -> {dst} {r[dst]:.2f} ({r[dst+'dd']:.1f})  F {r.F:.2f} ({r.Fdd:.1f})")
