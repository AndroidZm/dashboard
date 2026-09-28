"""v_e_select 检查：波动率窗口长度扰动（20/40/50/60/80/120 根成交 K 线，独立重算）x 阈值，C1/C2/C3 三种仓位设定，80 种子，A/B/B去2024-02。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
t = time.time()
setts = {'C1': (1.0, 0.9), 'C2': (1.25, 0.9), 'C3': (1.5, 1.0)}
rows = []
for sname, (k, cap) in setts.items():
    b = {c: stats(cfg_of(None, k, cap), c[0], drop=c[1])['cagr'] for c in [('A', ()), ('B', ()), ('B', ('2024-02',))]}
    b0 = {c: stats(cfg_of(), c[0], drop=c[1])['cagr'] for c in b}
    for N in (20, 40, 50, 60, 80, 120):
        for thr in (0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44):
            r = dict(sett=sname, N=N, thr=thr, remA=int(((VI[N] < thr) & (D_win := (data().sig.signal_date.values < '2016-01-01'))).sum()), remB=int(((VI[N] < thr) & ~D_win).sum()))
            for c in b:
                m = stats(cfg_of(thr, k, cap, vol=VI[N]), c[0], drop=c[1])
                key = c[0] + ('x' if c[1] else '')
                r[key] = m['cagr'] - b0[c]
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/v_e_select_out/volwin80.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for s in setts:
    for col in ('A', 'B', 'Bx'):
        print(s, col, '(vs 2/8/6 baseline, pp)'); print(df[df.sett == s].pivot(index='N', columns='thr', values=col).round(2).to_string())
print('time', time.time() - t)
