"""v_e_select 检查 2：thr x k x cap 全网格（80 种子，A/B），双向选参样本外、邻域平台；vol 窗口长度扰动。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
t = time.time()
rows = []
thrs = [None, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]
for cap in (0.9, 0.95, 1.0):
    for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        for thr in thrs:
            r = dict(cap=cap, k=k, thr=thr if thr else 0.0)
            for w in ('A', 'B', 'F'):
                m = stats(cfg_of(thr, k, cap), w)
                r[w] = m['cagr']; r[w + 'dd'] = m['dd']; r[w + 'fin'] = m['final']
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/v_e_select_out/grid80.csv', index=False)
print('time', time.time() - t)
