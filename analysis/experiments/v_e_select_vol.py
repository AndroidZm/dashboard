"""v_e_select: 独立重算信号日波动率（直接读价格 CSV，不用引擎缓存），与 e_select_rule.vol60 对比；另算 N 日窗口变体。"""
import sys, glob, time
sys.path.insert(0, '/home/user/dashboard/analysis'); sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from engine import Data
from e_select_rule import vol60
t0 = time.time()
D = Data.load()
sig = D.sig
px = pd.concat(pd.read_csv(f, dtype={'code': str}, usecols=['code', 'date', 'close'])
               for f in sorted(glob.glob('/home/user/dashboard/prices/signal_stocks_adj_daily_part*.csv.gz')))
px = px[px.date <= '2026-08-21'].sort_values(['code', 'date'])
g = {c: (d.date.values.astype(str), d.close.values) for c, d in px.groupby('code')}
S = len(sig)
out = {}
# 独立口径：信号日（含）之前最后 N+1 根实际成交 K 线的 N 个对数收益（按 K 线数，不按日历）
for N in (20, 40, 50, 60, 80, 120):
    v = np.full(S, np.nan)
    for s in range(S):
        dts, cl = g[sig.code.values[s]]
        j = np.searchsorted(dts, sig.signal_date.values[s], side='right')  # 第一根 > 信号日
        assert dts[j - 1] == sig.signal_date.values[s], (s, dts[j-1])
        seg = cl[max(j - N - 1, 0):j]
        lr = np.diff(np.log(seg))
        v[s] = lr.std() * np.sqrt(244)
    out[N] = v
theirs = vol60(D)
d = out[60] - theirs
print('indep bar-based vol60 vs theirs: max|diff|', np.nanmax(np.abs(d)), ' corr', np.corrcoef(out[60], theirs)[0, 1])
print('n signals where (theirs>=0.38) != (indep>=0.38):', int(((theirs >= 0.38) != (out[60] >= 0.38)).sum()))
print('n |diff|>0.01:', int((np.abs(d) > 0.01).sum()))
bad = np.nonzero(np.abs(d) > 0.02)[0]
print(pd.DataFrame({'code': sig.code.values[bad], 'date': sig.signal_date.values[bad], 'bars': sig.bars_before_signal.values[bad], 'theirs': theirs[bad], 'indep': out[60][bad]}).head(20).to_string())
np.save('/home/user/dashboard/analysis/experiments/v_e_select_out/vols_indep.npy', np.vstack([out[N] for N in (20, 40, 50, 60, 80, 120)]))
np.save('/home/user/dashboard/analysis/experiments/v_e_select_out/vol60_theirs.npy', theirs)
# 分布：各年、各窗口被剔除比例
win = np.where(sig.signal_date.values < '2016-01-01', 'A', 'B')
yr = sig.signal_date.str[:4].values
lo = theirs < 0.38
print('removed frac A', lo[win == 'A'].mean().round(3), lo[win == 'A'].sum(), '/', (win == 'A').sum(), ' B', lo[win == 'B'].mean().round(3), lo[win == 'B'].sum(), '/', (win == 'B').sum())
print(pd.DataFrame({'yr': yr, 'lo': lo, 'tier': D.tier}).groupby('yr').agg(n=('lo', 'size'), removed=('lo', 'sum')).T.to_string())
print(pd.crosstab(D.tier, lo))
print('time', time.time() - t0)
