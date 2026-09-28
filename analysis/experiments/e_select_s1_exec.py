"""e_select s1: 基线复现 + 执行成本（次日收盘 lag=1、滑点 0.5%）在 A/B/全期三个窗口的影响（80 种子中位）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis')
from engine import Data, Config, run_seeds
D = Data.load()
W = {'B': ('2016-01-01', '2026-08-21'), 'A': ('2006-01-04', '2016-01-28'), 'F': ('2006-01-04', '2026-08-21')}
def fmt(m):
    return f"final {m['final']/1e4:7.1f}万 CAGR {m['cagr']*100:5.2f}% DD {m['max_dd']*100:6.1f}% n {m['n_trades']:.0f} exp {m['avg_exposure']*100:4.1f}% tiers(n0,n1,n2) med"
for wn, (a, b) in W.items():
    for lag, slip in [(0, 0), (1, 0), (0, 0.005), (1, 0.005)]:
        m = run_seeds(D, Config(start=a, end=b, lag=lag, slip=slip))
        print(wn, f"lag={lag} slip={slip}", fmt(m), flush=True)
