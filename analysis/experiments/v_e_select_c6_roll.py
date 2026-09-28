"""v_e_select 检查：起点/终点滚动（子区间）、恐慌持仓上限与总仓位上限变化下的增益，80 种子。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
D = data()
t = time.time()
C = {'C1': cfg_of(0.38), 'C2': cfg_of(0.38, 1.25), 'C3': cfg_of(0.40, 1.5, 1.0)}
base = cfg_of()
def dd(c, a, b):
    m1 = stats(replace(c, start=a, end=b), 'F'); return m1
periods = [(f'{y}-01-01', '2026-08-21') for y in range(2016, 2025)] + [('2016-01-01', f'{y}-12-31') for y in (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)] + \
          [(f'{y}-01-01', '2016-01-28') for y in (2006, 2008, 2010, 2011, 2012, 2013)] + [('2006-01-04', f'{y}-12-31') for y in (2009, 2011, 2012, 2013, 2014)] + \
          [(f'{y}-01-01', f'{y+2}-12-31') for y in range(2016, 2024)]
rows = []
for a, b in periods:
    # stats() 用 WIN[win]，这里直接改 start/end
    import v_e_select_lib as L
    L.WIN['X'] = (a, b)
    m0 = stats(base, 'X')
    r = dict(start=a, end=b, base=m0['cagr'], base_dd=m0['dd'])
    for n, c in C.items():
        m = stats(c, 'X'); r[n] = m['cagr'] - m0['cagr']; r[n + '_dd'] = m['dd']
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(df.round(2).to_string(index=False))
# 机制旋钮：恐慌持仓上限 / 总仓位上限
rows = []
for pmp in (10, 15, 20, 30, None):
    for cap in (0.8, 0.9):
        r = dict(pmp=pmp, cap=cap)
        for w in ('A', 'B'):
            m0 = stats(replace(base, panic_max_pos=pmp, cap=cap), w); m1 = stats(replace(C['C1'], panic_max_pos=pmp, cap=cap), w)
            r[w + '_base'] = m0['cagr']; r[w + '_C1'] = m1['cagr'] - m0['cagr']; r[w + '_pan_base'] = m0['npanic']; r[w + '_pan_C1'] = m1['npanic']
        rows.append(r)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print('time', time.time() - t)
