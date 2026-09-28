"""v_e_select 检查 3+4：去 2024-02、逐个去大簇、去全部大簇；次日成交+0.5%滑点。每个候选对基线、对“同 k/cap 不筛选”两种参照。80 种子。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
t = time.time()
C = {'C1': (cfg_of(0.38), cfg_of()), 'C2': (cfg_of(0.38, 1.25), cfg_of(None, 1.25)), 'C3': (cfg_of(0.40, 1.5, 1.0), cfg_of(None, 1.5, 1.0))}
base = cfg_of()
cases = [('B', ()), ('B', ('2024-02',)), ('B', ('2018-02',)), ('B', ('2022-05',)), ('B', ('2022-10',)), ('B', ('2018-02', '2022-05', '2022-10', '2024-02')),
         ('A', ()), ('A', ('2012-01',)),
         ('F', ())] + [('F', (m,)) for m in BIG] + [('F', tuple(BIG))]
rows = []
for w, drop in cases:
    m0 = stats(base, w, drop=drop)
    for name, (c, ctl) in C.items():
        m1 = stats(c, w, drop=drop); mc = stats(ctl, w, drop=drop)
        rows.append(dict(cand=name, win=w, drop='+'.join(drop) or '-', cagr=m1['cagr'], dd=m1['dd'], base=m0['cagr'], base_dd=m0['dd'],
                         d_base=m1['cagr'] - m0['cagr'], d_same_size=m1['cagr'] - mc['cagr'], ctl=mc['cagr']))
# 执行成本
for w in ('A', 'B', 'F'):
    m0 = stats(replace(base, lag=1, slip=0.005), w)
    for name, (c, ctl) in C.items():
        m1 = stats(replace(c, lag=1, slip=0.005), w); mc = stats(replace(ctl, lag=1, slip=0.005), w)
        rows.append(dict(cand=name, win=w, drop='lag1+slip', cagr=m1['cagr'], dd=m1['dd'], base=m0['cagr'], base_dd=m0['dd'],
                         d_base=m1['cagr'] - m0['cagr'], d_same_size=m1['cagr'] - mc['cagr'], ctl=mc['cagr']))
    for slip in (0.01,):
        m0 = stats(replace(base, lag=1, slip=slip), w)
        for name, (c, ctl) in C.items():
            m1 = stats(replace(c, lag=1, slip=slip), w)
            rows.append(dict(cand=name, win=w, drop=f'lag1+slip{slip}', cagr=m1['cagr'], dd=m1['dd'], base=m0['cagr'], base_dd=m0['dd'], d_base=m1['cagr'] - m0['cagr']))
df = pd.DataFrame(rows)
df.to_csv('/home/user/dashboard/analysis/experiments/v_e_select_out/loco80.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for name in C:
    print(df[df.cand == name].round(2).to_string(index=False))
print('time', time.time() - t)
