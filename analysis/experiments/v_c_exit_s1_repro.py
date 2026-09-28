"""v_c_exit 第 1 步：独立出场实现与 engine 基线一致性；四个候选 80 种子 A/B/F 复现；与 c_exit_lib 出场数组对比。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *
from engine import run_seeds
D = getD()
t1 = len(D.dates) - 1
# 1) fixed 规则与 D.exits 完全一致（lag 0/1）
for lag in (0, 1):
    for end in (t1, D.day_le(WIN['A'][1])):
        a = D.exits(0.4, 0.4, None, end, lag); b = my_exits(D, CANDS['base'], end, lag)
        print('fixed==D.exits', lag, end, all((a[i] == b[i]).all() if i < 2 else abs(a[i]-b[i]).max() < 1e-12 for i in range(3)))
# 2) 引擎 80 种子基线
for w in 'BAF':
    m = run_seeds(D, cfg_for(w)); m2 = seeds_stats(D, CANDS['base'], cfg_for(w))
    print(w, 'engine', round(m['final']/1e4, 1), round(m['cagr']*100, 2), round(m['max_dd']*100, 1), '| patched', round(m2['final'], 1), round(m2['cagr'], 2), round(m2['dd'], 1))
# 3) 与 c_exit_lib 的出场数组对比
import c_exit_lib as CL
CL._D = D; D._rule_cache = {}
for n in ['C1', 'C2', 'C3', 'C4']:
    for end in (t1, D.day_le(WIN['A'][1])):
        e1, legs = CL.exits_for(D, CANDS[n], end)
        e2, x2, r2 = my_exits(D, CANDS[n], end)
        print(n, end, 'x same:', (legs[0][1] == x2).all(), 'n diff', int((legs[0][1] != x2).sum()))
# 4) 候选复现
rows = []
for n, r in CANDS.items():
    for w in 'BAF':
        m = seeds_stats(D, r, cfg_for(w))
        rows.append(dict(name=n, win=w, **{k: v for k, v in m.items() if k not in ('finals', 'cagrs')}))
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(2).to_string(index=False))
df.to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s1_repro.csv', index=False)
