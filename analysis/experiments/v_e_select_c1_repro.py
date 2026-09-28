"""v_e_select 检查 1+5：三个候选与基线 80 种子复现（A/B/F），分位与最差种子回撤。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
t = time.time()
C = {'base': cfg_of(), 'C1': cfg_of(0.38), 'C2': cfg_of(0.38, 1.25), 'C3': cfg_of(0.40, 1.5, 1.0)}
for name, c in C.items():
    for w in ('B', 'A', 'F'):
        m = stats(c, w)
        print(f'{name:5s} {w} {fmt(m)}', flush=True)
print('time', time.time() - t)
