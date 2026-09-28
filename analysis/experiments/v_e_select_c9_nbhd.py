"""v_e_select 检查：每个候选的 ±1 步邻域（thr±0.02、k±0.25、cap±0.05、vol 窗口 50/80），A/B/B去2024-02/F 及 F 上最差 LOCO，80 种子，相对 2/8/6 基线。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
b0 = {}
cases = [('A', ()), ('B', ()), ('B', ('2024-02',)), ('F', ())] + [('F', (m,)) for m in BIG]
for c in cases:
    b0[c] = stats(cfg_of(), c[0], drop=c[1])['cagr']
def row(label, cfg):
    r = dict(label=label)
    loco = []
    for c in cases:
        m = stats(cfg, c[0], drop=c[1]); d = m['cagr'] - b0[c]
        if c[0] == 'F' and c[1]:
            loco.append(d)
        else:
            r[c[0] + ('x' if c[1] else '')] = d; r[c[0] + ('x' if c[1] else '') + '_dd'] = m['dd']
    r['LOCO_min'] = min(loco); r['LOCO_max'] = max(loco)
    return r
C = {'C1': (0.38, 1.0, 0.9), 'C2': (0.38, 1.25, 0.9), 'C3': (0.40, 1.5, 1.0)}
rows = []
for n, (thr, k, cap) in C.items():
    rows.append(row(f'{n} center', cfg_of(thr, k, cap)))
    for dt in (-0.02, 0.02):
        rows.append(row(f'{n} thr{thr+dt:.2f}', cfg_of(round(thr + dt, 2), k, cap)))
    for dk in (-0.25, 0.25):
        rows.append(row(f'{n} k{k+dk:.2f}', cfg_of(thr, k + dk, cap)))
    for dc in (-0.05, 0.05):
        if cap + dc <= 1.0 + 1e-9:
            rows.append(row(f'{n} cap{cap+dc:.2f}', cfg_of(thr, k, round(cap + dc, 2))))
    for N in (50, 80):
        rows.append(row(f'{n} N{N}', cfg_of(thr, k, cap, vol=VI[N])))
    rows.append(row(f'{n} nofilter', cfg_of(None, k, cap)))
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(2).to_string(index=False))
df.to_csv('/home/user/dashboard/analysis/experiments/v_e_select_out/nbhd80.csv', index=False)
