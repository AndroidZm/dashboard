"""c_exit_lib.run_legs 复现 engine.run 的检查。"""
from c_exit_lib import *
from engine import run, run_seeds
D = getD()
fixed = {"kind": "fixed", "tp": 0.4, "sl": 0.4, "mh": None}
same2 = {"kind": "legs", "legs": [(0.5, fixed), (0.5, fixed)]}
trail_eq = {"kind": "trail", "act": 9.0, "trail": 0.5, "tp": 0.4, "sl": 0.4}   # 永不启动 = fixed
for win in "ABF":
    for seed in (0, 13, 55):
        c = cfg_for(win)
        a = run(D, c, seed); b = run_legs(D, c, fixed, seed); b2 = run_legs(D, c, same2, seed); b3 = run_legs(D, c, trail_eq, seed)
        print(win, seed, round(a['final'], 2), round(b['final'], 2), round(b2['final'], 2), round(b3['final'], 2),
              round(a['max_dd'], 5), round(b['max_dd'], 5), a['n_trades'], b['n_trades'])
    for kw in ({"tp": 0.6, "sl": None, "max_hold": 500},):
        c = cfg_for(win, **kw)
        a = run(D, c, 3); b = run_legs(D, cfg_for(win), {"kind": "fixed", "tp": 0.6, "sl": None, "mh": 500}, 3)
        print(win, 'tp.6 mh500', round(a['final'], 2), round(b['final'], 2))
    m = run_seeds(D, cfg_for(win)); m2 = run_seeds_legs(D, cfg_for(win), fixed)
    print(win, '80seeds', round(m['final']/1e4, 2), round(m2['final']/1e4, 2), round(m['cagr']*100, 2), round(m2['cagr']*100, 2), round(m['max_dd']*100, 2), round(m2['max_dd']*100, 2))
