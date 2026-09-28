"""v_c_exit 第 5 步：B 窗增益归因——止损放宽（-40% -> -70%）改变结局的信号；组合层面的成交差异。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *
from engine import run
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
D = getD()
t0, t1 = D.day(WIN['B'][0]), D.day_le(WIN['B'][1])
c1 = CANDS['C1']; c1s = dict(c1, sl=0.4)
e, xa, ra = my_exits(D, c1, t1); _, xb, rb = my_exits(D, c1s, t1); _, xc, rc = my_exits(D, CANDS['base'], t1)
sel = (D.entry >= t0)
dif = sel & (xa != xb)
S = D.sig
tab = pd.DataFrame({'code': S.code, 'date': S.signal_date, 'tier': S.tier, 'r_C1': ra - 1, 'hold_C1': xa - e, 'r_C1sl40': rb - 1, 'r_base': rc - 1,
                    'exit_C1': D.dates[xa], 'minr': [ (D.close[s, e[s]:xa[s]+1] / D.close[s, e[s]]).min() - 1 for s in range(len(e))]})[dif]
print('B 窗信号数', int(sel.sum()), ' 止损放宽后结局改变的信号', int(dif.sum()))
print(tab.sort_values('date').round(3).to_string())
print('这些信号 C1 平均收益 %.3f vs sl40 %.3f; 最低点 < -70%% 的: %d' % (tab.r_C1.mean(), tab.r_C1sl40.mean(), int((tab.minr <= -0.7).sum())))
# 组合层面：80 种子，按信号统计实际成交与 pnl 差
agg = {}
for name, rule in (('base', CANDS['base']), ('C1', c1), ('C1sl40', c1s)):
    Dx = patched(D, rule)
    pn = np.zeros(len(e)); cnt = np.zeros(len(e))
    for sd in range(80):
        r = run(Dx, Config(start=WIN['B'][0], record=True), sd)
        tr = r['trades']
        np.add.at(pn, tr.s.values, tr.pnl.values); np.add.at(cnt, tr.s.values, 1)
    agg[name] = (pn / 80, cnt / 80)
g = pd.DataFrame({'code': S.code, 'date': S.signal_date, 'ym': S.signal_date.str[:7],
                  **{f'pnl_{k}': v[0] / 1e4 for k, v in agg.items()}, **{f'freq_{k}': v[1] for k, v in agg.items()}})
g['d_C1_vs_sl40'] = g.pnl_C1 - g.pnl_C1sl40
g['d_C1_vs_base'] = g.pnl_C1 - g.pnl_base
g['sl_changed'] = dif
print('\n平均每种子 pnl 合计（万）:', {k: round(v[0].sum() / 1e4, 1) for k, v in agg.items()})
print('C1 - C1sl40 pnl 差合计 %.1f 万，其中止损改变结局的信号贡献 %.1f 万' % (g.d_C1_vs_sl40.sum(), g[g.sl_changed].d_C1_vs_sl40.sum()))
print(g.groupby('ym')[['d_C1_vs_sl40', 'd_C1_vs_base']].sum().round(1).query('abs(d_C1_vs_sl40) > 1 or abs(d_C1_vs_base) > 1').to_string())
print(g.reindex(g.d_C1_vs_sl40.abs().sort_values(ascending=False).index).head(20).round(2).to_string())
